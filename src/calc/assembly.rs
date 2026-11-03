//! Conformational search for molecular assemblies (dimers, clusters,
//! host–guest complexes, aggregates) by λ-continuation of soft-core
//! intermolecular interactions.
//!
//! Each trial
//!
//! 1. takes one conformer per molecule (the input geometry or an ETKDG
//!    conformer, minimized with the inner force field) and places the
//!    molecules with random orientations and random centroids inside a
//!    sphere, rejecting placements where atoms of different molecules come
//!    closer than [`PlacementOptions::min_distance`] (deterministic per trial
//!    seed);
//! 2. ([`AssemblyMode::LambdaContinuation`]) minimizes a
//!    [`SoftCoreBuilder`] force field along a schedule of increasing
//!    (λ_vdw, λ_ele): the intermolecular interactions start weak and soft,
//!    so molecules can slide past each other and rearrange, and become full
//!    strength at λ = 1. A flat-bottom wall on the fragment centroids
//!    (radius schedule) keeps the assembly together while the interactions
//!    are weak;
//! 3. minimizes the result without restraints with the plain inner builder
//!    (`ignore_interfrag_interactions = false`), so final energies are
//!    directly comparable with RDKit's.
//!
//! [`AssemblyMode::Direct`] skips step 2 (plain minimization from the same
//! placements) and serves as the baseline.
//!
//! Unique structures are found by energy and RMSD after Kabsch alignment,
//! trying all permutations of identical molecules (same molecular graph with
//! the same atom order) and of equivalent terminal atoms (e.g. the hydrogens
//! of a methyl group or of water). The binding energy is
//! `E(complex) − Σ_copies E(isolated optimized monomer)`, where the monomer
//! energy is the lowest minimized energy of that monomer's conformer pool.

use super::conformer::ConformerGenerator;
use super::{Frame, to_flat, to_points};
use crate::core::Molecule;
use crate::core::geometry::{V3, centroid, kabsch};
use crate::distgeom::{EmbedParams, EtkdgGenerator};
use crate::error::{Error, Result};
use crate::ff::softcore::{Confinement, SoftCoreBuilder, SoftCoreInner, SoftCoreParams};
use crate::optim::{self, MinimizeOptions, MinimizeResult, MinimizeStatus};
use crate::perception::rings::RingInfo;
use crate::tasks::{CtxObserver, Engine, JobContext, JobHandle};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// What to assemble.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum AssemblyInput {
    /// Molecules with copy counts. The complex lists the atoms molecule by
    /// molecule, copy by copy, in input order.
    Molecules(Vec<(Molecule, usize)>),
    /// One multi-fragment molecule; each connected component is one
    /// molecule. The atom order of the input is kept.
    Complex(Molecule),
}

/// Source of the per-molecule conformers.
#[derive(Debug, Clone)]
pub enum MonomerConformers {
    /// The input geometry of each molecule (minimized).
    Input,
    /// `num_conformers` ETKDG conformers per distinct molecule (conformer `i`
    /// = conformer `i` of RDKit `EmbedMultipleConfs` with `params`),
    /// minimized; each trial picks one at random per copy.
    Etkdg { params: EmbedParams, num_conformers: usize },
}

/// Search strategy of each trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssemblyMode {
    /// Soft-core λ-continuation followed by an unrestrained minimization.
    #[default]
    LambdaContinuation,
    /// Unrestrained full-strength minimization from the random placement.
    Direct,
}

/// Random rigid-body placement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacementOptions {
    /// Radius (Å) of the sphere the molecular centroids are drawn from.
    /// `None` = `(Σ_i (r_i + d_min/2)³)^(1/3)` with `r_i` the largest
    /// centroid–atom distance of molecule `i` (a dense packing volume).
    pub radius: Option<f64>,
    /// Minimum distance (Å) between atoms of different molecules.
    pub min_distance: f64,
    /// Attempts per molecule before the radius is enlarged by 10 %.
    pub max_attempts: usize,
}

impl Default for PlacementOptions {
    fn default() -> Self {
        PlacementOptions { radius: None, min_distance: 2.0, max_attempts: 200 }
    }
}

/// Flat-bottom wall on the fragment centroids during the λ stages; the
/// radius is interpolated linearly from `radius_start` (first λ step) to
/// `radius_end` (last λ step).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfinementSchedule {
    /// `None` = placement radius + 1 Å.
    pub radius_start: Option<f64>,
    /// `None` = same as the start radius.
    pub radius_end: Option<f64>,
    /// kcal/mol/Å².
    pub force_constant: f64,
}

impl Default for ConfinementSchedule {
    fn default() -> Self {
        ConfinementSchedule { radius_start: None, radius_end: None, force_constant: 10.0 }
    }
}

/// Geometric λ schedule from `start` to 1 in `steps` steps, the same value
/// for λ_vdw and λ_ele.
pub fn geometric_schedule(start: f64, steps: usize) -> Vec<(f64, f64)> {
    let steps = steps.max(1);
    if steps == 1 {
        return vec![(1.0, 1.0)];
    }
    let start = start.clamp(1e-12, 1.0);
    (0..steps)
        .map(|k| {
            let l = if k + 1 == steps { 1.0 } else { start * (1.0 / start).powf(k as f64 / (steps - 1) as f64) };
            (l, l)
        })
        .collect()
}

/// Options for [`assembly_search`].
#[derive(Debug, Clone)]
pub struct AssemblySearchOptions {
    /// Number of random placements (trials).
    pub num_trials: usize,
    /// Master seed; trial `t` uses a seed derived from `(seed, t)`.
    pub seed: u64,
    pub monomer_conformers: MonomerConformers,
    pub placement: PlacementOptions,
    pub mode: AssemblyMode,
    /// (λ_vdw, λ_ele) per continuation step (default
    /// [`geometric_schedule`]`(0.02, 12)`).
    pub lambda_schedule: Vec<(f64, f64)>,
    /// Soft-core α, p, n (`include_all_pairs` is always enabled during the
    /// search).
    pub softcore: SoftCoreParams,
    /// Minimizer options of every λ step.
    pub step_minimize: MinimizeOptions,
    /// Wall during the λ steps (`None` = no wall).
    pub confinement: Option<ConfinementSchedule>,
    /// Minimizer options of the final unrestrained minimization (and of the
    /// monomer minimizations).
    pub final_minimize: MinimizeOptions,
    /// Duplicates differ by less than this energy (kcal/mol) ...
    pub energy_tolerance: f64,
    /// ... and this RMSD (Å, after alignment and permutation).
    pub rmsd_threshold: f64,
    /// Include hydrogens in the RMSD (default `true`).
    pub rmsd_with_hydrogens: bool,
    /// Keep structures within this window above the minimum (kcal/mol).
    pub energy_window: f64,
    pub max_structures: Option<usize>,
    /// Two molecules are in contact if any atom pair is closer than this (Å)
    /// (for [`AssemblyStructure::connected`]).
    pub contact_distance: f64,
}

impl Default for AssemblySearchOptions {
    fn default() -> Self {
        AssemblySearchOptions {
            num_trials: 32,
            seed: 0x5eed_a55e_b1e5,
            monomer_conformers: MonomerConformers::Input,
            placement: PlacementOptions::default(),
            mode: AssemblyMode::LambdaContinuation,
            lambda_schedule: geometric_schedule(0.02, 12),
            softcore: SoftCoreParams::default(),
            step_minimize: MinimizeOptions { max_iters: 500, ..Default::default() },
            confinement: Some(ConfinementSchedule::default()),
            final_minimize: MinimizeOptions { max_iters: 5000, ..Default::default() },
            energy_tolerance: 0.01,
            rmsd_threshold: 0.25,
            rmsd_with_hydrogens: true,
            energy_window: f64::INFINITY,
            max_structures: None,
            contact_distance: 4.5,
        }
    }
}

/// One optimized assembly.
#[derive(Debug, Clone, PartialEq)]
pub struct AssemblyStructure {
    /// Trial (placement) it came from.
    pub trial: usize,
    /// Coordinates in the atom order of [`AssemblySearchResult::complex`].
    pub coords: Vec<V3>,
    /// Energy with the plain inner force field (intermolecular terms on).
    pub energy: f64,
    /// `energy − Σ_copies E(isolated optimized monomer)`.
    pub binding_energy: f64,
    /// All molecules are linked by contacts (see
    /// [`AssemblySearchOptions::contact_distance`]).
    pub connected: bool,
    /// Status of the final minimization.
    pub status: MinimizeStatus,
    /// Total minimizer iterations (λ steps + final).
    pub iterations: usize,
    /// Soft-core energy after each λ step (empty for [`AssemblyMode::Direct`]).
    pub lambda_energies: Vec<f64>,
}

/// Stage of an in-flight trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialStage {
    Placing,
    /// Minimizing at λ step `step` of the schedule.
    Lambda {
        step: usize,
    },
    /// Final unrestrained minimization.
    Final,
}

/// Current state of an in-flight trial.
#[derive(Debug, Clone, PartialEq)]
pub struct TrialProgress {
    pub stage: TrialStage,
    pub lambda_vdw: f64,
    pub lambda_ele: f64,
    /// Current coordinates and energy (soft-core energy incl. wall during
    /// the λ stages, plain force-field energy in the final stage).
    pub frame: Frame,
}

/// Snapshot of a running assembly search.
#[derive(Debug, Clone, Default)]
pub struct AssemblySearchSnapshot {
    /// `true` while monomer conformers are being prepared.
    pub preparing: bool,
    pub in_flight: BTreeMap<usize, TrialProgress>,
    /// Unique assemblies found so far, sorted by energy.
    pub found: Vec<AssemblyStructure>,
    pub completed: usize,
    pub failed: usize,
    pub total: usize,
}

impl AssemblySearchSnapshot {
    pub fn best(&self) -> Option<&AssemblyStructure> {
        self.found.first()
    }
}

/// One molecule of the assembly.
#[derive(Debug, Clone)]
pub struct AssemblyCopy {
    /// Index into [`AssemblySearchResult::monomers`].
    pub monomer: usize,
    /// Complex atom index of each monomer atom.
    pub atoms: Vec<usize>,
}

/// Result of an assembly search.
#[derive(Debug, Clone)]
pub struct AssemblySearchResult {
    /// The complex (all molecules; coordinates = the lowest structure, or
    /// the input arrangement if none was found).
    pub complex: Molecule,
    /// Distinct molecules.
    pub monomers: Vec<Molecule>,
    /// Lowest minimized energy of each distinct molecule.
    pub monomer_energies: Vec<f64>,
    pub copies: Vec<AssemblyCopy>,
    /// `Σ_copies monomer_energies[copy.monomer]`.
    pub reference_energy: f64,
    /// Unique structures sorted by energy (lowest first).
    pub structures: Vec<AssemblyStructure>,
    /// Every completed trial, sorted by trial index.
    pub all: Vec<AssemblyStructure>,
    pub failed: usize,
    pub cancelled: bool,
}

impl AssemblySearchResult {
    pub fn global_minimum(&self) -> Option<&AssemblyStructure> {
        self.structures.first()
    }
}

// ---------------------------------------------------------------------------
// molecule helpers
// ---------------------------------------------------------------------------

/// Appends `src` to `dst`, returning the new indices of `src`'s atoms.
fn append_molecule(dst: &mut Molecule, src: &Molecule, coords: &[V3]) -> Vec<usize> {
    let off = dst.num_atoms();
    for (i, a) in src.atoms().iter().enumerate() {
        dst.add_atom(a.clone(), coords[i]);
    }
    for b in src.bonds() {
        let nb = dst.add_bond(b.begin + off, b.end + off, b.order);
        let d = dst.bond_mut(nb);
        d.is_aromatic = b.is_aromatic;
        d.is_conjugated = b.is_conjugated;
        d.stereo = b.stereo;
        d.stereo_atoms = b.stereo_atoms.map(|(x, y)| (x + off, y + off));
    }
    (off..off + src.num_atoms()).collect()
}

/// Concatenates molecules (each `count` times, with its own coordinates)
/// into one multi-fragment molecule, in the same atom order as
/// [`AssemblyInput::Molecules`] uses for the complex.
pub fn combine_molecules(list: &[(Molecule, usize)]) -> Molecule {
    let mut cx = Molecule::new();
    for (m, count) in list {
        for _ in 0..*count {
            append_molecule(&mut cx, m, &m.coords);
        }
    }
    cx.name = "assembly".into();
    cx.ring_info = Some(RingInfo::compute(&cx));
    cx
}

/// The sub-molecule made of `atoms` (in that order).
fn extract_molecule(mol: &Molecule, atoms: &[usize]) -> Molecule {
    let mut map = vec![usize::MAX; mol.num_atoms()];
    let mut m = Molecule::new();
    for (k, &a) in atoms.iter().enumerate() {
        map[a] = k;
        m.add_atom(mol.atom(a).clone(), mol.coords[a]);
    }
    for b in mol.bonds() {
        if map[b.begin] != usize::MAX && map[b.end] != usize::MAX {
            let nb = m.add_bond(map[b.begin], map[b.end], b.order);
            let d = m.bond_mut(nb);
            d.is_aromatic = b.is_aromatic;
            d.is_conjugated = b.is_conjugated;
            d.stereo = b.stereo;
            d.stereo_atoms = b
                .stereo_atoms
                .and_then(|(x, y)| (map[x] != usize::MAX && map[y] != usize::MAX).then_some((map[x], map[y])));
        }
    }
    m.name = mol.name.clone();
    m.ring_info = Some(RingInfo::compute(&m));
    m
}

/// Same molecular graph with the same atom order.
fn same_graph(a: &Molecule, b: &Molecule) -> bool {
    a.num_atoms() == b.num_atoms()
        && a.num_bonds() == b.num_bonds()
        && a.atoms().iter().zip(b.atoms()).all(|(x, y)| {
            x.atomic_num == y.atomic_num
                && x.formal_charge == y.formal_charge
                && x.isotope == y.isotope
                && x.implicit_hs == y.implicit_hs
                && x.is_aromatic == y.is_aromatic
        })
        && a.bonds().iter().zip(b.bonds()).all(|(x, y)| x.begin == y.begin && x.end == y.end && x.order == y.order)
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn random_rotation(rng: &mut ChaCha8Rng) -> [[f64; 3]; 3] {
    let q = loop {
        let mut q = [0.0f64; 4];
        for v in &mut q {
            *v = rng.gen_range(-1.0..1.0);
        }
        let l2: f64 = q.iter().map(|v| v * v).sum();
        if l2 > 1e-3 && l2 <= 1.0 {
            let l = l2.sqrt();
            break q.map(|v| v / l);
        }
    };
    let [w, x, y, z] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

fn random_in_ball(rng: &mut ChaCha8Rng, r: f64) -> V3 {
    loop {
        let p = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
        if p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= 1.0 {
            return p.map(|v| v * r);
        }
    }
}

fn mat_vec(m: &[[f64; 3]; 3], v: V3) -> V3 {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

fn dist2(a: V3, b: V3) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn rec(cur: &mut Vec<usize>, used: &mut [bool], out: &mut Vec<Vec<usize>>) {
        if cur.len() == used.len() {
            out.push(cur.clone());
            return;
        }
        for i in 0..used.len() {
            if !used[i] {
                used[i] = true;
                cur.push(i);
                rec(cur, used, out);
                cur.pop();
                used[i] = false;
            }
        }
    }
    let mut out = Vec::new();
    rec(&mut Vec::new(), &mut vec![false; n], &mut out);
    out
}

// ---------------------------------------------------------------------------
// prepared system
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Monomer {
    mol: Molecule,
    /// Minimized conformers.
    confs: Vec<Vec<V3>>,
    /// Energy of each conformer.
    energies: Vec<f64>,
}

#[derive(Debug, Clone)]
struct CopyInfo {
    monomer: usize,
    atoms: Vec<usize>,
    /// Fixed conformer index (input geometry of this copy), else random.
    own_conf: Option<usize>,
}

/// Graph automorphisms of `m` (each maps atom `k` to atom `p[k]`),
/// preserving element, charge, hydrogen count, aromaticity and bond types;
/// at most `cap` are returned (the identity is always first).
fn automorphisms(m: &Molecule, cap: usize) -> Vec<Vec<usize>> {
    use std::hash::{Hash, Hasher};
    let n = m.num_atoms();
    let bt = |b: usize| {
        let bd = m.bond(b);
        if bd.is_aromatic { 4u8 } else { bd.order as u8 }
    };
    let h = |x: &dyn Fn(&mut std::collections::hash_map::DefaultHasher)| {
        let mut s = std::collections::hash_map::DefaultHasher::new();
        x(&mut s);
        s.finish()
    };
    let mut cls: Vec<u64> = (0..n)
        .map(|i| {
            let a = m.atom(i);
            h(&|s| (a.atomic_num, a.formal_charge, a.isotope, a.implicit_hs, a.is_aromatic, m.degree(i)).hash(s))
        })
        .collect();
    let count = |c: &[u64]| c.iter().collect::<std::collections::BTreeSet<_>>().len();
    let mut nc = count(&cls);
    for _ in 0..n {
        let next: Vec<u64> = (0..n)
            .map(|i| {
                let mut nb: Vec<(u64, u8)> = m.atom_bonds(i).iter().map(|&(j, b)| (cls[j], bt(b))).collect();
                nb.sort_unstable();
                h(&|s| (cls[i], &nb).hash(s))
            })
            .collect();
        let c = count(&next);
        cls = next;
        if c == nc {
            break;
        }
        nc = c;
    }
    // BFS order: every atom after a fragment's first has an earlier neighbor
    let mut order = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    for s in 0..n {
        if seen[s] {
            continue;
        }
        seen[s] = true;
        let mut k = order.len();
        order.push(s);
        while k < order.len() {
            let u = order[k];
            k += 1;
            for v in m.neighbors(u) {
                if !seen[v] {
                    seen[v] = true;
                    order.push(v);
                }
            }
        }
    }
    let mut pos_of = vec![0usize; n];
    for (k, &a) in order.iter().enumerate() {
        pos_of[a] = k;
    }
    struct St<'a> {
        m: &'a Molecule,
        order: &'a [usize],
        pos_of: &'a [usize],
        cls: &'a [u64],
        map: Vec<usize>,
        used: Vec<bool>,
        out: Vec<Vec<usize>>,
        cap: usize,
    }
    fn rec(st: &mut St, k: usize, bt: &dyn Fn(usize) -> u8) {
        if st.out.len() >= st.cap {
            return;
        }
        if k == st.order.len() {
            st.out.push(st.map.clone());
            return;
        }
        let i = st.order[k];
        // identity first so that it is always found
        let mut cands: Vec<usize> = (0..st.order.len()).filter(|&j| !st.used[j] && st.cls[j] == st.cls[i]).collect();
        if let Some(p) = cands.iter().position(|&j| j == i) {
            cands.swap(0, p);
        }
        for j in cands {
            let mut ok = true;
            let mut mapped_i = 0;
            for &(u, b) in st.m.atom_bonds(i) {
                if st.pos_of[u] < k {
                    mapped_i += 1;
                    match st.m.bond_between(st.map[u], j) {
                        Some(b2) if bt(b2) == bt(b) => {}
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            if !ok {
                continue;
            }
            let mapped_j = st.m.neighbors(j).filter(|&v| st.used[v]).count();
            if mapped_j != mapped_i {
                continue;
            }
            st.map[i] = j;
            st.used[j] = true;
            rec(st, k + 1, bt);
            st.used[j] = false;
            if st.out.len() >= st.cap {
                return;
            }
        }
    }
    let mut st = St {
        m,
        order: &order,
        pos_of: &pos_of,
        cls: &cls,
        map: vec![usize::MAX; n],
        used: vec![false; n],
        out: vec![],
        cap: cap.max(1),
    };
    rec(&mut st, 0, &bt);
    if st.out.is_empty() {
        st.out.push((0..n).collect());
    }
    st.out
}

/// Permutation-aware RMSD between two complexes: minimized over
/// permutations of identical molecules and over the graph automorphisms of
/// each molecule (e.g. ring rotations of benzene, methyl or water
/// hydrogens).
struct Matcher {
    copies: Vec<CopyInfo>,
    /// Monomer-local atoms used in the RMSD, per monomer.
    used: Vec<Vec<usize>>,
    /// Automorphisms per monomer.
    autos: Vec<Vec<Vec<usize>>>,
}

/// Automorphisms enumerated per molecule.
const MAX_AUTOMORPHISMS: usize = 1000;
/// Combinations (copy permutations × automorphisms) up to which all are
/// tried; above, alignments anchored on one molecule are used.
const MAX_EXHAUSTIVE: usize = 2000;

fn transform(rot: &[[f64; 3]; 3], t: V3, p: V3) -> V3 {
    let x = mat_vec(rot, p);
    [x[0] + t[0], x[1] + t[1], x[2] + t[2]]
}

impl Matcher {
    fn new(monomers: &[Monomer], copies: &[CopyInfo], with_h: bool) -> Self {
        let used = monomers
            .iter()
            .map(|m| (0..m.mol.num_atoms()).filter(|&k| with_h || m.mol.atom(k).atomic_num != 1).collect())
            .collect();
        let autos = monomers.iter().map(|m| automorphisms(&m.mol, MAX_AUTOMORPHISMS)).collect();
        Matcher { copies: copies.to_vec(), used, autos }
    }

    /// Atom pairs (A index, B index) for copy assignment `assign` (B copy of
    /// each A copy) and automorphism choice `auto` (per A copy).
    fn pairs(&self, assign: &[usize], auto: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let (mut ai, mut bi) = (Vec::new(), Vec::new());
        for (c, cp) in self.copies.iter().enumerate() {
            let p = &self.autos[cp.monomer][auto[c]];
            let cb = &self.copies[assign[c]].atoms;
            for &k in &self.used[cp.monomer] {
                ai.push(cp.atoms[k]);
                bi.push(cb[p[k]]);
            }
        }
        (ai, bi)
    }

    fn eval(&self, a: &[V3], b: &[V3], assign: &[usize], auto: &[usize]) -> (f64, [[f64; 3]; 3], V3) {
        let (ai, bi) = self.pairs(assign, auto);
        let pa: Vec<V3> = ai.iter().map(|&i| a[i]).collect();
        let pb: Vec<V3> = bi.iter().map(|&i| b[i]).collect();
        kabsch(&pa, &pb)
    }

    /// Σ squared distances of copy `c` of A to copy `cb` of B (transformed
    /// by `rot`, `t`) under automorphism `p`.
    fn copy_cost(&self, a: &[V3], b: &[V3], (c, cb): (usize, usize), p: &[usize], rot: &[[f64; 3]; 3], t: V3) -> f64 {
        let ca = &self.copies[c];
        let atoms_b = &self.copies[cb].atoms;
        self.used[ca.monomer].iter().map(|&k| dist2(a[ca.atoms[k]], transform(rot, t, b[atoms_b[p[k]]]))).sum()
    }

    fn best_auto(&self, a: &[V3], b: &[V3], c: usize, cb: usize, rot: &[[f64; 3]; 3], t: V3) -> (f64, usize) {
        let mon = self.copies[c].monomer;
        let mut best = (f64::INFINITY, 0);
        for (q, p) in self.autos[mon].iter().enumerate() {
            let x = self.copy_cost(a, b, (c, cb), p, rot, t);
            if x < best.0 - 1e-12 {
                best = (x, q);
            }
        }
        best
    }

    fn rmsd(&self, a: &[V3], b: &[V3]) -> f64 {
        let nc = self.copies.len();
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (k, c) in self.copies.iter().enumerate() {
            groups.entry(c.monomer).or_default().push(k);
        }
        let mut total: usize = 1;
        for g in groups.values() {
            total = total.saturating_mul((1..=g.len()).product::<usize>());
        }
        for c in &self.copies {
            total = total.saturating_mul(self.autos[c.monomer].len());
        }
        if total <= MAX_EXHAUSTIVE {
            // all copy permutations × automorphisms
            let mut assigns = vec![(0..nc).collect::<Vec<_>>()];
            for g in groups.values().filter(|g| g.len() > 1) {
                let ps = permutations(g.len());
                let mut next = Vec::with_capacity(assigns.len() * ps.len());
                for base in &assigns {
                    for p in &ps {
                        let mut q = base.clone();
                        for (i, &pi) in p.iter().enumerate() {
                            q[g[i]] = base[g[pi]];
                        }
                        next.push(q);
                    }
                }
                assigns = next;
            }
            let radix: Vec<usize> = self.copies.iter().map(|c| self.autos[c.monomer].len()).collect();
            let ncombo: usize = radix.iter().product();
            let mut best = f64::INFINITY;
            let mut auto = vec![0; nc];
            for asg in &assigns {
                for mut code in 0..ncombo {
                    for c in 0..nc {
                        auto[c] = code % radix[c];
                        code /= radix[c];
                    }
                    best = best.min(self.eval(a, b, asg, &auto).0);
                }
            }
            return best;
        }
        // anchored: align A copy 0 onto every same-type B copy under every
        // automorphism, assign the other copies optimally in that frame,
        // then refine with a global alignment
        let m0 = self.copies[0].monomer;
        let mut best = f64::INFINITY;
        for &b0 in &groups[&m0] {
            for q0 in 0..self.autos[m0].len() {
                let p0 = &self.autos[m0][q0];
                let ca = &self.copies[0];
                let cb = &self.copies[b0];
                let pa: Vec<V3> = self.used[m0].iter().map(|&k| a[ca.atoms[k]]).collect();
                let pb: Vec<V3> = self.used[m0].iter().map(|&k| b[cb.atoms[p0[k]]]).collect();
                let (_, rot, t) = kabsch(&pa, &pb);
                let mut assign = vec![usize::MAX; nc];
                let mut auto = vec![0; nc];
                assign[0] = b0;
                auto[0] = q0;
                for g in groups.values() {
                    let ra: Vec<usize> = g.iter().copied().filter(|&c| c != 0).collect();
                    let rb: Vec<usize> = g.iter().copied().filter(|&c| c != b0).collect();
                    if ra.is_empty() {
                        continue;
                    }
                    let cost: Vec<Vec<(f64, usize)>> =
                        ra.iter().map(|&c| rb.iter().map(|&x| self.best_auto(a, b, c, x, &rot, t)).collect()).collect();
                    if ra.len() <= 6 {
                        let mut bp = (f64::INFINITY, Vec::new());
                        for p in permutations(ra.len()) {
                            let s: f64 = p.iter().enumerate().map(|(i, &j)| cost[i][j].0).sum();
                            if s < bp.0 {
                                bp = (s, p);
                            }
                        }
                        for (i, &j) in bp.1.iter().enumerate() {
                            assign[ra[i]] = rb[j];
                            auto[ra[i]] = cost[i][j].1;
                        }
                    } else {
                        let mut taken = vec![false; rb.len()];
                        for i in 0..ra.len() {
                            let j = (0..rb.len())
                                .filter(|&j| !taken[j])
                                .min_by(|&x, &y| cost[i][x].0.total_cmp(&cost[i][y].0))
                                .unwrap();
                            taken[j] = true;
                            assign[ra[i]] = rb[j];
                            auto[ra[i]] = cost[i][j].1;
                        }
                    }
                }
                let (mut r, mut rot, mut t) = self.eval(a, b, &assign, &auto);
                for _ in 0..3 {
                    let mut changed = false;
                    for c in 0..nc {
                        let q = self.best_auto(a, b, c, assign[c], &rot, t).1;
                        if q != auto[c] {
                            auto[c] = q;
                            changed = true;
                        }
                    }
                    if !changed {
                        break;
                    }
                    let e = self.eval(a, b, &assign, &auto);
                    (r, rot, t) = (r.min(e.0), e.1, e.2);
                }
                best = best.min(r);
            }
        }
        best
    }
}

struct System {
    complex: Molecule,
    monomers: Vec<Monomer>,
    copies: Vec<CopyInfo>,
    monomer_energies: Vec<f64>,
    reference_energy: f64,
    placement_radius: f64,
    matcher: Matcher,
}

fn minimize_monomer<B: SoftCoreInner>(
    builder: &B,
    mol: &Molecule,
    coords: &[V3],
    opts: &MinimizeOptions,
) -> Result<(Vec<V3>, f64)> {
    let mut m = mol.clone();
    m.coords = coords.to_vec();
    let ff = builder.with_interfrag(true).build_dyn(&m, &[])?;
    let mut pos = to_flat(coords);
    let r = optim::minimize_with(&ff, &mut pos, opts, false, &mut optim::NoObserver);
    Ok((to_points(&pos), r.energy))
}

fn prepare<B: SoftCoreInner>(input: &AssemblyInput, builder: &B, opts: &AssemblySearchOptions) -> Result<System> {
    // distinct molecules + copies (complex atom indices) + input geometry per copy
    let mut monomers: Vec<Molecule> = Vec::new();
    let mut copies: Vec<(usize, Vec<usize>, Vec<V3>)> = Vec::new();
    let complex = match input {
        AssemblyInput::Molecules(list) => {
            let mut cx = Molecule::new();
            for (m, count) in list {
                if m.num_atoms() == 0 {
                    return Err(Error::InvalidArgument("assembly: empty molecule".into()));
                }
                if m.fragments().len() != 1 {
                    return Err(Error::InvalidArgument("assembly: each input molecule must be connected".into()));
                }
                let idx = match monomers.iter().position(|x| same_graph(x, m)) {
                    Some(i) => i,
                    None => {
                        monomers.push(m.clone());
                        monomers.len() - 1
                    }
                };
                for _ in 0..*count {
                    let atoms = append_molecule(&mut cx, m, &m.coords);
                    copies.push((idx, atoms, m.coords.clone()));
                }
            }
            cx.name = "assembly".into();
            cx.ring_info = Some(RingInfo::compute(&cx));
            cx
        }
        AssemblyInput::Complex(m) => {
            let mut frags = m.fragments();
            for f in &mut frags {
                f.sort_unstable();
            }
            frags.sort_by_key(|f| f[0]);
            for f in frags {
                let sub = extract_molecule(m, &f);
                let idx = match monomers.iter().position(|x| same_graph(x, &sub)) {
                    Some(i) => i,
                    None => {
                        monomers.push(sub.clone());
                        monomers.len() - 1
                    }
                };
                copies.push((idx, f, sub.coords.clone()));
            }
            let mut cx = m.clone();
            if cx.ring_info.is_none() {
                cx.ring_info = Some(RingInfo::compute(&cx));
            }
            cx
        }
    };
    if copies.len() < 2 {
        return Err(Error::InvalidArgument("assembly: at least two molecules are needed".into()));
    }
    if complex.coords.len() != complex.num_atoms() {
        return Err(Error::InvalidArgument("assembly: molecule has no coordinates".into()));
    }

    // starting geometries per monomer (and the fixed conformer of each copy)
    let mut starts: Vec<Vec<Vec<V3>>> = vec![Vec::new(); monomers.len()];
    let mut own: Vec<Option<usize>> = vec![None; copies.len()];
    match &opts.monomer_conformers {
        MonomerConformers::Input => {
            for (k, (mi, _, c)) in copies.iter().enumerate() {
                let pos = match starts[*mi].iter().position(|x| x == c) {
                    Some(p) => p,
                    None => {
                        starts[*mi].push(c.clone());
                        starts[*mi].len() - 1
                    }
                };
                own[k] = Some(pos);
            }
        }
        MonomerConformers::Etkdg { params, num_conformers } => {
            for (mi, m) in monomers.iter().enumerate() {
                let generator = EtkdgGenerator::new(params.clone());
                let confs: Vec<Option<Vec<V3>>> = (0..*num_conformers)
                    .into_par_iter()
                    .map(|id| generator.generate(m, id, &mut optim::NoObserver).ok())
                    .collect();
                starts[mi] = confs.into_iter().flatten().collect();
                if starts[mi].is_empty() {
                    return Err(Error::Embedding(format!(
                        "assembly: no conformer could be embedded for molecule {mi}"
                    )));
                }
            }
        }
    }
    let mut mons = Vec::with_capacity(monomers.len());
    for (mi, m) in monomers.into_iter().enumerate() {
        let res: Vec<Result<(Vec<V3>, f64)>> =
            starts[mi].par_iter().map(|c| minimize_monomer(builder, &m, c, &opts.final_minimize)).collect();
        let mut confs = Vec::new();
        let mut energies = Vec::new();
        for r in res {
            let (c, e) = r?;
            confs.push(c);
            energies.push(e);
        }
        mons.push(Monomer { mol: m, confs, energies });
    }
    let monomer_energies: Vec<f64> =
        mons.iter().map(|m| m.energies.iter().copied().fold(f64::INFINITY, f64::min)).collect();
    let copies: Vec<CopyInfo> = copies
        .into_iter()
        .zip(own)
        .map(|((monomer, atoms, _), own_conf)| CopyInfo { monomer, atoms, own_conf })
        .collect();
    let reference_energy: f64 = copies.iter().map(|c| monomer_energies[c.monomer]).sum();
    let placement_radius = match opts.placement.radius {
        Some(r) => r,
        None => {
            let half = 0.5 * opts.placement.min_distance;
            let v: f64 = copies
                .iter()
                .map(|c| {
                    let conf = &mons[c.monomer].confs[c.own_conf.unwrap_or(0)];
                    let cc = centroid(conf);
                    let r = conf.iter().map(|&p| dist2(p, cc).sqrt()).fold(0.0, f64::max);
                    (r + half).powi(3)
                })
                .sum();
            v.cbrt()
        }
    };
    let matcher = Matcher::new(&mons, &copies, opts.rmsd_with_hydrogens);
    Ok(System { complex, monomers: mons, copies, monomer_energies, reference_energy, placement_radius, matcher })
}

impl System {
    fn place(&self, opts: &AssemblySearchOptions, rng: &mut ChaCha8Rng) -> Vec<V3> {
        let mut coords = self.complex.coords.clone();
        let mut placed: Vec<V3> = Vec::new();
        let dmin2 = opts.placement.min_distance * opts.placement.min_distance;
        for c in &self.copies {
            let mon = &self.monomers[c.monomer];
            let k = c.own_conf.unwrap_or_else(|| rng.gen_range(0..mon.confs.len()));
            let conf = &mon.confs[k];
            let cc = centroid(conf);
            let local: Vec<V3> = conf.iter().map(|&p| [p[0] - cc[0], p[1] - cc[1], p[2] - cc[2]]).collect();
            let mut radius = self.placement_radius;
            let mut attempts = 0;
            let pts = loop {
                let rot = random_rotation(rng);
                let center = random_in_ball(rng, radius);
                let pts: Vec<V3> = local
                    .iter()
                    .map(|&p| {
                        let x = mat_vec(&rot, p);
                        [x[0] + center[0], x[1] + center[1], x[2] + center[2]]
                    })
                    .collect();
                if pts.iter().all(|&p| placed.iter().all(|&q| dist2(p, q) >= dmin2)) {
                    break pts;
                }
                attempts += 1;
                if attempts >= opts.placement.max_attempts.max(1) {
                    attempts = 0;
                    radius *= 1.1;
                }
            };
            for (k, &a) in c.atoms.iter().enumerate() {
                coords[a] = pts[k];
            }
            placed.extend(pts);
        }
        coords
    }

    fn connected(&self, coords: &[V3], contact: f64) -> bool {
        let n = self.copies.len();
        let c2 = contact * contact;
        let touch = |a: usize, b: usize| {
            self.copies[a].atoms.iter().any(|&i| self.copies[b].atoms.iter().any(|&j| dist2(coords[i], coords[j]) < c2))
        };
        let mut seen = vec![false; n];
        let mut stack = vec![0];
        seen[0] = true;
        while let Some(u) = stack.pop() {
            for v in 0..n {
                if !seen[v] && touch(u, v) {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
        seen.into_iter().all(|s| s)
    }
}

type Ctx = JobContext<AssemblySearchSnapshot>;

fn minimize_observed(
    ff: &dyn crate::ff::ForceField,
    pos: &mut [f64],
    mopts: &MinimizeOptions,
    job: Option<&Ctx>,
    trial: usize,
    stage: TrialStage,
    lambda: (f64, f64),
) -> MinimizeResult {
    match job {
        Some(ctx) => {
            ctx.update_silent(|s| {
                s.in_flight.insert(
                    trial,
                    TrialProgress {
                        stage,
                        lambda_vdw: lambda.0,
                        lambda_ele: lambda.1,
                        frame: Frame { coords: to_points(pos), energy: f64::NAN, iteration: 0 },
                    },
                );
            });
            let mut obs = CtxObserver::new(ctx, |ctx, info| {
                let coords = to_points(info.pos);
                let (e, it) = (info.energy, info.iteration);
                ctx.update(|s| {
                    s.in_flight.insert(
                        trial,
                        TrialProgress {
                            stage,
                            lambda_vdw: lambda.0,
                            lambda_ele: lambda.1,
                            frame: Frame { coords, energy: e, iteration: it },
                        },
                    );
                });
            });
            optim::minimize_with(ff, pos, mopts, false, &mut obs)
        }
        None => optim::minimize_with(ff, pos, mopts, false, &mut optim::NoObserver),
    }
}

fn run_trial<B: SoftCoreInner>(
    sys: &System,
    builder: &B,
    opts: &AssemblySearchOptions,
    trial: usize,
    job: Option<&Ctx>,
) -> Result<AssemblyStructure> {
    if let Some(ctx) = job {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        ctx.update_silent(|s| {
            s.in_flight.insert(
                trial,
                TrialProgress { stage: TrialStage::Placing, lambda_vdw: 0.0, lambda_ele: 0.0, frame: Frame::default() },
            );
        });
    }
    let mut rng = ChaCha8Rng::seed_from_u64(splitmix64(opts.seed ^ splitmix64(trial as u64)));
    let start = sys.place(opts, &mut rng);
    let mut mol = sys.complex.clone();
    mol.coords = start;
    let mut pos = to_flat(&mol.coords);
    let mut iterations = 0;
    let mut lambda_energies = Vec::new();
    if opts.mode == AssemblyMode::LambdaContinuation && !opts.lambda_schedule.is_empty() {
        let sched = &opts.lambda_schedule;
        let (r0, r1, k) = match opts.confinement {
            Some(c) => {
                let r0 = c.radius_start.unwrap_or(sys.placement_radius + 1.0);
                (r0, c.radius_end.unwrap_or(r0), Some(c.force_constant))
            }
            None => (f64::INFINITY, f64::INFINITY, None),
        };
        let sc = SoftCoreBuilder::new(builder.clone())
            .with_params(SoftCoreParams { include_all_pairs: true, ..opts.softcore })
            .with_confinement(k.map(|k| Confinement { radius: r0, force_constant: k }));
        let (ff, handle) = sc.build_with_handle(&mol, &[])?;
        for (step, &(lv, le)) in sched.iter().enumerate() {
            handle.set_lambda(lv, le);
            if k.is_some() {
                let f = if sched.len() > 1 { step as f64 / (sched.len() - 1) as f64 } else { 1.0 };
                handle.set_confinement_radius(r0 + (r1 - r0) * f);
            }
            let r = minimize_observed(
                &ff,
                &mut pos,
                &opts.step_minimize,
                job,
                trial,
                TrialStage::Lambda { step },
                (lv, le),
            );
            if r.status == MinimizeStatus::Cancelled {
                return Err(Error::Cancelled);
            }
            iterations += r.iterations;
            lambda_energies.push(r.energy);
        }
    }
    mol.coords = to_points(&pos);
    let ff = builder.with_interfrag(true).build_dyn(&mol, &[])?;
    let r = minimize_observed(ff.as_ref(), &mut pos, &opts.final_minimize, job, trial, TrialStage::Final, (1.0, 1.0));
    if r.status == MinimizeStatus::Cancelled {
        return Err(Error::Cancelled);
    }
    iterations += r.iterations;
    let coords = to_points(&pos);
    let connected = sys.connected(&coords, opts.contact_distance);
    Ok(AssemblyStructure {
        trial,
        coords,
        energy: r.energy,
        binding_energy: r.energy - sys.reference_energy,
        connected,
        status: r.status,
        iterations,
        lambda_energies,
    })
}

fn is_duplicate(m: &Matcher, a: &AssemblyStructure, b: &AssemblyStructure, opts: &AssemblySearchOptions) -> bool {
    (a.energy - b.energy).abs() < opts.energy_tolerance && m.rmsd(&a.coords, &b.coords) < opts.rmsd_threshold
}

fn insert_found(found: &mut Vec<AssemblyStructure>, s: AssemblyStructure, m: &Matcher, opts: &AssemblySearchOptions) {
    if let Some(p) = found.iter().position(|o| is_duplicate(m, o, &s, opts)) {
        if (s.energy, s.trial) < (found[p].energy, found[p].trial) {
            found[p] = s;
        } else {
            return;
        }
    } else {
        found.push(s);
    }
    found.sort_by(|a, b| a.energy.total_cmp(&b.energy).then(a.trial.cmp(&b.trial)));
}

fn deduplicate(mut all: Vec<AssemblyStructure>, m: &Matcher, opts: &AssemblySearchOptions) -> Vec<AssemblyStructure> {
    all.sort_by(|a, b| a.energy.total_cmp(&b.energy).then(a.trial.cmp(&b.trial)));
    let emin = all.first().map(|c| c.energy).unwrap_or(0.0);
    let mut out: Vec<AssemblyStructure> = Vec::new();
    for s in all {
        if s.energy - emin > opts.energy_window {
            break;
        }
        if !out.iter().any(|o| is_duplicate(m, o, &s, opts)) {
            out.push(s);
        }
        if let Some(mx) = opts.max_structures {
            if out.len() >= mx {
                break;
            }
        }
    }
    out
}

impl AssemblySearchResult {
    /// RMSD (Å) between two structures of this result, minimized over
    /// permutations of identical molecules and of equivalent terminal atoms.
    pub fn rmsd(&self, a: &[V3], b: &[V3], with_hydrogens: bool) -> f64 {
        let mons: Vec<Monomer> =
            self.monomers.iter().map(|m| Monomer { mol: m.clone(), confs: vec![], energies: vec![] }).collect();
        let copies: Vec<CopyInfo> = self
            .copies
            .iter()
            .map(|c| CopyInfo { monomer: c.monomer, atoms: c.atoms.clone(), own_conf: None })
            .collect();
        Matcher::new(&mons, &copies, with_hydrogens).rmsd(a, b)
    }
}

fn search<B: SoftCoreInner>(
    input: &AssemblyInput,
    builder: &B,
    opts: &AssemblySearchOptions,
    job: Option<&Ctx>,
) -> Result<AssemblySearchResult> {
    let sys = prepare(input, builder, opts)?;
    if let Some(ctx) = job {
        ctx.update(|s| s.preparing = false);
    }
    let results: Vec<Result<AssemblyStructure>> = (0..opts.num_trials)
        .into_par_iter()
        .map(|t| {
            let r = run_trial(&sys, builder, opts, t, job);
            if let Some(ctx) = job {
                ctx.update(|s| {
                    s.in_flight.remove(&t);
                    match &r {
                        Ok(st) => {
                            s.completed += 1;
                            insert_found(&mut s.found, st.clone(), &sys.matcher, opts);
                        }
                        Err(Error::Cancelled) => {}
                        Err(_) => s.failed += 1,
                    }
                });
                ctx.inc_progress();
            }
            r
        })
        .collect();
    let mut all = Vec::new();
    let mut failed = 0;
    let mut cancelled = false;
    for r in results {
        match r {
            Ok(s) => all.push(s),
            Err(Error::Cancelled) => cancelled = true,
            Err(_) => failed += 1,
        }
    }
    if all.is_empty() && !cancelled && opts.num_trials > 0 {
        return Err(Error::ForceField("assembly search: every trial failed".into()));
    }
    let structures = deduplicate(all.clone(), &sys.matcher, opts);
    let mut complex = sys.complex.clone();
    if let Some(b) = structures.first() {
        complex.coords = b.coords.clone();
    }
    Ok(AssemblySearchResult {
        complex,
        monomers: sys.monomers.iter().map(|m| m.mol.clone()).collect(),
        monomer_energies: sys.monomer_energies.clone(),
        copies: sys.copies.iter().map(|c| AssemblyCopy { monomer: c.monomer, atoms: c.atoms.clone() }).collect(),
        reference_energy: sys.reference_energy,
        structures,
        all,
        failed,
        cancelled,
    })
}

/// Synchronous assembly search (trials in parallel on the current rayon
/// pool). `builder` is the inner force field (MMFF or UFF); its
/// `ignore_interfrag_interactions` flag is ignored. Results do not depend
/// on the number of threads.
pub fn assembly_search<B: SoftCoreInner>(
    input: &AssemblyInput,
    builder: &B,
    opts: &AssemblySearchOptions,
) -> Result<AssemblySearchResult> {
    search(input, builder, opts, None)
}

impl Engine {
    /// Asynchronous assembly search. The snapshot shows every in-flight
    /// trial (stage, current λ, coordinates, energy) and the unique
    /// assemblies found so far; `cancel()` returns the completed trials.
    pub fn assembly_search<B: SoftCoreInner>(
        &self,
        input: AssemblyInput,
        builder: B,
        opts: AssemblySearchOptions,
    ) -> JobHandle<AssemblySearchResult, AssemblySearchSnapshot> {
        let n = opts.num_trials;
        self.spawn(AssemblySearchSnapshot { total: n, preparing: true, ..Default::default() }, n, move |ctx| {
            search(&input, &builder, &opts, Some(&ctx))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_is_geometric() {
        let s = geometric_schedule(0.02, 12);
        assert_eq!(s.len(), 12);
        assert!((s[0].0 - 0.02).abs() < 1e-15);
        assert_eq!(s[11], (1.0, 1.0));
        for w in s.windows(2) {
            assert!(w[1].0 > w[0].0);
        }
    }

    #[test]
    fn permutation_counts() {
        assert_eq!(permutations(3).len(), 6);
        assert_eq!(permutations(1), vec![vec![0]]);
    }
}
