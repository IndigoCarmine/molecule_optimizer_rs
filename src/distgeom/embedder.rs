//! The ETKDG embedding driver.
//!
//! Port of RDKit `Code/GraphMol/DistGeomHelpers/Embedder.cpp` (legacy,
//! sequential implementation used by ETKDGv3: `embedPoints`,
//! `generateInitialCoords`, `firstMinimization`, `minimizeFourthDimension`,
//! `minimizeWithExpTorsions`, the chirality / double-bond checks,
//! `findChiralSets`, `findDoubleBonds`, `setupInitialBoundsMatrix`,
//! `initETKDG`, the per-conformer seeding of `embedHelper_` and the RMS
//! pruning of `EmbedMultipleConfs`).
//! RDKit: Copyright (C) 2004-2026 Greg Landrum and other RDKit contributors,
//! BSD-3-Clause.

use super::EmbedParams;
use super::bounds::BoundsMatrix;
use super::bounds_builder::{TopolOptions, collect_bonds_and_angles, set_topol_bounds};
use super::contribs::calc_chiral_volume;
use super::dg_utils::{SymmMatrix, compute_initial_coords, compute_random_coords, pick_random_dist_mat};
use super::force_fields::*;
use super::point::{P3, compute_dihedral_angle};
use super::rng::{DoubleSource, unseeded_source};
use super::torsion_prefs::{EtkdgDetails, ForceConsts, TorsionOptions, get_experimental_torsions};
use crate::core::{BondOrder, BondStereo, ChiralTag, Molecule};
use crate::error::{Error, Result};
use crate::ff::{ContribForceField, ForceField};
use crate::optim::{MinimizeOptions, MinimizeStatus, Observer, bfgs::minimize_bfgs};

const ERROR_TOL: f64 = 0.00001;
const MAX_MINIMIZED_E_PER_ATOM: f64 = 0.05;
const MIN_TETRAHEDRAL_CHIRAL_VOL: f64 = 0.50;
const TETRAHEDRAL_CENTERINVOLUME_TOL: f64 = 0.30;

/// Everything needed to embed one connected fragment (shared by all
/// conformers; `Send + Sync`).
#[derive(Debug, Clone)]
pub struct FragmentSetup {
    /// Atom indices of this fragment in the input molecule (ascending).
    pub atoms: Vec<usize>,
    /// The fragment as a molecule (atoms renumbered 0..n).
    pub mol: Molecule,
    /// Smoothed bounds matrix.
    pub bounds: BoundsMatrix,
    pub chiral_centers: Vec<ChiralSet>,
    pub tetrahedral_centers: Vec<ChiralSet>,
    pub double_bond_ends: Vec<(usize, usize, usize)>,
    pub stereo_double_bonds: Vec<([usize; 4], i32)>,
    pub details: EtkdgDetails,
    pub four_d: bool,
}

/// Builds the fragment molecules (RDKit `getMolFrags`).
pub(crate) fn split_fragments(mol: &Molecule, separately: bool) -> Vec<(Vec<usize>, Molecule)> {
    let mut frags = if separately { mol.fragments() } else { vec![(0..mol.num_atoms()).collect()] };
    for f in frags.iter_mut() {
        f.sort_unstable();
    }
    frags.sort_by_key(|f| f[0]);
    if frags.len() == 1 && frags[0].len() == mol.num_atoms() {
        let mut m = mol.clone();
        m.rings();
        return vec![(frags.remove(0), m)];
    }
    let mut out = Vec::new();
    for f in frags {
        let mut map = vec![usize::MAX; mol.num_atoms()];
        let mut m = Molecule::new();
        for (k, &a) in f.iter().enumerate() {
            map[a] = k;
            m.add_atom(mol.atom(a).clone(), mol.pos(a));
        }
        for b in mol.bonds() {
            if map[b.begin] != usize::MAX {
                let bi = m.add_bond(map[b.begin], map[b.end], b.order);
                let nb = m.bond_mut(bi);
                nb.is_aromatic = b.is_aromatic;
                nb.is_conjugated = b.is_conjugated;
                nb.stereo = b.stereo;
                nb.stereo_atoms = b.stereo_atoms.map(|(x, y)| (map[x], map[y]));
            }
        }
        m.rings();
        out.push((f, m));
    }
    out
}

/// RDKit `findChiralSets` (no coordMap, no atropisomers).
pub fn find_chiral_sets(mol: &Molecule) -> (Vec<ChiralSet>, Vec<ChiralSet>) {
    let ri = mol.ring_info().cloned().unwrap_or_else(|| crate::perception::rings::RingInfo::compute(mol));
    let mut chiral = Vec::new();
    let mut tetra = Vec::new();
    for (i, atom) in mol.atoms().iter().enumerate() {
        if atom.atomic_num == 1 {
            continue;
        }
        let tag = atom.chiral_tag;
        if !(tag != ChiralTag::Unspecified || ((atom.atomic_num == 6 || atom.atomic_num == 7) && mol.degree(i) == 4)) {
            continue;
        }
        let mut nbrs: Vec<usize> = mol.neighbors(i).collect();
        if nbrs.len() < 3 {
            continue;
        }
        let (mut lo, hi) = (5.0, 100.0);
        if nbrs.len() < 4 {
            lo = 2.0;
            nbrs.push(i);
        }
        let n_small = ri.atom_ring_ids(i).iter().filter(|&&r| ri.atom_rings[r].len() < 5).count();
        let fused = n_small > 1;
        let idx = [nbrs[0], nbrs[1], nbrs[2], nbrs[3]];
        match tag {
            ChiralTag::Ccw => chiral.push(ChiralSet { idx0: i, idx, lower: lo, upper: hi, in_fused_small_rings: fused }),
            ChiralTag::Cw => chiral.push(ChiralSet { idx0: i, idx, lower: -hi, upper: -lo, in_fused_small_rings: fused }),
            ChiralTag::Unspecified => {
                if ri.num_atom_rings(i) < 2 || ri.is_atom_in_ring_of_size(i, 3) {
                    continue;
                }
                tetra.push(ChiralSet { idx0: i, idx, lower: 0.0, upper: 0.0, in_fused_small_rings: fused });
            }
        }
    }
    (chiral, tetra)
}

/// RDKit `findDoubleBonds`.
#[allow(clippy::type_complexity)]
pub fn find_double_bonds(mol: &Molecule) -> (Vec<(usize, usize, usize)>, Vec<([usize; 4], i32)>) {
    let mut ends = Vec::new();
    let mut stereo = Vec::new();
    for b in mol.bonds() {
        if b.order != BondOrder::Double {
            continue;
        }
        for (atm, oatm) in [(b.begin, b.end), (b.end, b.begin)] {
            if mol.degree(atm) < 2 {
                continue;
            }
            for &(nbr, ob) in mol.atom_bonds(atm) {
                if nbr == oatm {
                    continue;
                }
                if mol.bond(ob).order != BondOrder::Single && mol.degree(atm) == 2 {
                    continue;
                }
                ends.push((nbr, atm, oatm));
            }
        }
        if b.stereo != BondStereo::None {
            if let Some((s0, s1)) = b.stereo_atoms {
                let sign = if b.stereo == BondStereo::Cis { -1 } else { 1 };
                stereo.push(([s0, b.begin, b.end, s1], sign));
            }
        }
    }
    (ends, stereo)
}

fn topol_opts(p: &EmbedParams, set15: bool) -> TopolOptions {
    TopolOptions {
        scale_vdw: !set15,
        set15bounds: set15,
        set14bounds: true,
        set13bounds: true,
        use_macrocycle14config: p.use_macrocycle14config,
        force_trans_amides: p.force_trans_amides,
    }
}

/// RDKit `setupInitialBoundsMatrix` (without coordMap). Returns bounds and
/// the 1-4 path configurations.
pub fn setup_bounds(mol: &Molecule, p: &EmbedParams) -> Result<(BoundsMatrix, Vec<super::bounds_builder::Path14>)> {
    let n = mol.num_atoms();
    let mut mmat = BoundsMatrix::new_initialized(n);
    let paths = set_topol_bounds(mol, &mut mmat, &topol_opts(p, true));
    if !mmat.triangle_smooth(0.0) {
        mmat = BoundsMatrix::new_initialized(n);
        set_topol_bounds(mol, &mut mmat, &topol_opts(p, false));
        if !mmat.triangle_smooth(0.0) {
            if p.ignore_smoothing_failures {
                mmat = BoundsMatrix::new_initialized(n);
                set_topol_bounds(mol, &mut mmat, &topol_opts(p, false));
            } else {
                return Err(Error::Embedding("could not triangle bounds smooth molecule".into()));
            }
        }
    }
    Ok((mmat, paths))
}

/// Prepares one fragment.
pub fn setup_fragment(atoms: Vec<usize>, mol: Molecule, p: &EmbedParams) -> Result<FragmentSetup> {
    let (bounds, path14) = setup_bounds(&mol, p)?;
    let mut details = EtkdgDetails {
        force_consts: ForceConsts::default(),
        bounds_mat_force_scaling: p.bounds_mat_force_scaling,
        path14,
        ..Default::default()
    };
    if p.use_exp_torsion_angle_prefs || p.use_basic_knowledge {
        let (bonds, angles) = collect_bonds_and_angles(&mol);
        details.bonds = bonds;
        details.angles = angles;
        get_experimental_torsions(
            &mol,
            &mut details,
            &TorsionOptions {
                use_exp_torsions: p.use_exp_torsion_angle_prefs,
                use_small_ring_torsions: p.use_small_ring_torsions,
                use_macrocycle_torsions: p.use_macrocycle_torsions,
                use_basic_knowledge: p.use_basic_knowledge,
                version: p.et_version,
            },
        )?;
        details.atom_nums = mol.atoms().iter().map(|a| a.atomic_num).collect();
    }
    let (chiral_centers, tetrahedral_centers) = find_chiral_sets(&mol);
    let (double_bond_ends, stereo_double_bonds) = find_double_bonds(&mol);
    let four_d = p.use_random_coords || !chiral_centers.is_empty();
    Ok(FragmentSetup { atoms, mol, bounds, chiral_centers, tetrahedral_centers, double_bond_ends, stereo_double_bonds, details, four_d })
}

// ------------------------------------------------------------------ checks

fn pt(pos: &[f64], i: usize, dim: usize) -> P3 {
    P3::from_pos(pos, i, dim)
}

fn volume_test(cs: &ChiralSet, pos: &[f64], dim: usize) -> bool {
    let p0 = pt(pos, cs.idx0, dim);
    let mut v = [P3::default(); 4];
    for k in 0..4 {
        v[k] = p0 - pt(pos, cs.idx[k], dim);
        v[k].normalize();
    }
    let scale = if cs.in_fused_small_rings { 0.25 } else { 1.0 };
    let lim = scale * MIN_TETRAHEDRAL_CHIRAL_VOL;
    for (a, b, c) in [(0, 1, 2), (0, 1, 3), (0, 2, 3)] {
        if v[a].cross(&v[b]).dot(&v[c]).abs() < lim {
            return false;
        }
    }
    v[1].cross(&v[2]).dot(&v[3]).abs() >= lim
}

fn same_side(v1: P3, v2: P3, v3: P3, v4: P3, p0: P3, tol: f64) -> bool {
    let normal = (v2 - v1).cross(&(v3 - v1));
    let d1 = normal.dot(&(v4 - v1));
    let d2 = normal.dot(&(p0 - v1));
    if d1.abs() < tol || d2.abs() < tol {
        return false;
    }
    !((d1 < 0.0) ^ (d2 < 0.0))
}

fn center_in_volume(cs: &ChiralSet, pos: &[f64], dim: usize, tol: f64) -> bool {
    if cs.idx0 == cs.idx[3] {
        return true;
    }
    let p0 = pt(pos, cs.idx0, dim);
    let [p1, p2, p3, p4] = cs.idx.map(|i| pt(pos, i, dim));
    same_side(p1, p2, p3, p4, p0, tol)
        && same_side(p2, p3, p4, p1, p0, tol)
        && same_side(p3, p4, p1, p2, p0, tol)
        && same_side(p4, p1, p2, p3, p0, tol)
}

fn check_tetrahedral_centers(f: &FragmentSetup, pos: &[f64], dim: usize) -> bool {
    f.tetrahedral_centers
        .iter()
        .all(|t| volume_test(t, pos, dim) && center_in_volume(t, pos, dim, TETRAHEDRAL_CENTERINVOLUME_TOL))
}

fn check_chiral_centers(f: &FragmentSetup, pos: &[f64], dim: usize) -> bool {
    let opp = |a: f64, b: f64| a.is_sign_negative() ^ b.is_sign_negative();
    for cs in &f.chiral_centers {
        let vol = calc_chiral_volume(cs.idx[0], cs.idx[1], cs.idx[2], cs.idx[3], pos, dim);
        let (lb, ub) = (cs.lower, cs.upper);
        if (lb > 0.0 && vol < lb && (vol / lb < 0.8 || opp(vol, lb))) || (ub < 0.0 && vol > ub && (vol / ub < 0.8 || opp(vol, ub))) {
            return false;
        }
    }
    true
}

fn bounds_fulfilled(atoms: &[usize], mmat: &BoundsMatrix, pos: &[f64], dim: usize) -> bool {
    for i in 0..atoms.len().saturating_sub(1) {
        for j in i + 1..atoms.len() {
            let (a1, a2) = (atoms[i], atoms[j]);
            let d = (pt(pos, a1, dim) - pt(pos, a2, dim)).length();
            let lb = mmat.lower(a1, a2);
            let ub = mmat.upper(a1, a2);
            if (d < lb && (d - lb).abs() > 0.1 * ub) || (d > ub && (d - ub).abs() > 0.1 * ub) {
                return false;
            }
        }
    }
    true
}

fn final_chiral_checks(f: &FragmentSetup, pos: &[f64], dim: usize) -> bool {
    if !check_chiral_centers(f, pos, dim) {
        return false;
    }
    let mut atoms = std::collections::BTreeSet::new();
    for cs in &f.chiral_centers {
        if cs.idx0 != cs.idx[3] {
            atoms.insert(cs.idx0);
            atoms.extend(cs.idx);
        }
    }
    let atoms: Vec<usize> = atoms.into_iter().collect();
    if !atoms.is_empty() && !bounds_fulfilled(&atoms, &f.bounds, pos, dim) {
        return false;
    }
    f.chiral_centers.iter().all(|cs| center_in_volume(cs, pos, dim, 0.1))
}

fn double_bond_geometry_checks(f: &FragmentSetup, pos: &[f64], dim: usize) -> bool {
    const LINEAR_TOL: f64 = 1e-3;
    for &(a0, a1, a2) in &f.double_bond_ends {
        let (p0, p1, p2) = (pt(pos, a0, dim), pt(pos, a1, dim), pt(pos, a2, dim));
        let mut v1 = p1 - p0;
        v1.normalize();
        let mut v2 = p1 - p2;
        v2.normalize();
        if v1.dot(&v2) + 1.0 < LINEAR_TOL {
            return false;
        }
    }
    true
}

fn double_bond_stereo_checks(f: &FragmentSetup, pos: &[f64], dim: usize) -> bool {
    for (a, sign) in &f.stereo_double_bonds {
        let d = compute_dihedral_angle(pt(pos, a[0], dim), pt(pos, a[1], dim), pt(pos, a[2], dim), pt(pos, a[3], dim));
        if (d - std::f64::consts::FRAC_PI_2) * (*sign as f64) < 0.0 {
            return false;
        }
    }
    true
}

// ------------------------------------------------------------------ minimization

/// RDKit `ForceField::minimize(maxIts, forceTol)`; returns `needMore`.
fn ff_minimize(ff: &ContribForceField, pos: &mut [f64], max_its: usize, tol: f64, obs: &mut dyn Observer) -> Result<bool> {
    if ff.num_contribs() == 0 {
        return Ok(false);
    }
    let opts = MinimizeOptions { max_iters: max_its, force_tol: tol, ..Default::default() };
    let r = minimize_bfgs(ff, pos, &opts, obs);
    match r.status {
        MinimizeStatus::Cancelled => Err(Error::Cancelled),
        MinimizeStatus::MaxIterations => Ok(true),
        MinimizeStatus::Converged => Ok(false),
    }
}

fn first_minimization(f: &FragmentSetup, pos: &mut [f64], dim: usize, p: &EmbedParams, basin: f64, obs: &mut dyn Observer) -> Result<bool> {
    let fc = &f.details.force_consts;
    let ff = construct_force_field(&f.bounds, dim, &f.chiral_centers, fc.distance, fc.chiral, 0.1, basin);
    if ff.energy(pos) > ERROR_TOL {
        while ff_minimize(&ff, pos, 400, p.optimizer_force_tol, obs)? {}
    }
    let e = ff.energy(pos);
    Ok(e / (f.mol.num_atoms() as f64) < MAX_MINIMIZED_E_PER_ATOM)
}

fn minimize_fourth_dimension(f: &FragmentSetup, pos: &mut [f64], dim: usize, p: &EmbedParams, basin: f64, obs: &mut dyn Observer) -> Result<bool> {
    let fc = &f.details.force_consts;
    let ff = construct_force_field(&f.bounds, dim, &f.chiral_centers, fc.distance, 0.2, fc.fourth_dim, basin);
    if ff.energy(pos) > ERROR_TOL {
        while ff_minimize(&ff, pos, 200, p.optimizer_force_tol, obs)? {}
    }
    Ok(true)
}

fn minimize_with_exp_torsions(f: &FragmentSetup, pos: &mut [f64], dim: usize, p: &EmbedParams, obs: &mut dyn Observer) -> Result<bool> {
    let n = f.mol.num_atoms();
    let mut p3: Vec<f64> = (0..n).flat_map(|i| [pos[i * dim], pos[i * dim + 1], pos[i * dim + 2]]).collect();
    let ff = if p.use_basic_knowledge {
        construct_3d_force_field(&f.bounds, &p3, &f.details)
    } else {
        construct_plain_3d_force_field(&f.bounds, &p3, &f.details)
    };
    if ff.energy(&p3) > ERROR_TOL {
        ff_minimize(&ff, &mut p3, 300, p.optimizer_force_tol, obs)?;
    }
    let mut planar = true;
    if p.use_basic_knowledge {
        let ff2 = construct_3d_improper_force_field(n, &f.details);
        let n_centers = f.details.improper_atoms.len() as f64;
        if ff2.energy(&p3) > n_centers * 0.7 {
            planar = false;
        }
    }
    for i in 0..n {
        pos[i * dim..i * dim + 3].copy_from_slice(&p3[i * 3..i * 3 + 3]);
    }
    Ok(planar)
}

/// Which stage an embedding attempt reached (for diagnostics/tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StopAfter {
    /// Return right after the initial (metric-matrix / random) coordinates.
    pub initial_only: bool,
}

/// RDKit `embedPoints` for one fragment. Returns flat coordinates of
/// dimension `dim` (3 or 4) on success.
pub fn embed_points(f: &FragmentSetup, p: &EmbedParams, seed: i32, stop: StopAfter, obs: &mut dyn Observer) -> Result<Option<(Vec<f64>, usize)>> {
    let n = f.mol.num_atoms();
    let dim = if f.four_d { 4 } else { 3 };
    let max_iters = if p.max_iterations == 0 { 10 * n } else { p.max_iterations };
    let basin = if p.use_random_coords { 1e8 } else { p.basin_thresh };
    let mut rng = if seed > -1 { DoubleSource::new(seed as u32) } else { unseeded_source() };
    let mut pos = vec![0.0; n * dim];
    let mut dist_mat = SymmMatrix::new(n, 0.0);
    for _ in 0..max_iters {
        let got = if p.use_random_coords {
            let bs = if p.box_size_mult > 0.0 { 5.0 * p.box_size_mult } else { -p.box_size_mult };
            compute_random_coords(&mut pos, bs, &mut rng)
        } else {
            pick_random_dist_mat(&f.bounds, &mut dist_mat, &mut rng);
            compute_initial_coords(&dist_mat, &mut pos, dim, &mut rng, p.rand_neg_eig, p.num_zero_fail)
        };
        if !got {
            continue;
        }
        if stop.initial_only {
            return Ok(Some((pos, dim)));
        }
        let mut ok = first_minimization(f, &mut pos, dim, p, basin, obs)?;
        if ok {
            ok = check_tetrahedral_centers(f, &pos, dim);
        }
        if ok && p.enforce_chirality && !f.chiral_centers.is_empty() {
            ok = check_chiral_centers(f, &pos, dim);
        }
        if ok && (!f.chiral_centers.is_empty() || p.use_random_coords) {
            ok = minimize_fourth_dimension(f, &mut pos, dim, p, basin, obs)?;
        }
        if ok && (p.use_exp_torsion_angle_prefs || p.use_basic_knowledge) {
            ok = minimize_with_exp_torsions(f, &mut pos, dim, p, obs)?;
        }
        if ok {
            ok = double_bond_geometry_checks(f, &pos, dim);
        }
        if p.enforce_chirality && ok {
            if !f.chiral_centers.is_empty() {
                ok = final_chiral_checks(f, &pos, dim);
            }
            if ok && !f.stereo_double_bonds.is_empty() {
                ok = double_bond_stereo_checks(f, &pos, dim);
            }
        }
        if ok {
            return Ok(Some((pos, dim)));
        }
    }
    Ok(None)
}

/// Per-conformer seed used by RDKit's `embedHelper_` for conformer `ci`.
pub fn conformer_seed(random_seed: i32, ci: usize, sequential: bool) -> i32 {
    if random_seed <= -1 {
        return -1;
    }
    if sequential {
        return random_seed.wrapping_add(ci as i32 + 1);
    }
    let a = (ci + 1) as i64;
    let b = random_seed as i64;
    let max = i32::MAX as i64;
    if a == 0 || b == 0 || a <= max / b {
        (a * b) as i32
    } else {
        let big_seed = random_seed as u64;
        let max_val = ((ci + 1) as u64).max(big_seed);
        let big_num = big_seed.wrapping_add(max_val.wrapping_mul((ci + 1) as u64));
        let mask = 0x7fff_ffffu64;
        let folded = (big_num & mask) ^ (big_num >> 31);
        (folded & mask) as i32
    }
}
