//! Approximate nonbonded evaluation: cutoffs, buffered cluster-pair lists,
//! periodic boundary conditions and particle-mesh Ewald.
//!
//! [`ApproxNonbonded`] evaluates the van der Waals and electrostatic terms
//! of MMFF94 / UFF with an interaction cutoff:
//!
//! * pairs are found with a buffered cluster-pair list (clusters of
//!   four atoms, `rlist = cutoff + buffer`; full list memory is `O(N²/16)`
//!   with `Full` electrostatics, which needs no rebuilds), rebuilt
//!   lazily whenever an atom moved more than `buffer / 2` since the last
//!   build (checked on every evaluation and in
//!   [`NonbondedBackend::on_accepted_step`]);
//! * 4×4 cluster kernels use `wide::f64x4`; scalar per-pair reference
//!   kernels are used when SIMD is disabled;
//! * electrostatics: [`Electrostatics::Full`] (all pairs, no cutoff,
//!   isolated systems only), [`Electrostatics::ReactionField`] or
//!   [`Electrostatics::Pme`] (smooth PME, periodic systems only);
//! * van der Waals: [`VdwModifier::PotentialShift`] or
//!   [`VdwModifier::Switch`]; `Switch { r_switch }` with `r_switch ≥
//!   cutoff` is a plain truncation (no modification).
//!
//! # Parameters and exclusions
//!
//! Per-type-pair van der Waals parameters, partial charges and exclusions
//! follow the wrapped builder ([`MmffBuilder`] / [`UffBuilder`]): 1-2 and
//! 1-3 pairs are excluded; MMFF 1-4 pairs interact with full van der Waals
//! and 0.75 × electrostatics (evaluated in an explicit pair list); UFF 1-4
//! pairs are ordinary pairs and UFF has no electrostatics;
//! `ignore_interfrag_interactions` removes pairs between different
//! connected components (not supported with PME for multi-fragment
//! systems, since the reciprocal sum cannot exclude them). The build-time
//! distance thresholds of the RDKit-compatible builders
//! (`nonbonded_thresh`, `vdw_thresh`) are replaced by the cutoff.
//!
//! # Electrostatic functional forms
//!
//! MMFF uses `332.0716 qi qj / (D (r + 0.05))` (or `/(D (r + 0.05)²)` with
//! the distance-dependent dielectric). With `Full` and `ReactionField` the
//! buffered distance `ρ = r + 0.05` is kept (reaction field:
//! `f = 1/ρ^n + k r² - c`, `k = κ n / (rc ρc^(n+1))`, `κ = (eps_rf - D)/(2 eps_rf + D)`,
//! `c = 1/ρc^n + k rc²`, so that for `eps_rf = ∞` energy and force vanish
//! at the cutoff). The buffer is incompatible with the Ewald
//! splitting `1/r = erfc(βr)/r + erf(βr)/r`, so **PME uses the plain
//! Coulomb law `332.0716 qi qj / (D r)`** for all pairs (including 1-4
//! pairs, scaled by 0.75); the relative difference to the buffered form,
//! `≈ 0.05/r` (0.5 % at 10 Å), is ignored. The distance-dependent dielectric
//! is rejected with PME.
//!
//! # Periodic systems
//!
//! The cell is taken from [`Molecule::cell`]. Nonbonded interactions use
//! the minimum image; the cutoff must not exceed half the shortest
//! perpendicular box width and `cutoff + buffer` must be shorter than that
//! width. Positions passed to the force field must keep molecules whole
//! (bonded terms use them as they are); atoms may leave the box.
//!
//! # Ewald / PME
//!
//! `E = E_real + E_recip + E_self + E_excl + E_net`, with `β` chosen so that
//! `erfc(β rc) = ewald_rtol`; the real-space part `erfc(βr)/r` is evaluated
//! in the cluster kernels, the reciprocal part with smooth PME (Essmann et al. 1995),
//! `E_self = -β/√π Σ qi²`, the exclusion corrections subtract
//! `erf(βr)/r` for 1-2/1-3 pairs and replace the 1-4 pairs' Coulomb energy
//! by its 0.75 fraction, and `E_net = -π Q²/(2 V β²)` is the neutralizing
//! background correction for a net charge `Q` (all with the prefactor
//! `332.0716/D`).
//!
//! # Determinism
//!
//! All reductions use fixed partitions of the work combined in a fixed
//! order, so energies and gradients are bitwise independent of the number
//! of threads.

pub mod erfc;
mod kernel;
mod pairlist;
mod params;
mod pme;

pub use pme::{bspline, fft_friendly_size};

use super::backend::{NonbondedBackend, NonbondedEnergy};
use super::{ApproxOptions, Electrostatics, VdwModifier};
use crate::core::{Molecule, UnitCell};
use crate::error::{Error, Result};
#[allow(unused_imports)]
use crate::ff::{mmff::MmffBuilder, uff::UffBuilder};
use arc_swap::ArcSwap;
use kernel::{ClusterKernel, Ctx, ElecKind, KernelParams, Modifier, elec_shape, pair_ref, select_kernel, vdw_modified};
use pairlist::PairList;
use params::NbParams;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Statistics of the current pair list.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairListStats {
    /// Number of 4-atom clusters (including padding).
    pub clusters: usize,
    /// Number of stored (i, j) cluster pairs (the list is symmetric).
    pub cluster_pairs: usize,
    /// Number of atom pairs in the list (each unordered pair once).
    pub atom_pairs: usize,
    /// List radius `cutoff + buffer` (infinite for `Full` electrostatics).
    pub rlist: f64,
    /// Number of list builds so far (including the initial one).
    pub builds: usize,
}

/// Per-atom explicit pair list (MMFF 1-4 pairs and, with PME, the
/// excluded 1-2/1-3 pairs), stored in both directions.
#[derive(Debug, Clone, Default)]
struct Explicit {
    start: Vec<u32>,
    partner: Vec<u32>,
    /// `true` = 1-4 pair, `false` = excluded pair (Ewald correction only).
    is14: Vec<bool>,
}

/// Approximate nonbonded backend.
pub struct ApproxNonbonded {
    p: NbParams,
    kp: KernelParams,
    opts: ApproxOptions,
    simd: bool,
    cell: Option<UnitCell>,
    rlist: f64,
    all_pairs: bool,
    list: ArcSwap<PairList>,
    rebuild_lock: Mutex<()>,
    builds: AtomicUsize,
    pme: Option<pme::Pme>,
    /// Constant Ewald terms (self energy + net-charge correction).
    ewald_const: f64,
    explicit: Explicit,
    /// Charges times `sqrt(C)`.
    qs: Vec<f64>,
    name: &'static str,
}

impl std::fmt::Debug for ApproxNonbonded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApproxNonbonded")
            .field("force_field", &self.p.ff_name)
            .field("atoms", &self.p.n)
            .field("options", &self.opts)
            .field("simd", &self.simd)
            .field("cell", &self.cell)
            .field("stats", &self.pair_list_stats())
            .finish()
    }
}

/// `β` such that `erfc(β rc) = rtol`.
pub fn ewald_beta(cutoff: f64, rtol: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, 1.0);
    while erfc::erfc(hi * cutoff) > rtol {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if erfc::erfc(mid * cutoff) > rtol {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Builds the approximate nonbonded backend for `mol`.
///
/// `inner_builder` is the wrapped force-field builder (`&MmffBuilder` or
/// `&UffBuilder` as `&dyn Any`); its settings (dielectric, interfragment
/// handling, ...) and the molecule's typing define the per-pair parameters
/// and exclusions. The periodic cell, if any, is taken from the molecule.
pub fn build_approx_nonbonded(
    mol: &Molecule,
    inner_builder: &dyn std::any::Any,
    opts: &ApproxOptions,
    simd: bool,
) -> Result<ApproxNonbonded> {
    let rc = opts.cutoff;
    if !(rc > 0.0 && rc.is_finite()) {
        return Err(Error::InvalidArgument(format!("cutoff must be positive (got {rc})")));
    }
    if !(opts.buffer >= 0.0 && opts.buffer.is_finite()) {
        return Err(Error::InvalidArgument(format!("buffer must be non-negative (got {})", opts.buffer)));
    }
    if mol.coords.len() != mol.num_atoms() {
        return Err(Error::InvalidArgument("molecule has no coordinates".into()));
    }
    let p = params::extract(mol, inner_builder)?;
    let cell = mol.cell;
    let rlist = rc + opts.buffer;
    if let Some(c) = &cell {
        if rc > c.max_cutoff() {
            return Err(Error::InvalidArgument(format!(
                "cutoff {rc} Å exceeds half the shortest perpendicular box width ({:.4} Å)",
                c.max_cutoff()
            )));
        }
        if rlist >= c.min_perpendicular_width() {
            return Err(Error::InvalidArgument(format!(
                "cutoff + buffer ({rlist} Å) must be shorter than the shortest perpendicular box width ({:.4} Å)",
                c.min_perpendicular_width()
            )));
        }
    }
    let modifier = match opts.vdw_modifier {
        VdwModifier::PotentialShift => Modifier::Shift,
        VdwModifier::Switch { r_switch } => {
            if r_switch.is_nan() || r_switch < 0.0 {
                return Err(Error::InvalidArgument(format!("r_switch must be non-negative (got {r_switch})")));
            }
            if r_switch >= rc { Modifier::None } else { Modifier::Switch { rsw: r_switch, inv_w: 1.0 / (rc - r_switch) } }
        }
    };
    let (mut eps_rf, mut beta) = (f64::INFINITY, 0.0);
    let mut pme_opts = None;
    let mut elec = match opts.electrostatics {
        Electrostatics::Full => {
            if cell.is_some() {
                return Err(Error::InvalidArgument(
                    "Electrostatics::Full is for isolated systems; use ReactionField or Pme with a periodic cell".into(),
                ));
            }
            ElecKind::Full
        }
        Electrostatics::ReactionField { eps_rf: e } => {
            if e.is_nan() || e <= 0.0 {
                return Err(Error::InvalidArgument(format!("eps_rf must be positive (got {e})")));
            }
            eps_rf = e;
            ElecKind::ReactionField
        }
        Electrostatics::Pme { ewald_rtol, grid_spacing, order } => {
            let Some(c) = &cell else {
                return Err(Error::InvalidArgument("PME requires a periodic cell (Molecule::cell)".into()));
            };
            if !(ewald_rtol > 0.0 && ewald_rtol < 1.0) {
                return Err(Error::InvalidArgument(format!("ewald_rtol must be in (0, 1) (got {ewald_rtol})")));
            }
            if p.distance_dielectric {
                return Err(Error::InvalidArgument("the distance-dependent dielectric cannot be used with PME".into()));
            }
            if p.has_charges && p.ignore_interfrag && p.num_frags > 1 {
                return Err(Error::InvalidArgument(
                    "PME cannot ignore interfragment interactions: set ignore_interfrag_interactions = false".into(),
                ));
            }
            beta = ewald_beta(rc, ewald_rtol);
            pme_opts = Some((*c, grid_spacing, order));
            ElecKind::Ewald
        }
    };
    if !p.has_charges {
        elec = ElecKind::None;
    }
    let all_pairs = matches!(opts.electrostatics, Electrostatics::Full) && elec != ElecKind::None;
    let kp = KernelParams::new(&p, rc, modifier, elec, eps_rf, beta);
    let sq = p.coulomb.sqrt();
    let qs: Vec<f64> = p.q.iter().map(|&q| q * sq).collect();
    let (mut pme, mut ewald_const) = (None, 0.0);
    if let (ElecKind::Ewald, Some((c, spacing, order))) = (elec, pme_opts) {
        pme = Some(pme::Pme::new(&c, beta, spacing, order, &qs)?);
        let q2: f64 = qs.iter().map(|q| q * q).sum();
        let qt: f64 = qs.iter().sum();
        ewald_const = -beta / std::f64::consts::PI.sqrt() * q2 - std::f64::consts::PI * qt * qt / (2.0 * c.volume() * beta * beta);
    } else if let (Some((c, spacing, order)), false) = (pme_opts, p.has_charges) {
        // validate the grid settings even without charges
        let _ = pme::Pme::new(&c, ewald_beta(rc, 1e-5), spacing, order, &[])?;
    }
    // explicit pairs
    let mut lists: Vec<Vec<(u32, bool)>> = vec![Vec::new(); p.n];
    if elec != ElecKind::None || p.kind == params::VdwKind::Mmff {
        for &(i, j) in &p.pairs14 {
            lists[i as usize].push((j, true));
            lists[j as usize].push((i, true));
        }
    }
    if elec == ElecKind::Ewald {
        for &(i, j) in &p.excluded {
            if p.q[i as usize] != 0.0 && p.q[j as usize] != 0.0 {
                lists[i as usize].push((j, false));
                lists[j as usize].push((i, false));
            }
        }
    }
    let mut explicit = Explicit::default();
    explicit.start.push(0);
    for l in &mut lists {
        l.sort_unstable();
        for &(j, k) in l.iter() {
            explicit.partner.push(j);
            explicit.is14.push(k);
        }
        explicit.start.push(explicit.partner.len() as u32);
    }
    let pos = mol.flat_coords();
    let list = pairlist::build(&p, &pos, cell.as_ref(), rlist, all_pairs);
    Ok(ApproxNonbonded {
        kp,
        opts: opts.clone(),
        simd,
        cell,
        rlist: if all_pairs { f64::INFINITY } else { rlist },
        all_pairs,
        list: ArcSwap::from_pointee(list),
        rebuild_lock: Mutex::new(()),
        builds: AtomicUsize::new(1),
        pme,
        ewald_const,
        explicit,
        qs,
        name: if simd { "cpu-simd-cluster" } else { "cpu-scalar-cluster" },
        p,
    })
}

const BLOCK: usize = 16;
const ATOM_BLOCK: usize = 256;

impl ApproxNonbonded {
    /// Options used to build this backend.
    pub fn options(&self) -> &ApproxOptions {
        &self.opts
    }
    /// Periodic cell, if any.
    pub fn cell(&self) -> Option<&UnitCell> {
        self.cell.as_ref()
    }
    /// Number of atoms.
    pub fn num_atoms(&self) -> usize {
        self.p.n
    }
    /// Ewald splitting parameter `β` (1/Å) with PME.
    pub fn ewald_beta(&self) -> Option<f64> {
        self.pme.as_ref().map(|p| p.beta)
    }
    /// PME grid dimensions.
    pub fn pme_grid(&self) -> Option<[usize; 3]> {
        self.pme.as_ref().map(|p| p.k)
    }
    /// Uses SIMD kernels (`true`) or the scalar reference kernels.
    pub fn simd(&self) -> bool {
        self.simd
    }
    /// Statistics of the current pair list.
    pub fn pair_list_stats(&self) -> PairListStats {
        let l = self.list.load();
        PairListStats {
            clusters: l.nclusters,
            cluster_pairs: l.num_cluster_pairs(),
            atom_pairs: l.num_masked_pairs() / 2,
            rlist: self.rlist,
            builds: self.builds.load(Ordering::Relaxed),
        }
    }
    /// Number of pair-list builds so far (including the initial one).
    pub fn num_builds(&self) -> usize {
        self.builds.load(Ordering::Relaxed)
    }
    /// Largest displacement (Å) of any atom since the last list build.
    pub fn max_displacement(&self, pos: &[f64]) -> f64 {
        self.list.load().max_displacement2(pos).sqrt()
    }
    /// `true` if `pos` would trigger a list rebuild.
    pub fn needs_rebuild(&self, pos: &[f64]) -> bool {
        let l = self.list.load();
        !l.all_pairs && l.max_displacement2(pos) > 0.25 * self.opts.buffer * self.opts.buffer
    }
    /// Rebuilds the pair list for `pos` unconditionally.
    pub fn rebuild(&self, pos: &[f64]) {
        let _g = self.rebuild_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.store_new_list(pos);
    }

    fn store_new_list(&self, pos: &[f64]) -> Arc<PairList> {
        let l = Arc::new(pairlist::build(&self.p, pos, self.cell.as_ref(), self.rlist, self.all_pairs));
        self.list.store(l.clone());
        self.builds.fetch_add(1, Ordering::Relaxed);
        l
    }

    fn current_list(&self, pos: &[f64]) -> Arc<PairList> {
        let half2 = 0.25 * self.opts.buffer * self.opts.buffer;
        let l = self.list.load_full();
        if l.all_pairs || l.max_displacement2(pos) <= half2 {
            return l;
        }
        let _g = self.rebuild_lock.lock().unwrap_or_else(|e| e.into_inner());
        let l = self.list.load_full();
        if l.max_displacement2(pos) <= half2 {
            return l;
        }
        self.store_new_list(pos)
    }

    #[inline]
    fn delta(&self, pos: &[f64], i: usize, j: usize) -> [f64; 3] {
        let d = [pos[3 * i] - pos[3 * j], pos[3 * i + 1] - pos[3 * j + 1], pos[3 * i + 2] - pos[3 * j + 2]];
        match &self.cell {
            Some(c) => c.min_image(d),
            None => d,
        }
    }

    /// Explicit pair `(i, j)`: `(E_vdw, E_elec, (dE/dr)/r, d)`.
    fn explicit_pair(&self, pos: &[f64], i: usize, j: usize, is14: bool) -> (f64, f64, f64, [f64; 3]) {
        let kp = &self.kp;
        let d = self.delta(pos, i, j);
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let r = r2.sqrt();
        let qq = self.qs[i] * self.qs[j];
        let (mut ev, mut dev) = (0.0, 0.0);
        let (mut ee, mut dee) = (0.0, 0.0);
        if is14 {
            if r2 < kp.rc2 {
                let idx = self.p.atom_type[i] as usize * self.p.ntypes + self.p.atom_type[j] as usize;
                (ev, dev) = vdw_modified(kp, &self.p, idx, r);
            }
            let s = self.p.scale14;
            match kp.elec {
                ElecKind::None => {}
                ElecKind::Full => {
                    let (f, df) = elec_shape(kp, r);
                    (ee, dee) = (s * qq * f, s * qq * df);
                }
                ElecKind::ReactionField => {
                    if r2 < kp.rc2 {
                        let (f, df) = elec_shape(kp, r);
                        (ee, dee) = (s * qq * f, s * qq * df);
                    }
                }
                ElecKind::Ewald => {
                    // s/r - erf(βr)/r = (s - 1)/r + erfc(βr)/r
                    let (f, df) = elec_shape(kp, r);
                    let inv = 1.0 / r;
                    ee = qq * ((s - 1.0) * inv + f);
                    dee = qq * (-(s - 1.0) * inv * inv + df);
                }
            }
        } else {
            // excluded pair, Ewald: -erf(βr)/r = erfc(βr)/r - 1/r
            let (f, df) = elec_shape(kp, r);
            let inv = 1.0 / r;
            ee = qq * (f - inv);
            dee = qq * (df + inv * inv);
        }
        (ev, ee, (dev + dee) / r, d)
    }

    fn evaluate(&self, pos: &[f64], grad: Option<&mut [f64]>) -> NonbondedEnergy {
        let n = self.p.n;
        assert!(pos.len() >= 3 * n, "position vector too short");
        let list = self.current_list(pos);
        let coords = list.gather(pos);
        let do_grad = grad.is_some();
        let kern: ClusterKernel = select_kernel(&self.kp, self.simd, do_grad);
        let ctx = Ctx { kp: &self.kp, p: &self.p, list: &list, c: &coords };
        let nc = list.nclusters;
        let mut cg: Vec<[f64; 12]> = if do_grad { vec![[0.0; 12]; nc] } else { Vec::new() };
        let parts: Vec<(f64, f64)> = if do_grad {
            cg.par_chunks_mut(BLOCK)
                .enumerate()
                .map(|(b, ch)| {
                    let (mut v, mut e) = (0.0, 0.0);
                    for (k, g) in ch.iter_mut().enumerate() {
                        let (a, c) = kern(&ctx, b * BLOCK + k, g);
                        v += a;
                        e += c;
                    }
                    (v, e)
                })
                .collect()
        } else {
            (0..nc.div_ceil(BLOCK))
                .into_par_iter()
                .map(|b| {
                    let (mut v, mut e) = (0.0, 0.0);
                    let mut g = [0.0; 12];
                    for ci in b * BLOCK..((b + 1) * BLOCK).min(nc) {
                        let (a, c) = kern(&ctx, ci, &mut g);
                        v += a;
                        e += c;
                    }
                    (v, e)
                })
                .collect()
        };
        let (mut ev, mut ee) = (0.0, 0.0);
        for (a, b) in parts {
            ev += a;
            ee += b;
        }
        // explicit pairs + scatter of the cluster gradient (per atom)
        let ex = &self.explicit;
        let atom_part = |i: usize, g: Option<&mut [f64]>| -> (f64, f64) {
            let (mut v, mut e) = (0.0, 0.0);
            let mut gi = [0.0; 3];
            for k in ex.start[i] as usize..ex.start[i + 1] as usize {
                let (a, b, gs, d) = self.explicit_pair(pos, i, ex.partner[k] as usize, ex.is14[k]);
                v += a;
                e += b;
                for c in 0..3 {
                    gi[c] += gs * d[c];
                }
            }
            if let Some(g) = g {
                let s = list.slot[i] as usize;
                let cgs = &cg[s / 4];
                let l = s % 4;
                for c in 0..3 {
                    g[c] += cgs[4 * c + l] + gi[c];
                }
            }
            (0.5 * v, 0.5 * e)
        };
        let has_explicit = !ex.partner.is_empty();
        let mut grad = grad;
        let parts: Vec<(f64, f64)> = match grad.as_deref_mut() {
            Some(g) => g[..3 * n]
                .par_chunks_mut(3 * ATOM_BLOCK)
                .enumerate()
                .map(|(b, ch)| {
                    let (mut v, mut e) = (0.0, 0.0);
                    for (k, gi) in ch.chunks_mut(3).enumerate() {
                        let (a, c) = atom_part(b * ATOM_BLOCK + k, Some(gi));
                        v += a;
                        e += c;
                    }
                    (v, e)
                })
                .collect(),
            None if has_explicit => (0..n.div_ceil(ATOM_BLOCK))
                .into_par_iter()
                .map(|b| {
                    let (mut v, mut e) = (0.0, 0.0);
                    for i in b * ATOM_BLOCK..((b + 1) * ATOM_BLOCK).min(n) {
                        let (a, c) = atom_part(i, None);
                        v += a;
                        e += c;
                    }
                    (v, e)
                })
                .collect(),
            None => Vec::new(),
        };
        for (a, b) in parts {
            ev += a;
            ee += b;
        }
        if let Some(pme) = &self.pme {
            ee += pme.energy_grad(pos, grad) + self.ewald_const;
        }
        NonbondedEnergy { vdw: ev, electrostatic: ee }
    }

    /// Brute-force O(N²) evaluation with the scalar per-pair reference
    /// functions (minimum image for periodic systems), without a pair list.
    /// Same model as [`NonbondedBackend::energy_grad`]; intended for tests.
    /// Adds the gradient to `grad` if given.
    pub fn reference_energy_grad(&self, pos: &[f64], grad: Option<&mut [f64]>) -> NonbondedEnergy {
        let n = self.p.n;
        let mut g = vec![0.0; 3 * n];
        let (mut ev, mut ee) = (0.0, 0.0);
        let nt = self.p.ntypes;
        for i in 0..n {
            for j in i + 1..n {
                if self.p.is_masked(i, j) {
                    continue;
                }
                let d = self.delta(pos, i, j);
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let idx = self.p.atom_type[i] as usize * nt + self.p.atom_type[j] as usize;
                let (v, e, gs) = pair_ref(&self.kp, &self.p, idx, self.qs[i] * self.qs[j], r2);
                ev += v;
                ee += e;
                for c in 0..3 {
                    g[3 * i + c] += gs * d[c];
                    g[3 * j + c] -= gs * d[c];
                }
            }
        }
        for i in 0..n {
            for k in self.explicit.start[i] as usize..self.explicit.start[i + 1] as usize {
                let j = self.explicit.partner[k] as usize;
                if j < i {
                    continue;
                }
                let (v, e, gs, d) = self.explicit_pair(pos, i, j, self.explicit.is14[k]);
                ev += v;
                ee += e;
                for c in 0..3 {
                    g[3 * i + c] += gs * d[c];
                    g[3 * j + c] -= gs * d[c];
                }
            }
        }
        if let Some(pme) = &self.pme {
            ee += pme.energy_grad(pos, Some(&mut g)) + self.ewald_const;
        }
        if let Some(out) = grad {
            for (o, v) in out.iter_mut().zip(&g) {
                *o += v;
            }
        }
        NonbondedEnergy { vdw: ev, electrostatic: ee }
    }

    /// Reciprocal-space PME energy alone (0 without PME).
    pub fn pme_reciprocal_energy(&self, pos: &[f64]) -> f64 {
        self.pme.as_ref().map(|p| p.energy_grad(pos, None)).unwrap_or(0.0)
    }
}

impl NonbondedBackend for ApproxNonbonded {
    fn name(&self) -> &str {
        self.name
    }
    fn energy(&self, pos: &[f64]) -> NonbondedEnergy {
        self.evaluate(pos, None)
    }
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> NonbondedEnergy {
        self.evaluate(pos, Some(grad))
    }
    fn on_accepted_step(&self, pos: &[f64]) {
        let _ = self.current_list(pos);
    }
}
