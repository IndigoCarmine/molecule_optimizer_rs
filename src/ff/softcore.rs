//! Soft-core intermolecular interactions (λ-coupled van der Waals and
//! electrostatics between fragments).
//!
//! [`SoftCoreBuilder`] wraps an MMFF94 ([`MmffBuilder`]) or UFF
//! ([`UffBuilder`]) builder. The **intramolecular** part (everything inside
//! one connected component, i.e. what the inner builder produces with
//! `ignore_interfrag_interactions = true`) is built by the inner builder
//! unchanged. Every **intermolecular** non-bonded pair — exactly the pairs,
//! with exactly the per-pair parameters and charges, that the inner builder
//! adds when `ignore_interfrag_interactions = false` — goes into one
//! [`SoftCoreInterContrib`], whose strength is controlled by two coupling
//! parameters λ_vdw, λ_ele ∈ [0, 1].
//!
//! # Functional form
//!
//! Soft-core potentials after Beutler, Mark, van Schaik, Gerber and
//! van Gunsteren, *Chem. Phys. Lett.* **222**, 529 (1994), with the energy
//! scaling λⁿ and the soft-core power p of Zacharias, Straatsma and
//! McCammon, *J. Chem. Phys.* **100**, 9025 (1994) / Pitera and van
//! Gunsteren, *Mol. Simul.* **28**, 45 (2002).
//!
//! For a pair at distance r with van der Waals minimum-energy distance R*
//! (MMFF `R*_ij` after donor/acceptor scaling; UFF `x_ij`) the effective
//! van der Waals distance is
//!
//! ```text
//!   r_vdw = ( α_vdw · σ⁶ · (1 − λ_vdw)^p + r⁶ )^(1/6),      σ⁶ = R*⁶ / 2
//! ```
//!
//! (σ = 2^(-1/6) R* is the zero crossing of a Lennard-Jones potential with
//! minimum at R*, so α_vdw has its conventional meaning for both the 12-6
//! and the buffered 14-7 form; the same r⁶ form is used for the 14-7 term,
//! which, although finite at r = 0, has an enormous value there), and
//!
//! ```text
//!   E_vdw,sc(r) = λ_vdw^n · E_vdw(r_vdw)
//! ```
//!
//! where `E_vdw` is the unmodified pair function (MMFF buffered 14-7
//! `ε (1.07 R*/(ρ + 0.07 R*))⁷ (1.12 R*⁷/(ρ⁷ + 0.12 R*⁷) − 2)`, UFF
//! `D ((x/ρ)¹² − 2 (x/ρ)⁶)`). For electrostatics (MMFF only; UFF has
//! none) the soft-core Coulomb distance is
//!
//! ```text
//!   r_ele = ( α_ele · (1 − λ_ele)^p + r² )^(1/2)          (α_ele in Å²)
//!   E_ele,sc(r) = λ_ele^n · 332.0716 q_i q_j / (D (r_ele + 0.05)^m)
//! ```
//!
//! with m = 1 (constant dielectric) or 2 (distance-dependent). At
//! λ = 1 the shift terms vanish identically (`r_vdw = r_ele = r` exactly,
//! not via a sixth root) and λⁿ = 1, so the force field reproduces the inner
//! builder with `ignore_interfrag_interactions = false` (RDKit-exact terms;
//! only the summation order of the non-bonded pairs differs). For λ < 1 the
//! energy and its gradient are finite at r → 0 (the gradient vanishes
//! there). The UFF evaluation cutoff (energy zero beyond 10 x_ij) is applied
//! to the real distance r.
//!
//! The derivatives used for thermodynamic integration are analytic:
//!
//! ```text
//!   ∂E/∂λ = n λ^(n−1) E(r_sc) + λⁿ E'(r_sc) ∂r_sc/∂λ
//!   ∂r_vdw/∂λ = −α σ⁶ p (1−λ)^(p−1) / (6 r_vdw⁵)
//!   ∂r_ele/∂λ = −α_ele p (1−λ)^(p−1) / (2 r_ele)
//! ```
//!
//! (see [`SoftCoreHandle::de_dlambda`]). Requirements: `n ≥ 1`, `p ≥ 1`
//! (so the derivatives exist on the closed interval).
//!
//! # Changing λ without rebuilding
//!
//! λ_vdw, λ_ele and the confinement radius live in an `Arc`-shared
//! [`SoftCoreState`]. [`SoftCoreBuilder::build_with_handle`] returns a
//! [`SoftCoreHandle`] whose setters change the values seen by the already
//! built force field. Force fields created through the
//! [`ForceFieldBuilder`] trait share the builder's own state
//! ([`SoftCoreBuilder::set_lambda`]).
//!
//! # Confinement
//!
//! Optionally a flat-bottom spherical wall acts on the fragment centroids
//! (unweighted mean of the atom positions of each fragment):
//!
//! ```text
//!   E_wall = k Σ_f max(0, |c_f − C| − R)²,   C = (1/F) Σ_f c_f
//! ```
//!
//! It is translation invariant and keeps an assembly together while the
//! intermolecular interactions are weak.

use super::mmff::{DielectricModel, MmffBuilder, MmffMolProperties, terms::calc_ele_energy, terms::calc_vdw_energy};
use super::uff::nonbonded::{calc_nonbonded_depth, calc_nonbonded_minimum};
use super::uff::{UffBuilder, atom_typer};
use super::{Constraint, Contrib, ContribForceField, ForceFieldBuilder};
use crate::core::Molecule;
use crate::error::{Error, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// An `f64` that can be shared and changed through `&self`.
#[derive(Debug, Default)]
pub struct AtomicF64(AtomicU64);

impl AtomicF64 {
    pub fn new(v: f64) -> Self {
        AtomicF64(AtomicU64::new(v.to_bits()))
    }
    pub fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Acquire))
    }
    pub fn set(&self, v: f64) {
        self.0.store(v.to_bits(), Ordering::Release)
    }
}

/// Soft-core parameters (see the [module documentation](self)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftCoreParams {
    /// α_vdw (dimensionless, default 0.5).
    pub alpha_vdw: f64,
    /// α_ele (Å², default 1.0).
    pub alpha_ele: f64,
    /// Soft-core power p of (1 − λ) (default 1, must be ≥ 1).
    pub p: f64,
    /// Energy scaling exponent n of λ (default 1, must be ≥ 1).
    pub n: f64,
    /// Include every intermolecular pair instead of applying the inner
    /// builder's build-time distance filter (MMFF `nonbonded_thresh`, UFF
    /// `vdw_thresh · x_ij`, evaluated at the build coordinates). Default
    /// `false` (= the same pair list as the inner builder).
    pub include_all_pairs: bool,
}

impl Default for SoftCoreParams {
    fn default() -> Self {
        SoftCoreParams { alpha_vdw: 0.5, alpha_ele: 1.0, p: 1.0, n: 1.0, include_all_pairs: false }
    }
}

/// Flat-bottom spherical wall on fragment centroids.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confinement {
    /// Radius R (Å) of the flat bottom around the mean fragment centroid.
    pub radius: f64,
    /// Force constant k (kcal/mol/Å²).
    pub force_constant: f64,
}

/// Shared, mutable coupling state of soft-core force fields.
#[derive(Debug)]
pub struct SoftCoreState {
    lambda_vdw: AtomicF64,
    lambda_ele: AtomicF64,
    wall_radius: AtomicF64,
}

impl SoftCoreState {
    pub fn new(lambda_vdw: f64, lambda_ele: f64, wall_radius: f64) -> Self {
        SoftCoreState {
            lambda_vdw: AtomicF64::new(lambda_vdw.clamp(0.0, 1.0)),
            lambda_ele: AtomicF64::new(lambda_ele.clamp(0.0, 1.0)),
            wall_radius: AtomicF64::new(wall_radius),
        }
    }
    /// `(λ_vdw, λ_ele)`.
    pub fn lambda(&self) -> (f64, f64) {
        (self.lambda_vdw.get(), self.lambda_ele.get())
    }
    /// Sets λ_vdw and λ_ele (clamped to [0, 1]).
    pub fn set_lambda(&self, vdw: f64, ele: f64) {
        self.lambda_vdw.set(vdw.clamp(0.0, 1.0));
        self.lambda_ele.set(ele.clamp(0.0, 1.0));
    }
    pub fn confinement_radius(&self) -> f64 {
        self.wall_radius.get()
    }
    pub fn set_confinement_radius(&self, r: f64) {
        self.wall_radius.set(r)
    }
}

/// van der Waals functional form of the inner force field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VdwForm {
    /// MMFF buffered 14-7.
    Buffered14_7,
    /// UFF Lennard-Jones 12-6.
    Lj12_6,
}

/// Intermolecular pair list with per-pair parameters.
#[derive(Debug, Clone)]
pub struct InterPairs {
    pub form: VdwForm,
    /// Dielectric model of the electrostatic term (MMFF).
    pub dielectric: DielectricModel,
    pub i: Vec<usize>,
    pub j: Vec<usize>,
    /// van der Waals minimum distance (MMFF R*_ij, UFF x_ij); 0 = no vdW.
    pub r_star: Vec<f64>,
    /// Well depth (MMFF ε_ij, UFF D_ij).
    pub well_depth: Vec<f64>,
    /// `true` if the pair has a vdW term.
    pub has_vdw: Vec<bool>,
    /// q_i q_j / D, `None` = no electrostatic term.
    pub charge_term: Vec<Option<f64>>,
    /// Evaluation cutoff on the real distance (UFF 10 x_ij; MMFF ∞).
    pub cutoff: Vec<f64>,
    /// Fragments (connected components) of the molecule.
    pub fragments: Vec<Vec<usize>>,
}

impl InterPairs {
    fn new(form: VdwForm, dielectric: DielectricModel, fragments: Vec<Vec<usize>>) -> Self {
        InterPairs {
            form,
            dielectric,
            i: vec![],
            j: vec![],
            r_star: vec![],
            well_depth: vec![],
            has_vdw: vec![],
            charge_term: vec![],
            cutoff: vec![],
            fragments,
        }
    }
    fn push(&mut self, i: usize, j: usize, vdw: Option<(f64, f64)>, q: Option<f64>, cutoff: f64) {
        if vdw.is_none() && q.is_none() {
            return;
        }
        self.i.push(i);
        self.j.push(j);
        let (r, e) = vdw.unwrap_or((0.0, 0.0));
        self.r_star.push(r);
        self.well_depth.push(e);
        self.has_vdw.push(vdw.is_some());
        self.charge_term.push(q);
        self.cutoff.push(cutoff);
    }
    pub fn len(&self) -> usize {
        self.i.len()
    }
    pub fn is_empty(&self) -> bool {
        self.i.is_empty()
    }
}

/// Result of one pair evaluation.
#[derive(Debug, Clone, Copy, Default)]
struct PairOut {
    e_vdw: f64,
    e_ele: f64,
    /// (dE/dr)/r: gradient on atom i is `g * (x_i - x_j)`.
    g: f64,
    de_dl_vdw: f64,
    de_dl_ele: f64,
}

const VDW1: f64 = 1.07;
const VDW1M1: f64 = VDW1 - 1.0;
const VDW2: f64 = 1.12;
const VDW2M1: f64 = VDW2 - 1.0;
const VDW2T7: f64 = VDW2 * 7.0;
const ELE_PREFACTOR: f64 = 332.0716;

/// MMFF buffered 14-7 dE/dr (RDKit `NonbondedContrib::getGrad` operations).
fn mmff_vdw_de_dr(dist: f64, r: f64, well_depth: f64) -> f64 {
    let q = dist / r;
    let q2 = q * q;
    let q6 = q2 * q2 * q2;
    let q7 = q6 * q;
    let q7pvdw2m1 = q7 + VDW2M1;
    let tt = VDW1 / (q + VDW1 - 1.0);
    let t2 = tt * tt;
    let t7 = t2 * t2 * t2 * tt;
    well_depth / r * t7 * (-VDW2T7 * q6 / (q7pvdw2m1 * q7pvdw2m1) + ((-VDW2T7 / q7pvdw2m1 + 14.0) / (q + VDW1M1)))
}

/// UFF LJ energy (RDKit `vdWContrib::getEnergy` operations).
fn uff_vdw_energy(dist: f64, xij: f64, d: f64) -> f64 {
    let r = xij / dist;
    let r6 = super::uff::int_pow::<6>(r);
    let r12 = r6 * r6;
    d * (r12 - 2.0 * r6)
}

/// UFF LJ dE/dr (RDKit `vdWContrib::getGrad` operations).
fn uff_vdw_de_dr(dist: f64, xij: f64, d: f64) -> f64 {
    let r = xij / dist;
    let r7 = super::uff::int_pow::<7>(r);
    let r13 = super::uff::int_pow::<13>(r);
    12. * d / xij * (r7 - r13)
}

/// The λ-coupled intermolecular contrib (see the [module documentation](self)).
#[derive(Debug, Clone)]
pub struct SoftCoreInterContrib {
    pairs: Arc<InterPairs>,
    params: SoftCoreParams,
    state: Arc<SoftCoreState>,
}

impl SoftCoreInterContrib {
    pub fn new(pairs: Arc<InterPairs>, params: SoftCoreParams, state: Arc<SoftCoreState>) -> Self {
        SoftCoreInterContrib { pairs, params, state }
    }
    pub fn pairs(&self) -> &InterPairs {
        &self.pairs
    }

    #[inline]
    fn eval_pair(&self, t: usize, dist: f64, lv: f64, le: f64, deriv: bool) -> PairOut {
        let pr = &*self.pairs;
        let sp = &self.params;
        let mut out = PairOut::default();
        if pr.has_vdw[t] && dist <= pr.cutoff[t] {
            let r_star = pr.r_star[t];
            let eps = pr.well_depth[t];
            let one_m = 1.0 - lv;
            let shift = if one_m == 0.0 {
                0.0
            } else {
                let r2 = r_star * r_star;
                sp.alpha_vdw * 0.5 * (r2 * r2 * r2) * one_m.powf(sp.p)
            };
            let rs = if shift == 0.0 {
                dist
            } else {
                let d2 = dist * dist;
                (shift + d2 * d2 * d2).powf(1.0 / 6.0)
            };
            let scale = if lv == 1.0 { 1.0 } else { lv.powf(sp.n) };
            let (e0, de0) = match pr.form {
                VdwForm::Buffered14_7 => {
                    (calc_vdw_energy(rs, r_star, eps), if deriv { mmff_vdw_de_dr(rs, r_star, eps) } else { 0.0 })
                }
                VdwForm::Lj12_6 => {
                    if rs <= 0.0 {
                        (0.0, 0.0)
                    } else {
                        (uff_vdw_energy(rs, r_star, eps), if deriv { uff_vdw_de_dr(rs, r_star, eps) } else { 0.0 })
                    }
                }
            };
            out.e_vdw = scale * e0;
            if deriv {
                if shift == 0.0 {
                    if dist > 0.0 {
                        out.g += scale * de0 / dist;
                    }
                } else {
                    // dr_sc/dr = r⁵ / r_sc⁵  →  (dE/dr)/r = E'(r_sc) r⁴ / r_sc⁵
                    let rs2 = rs * rs;
                    let rs5 = rs2 * rs2 * rs;
                    let d2 = dist * dist;
                    out.g += scale * de0 * (d2 * d2) / rs5;
                }
                let dscale = if sp.n == 1.0 { 1.0 } else { sp.n * lv.powf(sp.n - 1.0) };
                let mut d = dscale * e0;
                if one_m != 0.0 || sp.p == 1.0 {
                    let r2 = r_star * r_star;
                    let sig6 = 0.5 * (r2 * r2 * r2);
                    let dpow = if sp.p == 1.0 { 1.0 } else { sp.p * one_m.powf(sp.p - 1.0) };
                    let rs2 = rs * rs;
                    let rs5 = rs2 * rs2 * rs;
                    if rs5 > 0.0 {
                        d += scale * de0 * (-sp.alpha_vdw * sig6 * dpow / (6.0 * rs5));
                    }
                }
                out.de_dl_vdw = d;
            }
        }
        if let Some(ct) = pr.charge_term[t] {
            let one_m = 1.0 - le;
            let shift = if one_m == 0.0 { 0.0 } else { sp.alpha_ele * one_m.powf(sp.p) };
            let re = if shift == 0.0 { dist } else { (shift + dist * dist).sqrt() };
            let scale = if le == 1.0 { 1.0 } else { le.powf(sp.n) };
            let model = pr.dielectric;
            let e0 = calc_ele_energy(re, ct, model, false);
            out.e_ele = scale * e0;
            if deriv {
                let mut corr = re + 0.05;
                corr *= if model == DielectricModel::Distance { corr * corr } else { corr };
                let de0 = -ELE_PREFACTOR * model.as_f64() * ct / corr;
                if shift == 0.0 {
                    if dist > 0.0 {
                        out.g += scale * de0 / dist;
                    }
                } else {
                    // dr_ele/dr = r / r_ele  →  (dE/dr)/r = E'(r_ele) / r_ele
                    out.g += scale * de0 / re;
                }
                let dscale = if sp.n == 1.0 { 1.0 } else { sp.n * le.powf(sp.n - 1.0) };
                let mut d = dscale * e0;
                if one_m != 0.0 || sp.p == 1.0 {
                    let dpow = if sp.p == 1.0 { 1.0 } else { sp.p * one_m.powf(sp.p - 1.0) };
                    if re > 0.0 {
                        d += scale * de0 * (-sp.alpha_ele * dpow / (2.0 * re));
                    }
                }
                out.de_dl_ele = d;
            }
        }
        out
    }

    #[inline]
    fn dist(pos: &[f64], i: usize, j: usize) -> (f64, [f64; 3]) {
        let d = [pos[3 * i] - pos[3 * j], pos[3 * i + 1] - pos[3 * j + 1], pos[3 * i + 2] - pos[3 * j + 2]];
        ((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt(), d)
    }

    /// `(E_vdw, E_ele)` of the intermolecular pairs at the current λ.
    pub fn energies(&self, pos: &[f64]) -> (f64, f64) {
        let (lv, le) = self.state.lambda();
        let (mut ev, mut ee) = (0.0, 0.0);
        for t in 0..self.pairs.len() {
            let (r, _) = Self::dist(pos, self.pairs.i[t], self.pairs.j[t]);
            let o = self.eval_pair(t, r, lv, le, false);
            ev += o.e_vdw;
            ee += o.e_ele;
        }
        (ev, ee)
    }

    /// `(∂E/∂λ_vdw, ∂E/∂λ_ele)` at the current λ.
    pub fn de_dlambda(&self, pos: &[f64]) -> (f64, f64) {
        let (lv, le) = self.state.lambda();
        let (mut dv, mut de) = (0.0, 0.0);
        for t in 0..self.pairs.len() {
            let (r, _) = Self::dist(pos, self.pairs.i[t], self.pairs.j[t]);
            let o = self.eval_pair(t, r, lv, le, true);
            dv += o.de_dl_vdw;
            de += o.de_dl_ele;
        }
        (dv, de)
    }
}

impl Contrib for SoftCoreInterContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        let (ev, ee) = self.energies(pos);
        ev + ee
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let (lv, le) = self.state.lambda();
        for t in 0..self.pairs.len() {
            let (i, j) = (self.pairs.i[t], self.pairs.j[t]);
            let (r, d) = Self::dist(pos, i, j);
            let o = self.eval_pair(t, r, lv, le, true);
            if o.g != 0.0 {
                for k in 0..3 {
                    let dg = o.g * d[k];
                    grad[3 * i + k] += dg;
                    grad[3 * j + k] -= dg;
                }
            }
        }
    }
    fn label(&self) -> &'static str {
        "SoftCoreInter"
    }
    fn energy_terms(&self, pos: &[f64], out: &mut dyn FnMut(&'static str, f64)) {
        let (ev, ee) = self.energies(pos);
        if self.pairs.has_vdw.iter().any(|&b| b) {
            out("VdW", ev);
        }
        if self.pairs.charge_term.iter().any(|q| q.is_some()) {
            out("Electrostatic", ee);
        }
    }
}

/// Flat-bottom spherical wall on fragment centroids (see the
/// [module documentation](self)). The radius is read from the shared
/// [`SoftCoreState`]; a non-finite or negative radius disables the wall.
#[derive(Debug, Clone)]
pub struct CentroidWallContrib {
    fragments: Arc<Vec<Vec<usize>>>,
    force_constant: f64,
    state: Arc<SoftCoreState>,
}

impl CentroidWallContrib {
    pub fn new(fragments: Arc<Vec<Vec<usize>>>, force_constant: f64, state: Arc<SoftCoreState>) -> Self {
        CentroidWallContrib { fragments, force_constant, state }
    }

    fn centroids(&self, pos: &[f64]) -> (Vec<[f64; 3]>, [f64; 3]) {
        let mut cs = Vec::with_capacity(self.fragments.len());
        let mut all = [0.0; 3];
        for f in self.fragments.iter() {
            let mut c = [0.0; 3];
            for &a in f {
                for k in 0..3 {
                    c[k] += pos[3 * a + k];
                }
            }
            let n = f.len().max(1) as f64;
            for k in 0..3 {
                c[k] /= n;
                all[k] += c[k];
            }
            cs.push(c);
        }
        let nf = cs.len().max(1) as f64;
        for v in &mut all {
            *v /= nf;
        }
        (cs, all)
    }
}

impl Contrib for CentroidWallContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        let r0 = self.state.confinement_radius();
        if !(r0.is_finite() && r0 >= 0.0) || self.fragments.len() < 2 {
            return 0.0;
        }
        let (cs, c0) = self.centroids(pos);
        let mut e = 0.0;
        for c in &cs {
            let d = crate::core::geometry::dist(*c, c0);
            if d > r0 {
                e += self.force_constant * (d - r0) * (d - r0);
            }
        }
        e
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let r0 = self.state.confinement_radius();
        if !(r0.is_finite() && r0 >= 0.0) || self.fragments.len() < 2 {
            return;
        }
        let (cs, c0) = self.centroids(pos);
        let nf = cs.len() as f64;
        // dE/dc_g = h_g - (1/F) Σ_f h_f with h_f = 2k (d_f - R) u_f
        let mut h = vec![[0.0; 3]; cs.len()];
        let mut hsum = [0.0; 3];
        for (f, c) in cs.iter().enumerate() {
            let v = crate::core::geometry::sub(*c, c0);
            let d = crate::core::geometry::norm(v);
            if d > r0 && d > 0.0 {
                let s = 2.0 * self.force_constant * (d - r0) / d;
                for k in 0..3 {
                    h[f][k] = s * v[k];
                    hsum[k] += h[f][k];
                }
            }
        }
        for (f, members) in self.fragments.iter().enumerate() {
            let n = members.len().max(1) as f64;
            for &a in members {
                for k in 0..3 {
                    grad[3 * a + k] += (h[f][k] - hsum[k] / nf) / n;
                }
            }
        }
    }
    fn label(&self) -> &'static str {
        "Confinement"
    }
}

/// Inner force fields that [`SoftCoreBuilder`] can wrap (MMFF, UFF).
pub trait SoftCoreInner: ForceFieldBuilder + Clone + std::fmt::Debug + 'static {
    /// A copy with `ignore_interfrag_interactions` set to `!on`.
    fn with_interfrag(&self, on: bool) -> Self;
    /// Builds the intramolecular force field (exactly as `self` with
    /// `ignore_interfrag_interactions = true`, including `constraints`) and
    /// the list of intermolecular pairs the inner builder would add with
    /// `ignore_interfrag_interactions = false`.
    fn build_parts(
        &self,
        mol: &Molecule,
        constraints: &[Constraint],
        fragments: Vec<Vec<usize>>,
        include_all_pairs: bool,
    ) -> Result<(ContribForceField, InterPairs)>;
}

fn fragment_index(n: usize, fragments: &[Vec<usize>]) -> Vec<usize> {
    let mut frag = vec![0usize; n];
    for (k, f) in fragments.iter().enumerate() {
        for &a in f {
            frag[a] = k;
        }
    }
    frag
}

fn coord_dist(c: &[[f64; 3]], i: usize, j: usize) -> f64 {
    let d = [c[i][0] - c[j][0], c[i][1] - c[j][1], c[i][2] - c[j][2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

impl SoftCoreInner for MmffBuilder {
    fn with_interfrag(&self, on: bool) -> Self {
        MmffBuilder { ignore_interfrag_interactions: !on, ..self.clone() }
    }
    fn build_parts(
        &self,
        mol: &Molecule,
        constraints: &[Constraint],
        fragments: Vec<Vec<usize>>,
        include_all_pairs: bool,
    ) -> Result<(ContribForceField, InterPairs)> {
        let props = MmffMolProperties::new(mol, self.variant)?;
        let intra = self.with_interfrag(false).build_with_properties(mol, &props, constraints)?;
        let n = mol.num_atoms();
        let frag = fragment_index(n, &fragments);
        let q = props.partial_charges();
        let is_zero = |x: f64| x < 1.0e-10 && x > -1.0e-10;
        let mut pairs = InterPairs::new(VdwForm::Buffered14_7, self.dielectric_model, fragments);
        for i in 0..n {
            for j in (i + 1)..n {
                if frag[i] == frag[j] {
                    continue;
                }
                if !include_all_pairs && coord_dist(&mol.coords, i, j) > self.nonbonded_thresh {
                    continue;
                }
                let vdw = props.vdw_params(i, j).map(|v| (v.r_ij_star, v.epsilon));
                let ele =
                    if !is_zero(q[i]) && !is_zero(q[j]) { Some(q[i] * q[j] / self.dielectric_constant) } else { None };
                pairs.push(i, j, vdw, ele, f64::INFINITY);
            }
        }
        Ok((intra, pairs))
    }
}

impl SoftCoreInner for UffBuilder {
    fn with_interfrag(&self, on: bool) -> Self {
        UffBuilder { ignore_interfrag_interactions: !on, ..self.clone() }
    }
    fn build_parts(
        &self,
        mol: &Molecule,
        constraints: &[Constraint],
        fragments: Vec<Vec<usize>>,
        include_all_pairs: bool,
    ) -> Result<(ContribForceField, InterPairs)> {
        if mol.coords.len() != mol.num_atoms() {
            return Err(Error::InvalidArgument("molecule has no coordinates".into()));
        }
        let (_, params, _) = atom_typer::atom_types_and_params(mol);
        let mut intra = super::uff::builder::construct_force_field(mol, &params, self.vdw_thresh, true)?;
        intra.add_constraints(&mol.coords, constraints);
        let n = mol.num_atoms();
        let frag = fragment_index(n, &fragments);
        let mut pairs = InterPairs::new(VdwForm::Lj12_6, DielectricModel::Constant, fragments);
        for i in 0..n {
            let Some(pi) = params[i] else { continue };
            for j in (i + 1)..n {
                let Some(pj) = params[j] else { continue };
                if frag[i] == frag[j] {
                    continue;
                }
                let xij = calc_nonbonded_minimum(pi, pj);
                if !include_all_pairs && coord_dist(&mol.coords, i, j) >= self.vdw_thresh * xij {
                    continue;
                }
                // RDKit vdWContrib uses a fixed evaluation cutoff of 10 x_ij.
                pairs.push(i, j, Some((xij, calc_nonbonded_depth(pi, pj))), None, 10.0 * xij);
            }
        }
        Ok((intra, pairs))
    }
}

/// Handle to the λ state and intermolecular pairs of a built soft-core
/// force field.
#[derive(Debug, Clone)]
pub struct SoftCoreHandle {
    state: Arc<SoftCoreState>,
    inter: SoftCoreInterContrib,
}

impl SoftCoreHandle {
    pub fn state(&self) -> &Arc<SoftCoreState> {
        &self.state
    }
    /// Sets λ_vdw and λ_ele (clamped to [0, 1]); takes effect immediately in
    /// the force field this handle belongs to.
    pub fn set_lambda(&self, vdw: f64, ele: f64) {
        self.state.set_lambda(vdw, ele)
    }
    /// `(λ_vdw, λ_ele)`.
    pub fn lambda(&self) -> (f64, f64) {
        self.state.lambda()
    }
    /// Sets the confinement radius (Å); `f64::INFINITY` disables the wall.
    pub fn set_confinement_radius(&self, r: f64) {
        self.state.set_confinement_radius(r)
    }
    pub fn confinement_radius(&self) -> f64 {
        self.state.confinement_radius()
    }
    /// Analytic `(∂E/∂λ_vdw, ∂E/∂λ_ele)` at `pos` (flat coordinates).
    pub fn de_dlambda(&self, pos: &[f64]) -> (f64, f64) {
        self.inter.de_dlambda(pos)
    }
    /// Intermolecular `(E_vdw, E_ele)` at `pos` and the current λ.
    pub fn inter_energy(&self, pos: &[f64]) -> (f64, f64) {
        self.inter.energies(pos)
    }
    /// The intermolecular pair list.
    pub fn pairs(&self) -> &InterPairs {
        self.inter.pairs()
    }
    /// Fragments (connected components).
    pub fn fragments(&self) -> &[Vec<usize>] {
        &self.inter.pairs().fragments
    }
}

/// Builder of λ-coupled soft-core force fields around an MMFF or UFF
/// builder (see the [module documentation](self)).
#[derive(Debug, Clone)]
pub struct SoftCoreBuilder<B: SoftCoreInner> {
    pub inner: B,
    pub params: SoftCoreParams,
    pub confinement: Option<Confinement>,
    state: Arc<SoftCoreState>,
    name: String,
}

impl<B: SoftCoreInner> SoftCoreBuilder<B> {
    /// Wraps `inner` with default soft-core parameters, λ = 1 and no
    /// confinement.
    pub fn new(inner: B) -> Self {
        let name = format!("{}+SoftCore", inner.name());
        SoftCoreBuilder {
            inner,
            params: SoftCoreParams::default(),
            confinement: None,
            state: Arc::new(SoftCoreState::new(1.0, 1.0, f64::INFINITY)),
            name,
        }
    }
    pub fn with_params(mut self, params: SoftCoreParams) -> Self {
        self.params = params;
        self
    }
    pub fn with_confinement(mut self, c: Option<Confinement>) -> Self {
        self.confinement = c;
        self.state.set_confinement_radius(c.map(|c| c.radius).unwrap_or(f64::INFINITY));
        self
    }
    /// Sets λ in the state shared by all force fields built through the
    /// [`ForceFieldBuilder`] trait.
    pub fn set_lambda(&self, vdw: f64, ele: f64) {
        self.state.set_lambda(vdw, ele)
    }
    /// State shared by all force fields built through [`ForceFieldBuilder::build`].
    pub fn shared_state(&self) -> &Arc<SoftCoreState> {
        &self.state
    }
    /// The plain inner builder with intermolecular interactions on (the
    /// λ = 1 reference).
    pub fn reference_builder(&self) -> B {
        self.inner.with_interfrag(true)
    }

    fn build_with_state(
        &self,
        mol: &Molecule,
        constraints: &[Constraint],
        state: Arc<SoftCoreState>,
    ) -> Result<(ContribForceField, SoftCoreHandle)> {
        if !(self.params.n >= 1.0 && self.params.p >= 1.0) {
            return Err(Error::InvalidArgument("soft-core: n and p must be >= 1".into()));
        }
        let mut frags = mol.fragments();
        for f in &mut frags {
            f.sort_unstable();
        }
        let (mut ff, pairs) = self.inner.build_parts(mol, constraints, frags, self.params.include_all_pairs)?;
        let pairs = Arc::new(pairs);
        let inter = SoftCoreInterContrib::new(pairs.clone(), self.params, state.clone());
        if !pairs.is_empty() {
            ff.add_contrib(Box::new(inter.clone()));
        }
        if let Some(c) = self.confinement {
            let frags = Arc::new(pairs.fragments.clone());
            ff.add_contrib(Box::new(CentroidWallContrib::new(frags, c.force_constant, state.clone())));
        }
        Ok((ff, SoftCoreHandle { state, inter }))
    }

    /// Builds a force field with its own λ state (initialized from the
    /// builder's shared state) and returns the handle to change it.
    pub fn build_with_handle(
        &self,
        mol: &Molecule,
        constraints: &[Constraint],
    ) -> Result<(ContribForceField, SoftCoreHandle)> {
        let (lv, le) = self.state.lambda();
        let st = Arc::new(SoftCoreState::new(lv, le, self.state.confinement_radius()));
        self.build_with_state(mol, constraints, st)
    }
}

impl<B: SoftCoreInner> ForceFieldBuilder for SoftCoreBuilder<B> {
    fn name(&self) -> &str {
        &self.name
    }
    /// Builds a force field that shares the builder's λ state.
    fn build(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<ContribForceField> {
        Ok(self.build_with_state(mol, constraints, self.state.clone())?.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, BondOrder};
    use crate::ff::ForceField;
    use crate::perception::sanitize;

    /// Two waters roughly in a hydrogen-bonded arrangement.
    fn water_dimer() -> Molecule {
        let mut m = Molecule::new();
        let w = |m: &mut Molecule, o: [f64; 3], h1: [f64; 3], h2: [f64; 3]| {
            let a = m.add_atom(Atom::new(8), o);
            let b = m.add_atom(Atom::new(1), h1);
            let c = m.add_atom(Atom::new(1), h2);
            m.add_bond(a, b, BondOrder::Single);
            m.add_bond(a, c, BondOrder::Single);
        };
        w(&mut m, [0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]);
        w(&mut m, [2.9, 0.1, 0.05], [3.25, -0.4, 0.75], [3.2, -0.3, -0.7]);
        sanitize(&mut m).unwrap();
        m
    }

    #[test]
    fn lambda_one_matches_mmff_interfrag() {
        let m = water_dimer();
        let reference =
            MmffBuilder { ignore_interfrag_interactions: false, ..Default::default() }.build(&m, &[]).unwrap();
        let sc = SoftCoreBuilder::new(MmffBuilder::default());
        let ff = sc.build(&m, &[]).unwrap();
        let pos = m.flat_coords();
        let (mut g1, mut g2) = (vec![0.0; pos.len()], vec![0.0; pos.len()]);
        let e1 = reference.energy_grad(&pos, &mut g1);
        let e2 = ff.energy_grad(&pos, &mut g2);
        assert!((e1 - e2).abs() <= 1e-12 * e1.abs().max(1.0), "{e1} vs {e2}");
        for (a, b) in g1.iter().zip(&g2) {
            assert!((a - b).abs() <= 1e-12 * a.abs().max(1.0));
        }
    }

    #[test]
    fn wall_gradient() {
        let m = water_dimer();
        let sc = SoftCoreBuilder::new(MmffBuilder::default())
            .with_confinement(Some(Confinement { radius: 0.5, force_constant: 3.0 }));
        let (ff, h) = sc.build_with_handle(&m, &[]).unwrap();
        h.set_lambda(0.3, 0.2);
        let pos = m.flat_coords();
        assert!(crate::ff::gradient_error(&ff, &pos) < 1e-5);
    }
}
