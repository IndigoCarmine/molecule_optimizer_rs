//! Energy terms used during distance-geometry embedding and the ETKDG
//! minimization. Each type collects many terms (like RDKit's `*Contribs`
//! classes) and sums them in insertion order, so energies and gradients are
//! accumulated in the same order as in RDKit.
//!
//! Ports of RDKit:
//! * `Code/DistGeom/DistViolationContribs.{h,cpp}`
//! * `Code/DistGeom/ChiralViolationContribs.{h,cpp}`
//! * `Code/DistGeom/FourthDimContribs.h`
//! * `Code/ForceField/DistanceConstraints.{h,cpp}`
//! * `Code/ForceField/AngleConstraints.{h,cpp}`
//! * `Code/ForceField/UFF/Inversions.{h,cpp}` and `UFF/Utils.cpp`
//! * `Code/GraphMol/ForceFieldHelpers/CrystalFF/TorsionAngleContribs.{h,cpp}`
//!   (with `Code/ForceField/MMFF/TorsionAngle.cpp` helpers)
//!
//! RDKit: Copyright (C) 2004-2026 Greg Landrum, Sereina Riniker and other
//! RDKit contributors, BSD-3-Clause.
//!
//! Note: like RDKit, the 4th-dimension and chiral-violation "gradients" are
//! half of the true derivative of the energy (RDKit omits the factor 2);
//! this is reproduced on purpose.

use super::point::P3;
use crate::ff::Contrib;

const RAD2DEG: f64 = 180.0 / std::f64::consts::PI;

#[inline]
fn is_double_zero(x: f64) -> bool {
    x < 1.0e-10 && x > -1.0e-10
}

#[inline]
fn distance2(pos: &[f64], i: usize, j: usize, dim: usize) -> f64 {
    let mut d2 = 0.0;
    for k in 0..dim {
        let d = pos[dim * i + k] - pos[dim * j + k];
        d2 += d * d;
    }
    d2
}

// ---------------------------------------------------------------- DistViolation

#[derive(Debug, Clone)]
struct DistViolationTerm {
    i: usize,
    j: usize,
    ub2: f64,
    lb2: f64,
    weight: f64,
}

/// Bounds-violation error function (RDKit `DistViolationContribs`).
#[derive(Debug, Clone)]
pub struct DistViolationContribs {
    dim: usize,
    terms: Vec<DistViolationTerm>,
}

impl DistViolationContribs {
    pub fn new(dim: usize) -> Self {
        DistViolationContribs { dim, terms: Vec::new() }
    }
    pub fn add(&mut self, i: usize, j: usize, ub: f64, lb: f64, weight: f64) {
        self.terms.push(DistViolationTerm { i, j, ub2: ub * ub, lb2: lb * lb, weight });
    }
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
    pub fn len(&self) -> usize {
        self.terms.len()
    }
}

impl Contrib for DistViolationContribs {
    fn energy(&self, pos: &[f64]) -> f64 {
        let mut accum = 0.0;
        for c in &self.terms {
            let d2 = distance2(pos, c.i, c.j, self.dim);
            let mut val = 0.0;
            if d2 > c.ub2 {
                val = (d2 / c.ub2) - 1.0;
            } else if d2 < c.lb2 {
                val = ((2.0 * c.lb2) / (c.lb2 + d2)) - 1.0;
            }
            if val > 0.0 {
                accum += c.weight * val * val;
            }
        }
        accum
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let dim = self.dim;
        for c in &self.terms {
            let d2 = distance2(pos, c.i, c.j, dim);
            let (d, pre) = if d2 > c.ub2 {
                let d = d2.sqrt();
                (d, 4. * (((d * d) / c.ub2) - 1.0) * (d / c.ub2))
            } else if d2 < c.lb2 {
                let d = d2.sqrt();
                let l2d2 = d2 + c.lb2;
                (d, 8. * c.lb2 * d * (1. - 2.0 * c.lb2 / l2d2) / (l2d2 * l2d2))
            } else {
                continue;
            };
            for k in 0..dim {
                let p1 = dim * c.i + k;
                let p2 = dim * c.j + k;
                let g = if d > 0.0 { c.weight * pre * (pos[p1] - pos[p2]) / d } else { c.weight * pre * (pos[p1] - pos[p2]) };
                grad[p1] += g;
                grad[p2] -= g;
            }
        }
    }
    fn label(&self) -> &'static str {
        "DistViolation"
    }
}

// ---------------------------------------------------------------- Chiral

/// RDKit `DistGeom::calcChiralVolume` on a flat coordinate array.
pub fn calc_chiral_volume(i1: usize, i2: usize, i3: usize, i4: usize, pos: &[f64], dim: usize) -> f64 {
    let p4 = P3::from_pos(pos, i4, dim);
    let v1 = P3::from_pos(pos, i1, dim) - p4;
    let v2 = P3::from_pos(pos, i2, dim) - p4;
    let v3 = P3::from_pos(pos, i3, dim) - p4;
    let v2xv3 = v2.cross(&v3);
    v1.dot(&v2xv3)
}

#[derive(Debug, Clone)]
struct ChiralTerm {
    idx: [usize; 4],
    vol_upper: f64,
    vol_lower: f64,
    weight: f64,
}

/// Chiral-volume violation (RDKit `ChiralViolationContribs`).
#[derive(Debug, Clone)]
pub struct ChiralViolationContribs {
    dim: usize,
    terms: Vec<ChiralTerm>,
}

impl ChiralViolationContribs {
    pub fn new(dim: usize) -> Self {
        ChiralViolationContribs { dim, terms: Vec::new() }
    }
    pub fn add(&mut self, idx: [usize; 4], vol_lower: f64, vol_upper: f64, weight: f64) {
        self.terms.push(ChiralTerm { idx, vol_upper, vol_lower, weight });
    }
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

impl Contrib for ChiralViolationContribs {
    fn energy(&self, pos: &[f64]) -> f64 {
        let mut res = 0.0;
        for c in &self.terms {
            let vol = calc_chiral_volume(c.idx[0], c.idx[1], c.idx[2], c.idx[3], pos, self.dim);
            if vol < c.vol_lower {
                res += c.weight * (vol - c.vol_lower) * (vol - c.vol_lower);
            } else if vol > c.vol_upper {
                res += c.weight * (vol - c.vol_upper) * (vol - c.vol_upper);
            }
        }
        res
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let dim = self.dim;
        for c in &self.terms {
            let [i1, i2, i3, i4] = c.idx;
            let p4 = P3::from_pos(pos, i4, dim);
            let v1 = P3::from_pos(pos, i1, dim) - p4;
            let v2 = P3::from_pos(pos, i2, dim) - p4;
            let v3 = P3::from_pos(pos, i3, dim) - p4;
            let v2xv3 = v2.cross(&v3);
            let vol = v1.dot(&v2xv3);
            let pre = if vol < c.vol_lower {
                c.weight * (vol - c.vol_lower)
            } else if vol > c.vol_upper {
                c.weight * (vol - c.vol_upper)
            } else {
                continue;
            };
            grad[dim * i1] += pre * ((v2.y) * (v3.z) - (v3.y) * (v2.z));
            grad[dim * i1 + 1] += pre * ((v3.x) * (v2.z) - (v2.x) * (v3.z));
            grad[dim * i1 + 2] += pre * ((v2.x) * (v3.y) - (v3.x) * (v2.y));

            grad[dim * i2] += pre * ((v3.y) * (v1.z) - (v3.z) * (v1.y));
            grad[dim * i2 + 1] += pre * ((v3.z) * (v1.x) - (v3.x) * (v1.z));
            grad[dim * i2 + 2] += pre * ((v3.x) * (v1.y) - (v3.y) * (v1.x));

            grad[dim * i3] += pre * ((v2.z) * (v1.y) - (v2.y) * (v1.z));
            grad[dim * i3 + 1] += pre * ((v2.x) * (v1.z) - (v2.z) * (v1.x));
            grad[dim * i3 + 2] += pre * ((v2.y) * (v1.x) - (v2.x) * (v1.y));

            let p = |i: usize, k: usize| pos[i * dim + k];
            grad[dim * i4] += pre
                * (p(i1, 2) * (p(i2, 1) - p(i3, 1)) + p(i2, 2) * (p(i3, 1) - p(i1, 1)) + p(i3, 2) * (p(i1, 1) - p(i2, 1)));
            grad[dim * i4 + 1] += pre
                * (p(i1, 0) * (p(i2, 2) - p(i3, 2)) + p(i2, 0) * (p(i3, 2) - p(i1, 2)) + p(i3, 0) * (p(i1, 2) - p(i2, 2)));
            grad[dim * i4 + 2] += pre
                * (p(i1, 1) * (p(i2, 0) - p(i3, 0)) + p(i2, 1) * (p(i3, 0) - p(i1, 0)) + p(i3, 1) * (p(i1, 0) - p(i2, 0)));
        }
    }
    fn label(&self) -> &'static str {
        "ChiralViolation"
    }
}

// ---------------------------------------------------------------- FourthDim

/// Penalty on the 4th coordinate (RDKit `FourthDimContribs`, dimension 4).
#[derive(Debug, Clone, Default)]
pub struct FourthDimContribs {
    terms: Vec<(usize, f64)>,
}

impl FourthDimContribs {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add(&mut self, idx: usize, weight: f64) {
        self.terms.push((idx, weight));
    }
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

impl Contrib for FourthDimContribs {
    fn energy(&self, pos: &[f64]) -> f64 {
        let mut res = 0.0;
        for &(i, w) in &self.terms {
            let pid = i * 4 + 3;
            res += w * pos[pid] * pos[pid];
        }
        res
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        for &(i, w) in &self.terms {
            let pid = i * 4 + 3;
            grad[pid] += w * pos[pid];
        }
    }
    fn label(&self) -> &'static str {
        "FourthDim"
    }
}

// ---------------------------------------------------------------- DistanceConstraint

#[derive(Debug, Clone)]
struct DistConstraintTerm {
    i: usize,
    j: usize,
    min_len: f64,
    max_len: f64,
    k: f64,
}

/// Flat-bottom harmonic distance constraints (RDKit
/// `DistanceConstraintContribs`).
#[derive(Debug, Clone)]
pub struct DistanceConstraintContribs {
    dim: usize,
    terms: Vec<DistConstraintTerm>,
}

impl DistanceConstraintContribs {
    pub fn new(dim: usize) -> Self {
        DistanceConstraintContribs { dim, terms: Vec::new() }
    }
    pub fn add(&mut self, i: usize, j: usize, min_len: f64, max_len: f64, k: f64) {
        self.terms.push(DistConstraintTerm { i, j, min_len, max_len, k });
    }
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

impl Contrib for DistanceConstraintContribs {
    fn energy(&self, pos: &[f64]) -> f64 {
        let mut accum = 0.0;
        for c in &self.terms {
            let (a, b) = if c.j < c.i { (c.j, c.i) } else { (c.i, c.j) };
            let d2 = distance2(pos, a, b, self.dim);
            let diff = if d2 < c.min_len * c.min_len {
                c.min_len - d2.sqrt()
            } else if d2 > c.max_len * c.max_len {
                d2.sqrt() - c.max_len
            } else {
                continue;
            };
            accum += 0.5 * c.k * diff * diff;
        }
        accum
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let dim = self.dim;
        for c in &self.terms {
            let (a, b) = if c.j < c.i { (c.j, c.i) } else { (c.i, c.j) };
            let d2 = distance2(pos, a, b, dim);
            let (dist, mut pre);
            if d2 < c.min_len * c.min_len {
                dist = d2.sqrt();
                pre = dist - c.min_len;
            } else if d2 > c.max_len * c.max_len {
                dist = d2.sqrt();
                pre = dist - c.max_len;
            } else {
                continue;
            }
            pre *= c.k;
            pre /= dist.max(1.0e-8);
            for k in 0..3 {
                let g = pre * (pos[dim * c.i + k] - pos[dim * c.j + k]);
                grad[dim * c.i + k] += g;
                grad[dim * c.j + k] -= g;
            }
        }
    }
    fn label(&self) -> &'static str {
        "DistanceConstraint"
    }
}

// ---------------------------------------------------------------- AngleConstraint

#[derive(Debug, Clone)]
struct AngleTerm {
    idx: [usize; 3],
    min_deg: f64,
    max_deg: f64,
    k: f64,
}

/// Flat-bottom angle constraints in degrees (RDKit `AngleConstraintContribs`).
#[derive(Debug, Clone)]
pub struct AngleConstraintContribs {
    dim: usize,
    terms: Vec<AngleTerm>,
}

impl AngleConstraintContribs {
    pub fn new(dim: usize) -> Self {
        AngleConstraintContribs { dim, terms: Vec::new() }
    }
    pub fn add(&mut self, i: usize, j: usize, k: usize, min_deg: f64, max_deg: f64, force: f64) {
        self.terms.push(AngleTerm { idx: [i, j, k], min_deg, max_deg, k: force });
    }
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
    /// Per-term energies (RDKit `getEnergy(positions, energies)`).
    pub fn term_energies(&self, pos: &[f64]) -> Vec<f64> {
        self.terms.iter().map(|c| self.term_energy(pos, c)).collect()
    }
    fn angle_of(&self, pos: &[f64], c: &AngleTerm) -> (P3, P3, [f64; 2], f64) {
        let p1 = P3::from_pos(pos, c.idx[0], self.dim);
        let p2 = P3::from_pos(pos, c.idx[1], self.dim);
        let p3 = P3::from_pos(pos, c.idx[2], self.dim);
        let r0 = p1 - p2;
        let r1 = p3 - p2;
        let l = [r0.length_sq().max(1.0e-5), r1.length_sq().max(1.0e-5)];
        let cos_t = (r0.dot(&r1) / (l[0] * l[1]).sqrt()).clamp(-1.0, 1.0);
        (r0, r1, l, RAD2DEG * cos_t.acos())
    }
    fn angle_term(c: &AngleTerm, angle: f64) -> f64 {
        if angle < c.min_deg {
            angle - c.min_deg
        } else if angle > c.max_deg {
            angle - c.max_deg
        } else {
            0.0
        }
    }
    fn term_energy(&self, pos: &[f64], c: &AngleTerm) -> f64 {
        let (_, _, _, angle) = self.angle_of(pos, c);
        let t = Self::angle_term(c, angle);
        c.k * t * t
    }
}

impl Contrib for AngleConstraintContribs {
    fn energy(&self, pos: &[f64]) -> f64 {
        let mut accum = 0.0;
        for c in &self.terms {
            accum += self.term_energy(pos, c);
        }
        accum
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let dim = self.dim;
        for c in &self.terms {
            let (r0, r1, l, angle) = self.angle_of(pos, c);
            let at = Self::angle_term(c, angle);
            let de_dtheta = 2.0 * RAD2DEG * c.k * at;
            let rp = r1.cross(&r0);
            let pre = de_dtheta / rp.length().max(1.0e-5);
            let t = [-pre / l[0], pre / l[1]];
            let d0 = r0.cross(&rp) * t[0];
            let d2 = r1.cross(&rp) * t[1];
            let d1 = (-d0) - d2;
            for (n, d) in [(c.idx[0], d0), (c.idx[1], d1), (c.idx[2], d2)] {
                grad[dim * n] += d.x;
                grad[dim * n + 1] += d.y;
                grad[dim * n + 2] += d.z;
            }
        }
    }
    fn label(&self) -> &'static str {
        "AngleConstraint"
    }
}

// ---------------------------------------------------------------- UFF inversion

/// RDKit `UFF::Utils::calcInversionCoefficientsAndForceConstant`:
/// returns `(forceConstant, C0, C1, C2)`.
pub fn inversion_coefficients(at2_atomic_num: i32, is_c_bound_to_o: bool) -> (f64, f64, f64, f64) {
    let (mut res, c0, c1, c2);
    if at2_atomic_num == 6 || at2_atomic_num == 7 || at2_atomic_num == 8 {
        c0 = 1.0;
        c1 = -1.0;
        c2 = 0.0;
        res = if is_c_bound_to_o { 50.0 } else { 6.0 };
    } else {
        let mut w0 = std::f64::consts::PI / 180.0;
        match at2_atomic_num {
            15 => w0 *= 84.4339,
            33 => w0 *= 86.9735,
            51 => w0 *= 87.7047,
            83 => w0 *= 90.0,
            _ => {}
        }
        c2 = 1.0;
        c1 = -4.0 * w0.cos();
        c0 = -(c1 * w0.cos() + c2 * (2.0 * w0).cos());
        res = 22.0 / (c0 + c1 + c2);
    }
    res /= 3.0;
    (res, c0, c1, c2)
}

/// RDKit `UFF::Utils::calculateCosY`.
pub fn calculate_cos_y(p1: P3, p2: P3, p3: P3, p4: P3) -> f64 {
    const ZERO_TOL: f64 = 1.0e-16;
    let r_ji = p1 - p2;
    let r_jk = p3 - p2;
    let r_jl = p4 - p2;
    let l2ji = r_ji.length_sq();
    let l2jk = r_jk.length_sq();
    let l2jl = r_jl.length_sq();
    if l2ji < ZERO_TOL || l2jk < ZERO_TOL || l2jl < ZERO_TOL {
        return 0.0;
    }
    let mut n = r_ji.cross(&r_jk);
    n /= l2ji.sqrt() * l2jk.sqrt();
    let l2n = n.length_sq();
    if l2n < ZERO_TOL {
        return 0.0;
    }
    n.dot(&r_jl) / (l2jl.sqrt() * l2n.sqrt())
}

#[derive(Debug, Clone)]
struct InversionTerm {
    idx: [usize; 4],
    c0: f64,
    c1: f64,
    c2: f64,
    k: f64,
}

/// UFF out-of-plane (inversion) terms (RDKit `UFF::InversionContribs`).
#[derive(Debug, Clone)]
pub struct InversionContribs {
    dim: usize,
    terms: Vec<InversionTerm>,
}

impl InversionContribs {
    pub fn new(dim: usize) -> Self {
        InversionContribs { dim, terms: Vec::new() }
    }
    pub fn add(&mut self, idx: [usize; 4], at2_atomic_num: i32, is_c_bound_to_o: bool, scaling: f64) {
        let (k, c0, c1, c2) = inversion_coefficients(at2_atomic_num, is_c_bound_to_o);
        self.terms.push(InversionTerm { idx, c0, c1, c2, k: k * scaling });
    }
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
    fn term_energy(&self, pos: &[f64], c: &InversionTerm) -> f64 {
        let p = |i| P3::from_pos(pos, i, self.dim);
        let cos_y = calculate_cos_y(p(c.idx[0]), p(c.idx[1]), p(c.idx[2]), p(c.idx[3]));
        let sin_y_sq = 1.0 - cos_y * cos_y;
        let sin_y = if sin_y_sq > 0.0 { sin_y_sq.sqrt() } else { 0.0 };
        let cos2w = 2.0 * sin_y * sin_y - 1.0;
        c.k * (c.c0 + c.c1 * sin_y + c.c2 * cos2w)
    }
    /// Per-term energies.
    pub fn term_energies(&self, pos: &[f64]) -> Vec<f64> {
        self.terms.iter().map(|c| self.term_energy(pos, c)).collect()
    }
}

impl Contrib for InversionContribs {
    fn energy(&self, pos: &[f64]) -> f64 {
        let mut accum = 0.0;
        for c in &self.terms {
            accum += self.term_energy(pos, c);
        }
        accum
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let dim = self.dim;
        for c in &self.terms {
            let p = |i| P3::from_pos(pos, i, dim);
            let (p1, p2, p3, p4) = (p(c.idx[0]), p(c.idx[1]), p(c.idx[2]), p(c.idx[3]));
            let mut r_ji = p1 - p2;
            let mut r_jk = p3 - p2;
            let mut r_jl = p4 - p2;
            let d_ji = r_ji.length();
            let d_jk = r_jk.length();
            let d_jl = r_jl.length();
            if is_double_zero(d_ji) || is_double_zero(d_jk) || is_double_zero(d_jl) {
                // RDKit returns from the whole loop here
                return;
            }
            r_ji.normalize();
            r_jk.normalize();
            r_jl.normalize();
            let mut n = (-r_ji).cross(&r_jk);
            n.normalize();
            let cos_y = n.dot(&r_jl).clamp(-1.0, 1.0);
            let sin_y_sq = 1.0 - cos_y * cos_y;
            let sin_y = sin_y_sq.sqrt().max(1.0e-8);
            let cos_t = r_ji.dot(&r_jk).clamp(-1.0, 1.0);
            let sin_t_sq = 1.0 - cos_t * cos_t;
            let sin_t = sin_t_sq.sqrt().max(1.0e-8);
            let de_dw = -c.k * (c.c1 * cos_y + 4.0 * c.c2 * cos_y * sin_y);
            let t1 = r_jl.cross(&r_jk);
            let t2 = r_ji.cross(&r_jl);
            let t3 = r_jk.cross(&r_ji);
            let term1 = sin_y * sin_t;
            let term2 = cos_y / (sin_y * sin_t_sq);
            let tg1 = [
                (t1.x / term1 - (r_ji.x - r_jk.x * cos_t) * term2) / d_ji,
                (t1.y / term1 - (r_ji.y - r_jk.y * cos_t) * term2) / d_ji,
                (t1.z / term1 - (r_ji.z - r_jk.z * cos_t) * term2) / d_ji,
            ];
            let tg3 = [
                (t2.x / term1 - (r_jk.x - r_ji.x * cos_t) * term2) / d_jk,
                (t2.y / term1 - (r_jk.y - r_ji.y * cos_t) * term2) / d_jk,
                (t2.z / term1 - (r_jk.z - r_ji.z * cos_t) * term2) / d_jk,
            ];
            let tg4 = [
                (t3.x / term1 - r_jl.x * cos_y / sin_y) / d_jl,
                (t3.y / term1 - r_jl.y * cos_y / sin_y) / d_jl,
                (t3.z / term1 - r_jl.z * cos_y / sin_y) / d_jl,
            ];
            for k in 0..3 {
                grad[dim * c.idx[0] + k] += de_dw * tg1[k];
                grad[dim * c.idx[1] + k] += -de_dw * (tg1[k] + tg3[k] + tg4[k]);
                grad[dim * c.idx[2] + k] += de_dw * tg3[k];
                grad[dim * c.idx[3] + k] += de_dw * tg4[k];
            }
        }
    }
    fn label(&self) -> &'static str {
        "Inversion"
    }
}

// ---------------------------------------------------------------- CrystalFF torsions

/// RDKit `CrystalFF::calcTorsionEnergy`.
pub fn calc_torsion_energy(v: &[f64; 6], s: &[i32; 6], cos_phi: f64) -> f64 {
    let c2 = cos_phi * cos_phi;
    let c3 = cos_phi * c2;
    let c4 = cos_phi * c3;
    let c5 = cos_phi * c4;
    let c6 = cos_phi * c5;
    let cos2 = 2.0 * c2 - 1.0;
    let cos3 = 4.0 * c3 - 3.0 * cos_phi;
    let cos4 = 8.0 * c4 - 8.0 * c2 + 1.0;
    let cos5 = 16.0 * c5 - 20.0 * c3 + 5.0 * cos_phi;
    let cos6 = 32.0 * c6 - 48.0 * c4 + 18.0 * c2 - 1.0;
    let sf = |k: usize| s[k] as f64;
    v[0] * (1.0 + sf(0) * cos_phi)
        + v[1] * (1.0 + sf(1) * cos2)
        + v[2] * (1.0 + sf(2) * cos3)
        + v[3] * (1.0 + sf(3) * cos4)
        + v[4] * (1.0 + sf(4) * cos5)
        + v[5] * (1.0 + sf(5) * cos6)
}

/// RDKit `MMFF::Utils::calcTorsionCosPhi`.
pub fn calc_torsion_cos_phi(p1: P3, p2: P3, p3: P3, p4: P3) -> f64 {
    let r1 = p1 - p2;
    let r2 = p3 - p2;
    let r3 = p2 - p3;
    let r4 = p4 - p3;
    let t1 = r1.cross(&r2);
    let t2 = r3.cross(&r4);
    let l1 = t1.length();
    let l2 = t2.length();
    if is_double_zero(l1) || is_double_zero(l2) {
        return 0.0;
    }
    (t1.dot(&t2) / (l1 * l2)).clamp(-1.0, 1.0)
}

/// RDKit `MMFF::Utils::calcTorsionGrad`.
fn calc_torsion_grad(r: &[P3; 4], t: &[P3; 2], d: &[f64; 2], g: &mut [[f64; 3]; 4], sin_term: f64, cos_phi: f64) {
    let dcos = [
        1.0 / d[0] * (t[1].x - cos_phi * t[0].x),
        1.0 / d[0] * (t[1].y - cos_phi * t[0].y),
        1.0 / d[0] * (t[1].z - cos_phi * t[0].z),
        1.0 / d[1] * (t[0].x - cos_phi * t[1].x),
        1.0 / d[1] * (t[0].y - cos_phi * t[1].y),
        1.0 / d[1] * (t[0].z - cos_phi * t[1].z),
    ];
    g[0][0] += sin_term * (dcos[2] * r[1].y - dcos[1] * r[1].z);
    g[0][1] += sin_term * (dcos[0] * r[1].z - dcos[2] * r[1].x);
    g[0][2] += sin_term * (dcos[1] * r[1].x - dcos[0] * r[1].y);

    g[1][0] += sin_term
        * (dcos[1] * (r[1].z - r[0].z) + dcos[2] * (r[0].y - r[1].y) + dcos[4] * (-r[3].z) + dcos[5] * (r[3].y));
    g[1][1] += sin_term
        * (dcos[0] * (r[0].z - r[1].z) + dcos[2] * (r[1].x - r[0].x) + dcos[3] * (r[3].z) + dcos[5] * (-r[3].x));
    g[1][2] += sin_term
        * (dcos[0] * (r[1].y - r[0].y) + dcos[1] * (r[0].x - r[1].x) + dcos[3] * (-r[3].y) + dcos[4] * (r[3].x));

    g[2][0] += sin_term
        * (dcos[1] * (r[0].z) + dcos[2] * (-r[0].y) + dcos[4] * (r[3].z - r[2].z) + dcos[5] * (r[2].y - r[3].y));
    g[2][1] += sin_term
        * (dcos[0] * (-r[0].z) + dcos[2] * (r[0].x) + dcos[3] * (r[2].z - r[3].z) + dcos[5] * (r[3].x - r[2].x));
    g[2][2] += sin_term
        * (dcos[0] * (r[0].y) + dcos[1] * (-r[0].x) + dcos[3] * (r[3].y - r[2].y) + dcos[4] * (r[2].x - r[3].x));

    g[3][0] += sin_term * (dcos[4] * r[2].z - dcos[5] * r[2].y);
    g[3][1] += sin_term * (dcos[5] * r[2].x - dcos[3] * r[2].z);
    g[3][2] += sin_term * (dcos[3] * r[2].y - dcos[4] * r[2].x);
}

#[derive(Debug, Clone)]
struct TorsionTerm {
    idx: [usize; 4],
    v: [f64; 6],
    s: [i32; 6],
}

/// Experimental torsion-angle preference terms (RDKit
/// `CrystalFF::TorsionAngleContribs`, cosine series).
#[derive(Debug, Clone)]
pub struct TorsionAngleContribs {
    dim: usize,
    terms: Vec<TorsionTerm>,
}

impl TorsionAngleContribs {
    pub fn new(dim: usize) -> Self {
        TorsionAngleContribs { dim, terms: Vec::new() }
    }
    pub fn add(&mut self, idx: [usize; 4], v: [f64; 6], s: [i32; 6]) {
        self.terms.push(TorsionTerm { idx, v, s });
    }
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

impl Contrib for TorsionAngleContribs {
    fn energy(&self, pos: &[f64]) -> f64 {
        let mut accum = 0.0;
        for c in &self.terms {
            let p = |i| P3::from_pos(pos, i, self.dim);
            accum += calc_torsion_energy(&c.v, &c.s, calc_torsion_cos_phi(p(c.idx[0]), p(c.idx[1]), p(c.idx[2]), p(c.idx[3])));
        }
        accum
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let dim = self.dim;
        for c in &self.terms {
            let p = |i| P3::from_pos(pos, i, dim);
            let (ip, jp, kp, lp) = (p(c.idx[0]), p(c.idx[1]), p(c.idx[2]), p(c.idx[3]));
            let r = [ip - jp, kp - jp, jp - kp, lp - kp];
            let mut t = [r[0].cross(&r[1]), r[2].cross(&r[3])];
            let d = [t[0].length(), t[1].length()];
            if is_double_zero(d[0]) || is_double_zero(d[1]) {
                continue;
            }
            t[0] /= d[0];
            t[1] /= d[1];
            let cos_phi = t[0].dot(&t[1]).clamp(-1.0, 1.0);
            let sin_phi_sq = 1.0 - cos_phi * cos_phi;
            let sin_phi = if sin_phi_sq > 0.0 { sin_phi_sq.sqrt() } else { 0.0 };
            let c2 = cos_phi * cos_phi;
            let c3 = cos_phi * c2;
            let c4 = cos_phi * c3;
            let c5 = cos_phi * c4;
            let v = &c.v;
            let s = |k: usize| c.s[k] as f64;
            // NB: RDKit uses forceConstants[4]/signs[4] for the 6-fold term.
            let de_dphi = -v[0] * s(0) * sin_phi
                - 2.0 * v[1] * s(1) * (2.0 * cos_phi * sin_phi)
                - 3.0 * v[2] * s(2) * (4.0 * c2 * sin_phi - sin_phi)
                - 4.0 * v[3] * s(3) * (8.0 * c3 * sin_phi - 4.0 * cos_phi * sin_phi)
                - 5.0 * v[4] * s(4) * (16.0 * c4 * sin_phi - 12.0 * c2 * sin_phi + sin_phi)
                - 6.0 * v[4] * s(4) * (32.0 * c5 * sin_phi - 32.0 * c3 * sin_phi + 6.0 * sin_phi);
            let sin_term = -de_dphi * (if is_double_zero(sin_phi) { 1.0 / cos_phi } else { 1.0 / sin_phi });
            let mut g = [[0.0; 3]; 4];
            for (gk, &ik) in g.iter_mut().zip(&c.idx) {
                *gk = [grad[dim * ik], grad[dim * ik + 1], grad[dim * ik + 2]];
            }
            calc_torsion_grad(&r, &t, &d, &mut g, sin_term, cos_phi);
            for (gk, &ik) in g.iter().zip(&c.idx) {
                grad[dim * ik] = gk[0];
                grad[dim * ik + 1] = gk[1];
                grad[dim * ik + 2] = gk[2];
            }
        }
    }
    fn label(&self) -> &'static str {
        "TorsionPreference"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::{ContribForceField, ForceField, numerical_gradient};

    /// Deterministic pseudo-random coordinates.
    fn coords(n: usize, dim: usize, seed: u32) -> Vec<f64> {
        let mut r = crate::distgeom::rng::DoubleSource::new(seed);
        (0..n * dim).map(|_| 3.0 * (r.next_f64() - 0.5)).collect()
    }

    /// max |scale * analytic - numeric|
    fn grad_err(c: Box<dyn Contrib>, n: usize, dim: usize, pos: &[f64], scale: f64) -> f64 {
        let mut ff = ContribForceField::new(n, dim);
        ff.add_contrib(c);
        let mut g = vec![0.0; n * dim];
        ff.energy_grad(pos, &mut g);
        let num = numerical_gradient(&ff, pos, 1e-6);
        g.iter().zip(&num).map(|(a, b)| (scale * a - b).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn dist_violation_gradient_3d_4d() {
        for dim in [3, 4] {
            let pos = coords(5, dim, 11);
            let mut c = DistViolationContribs::new(dim);
            c.add(0, 1, 0.5, 0.2, 1.0); // upper violated
            c.add(2, 3, 9.0, 8.0, 0.7); // lower violated
            c.add(1, 4, 1.4, 1.0, 1.0);
            let e = grad_err(Box::new(c), 5, dim, &pos, 1.0);
            assert!(e < 1e-5, "dim {dim}: {e}");
        }
    }

    #[test]
    fn chiral_violation_gradient_is_half_of_derivative() {
        for dim in [3, 4] {
            let pos = coords(5, dim, 3);
            let mut c = ChiralViolationContribs::new(dim);
            c.add([0, 1, 2, 3], 5.0, 100.0, 1.0);
            c.add([1, 2, 3, 4], -100.0, -5.0, 1.0);
            // RDKit omits the factor 2 of d/dV (V - V0)^2
            let e = grad_err(Box::new(c), 5, dim, &pos, 2.0);
            assert!(e < 1e-5, "dim {dim}: {e}");
        }
    }

    #[test]
    fn fourth_dim_gradient_is_half_of_derivative() {
        let pos = coords(4, 4, 5);
        let mut c = FourthDimContribs::new();
        for i in 0..4 {
            c.add(i, 0.3);
        }
        assert!(grad_err(Box::new(c), 4, 4, &pos, 2.0) < 1e-6);
    }

    #[test]
    fn distance_constraint_gradient_3d_4d() {
        for dim in [3, 4] {
            let mut pos = coords(4, dim, 7);
            if dim == 4 {
                // RDKit's gradient only covers x,y,z: keep the 4th coordinate equal
                for i in 0..4 {
                    pos[4 * i + 3] = 0.25;
                }
            }
            let mut c = DistanceConstraintContribs::new(dim);
            c.add(0, 1, 5.0, 6.0, 100.0);
            c.add(2, 3, 0.1, 0.2, 10.0);
            let e = grad_err(Box::new(c), 4, dim, &pos, 1.0);
            assert!(e < 1e-4, "dim {dim}: {e}");
        }
    }

    #[test]
    fn angle_constraint_gradient() {
        let pos = coords(4, 3, 9);
        let mut c = AngleConstraintContribs::new(3);
        c.add(0, 1, 2, 179.0, 180.0, 1.0);
        c.add(1, 2, 3, 10.0, 20.0, 1.0);
        assert!(grad_err(Box::new(c), 4, 3, &pos, 1.0) < 1e-4);
    }

    #[test]
    fn inversion_gradient() {
        let mut pos = coords(4, 3, 13);
        pos[3] += 1.0;
        let mut c = InversionContribs::new(3);
        c.add([0, 1, 2, 3], 6, false, 10.0);
        c.add([2, 1, 3, 0], 6, true, 10.0);
        c.add([0, 1, 3, 2], 15, false, 1.0);
        assert!(grad_err(Box::new(c), 4, 3, &pos, 1.0) < 1e-4);
    }

    #[test]
    fn torsion_gradient() {
        let pos = coords(4, 3, 17);
        let mut c = TorsionAngleContribs::new(3);
        // RDKit adds a bogus 6-fold term with V[4] to the gradient, so V[4] = V[5] = 0 here
        c.add([0, 1, 2, 3], [1.0, 5.0, 2.5, 0.7, 0.0, 0.0], [1, -1, 1, -1, 1, 1]);
        assert!(grad_err(Box::new(c), 4, 3, &pos, 1.0) < 1e-4);
    }
}
