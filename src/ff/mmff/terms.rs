//! MMFF94 energy terms with analytic gradients.
//!
//! Port of RDKit `Code/ForceField/MMFF/{BondStretch,AngleBend,StretchBend,
//! OopBend,TorsionAngle,Nonbonded}.cpp` and of
//! `ForceFieldsHelper::computeDihedral` (`Code/ForceField/ForceField.cpp`).
//! As in RDKit, every contrib holds all terms of one kind, and every
//! floating-point expression is evaluated in the same order as in RDKit so
//! that energies, gradients and minimization trajectories agree to the last
//! bit wherever the platform math library agrees.
//!
//! RDKit: Copyright (C) 2013-2025 Paolo Tosco, Greg Landrum and other RDKit
//! contributors. BSD 3-Clause license (see `LICENSE-RDKIT.txt`).

use super::params::{MmffVdw, VdwTable};
use super::{DEG2RAD, DielectricModel, MDYNE_A_TO_KCAL_MOL, RAD2DEG};
use crate::ff::Contrib;
use crate::ff::mmff::typer::is_double_zero;

// ---------------------------------------------------------------------------
// RDKit Point3D arithmetic (component-wise, same operation order)
// ---------------------------------------------------------------------------

type P3 = [f64; 3];

#[inline]
fn pt(pos: &[f64], i: usize) -> P3 {
    [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]
}
#[inline]
fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn neg(a: P3) -> P3 {
    [-a[0], -a[1], -a[2]]
}
#[inline]
fn div(a: P3, s: f64) -> P3 {
    [a[0] / s, a[1] / s, a[2] / s]
}
#[inline]
fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn length(a: P3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
/// `Point3D::crossProduct`
#[inline]
fn cross(a: P3, b: P3) -> P3 {
    [a[1] * b[2] - a[2] * b[1], -a[0] * b[2] + a[2] * b[0], a[0] * b[1] - a[1] * b[0]]
}
#[inline]
fn clip_to_one(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}
/// `ForceField::distance(i, j, pos)`
#[inline]
fn distance(pos: &[f64], i: usize, j: usize) -> f64 {
    let (i, j) = if j < i { (j, i) } else { (i, j) };
    let mut res = 0.0;
    for d in 0..3 {
        let tmp = pos[3 * i + d] - pos[3 * j + d];
        res += tmp * tmp;
    }
    res.sqrt()
}
#[inline]
fn add_to(grad: &mut [f64], i: usize, d: usize, v: f64) {
    grad[3 * i + d] += v;
}

// ---------------------------------------------------------------------------
// bond stretch
// ---------------------------------------------------------------------------

/// RDKit `MMFF::Utils::calcBondStretchEnergy`.
pub fn calc_bond_stretch_energy(r0: f64, kb: f64, distance: f64) -> f64 {
    let dist_term = distance - r0;
    let dist_term2 = dist_term * dist_term;
    let c1 = MDYNE_A_TO_KCAL_MOL;
    let cs = -2.0;
    let c3 = 7.0 / 12.0;
    0.5 * c1 * kb * dist_term2 * (1.0 + cs * dist_term + c3 * cs * cs * dist_term2)
}

/// All MMFF bond stretching terms (RDKit `BondStretchContrib`).
#[derive(Debug, Clone, Default)]
pub struct BondStretchContrib {
    pub(crate) at1: Vec<usize>,
    pub(crate) at2: Vec<usize>,
    pub(crate) r0: Vec<f64>,
    pub(crate) kb: Vec<f64>,
}

impl BondStretchContrib {
    pub fn add_term(&mut self, i: usize, j: usize, r0: f64, kb: f64) {
        self.at1.push(i);
        self.at2.push(j);
        self.r0.push(r0);
        self.kb.push(kb);
    }
    pub fn len(&self) -> usize {
        self.at1.len()
    }
    pub fn is_empty(&self) -> bool {
        self.at1.is_empty()
    }
}

impl Contrib for BondStretchContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        let mut e = 0.0;
        for t in 0..self.at1.len() {
            e += calc_bond_stretch_energy(self.r0[t], self.kb[t], distance(pos, self.at1[t], self.at2[t]));
        }
        e
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        const CS: f64 = -2.0;
        const C1: f64 = MDYNE_A_TO_KCAL_MOL;
        const C3: f64 = 7.0 / 12.0;
        for t in 0..self.at1.len() {
            let (i1, i2) = (self.at1[t], self.at2[t]);
            let dist = distance(pos, i1, i2);
            let dist_term = dist - self.r0[t];
            let de_dr = C1 * self.kb[t] * dist_term * (1.0 + 1.5 * CS * dist_term + 2.0 * C3 * CS * CS * dist_term * dist_term);
            for d in 0..3 {
                let dg = if dist > 0.0 { de_dr * (pos[3 * i1 + d] - pos[3 * i2 + d]) / dist } else { self.kb[t] * 0.01 };
                add_to(grad, i1, d, dg);
                add_to(grad, i2, d, -dg);
            }
        }
    }
    fn label(&self) -> &'static str {
        "BondStretch"
    }
}

// ---------------------------------------------------------------------------
// angle bend
// ---------------------------------------------------------------------------

/// RDKit `MMFF::Utils::calcCosTheta`.
pub fn calc_cos_theta(p1: P3, p2: P3, p3: P3, dist1: f64, dist2: f64) -> f64 {
    let p12 = sub(p1, p2);
    let p32 = sub(p3, p2);
    clip_to_one(dot(p12, p32) / (dist1 * dist2))
}

const ANGLE_CB: f64 = -0.006981317;
const ANGLE_C2: f64 = MDYNE_A_TO_KCAL_MOL * DEG2RAD * DEG2RAD;

/// RDKit `MMFF::Utils::calcAngleBendEnergy`.
pub fn calc_angle_bend_energy(theta0: f64, ka: f64, is_linear: bool, cos_theta: f64) -> f64 {
    let angle = RAD2DEG * cos_theta.acos() - theta0;
    if is_linear {
        MDYNE_A_TO_KCAL_MOL * ka * (1.0 + cos_theta)
    } else {
        0.5 * ANGLE_C2 * ka * angle * angle * (1.0 + ANGLE_CB * angle)
    }
}

/// All MMFF angle bending terms (RDKit `AngleBendContrib`).
#[derive(Debug, Clone, Default)]
pub struct AngleBendContrib {
    pub(crate) at: Vec<[usize; 3]>,
    pub(crate) is_linear: Vec<bool>,
    pub(crate) theta0: Vec<f64>,
    pub(crate) ka: Vec<f64>,
}

impl AngleBendContrib {
    pub fn add_term(&mut self, i: usize, j: usize, k: usize, theta0: f64, ka: f64, is_linear: bool) {
        self.at.push([i, j, k]);
        self.is_linear.push(is_linear);
        self.theta0.push(theta0);
        self.ka.push(ka);
    }
    pub fn len(&self) -> usize {
        self.at.len()
    }
    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }
}

/// RDKit `MMFF::Utils::calcAngleBendGrad`.
fn calc_angle_bend_grad(r: &[P3; 2], dist: &[f64; 2], g: &mut [f64], idx: [usize; 3], de_dtheta: f64, cos_theta: f64, sin_theta: f64) {
    let dcos = [
        1.0 / dist[0] * (r[1][0] - cos_theta * r[0][0]),
        1.0 / dist[0] * (r[1][1] - cos_theta * r[0][1]),
        1.0 / dist[0] * (r[1][2] - cos_theta * r[0][2]),
        1.0 / dist[1] * (r[0][0] - cos_theta * r[1][0]),
        1.0 / dist[1] * (r[0][1] - cos_theta * r[1][1]),
        1.0 / dist[1] * (r[0][2] - cos_theta * r[1][2]),
    ];
    for d in 0..3 {
        add_to(g, idx[0], d, de_dtheta * dcos[d] / (-sin_theta));
    }
    for d in 0..3 {
        add_to(g, idx[1], d, de_dtheta * (-dcos[d] - dcos[d + 3]) / (-sin_theta));
    }
    for d in 0..3 {
        add_to(g, idx[2], d, de_dtheta * dcos[d + 3] / (-sin_theta));
    }
}

impl Contrib for AngleBendContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        let mut res = 0.0;
        for t in 0..self.at.len() {
            let [i, j, k] = self.at[t];
            let dist1 = distance(pos, i, j);
            let dist2 = distance(pos, j, k);
            let cos = calc_cos_theta(pt(pos, i), pt(pos, j), pt(pos, k), dist1, dist2);
            res += calc_angle_bend_energy(self.theta0[t], self.ka[t], self.is_linear[t], cos);
        }
        res
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        for t in 0..self.at.len() {
            let [i, j, k] = self.at[t];
            let dist = [distance(pos, i, j), distance(pos, j, k)];
            let (p1, p2, p3) = (pt(pos, i), pt(pos, j), pt(pos, k));
            let r = [div(sub(p1, p2), dist[0]), div(sub(p3, p2), dist[1])];
            let cos_theta = clip_to_one(dot(r[0], r[1]));
            let sin_theta_sq = 1.0 - cos_theta * cos_theta;
            let sin_theta = (if sin_theta_sq > 0.0 { sin_theta_sq.sqrt() } else { 0.0 }).max(1.0e-8);
            let angle_term = RAD2DEG * cos_theta.acos() - self.theta0[t];
            let de_dtheta = if self.is_linear[t] {
                -MDYNE_A_TO_KCAL_MOL * self.ka[t] * sin_theta
            } else {
                RAD2DEG * ANGLE_C2 * self.ka[t] * angle_term * (1.0 + 1.5 * ANGLE_CB * angle_term)
            };
            calc_angle_bend_grad(&r, &dist, grad, [i, j, k], de_dtheta, cos_theta, sin_theta);
        }
    }
    fn label(&self) -> &'static str {
        "AngleBend"
    }
}

// ---------------------------------------------------------------------------
// stretch-bend
// ---------------------------------------------------------------------------

/// RDKit `MMFF::Utils::calcStretchBendEnergy`.
pub fn calc_stretch_bend_energy(delta_dist1: f64, delta_dist2: f64, delta_theta: f64, k: (f64, f64)) -> (f64, f64) {
    let factor = MDYNE_A_TO_KCAL_MOL * DEG2RAD * delta_theta;
    (factor * k.0 * delta_dist1, factor * k.1 * delta_dist2)
}

/// All MMFF stretch-bend terms (RDKit `StretchBendContrib`).
#[derive(Debug, Clone, Default)]
pub struct StretchBendContrib {
    pub(crate) at: Vec<[usize; 3]>,
    pub(crate) rest_len1: Vec<f64>,
    pub(crate) rest_len2: Vec<f64>,
    pub(crate) theta0: Vec<f64>,
    pub(crate) k1: Vec<f64>,
    pub(crate) k2: Vec<f64>,
}

impl StretchBendContrib {
    #[allow(clippy::too_many_arguments)]
    pub fn add_term(&mut self, i: usize, j: usize, k: usize, r1: f64, r2: f64, theta0: f64, kba_ijk: f64, kba_kji: f64) {
        self.at.push([i, j, k]);
        self.rest_len1.push(r1);
        self.rest_len2.push(r2);
        self.theta0.push(theta0);
        self.k1.push(kba_ijk);
        self.k2.push(kba_kji);
    }
    pub fn len(&self) -> usize {
        self.at.len()
    }
    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }
}

impl Contrib for StretchBendContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        let mut total = 0.0;
        for t in 0..self.at.len() {
            let [i, j, k] = self.at[t];
            let dist1 = distance(pos, i, j);
            let dist2 = distance(pos, j, k);
            let cos = calc_cos_theta(pt(pos, i), pt(pos, j), pt(pos, k), dist1, dist2);
            let e = calc_stretch_bend_energy(
                dist1 - self.rest_len1[t],
                dist2 - self.rest_len2[t],
                RAD2DEG * cos.acos() - self.theta0[t],
                (self.k1[t], self.k2[t]),
            );
            total += e.0 + e.1;
        }
        total
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        const C5: f64 = MDYNE_A_TO_KCAL_MOL * DEG2RAD;
        for t in 0..self.at.len() {
            let [i1, i2, i3] = self.at[t];
            let theta0 = self.theta0[t];
            let fc1 = self.k1[t];
            let fc2 = self.k2[t];
            let dist1 = distance(pos, i1, i2);
            let dist2 = distance(pos, i2, i3);
            let (p1, p2, p3) = (pt(pos, i1), pt(pos, i2), pt(pos, i3));
            let p12 = div(sub(p1, p2), dist1);
            let p32 = div(sub(p3, p2), dist2);
            let cos_theta = clip_to_one(dot(p12, p32));
            let sin_theta_sq = 1.0 - cos_theta * cos_theta;
            let sin_theta = sin_theta_sq.sqrt().max(1.0e-8);
            let angle_term = RAD2DEG * cos_theta.acos() - theta0;
            let dist_term = RAD2DEG * (fc1 * (dist1 - self.rest_len1[t]) + fc2 * (dist2 - self.rest_len2[t]));
            let dcos1 = [
                1.0 / dist1 * (p32[0] - cos_theta * p12[0]),
                1.0 / dist1 * (p32[1] - cos_theta * p12[1]),
                1.0 / dist1 * (p32[2] - cos_theta * p12[2]),
            ];
            let dcos2 = [
                1.0 / dist2 * (p12[0] - cos_theta * p32[0]),
                1.0 / dist2 * (p12[1] - cos_theta * p32[1]),
                1.0 / dist2 * (p12[2] - cos_theta * p32[2]),
            ];
            for d in 0..3 {
                add_to(grad, i1, d, C5 * (p12[d] * fc1 * angle_term + dcos1[d] / (-sin_theta) * dist_term));
            }
            for d in 0..3 {
                add_to(
                    grad,
                    i2,
                    d,
                    C5 * ((-p12[d] * fc1 - p32[d] * fc2) * angle_term + (-dcos1[d] - dcos2[d]) / (-sin_theta) * dist_term),
                );
            }
            for d in 0..3 {
                add_to(grad, i3, d, C5 * (p32[d] * fc2 * angle_term + dcos2[d] / (-sin_theta) * dist_term));
            }
        }
    }
    fn label(&self) -> &'static str {
        "StretchBend"
    }
}

// ---------------------------------------------------------------------------
// out-of-plane bend
// ---------------------------------------------------------------------------

/// RDKit `MMFF::Utils::calcOopChi` (degrees).
pub fn calc_oop_chi(i: P3, j: P3, k: P3, l: P3) -> f64 {
    let mut rji = sub(i, j);
    let mut rjk = sub(k, j);
    let mut rjl = sub(l, j);
    rji = div(rji, length(rji));
    rjk = div(rjk, length(rjk));
    rjl = div(rjl, length(rjl));
    let mut n = cross(rji, rjk);
    n = div(n, length(n));
    let sin_chi = clip_to_one(dot(n, rjl));
    RAD2DEG * sin_chi.asin()
}

/// RDKit `MMFF::Utils::calcOopBendEnergy`.
pub fn calc_oop_bend_energy(chi: f64, koop: f64) -> f64 {
    let c2 = MDYNE_A_TO_KCAL_MOL * DEG2RAD * DEG2RAD;
    0.5 * c2 * koop * chi * chi
}

/// All MMFF out-of-plane terms (RDKit `OopBendContrib`).
#[derive(Debug, Clone, Default)]
pub struct OopBendContrib {
    pub(crate) at: Vec<[usize; 4]>,
    pub(crate) koop: Vec<f64>,
}

impl OopBendContrib {
    pub fn add_term(&mut self, i: usize, j: usize, k: usize, l: usize, koop: f64) {
        self.at.push([i, j, k, l]);
        self.koop.push(koop);
    }
    pub fn len(&self) -> usize {
        self.at.len()
    }
    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }
}

impl Contrib for OopBendContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        let mut total = 0.0;
        for t in 0..self.at.len() {
            let [i, j, k, l] = self.at[t];
            total += calc_oop_bend_energy(calc_oop_chi(pt(pos, i), pt(pos, j), pt(pos, k), pt(pos, l)), self.koop[t]);
        }
        total
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        const C2: f64 = MDYNE_A_TO_KCAL_MOL * DEG2RAD * DEG2RAD;
        for t in 0..self.at.len() {
            let [i1, i2, i3, i4] = self.at[t];
            let (ip, jp, kp, lp) = (pt(pos, i1), pt(pos, i2), pt(pos, i3), pt(pos, i4));
            let mut rji = sub(ip, jp);
            let mut rjk = sub(kp, jp);
            let mut rjl = sub(lp, jp);
            let dji = length(rji);
            let djk = length(rjk);
            let djl = length(rjl);
            if is_double_zero(dji) || is_double_zero(djk) || is_double_zero(djl) {
                continue;
            }
            rji = div(rji, dji);
            rjk = div(rjk, djk);
            rjl = div(rjl, djl);
            let mut n = cross(neg(rji), rjk);
            n = div(n, length(n));
            let sin_chi = clip_to_one(dot(rjl, n));
            let cos_chi_sq = 1.0 - sin_chi * sin_chi;
            let cos_chi = (if cos_chi_sq > 0.0 { cos_chi_sq.sqrt() } else { 0.0 }).max(1.0e-8);
            let chi = RAD2DEG * sin_chi.asin();
            let cos_theta = clip_to_one(dot(rji, rjk));
            let sin_theta_sq = (1.0 - cos_theta * cos_theta).max(1.0e-8);
            let sin_theta = (if sin_theta_sq > 0.0 { sin_theta_sq.sqrt() } else { 0.0 }).max(1.0e-8);
            let de_dchi = RAD2DEG * C2 * self.koop[t] * chi;
            let t1 = cross(rjl, rjk);
            let t2 = cross(rji, rjl);
            let t3 = cross(rjk, rji);
            let term1 = cos_chi * sin_theta;
            let term2 = sin_chi / (cos_chi * sin_theta_sq);
            let mut tg1 = [0.0; 3];
            let mut tg3 = [0.0; 3];
            let mut tg4 = [0.0; 3];
            for d in 0..3 {
                tg1[d] = (t1[d] / term1 - (rji[d] - rjk[d] * cos_theta) * term2) / dji;
                tg3[d] = (t2[d] / term1 - (rjk[d] - rji[d] * cos_theta) * term2) / djk;
                tg4[d] = (t3[d] / term1 - rjl[d] * sin_chi / cos_chi) / djl;
            }
            for d in 0..3 {
                add_to(grad, i1, d, de_dchi * tg1[d]);
                add_to(grad, i2, d, -de_dchi * (tg1[d] + tg3[d] + tg4[d]));
                add_to(grad, i3, d, de_dchi * tg3[d]);
                add_to(grad, i4, d, de_dchi * tg4[d]);
            }
        }
    }
    fn label(&self) -> &'static str {
        "OopBend"
    }
}

// ---------------------------------------------------------------------------
// torsion
// ---------------------------------------------------------------------------

/// RDKit `MMFF::Utils::calcTorsionCosPhi`.
pub fn calc_torsion_cos_phi(i: P3, j: P3, k: P3, l: P3) -> f64 {
    let r1 = sub(i, j);
    let r2 = sub(k, j);
    let r3 = sub(j, k);
    let r4 = sub(l, k);
    let t1 = cross(r1, r2);
    let t2 = cross(r3, r4);
    let t1_len = length(t1);
    let t2_len = length(t2);
    if is_double_zero(t1_len) || is_double_zero(t2_len) {
        return 0.0;
    }
    clip_to_one(dot(t1, t2) / (t1_len * t2_len))
}

/// RDKit `MMFF::Utils::calcTorsionEnergy`.
pub fn calc_torsion_energy(v1: f64, v2: f64, v3: f64, cos_phi: f64) -> f64 {
    let cos2phi = 2.0 * cos_phi * cos_phi - 1.0;
    let cos3phi = cos_phi * (2.0 * cos2phi - 1.0);
    0.5 * (v1 * (1.0 + cos_phi) + v2 * (1.0 - cos2phi) + v3 * (1.0 + cos3phi))
}

/// RDKit `ForceFieldsHelper::computeDihedral` (vectors `r`, `t`, lengths
/// `d` and `cosPhi`).
fn compute_dihedral(pos: &[f64], i1: usize, i2: usize, i3: usize, i4: usize) -> ([P3; 4], [P3; 2], [f64; 2], f64) {
    let (p1, p2, p3, p4) = (pt(pos, i1), pt(pos, i2), pt(pos, i3), pt(pos, i4));
    let r0 = sub(p1, p2);
    let r1 = sub(p3, p2);
    let r2 = neg(r1);
    let r3 = sub(p4, p3);
    let mut t0 = cross(r0, r1);
    let d0 = length(t0).max(1.0e-5);
    t0 = div(t0, d0);
    let mut t1 = cross(r2, r3);
    let d1 = length(t1).max(1.0e-5);
    t1 = div(t1, d1);
    let cos_phi = dot(t0, t1).clamp(-1.0, 1.0);
    ([r0, r1, r2, r3], [t0, t1], [d0, d1], cos_phi)
}

/// RDKit `MMFF::Utils::calcTorsionGrad`.
fn calc_torsion_grad(r: &[P3; 4], t: &[P3; 2], d: &[f64; 2], g: &mut [f64], idx: [usize; 4], sin_term: f64, cos_phi: f64) {
    let dc = [
        1.0 / d[0] * (t[1][0] - cos_phi * t[0][0]),
        1.0 / d[0] * (t[1][1] - cos_phi * t[0][1]),
        1.0 / d[0] * (t[1][2] - cos_phi * t[0][2]),
        1.0 / d[1] * (t[0][0] - cos_phi * t[1][0]),
        1.0 / d[1] * (t[0][1] - cos_phi * t[1][1]),
        1.0 / d[1] * (t[0][2] - cos_phi * t[1][2]),
    ];
    let (x, y, z) = (0, 1, 2);
    add_to(g, idx[0], 0, sin_term * (dc[2] * r[1][y] - dc[1] * r[1][z]));
    add_to(g, idx[0], 1, sin_term * (dc[0] * r[1][z] - dc[2] * r[1][x]));
    add_to(g, idx[0], 2, sin_term * (dc[1] * r[1][x] - dc[0] * r[1][y]));

    add_to(
        g,
        idx[1],
        0,
        sin_term * (dc[1] * (r[1][z] - r[0][z]) + dc[2] * (r[0][y] - r[1][y]) + dc[4] * (-r[3][z]) + dc[5] * (r[3][y])),
    );
    add_to(
        g,
        idx[1],
        1,
        sin_term * (dc[0] * (r[0][z] - r[1][z]) + dc[2] * (r[1][x] - r[0][x]) + dc[3] * (r[3][z]) + dc[5] * (-r[3][x])),
    );
    add_to(
        g,
        idx[1],
        2,
        sin_term * (dc[0] * (r[1][y] - r[0][y]) + dc[1] * (r[0][x] - r[1][x]) + dc[3] * (-r[3][y]) + dc[4] * (r[3][x])),
    );

    add_to(
        g,
        idx[2],
        0,
        sin_term * (dc[1] * (r[0][z]) + dc[2] * (-r[0][y]) + dc[4] * (r[3][z] - r[2][z]) + dc[5] * (r[2][y] - r[3][y])),
    );
    add_to(
        g,
        idx[2],
        1,
        sin_term * (dc[0] * (-r[0][z]) + dc[2] * (r[0][x]) + dc[3] * (r[2][z] - r[3][z]) + dc[5] * (r[3][x] - r[2][x])),
    );
    add_to(
        g,
        idx[2],
        2,
        sin_term * (dc[0] * (r[0][y]) + dc[1] * (-r[0][x]) + dc[3] * (r[3][y] - r[2][y]) + dc[4] * (r[2][x] - r[3][x])),
    );

    add_to(g, idx[3], 0, sin_term * (dc[4] * r[2][z] - dc[5] * r[2][y]));
    add_to(g, idx[3], 1, sin_term * (dc[5] * r[2][x] - dc[3] * r[2][z]));
    add_to(g, idx[3], 2, sin_term * (dc[3] * r[2][y] - dc[4] * r[2][x]));
}

/// All MMFF torsion terms (RDKit `TorsionAngleContrib`).
#[derive(Debug, Clone, Default)]
pub struct TorsionAngleContrib {
    pub(crate) at: Vec<[usize; 4]>,
    pub(crate) v1: Vec<f64>,
    pub(crate) v2: Vec<f64>,
    pub(crate) v3: Vec<f64>,
}

impl TorsionAngleContrib {
    #[allow(clippy::too_many_arguments)]
    pub fn add_term(&mut self, i: usize, j: usize, k: usize, l: usize, v1: f64, v2: f64, v3: f64) {
        self.at.push([i, j, k, l]);
        self.v1.push(v1);
        self.v2.push(v2);
        self.v3.push(v3);
    }
    pub fn len(&self) -> usize {
        self.at.len()
    }
    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }
}

impl Contrib for TorsionAngleContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        let mut total = 0.0;
        for t in 0..self.at.len() {
            let [i, j, k, l] = self.at[t];
            total +=
                calc_torsion_energy(self.v1[t], self.v2[t], self.v3[t], calc_torsion_cos_phi(pt(pos, i), pt(pos, j), pt(pos, k), pt(pos, l)));
        }
        total
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        for t in 0..self.at.len() {
            let [i1, i2, i3, i4] = self.at[t];
            let (r, tt, d, cos_phi) = compute_dihedral(pos, i1, i2, i3, i4);
            let sin_phi_sq = 1.0 - cos_phi * cos_phi;
            let sin_phi = if sin_phi_sq > 0.0 { sin_phi_sq.sqrt() } else { 0.0 };
            let sin2phi = 2.0 * sin_phi * cos_phi;
            let sin3phi = 3.0 * sin_phi - 4.0 * sin_phi * sin_phi_sq;
            let de_dphi = 0.5 * (-(self.v1[t]) * sin_phi + 2.0 * self.v2[t] * sin2phi - 3.0 * self.v3[t] * sin3phi);
            let sin_term = -de_dphi * (if is_double_zero(sin_phi) { 1.0 / cos_phi } else { 1.0 / sin_phi });
            calc_torsion_grad(&r, &tt, &d, grad, [i1, i2, i3, i4], sin_term, cos_phi);
        }
    }
    fn label(&self) -> &'static str {
        "Torsion"
    }
}

// ---------------------------------------------------------------------------
// non-bonded: buffered 14-7 van der Waals and electrostatics
// ---------------------------------------------------------------------------

/// RDKit `MMFF::Utils::calcUnscaledVdWMinimum`.
pub fn calc_unscaled_vdw_minimum(tab: &VdwTable, pi: &MmffVdw, pj: &MmffVdw) -> f64 {
    let gamma_ij = (pi.r_star - pj.r_star) / (pi.r_star + pj.r_star);
    0.5 * (pi.r_star + pj.r_star)
        * (1.0 + (if pi.da == b'D' || pj.da == b'D' { 0.0 } else { tab.b * (1.0 - (-(tab.beta) * gamma_ij * gamma_ij).exp()) }))
}

/// RDKit `MMFF::Utils::calcUnscaledVdWWellDepth`.
pub fn calc_unscaled_vdw_well_depth(r_star_ij: f64, pi: &MmffVdw, pj: &MmffVdw) -> f64 {
    let r2 = r_star_ij * r_star_ij;
    let c4 = 181.16;
    c4 * pi.g_i * pj.g_i * pi.alpha_i * pj.alpha_i / (((pi.alpha_i / pi.n_i).sqrt() + (pj.alpha_i / pj.n_i).sqrt()) * r2 * r2 * r2)
}

/// RDKit `MMFF::Utils::scaleVdWParams`.
pub fn scale_vdw_params(r_star_ij: &mut f64, well_depth: &mut f64, tab: &VdwTable, pi: &MmffVdw, pj: &MmffVdw) {
    if (pi.da == b'D' && pj.da == b'A') || (pi.da == b'A' && pj.da == b'D') {
        *r_star_ij *= tab.darad;
        *well_depth *= tab.daeps;
    }
}

const VDW1: f64 = 1.07;
const VDW1M1: f64 = VDW1 - 1.0;
const VDW2: f64 = 1.12;
const VDW2M1: f64 = VDW2 - 1.0;
const VDW2T7: f64 = VDW2 * 7.0;

/// RDKit `MMFF::Utils::calcVdWEnergy`.
pub fn calc_vdw_energy(dist: f64, r_star_ij: f64, well_depth: f64) -> f64 {
    let dist2 = dist * dist;
    let dist7 = dist2 * dist2 * dist2 * dist;
    let a_term = VDW1 * r_star_ij / (dist + VDW1M1 * r_star_ij);
    let a_term2 = a_term * a_term;
    let a_term7 = a_term2 * a_term2 * a_term2 * a_term;
    let r2 = r_star_ij * r_star_ij;
    let r7 = r2 * r2 * r2 * r_star_ij;
    let b_term = VDW2 * r7 / (dist7 + VDW2M1 * r7) - 2.0;
    well_depth * a_term7 * b_term
}

/// RDKit `MMFF::Utils::calcEleEnergy`.
pub fn calc_ele_energy(dist: f64, charge_term: f64, diel_model: DielectricModel, is_1_4: bool) -> f64 {
    let mut corr_dist = dist + 0.05;
    let diel = 332.0716;
    let sc1_4 = 0.75;
    if diel_model == DielectricModel::Distance {
        corr_dist *= corr_dist;
    }
    diel * charge_term / corr_dist * (if is_1_4 { sc1_4 } else { 1.0 })
}

const VDW_BIT: u8 = 1;
const ELE_BIT: u8 = 2;

/// Combined van der Waals + electrostatic pair terms (RDKit
/// `NonbondedContrib`, which is what the RDKit MMFF builder uses).
/// [`Contrib::energy_terms`] reports the two parts separately as `"VdW"`
/// and `"Electrostatic"`.
#[derive(Debug, Clone, Default)]
pub struct NonbondedContrib {
    pub(crate) at1: Vec<usize>,
    pub(crate) at2: Vec<usize>,
    pub(crate) kind: Vec<u8>,
    pub(crate) r_ij_star: Vec<f64>,
    pub(crate) well_depth: Vec<f64>,
    pub(crate) charge_term: Vec<f64>,
    pub(crate) diel_model: Vec<Option<DielectricModel>>,
    pub(crate) is_1_4: Vec<bool>,
}

impl NonbondedContrib {
    /// Adds a pair; `vdw = Some((R*_ij, eps_ij))`, `ele = Some((qi*qj/D,
    /// model, is14))`.
    pub fn add_term(&mut self, i: usize, j: usize, vdw: Option<(f64, f64)>, ele: Option<(f64, DielectricModel, bool)>) {
        if vdw.is_none() && ele.is_none() {
            return;
        }
        self.at1.push(i);
        self.at2.push(j);
        let mut k = 0;
        match vdw {
            Some((r, e)) => {
                k |= VDW_BIT;
                self.r_ij_star.push(r);
                self.well_depth.push(e);
            }
            None => {
                self.r_ij_star.push(0.0);
                self.well_depth.push(0.0);
            }
        }
        match ele {
            Some((q, m, is14)) => {
                k |= ELE_BIT;
                self.charge_term.push(q);
                self.diel_model.push(Some(m));
                self.is_1_4.push(is14);
            }
            None => {
                self.charge_term.push(0.0);
                self.diel_model.push(None);
                self.is_1_4.push(false);
            }
        }
        self.kind.push(k);
    }
    pub fn len(&self) -> usize {
        self.at1.len()
    }
    pub fn is_empty(&self) -> bool {
        self.at1.is_empty()
    }
    fn vdw_energy(&self, t: usize, dist: f64) -> f64 {
        calc_vdw_energy(dist, self.r_ij_star[t], self.well_depth[t])
    }
    fn ele_energy(&self, t: usize, dist: f64) -> f64 {
        calc_ele_energy(dist, self.charge_term[t], self.diel_model[t].unwrap_or(DielectricModel::Constant), self.is_1_4[t])
    }
}

impl Contrib for NonbondedContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        let mut e = 0.0;
        for t in 0..self.at1.len() {
            let dist = distance(pos, self.at1[t], self.at2[t]);
            if self.kind[t] & VDW_BIT != 0 {
                e += self.vdw_energy(t, dist);
            }
            if self.kind[t] & ELE_BIT != 0 {
                e += self.ele_energy(t, dist);
            }
        }
        e
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        for t in 0..self.at1.len() {
            let (i1, i2) = (self.at1[t], self.at2[t]);
            let dist = distance(pos, i1, i2);
            let mut vdw_grad = 0.0;
            let mut ele_grad = 0.0;
            if dist <= 0.0 {
                // RDKit returns here (skipping the remaining pairs).
                if self.kind[t] & VDW_BIT != 0 {
                    for d in 0..3 {
                        add_to(grad, i1, d, self.r_ij_star[t] * 0.01);
                        add_to(grad, i2, d, -(self.r_ij_star[t] * 0.01));
                    }
                }
                if self.kind[t] & ELE_BIT != 0 {
                    for d in 0..3 {
                        add_to(grad, i1, d, 0.02);
                        add_to(grad, i2, d, -0.02);
                    }
                }
                return;
            }
            if self.kind[t] & VDW_BIT != 0 {
                let r = self.r_ij_star[t];
                let q = dist / r;
                let q2 = q * q;
                let q6 = q2 * q2 * q2;
                let q7 = q6 * q;
                let q7pvdw2m1 = q7 + VDW2M1;
                let tt = VDW1 / (q + VDW1 - 1.0);
                let t2 = tt * tt;
                let t7 = t2 * t2 * t2 * tt;
                let de_dr = self.well_depth[t] / r
                    * t7
                    * (-VDW2T7 * q6 / (q7pvdw2m1 * q7pvdw2m1) + ((-VDW2T7 / q7pvdw2m1 + 14.0) / (q + VDW1M1)));
                vdw_grad = de_dr / dist;
            }
            if self.kind[t] & ELE_BIT != 0 {
                let model = self.diel_model[t].unwrap_or(DielectricModel::Constant);
                let mut corr_dist = dist + 0.05;
                corr_dist *= if model == DielectricModel::Distance { corr_dist * corr_dist } else { corr_dist };
                let de_dr = -332.0716 * model.as_f64() * self.charge_term[t] / corr_dist * (if self.is_1_4[t] { 0.75 } else { 1.0 });
                ele_grad = de_dr / dist;
            }
            let de_dr = vdw_grad + ele_grad;
            for d in 0..3 {
                let dg = de_dr * (pos[3 * i1 + d] - pos[3 * i2 + d]);
                add_to(grad, i1, d, dg);
                add_to(grad, i2, d, -dg);
            }
        }
    }
    fn label(&self) -> &'static str {
        "Nonbonded"
    }
    fn energy_terms(&self, pos: &[f64], out: &mut dyn FnMut(&'static str, f64)) {
        let (mut ev, mut ee) = (0.0, 0.0);
        let (mut has_v, mut has_e) = (false, false);
        for t in 0..self.at1.len() {
            let dist = distance(pos, self.at1[t], self.at2[t]);
            if self.kind[t] & VDW_BIT != 0 {
                ev += self.vdw_energy(t, dist);
                has_v = true;
            }
            if self.kind[t] & ELE_BIT != 0 {
                ee += self.ele_energy(t, dist);
                has_e = true;
            }
        }
        if has_v {
            out("VdW", ev);
        }
        if has_e {
            out("Electrostatic", ee);
        }
    }
}
