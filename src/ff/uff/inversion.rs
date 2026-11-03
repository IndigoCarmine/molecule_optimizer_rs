//! UFF inversion (out-of-plane) term.
//!
//! Port of RDKit `Code/ForceField/UFF/Inversion.{h,cpp}` and the inversion
//! helpers of `Code/ForceField/UFF/Utils.cpp` (`calculateCosY`,
//! `calcInversionCoefficientsAndForceConstant`).
//!
//! Copyright (C) 2013-2024 Paolo Tosco and other RDKit contributors.
//! Distributed under the BSD 3-Clause license (see `LICENSE-RDKIT.txt`).

use super::is_double_zero;
use crate::core::geometry::{V3, cross, dot, norm, sub};
use crate::ff::Contrib;
use crate::ff::utils::p3;
use std::f64::consts::PI;

#[inline]
fn div(a: V3, s: f64) -> V3 {
    [a[0] / s, a[1] / s, a[2] / s]
}

#[inline]
fn len_sq(a: V3) -> f64 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}

/// RDKit `Utils::calculateCosY`: cosine of the angle between the I-J-K
/// plane normal and the J-L vector.
pub fn calculate_cos_y(i: V3, j: V3, k: V3, l: V3) -> f64 {
    const ZERO_TOL: f64 = 1.0e-16;
    let r_ji = sub(i, j);
    let r_jk = sub(k, j);
    let r_jl = sub(l, j);
    let l2_ji = len_sq(r_ji);
    let l2_jk = len_sq(r_jk);
    let l2_jl = len_sq(r_jl);
    if l2_ji < ZERO_TOL || l2_jk < ZERO_TOL || l2_jl < ZERO_TOL {
        return 0.0;
    }
    let n = div(cross(r_ji, r_jk), l2_ji.sqrt() * l2_jk.sqrt());
    let l2n = len_sq(n);
    if l2n < ZERO_TOL {
        return 0.0;
    }
    dot(n, r_jl) / (l2_jl.sqrt() * l2n.sqrt())
}

/// RDKit `Utils::calcInversionCoefficientsAndForceConstant`:
/// returns `(force_constant, C0, C1, C2)`.
pub fn calc_inversion_coefficients_and_force_constant(at2_atomic_num: u8, is_c_bound_to_o: bool) -> (f64, f64, f64, f64) {
    let (mut res, c0, c1, c2);
    if matches!(at2_atomic_num, 6..=8) {
        c0 = 1.0;
        c1 = -1.0;
        c2 = 0.0;
        res = if is_c_bound_to_o { 50.0 } else { 6.0 };
    } else {
        let mut w0 = PI / 180.0;
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

/// UFF inversion term (RDKit `InversionContrib`); atom 2 is the centre.
#[derive(Debug, Clone)]
pub struct InversionContrib {
    pub idx: [usize; 4],
    pub force_constant: f64,
    pub c0: f64,
    pub c1: f64,
    pub c2: f64,
}

impl InversionContrib {
    pub fn new(idx: [usize; 4], at2_atomic_num: u8, is_c_bound_to_o: bool, oob_force_scaling_factor: f64) -> Self {
        let (k, c0, c1, c2) = calc_inversion_coefficients_and_force_constant(at2_atomic_num, is_c_bound_to_o);
        InversionContrib { idx, force_constant: oob_force_scaling_factor * k, c0, c1, c2 }
    }
}

impl Contrib for InversionContrib {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        let [i1, i2, i3, i4] = self.idx;
        let cos_y = calculate_cos_y(p3(pos, i1), p3(pos, i2), p3(pos, i3), p3(pos, i4));
        let sin_y_sq = 1.0 - cos_y * cos_y;
        let sin_y = if sin_y_sq > 0.0 { sin_y_sq.sqrt() } else { 0.0 };
        let cos2w = 2.0 * sin_y * sin_y - 1.0;
        self.force_constant * (self.c0 + self.c1 * sin_y + self.c2 * cos2w)
    }

    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let [i1, i2, i3, i4] = self.idx;
        let p1 = p3(pos, i1);
        let p2 = p3(pos, i2);
        let p3_ = p3(pos, i3);
        let p4 = p3(pos, i4);
        let mut r_ji = sub(p1, p2);
        let mut r_jk = sub(p3_, p2);
        let mut r_jl = sub(p4, p2);
        let d_ji = norm(r_ji);
        let d_jk = norm(r_jk);
        let d_jl = norm(r_jl);
        if is_double_zero(d_ji) || is_double_zero(d_jk) || is_double_zero(d_jl) {
            return;
        }
        r_ji = div(r_ji, d_ji);
        r_jk = div(r_jk, d_jk);
        r_jl = div(r_jl, d_jl);

        let neg_ji = [-r_ji[0], -r_ji[1], -r_ji[2]]; // RDKit: `-rJI` (x * -1.0, exact)
        let mut n = cross(neg_ji, r_jk);
        n = div(n, norm(n));
        let cos_y = dot(n, r_jl).clamp(-1.0, 1.0);
        let sin_y_sq = 1.0 - cos_y * cos_y;
        let sin_y = sin_y_sq.sqrt().max(1.0e-8);
        let cos_theta = dot(r_ji, r_jk).clamp(-1.0, 1.0);
        let sin_theta_sq = 1.0 - cos_theta * cos_theta;
        let sin_theta = sin_theta_sq.sqrt().max(1.0e-8);
        let de_dw = -self.force_constant * (self.c1 * cos_y + 4.0 * self.c2 * cos_y * sin_y);
        let t1 = cross(r_jl, r_jk);
        let t2 = cross(r_ji, r_jl);
        let t3 = cross(r_jk, r_ji);
        let term1 = sin_y * sin_theta;
        let term2 = cos_y / (sin_y * sin_theta_sq);
        let mut tg1 = [0.0; 3];
        let mut tg3 = [0.0; 3];
        let mut tg4 = [0.0; 3];
        for k in 0..3 {
            tg1[k] = (t1[k] / term1 - (r_ji[k] - r_jk[k] * cos_theta) * term2) / d_ji;
            tg3[k] = (t2[k] / term1 - (r_jk[k] - r_ji[k] * cos_theta) * term2) / d_jk;
            tg4[k] = (t3[k] / term1 - r_jl[k] * cos_y / sin_y) / d_jl;
        }
        for k in 0..3 {
            grad[3 * i1 + k] += de_dw * tg1[k];
            grad[3 * i2 + k] += -de_dw * (tg1[k] + tg3[k] + tg4[k]);
            grad[3 * i3 + k] += de_dw * tg3[k];
            grad[3 * i4 + k] += de_dw * tg4[k];
        }
    }

    fn label(&self) -> &'static str {
        "Inversion"
    }
}
