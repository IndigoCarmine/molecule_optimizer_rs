//! UFF inversion (out-of-plane) term.
//!
//! Port of RDKit `Code/ForceField/UFF/Inversion.{h,cpp}` and the inversion
//! helpers of `Code/ForceField/UFF/Utils.cpp` (`calculateCosY`,
//! `calcInversionCoefficientsAndForceConstant`).
//!
//! Copyright (C) 2013-2024 Paolo Tosco and other RDKit contributors.
//! Distributed under the BSD 3-Clause license (see `LICENSE-RDKIT.txt`).

use crate::core::geometry::{V3, cross, dot};
use crate::ff::Contrib;
use crate::ff::utils::OutOfPlane;
use std::f64::consts::PI;

/// RDKit `Utils::calculateCosY`: cosine of the angle between the I-J-K
/// plane normal and the J-L vector.
pub fn calculate_cos_y(i: V3, j: V3, k: V3, l: V3) -> f64 {
    cos_y(&OutOfPlane::new(i, j, k, l))
}

#[inline(always)]
fn cos_y(g: &OutOfPlane) -> f64 {
    const ZERO_TOL: f64 = 1.0e-16;
    let [r_ji, r_jk, r_jl] = g.r;
    let [l2_ji, l2_jk, l2_jl] = g.l2;
    if l2_ji < ZERO_TOL || l2_jk < ZERO_TOL || l2_jl < ZERO_TOL {
        return 0.0;
    }
    let n = cross(r_ji, r_jk);
    let s = g.d[0] * g.d[1];
    let n = [n[0] / s, n[1] / s, n[2] / s];
    let l2n = dot(n, n);
    if l2n < ZERO_TOL {
        return 0.0;
    }
    dot(n, r_jl) / (g.d[2] * l2n.sqrt())
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

    #[inline(always)]
    fn energy_of(&self, g: &OutOfPlane) -> f64 {
        let cos_y = cos_y(g);
        let sin_y_sq = 1.0 - cos_y * cos_y;
        let sin_y = if sin_y_sq > 0.0 { sin_y_sq.sqrt() } else { 0.0 };
        let cos2w = 2.0 * sin_y * sin_y - 1.0;
        self.force_constant * (self.c0 + self.c1 * sin_y + self.c2 * cos2w)
    }

    #[inline(always)]
    fn grad_of(&self, g: &OutOfPlane, grad: &mut [f64]) {
        if g.is_degenerate() {
            return;
        }
        let u = g.unit();
        let (cos_y, cos_theta) = OutOfPlane::wilson(&u);
        let sin_y_sq = 1.0 - cos_y * cos_y;
        let sin_y = sin_y_sq.sqrt().max(1.0e-8);
        let sin_theta_sq = 1.0 - cos_theta * cos_theta;
        let sin_theta = sin_theta_sq.sqrt().max(1.0e-8);
        let de_dw = -self.force_constant * (self.c1 * cos_y + 4.0 * self.c2 * cos_y * sin_y);
        g.add_grad(grad, self.idx, &u, de_dw, cos_y, sin_y, cos_theta, sin_theta, sin_theta_sq);
    }
}

impl Contrib for InversionContrib {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        self.energy_of(&OutOfPlane::at(pos, self.idx))
    }

    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        self.grad_of(&OutOfPlane::at(pos, self.idx), grad);
    }

    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> f64 {
        let g = OutOfPlane::at(pos, self.idx);
        self.grad_of(&g, grad);
        self.energy_of(&g)
    }

    fn label(&self) -> &'static str {
        "Inversion"
    }
}
