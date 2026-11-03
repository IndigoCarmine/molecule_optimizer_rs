//! UFF angle bend term.
//!
//! Port of RDKit `Code/ForceField/UFF/AngleBend.{h,cpp}`.
//!
//! Copyright (C) 2004-2013 Rational Discovery LLC.
//! Distributed under the BSD 3-Clause license (see `LICENSE-RDKIT.txt`).

use super::bond_stretch::calc_bond_rest_length;
use super::params::{AtomicParams, G};
use super::{int_pow, rd_distance};
use crate::core::geometry::{V3, dot, sub};
use crate::ff::Contrib;
use crate::ff::utils::p3;
use std::f64::consts::PI;

const ANGLE_CORRECTION_THRESHOLD: f64 = 0.8660;

/// RDKit `Utils::calcAngleForceConstant`.
pub fn calc_angle_force_constant(
    theta0: f64,
    bond_order12: f64,
    bond_order23: f64,
    at1: &AtomicParams,
    at2: &AtomicParams,
    at3: &AtomicParams,
) -> f64 {
    let cos_theta0 = theta0.cos();
    let r12 = calc_bond_rest_length(bond_order12, at1, at2);
    let r23 = calc_bond_rest_length(bond_order23, at2, at3);
    let r13 = (r12 * r12 + r23 * r23 - 2. * r12 * r23 * cos_theta0).sqrt();
    let beta = 2. * G / (r12 * r23);
    let pre_factor = beta * at1.z1 * at3.z1 / int_pow::<5>(r13);
    let r_term = r12 * r23;
    let inner_bit = 3. * r_term * (1. - cos_theta0 * cos_theta0) - r13 * r13 * cos_theta0;
    pre_factor * r_term * inner_bit
}

/// RDKit `Utils::calcAngleBendGrad`.
pub(crate) fn calc_angle_bend_grad(
    r: &[V3; 2],
    dist: &[f64; 2],
    g: [usize; 3],
    grad: &mut [f64],
    de_dtheta: f64,
    cos_theta: f64,
    sin_theta: f64,
) {
    let dcos_ds = [
        1.0 / dist[0] * (r[1][0] - cos_theta * r[0][0]),
        1.0 / dist[0] * (r[1][1] - cos_theta * r[0][1]),
        1.0 / dist[0] * (r[1][2] - cos_theta * r[0][2]),
        1.0 / dist[1] * (r[0][0] - cos_theta * r[1][0]),
        1.0 / dist[1] * (r[0][1] - cos_theta * r[1][1]),
        1.0 / dist[1] * (r[0][2] - cos_theta * r[1][2]),
    ];
    for k in 0..3 {
        grad[3 * g[0] + k] += de_dtheta * dcos_ds[k] / (-sin_theta);
    }
    for k in 0..3 {
        grad[3 * g[1] + k] += de_dtheta * (-dcos_ds[k] - dcos_ds[k + 3]) / (-sin_theta);
    }
    for k in 0..3 {
        grad[3 * g[2] + k] += de_dtheta * dcos_ds[k + 3] / (-sin_theta);
    }
}

/// UFF angle bend (RDKit `AngleBendContrib`).
///
/// `order`: 0 = general (Fourier expansion around `theta0`), 1 = linear,
/// 2 = (used for trigonal-bipyramidal axial pairs), 3 = trigonal planar,
/// 4 = square planar / octahedral. Orders 30/35/40/45 are RDKit's special
/// cases for sp2 centres in 3- and 4-membered rings (theta0 = 150°, 60°,
/// 135°, 90°; treated as order 0).
#[derive(Debug, Clone)]
pub struct AngleBendContrib {
    pub idx1: usize,
    pub idx2: usize,
    pub idx3: usize,
    pub order: u32,
    pub force_constant: f64,
    pub c0: f64,
    pub c1: f64,
    pub c2: f64,
    pub theta0: f64,
}

impl AngleBendContrib {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        idx1: usize,
        idx2: usize,
        idx3: usize,
        bond_order12: f64,
        bond_order23: f64,
        at1: &AtomicParams,
        at2: &AtomicParams,
        at3: &AtomicParams,
        order: u32,
    ) -> Self {
        let mut theta0 = at2.theta0;
        let mut order = order;
        if order >= 30 {
            match order {
                30 => theta0 = 150.0 / 180.0 * PI,
                35 => theta0 = 60.0 / 180.0 * PI,
                40 => theta0 = 135.0 / 180.0 * PI,
                45 => theta0 = 90.0 / 180.0 * PI,
                _ => {}
            }
            order = 0;
        }
        let force_constant = calc_angle_force_constant(theta0, bond_order12, bond_order23, at1, at2, at3);
        let (mut c0, mut c1, mut c2) = (0.0, 0.0, 0.0);
        if order == 0 {
            let sin_theta0 = theta0.sin();
            let cos_theta0 = theta0.cos();
            c2 = 1. / (4. * (sin_theta0 * sin_theta0).max(1e-8));
            c1 = -4. * c2 * cos_theta0;
            c0 = c2 * (2. * cos_theta0 * cos_theta0 + 1.);
        }
        AngleBendContrib { idx1, idx2, idx3, order, force_constant, c0, c1, c2, theta0 }
    }

    fn energy_term(&self, cos_theta: f64, sin_theta_sq: f64) -> f64 {
        let cos2_theta = cos_theta * cos_theta - sin_theta_sq;
        if self.order == 0 {
            self.c0 + self.c1 * cos_theta + self.c2 * cos2_theta
        } else {
            let mut res = match self.order {
                1 => -cos_theta,
                2 => cos2_theta,
                3 => cos_theta * (cos_theta * cos_theta - 3. * sin_theta_sq),
                4 => {
                    int_pow::<4>(cos_theta) - 6. * cos_theta * cos_theta * sin_theta_sq
                        + sin_theta_sq * sin_theta_sq
                }
                _ => 0.0,
            };
            res = 1. - res;
            res /= (self.order * self.order) as f64;
            res
        }
    }

    fn theta_deriv(&self, cos_theta: f64, sin_theta: f64) -> f64 {
        let sin2_theta = 2. * sin_theta * cos_theta;
        if self.order == 0 {
            -self.force_constant * (self.c1 * sin_theta + 2. * self.c2 * sin2_theta)
        } else {
            let mut de = match self.order {
                1 => -sin_theta,
                2 => sin2_theta,
                3 => sin_theta * (3. - 4. * sin_theta * sin_theta),
                4 => cos_theta * sin_theta * (4. - 8. * sin_theta * sin_theta),
                _ => 0.0,
            };
            de *= self.force_constant / self.order as f64;
            de
        }
    }
}

impl Contrib for AngleBendContrib {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        let dist1 = rd_distance(pos, self.idx1, self.idx2);
        let dist2 = rd_distance(pos, self.idx2, self.idx3);
        let p1 = p3(pos, self.idx1);
        let p2 = p3(pos, self.idx2);
        let p3_ = p3(pos, self.idx3);
        let p12 = sub(p1, p2);
        let p32 = sub(p3_, p2);
        let cos_theta = (dot(p12, p32) / (dist1 * dist2)).clamp(-1.0, 1.0);
        let sin_theta_sq = 1. - cos_theta * cos_theta;
        let mut res = self.force_constant * self.energy_term(cos_theta, sin_theta_sq);
        if self.order != 0 && self.order < 5 && cos_theta > ANGLE_CORRECTION_THRESHOLD {
            let theta = cos_theta.acos();
            res += (-20.0 * (theta - self.theta0 + 0.25)).exp();
        }
        res
    }

    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let dist = [rd_distance(pos, self.idx1, self.idx2), rd_distance(pos, self.idx2, self.idx3)];
        let p1 = p3(pos, self.idx1);
        let p2 = p3(pos, self.idx2);
        let p3_ = p3(pos, self.idx3);
        let d0 = sub(p1, p2);
        let d1 = sub(p3_, p2);
        let r = [
            [d0[0] / dist[0], d0[1] / dist[0], d0[2] / dist[0]],
            [d1[0] / dist[1], d1[1] / dist[1], d1[2] / dist[1]],
        ];
        let cos_theta = dot(r[0], r[1]).clamp(-1.0, 1.0);
        let sin_theta_sq = 1.0 - cos_theta * cos_theta;
        let sin_theta = sin_theta_sq.sqrt().max(1.0e-8);
        let mut de_dtheta = self.theta_deriv(cos_theta, sin_theta);
        if self.order != 0 && self.order < 5 && cos_theta > ANGLE_CORRECTION_THRESHOLD {
            let theta = cos_theta.acos();
            let corr = -20.0 * (-20.0 * (theta - self.theta0 + 0.25)).exp();
            de_dtheta += corr;
        }
        calc_angle_bend_grad(&r, &dist, [self.idx1, self.idx2, self.idx3], grad, de_dtheta, cos_theta, sin_theta);
    }

    fn label(&self) -> &'static str {
        "AngleBend"
    }
}
