//! UFF bond stretch term.
//!
//! Port of RDKit `Code/ForceField/UFF/BondStretch.{h,cpp}`.
//!
//! Copyright (C) 2004-2006 Rational Discovery LLC.
//! Distributed under the BSD 3-Clause license (see `LICENSE-RDKIT.txt`).

use super::params::{AtomicParams, G, LAMBDA};
use super::rd_distance;
use crate::ff::Contrib;

/// RDKit `Utils::calcBondRestLength`.
pub fn calc_bond_rest_length(bond_order: f64, end1: &AtomicParams, end2: &AtomicParams) -> f64 {
    let ri = end1.r1;
    let rj = end2.r1;
    let r_bo = -LAMBDA * (ri + rj) * bond_order.ln();
    let xi = end1.gmp_xi;
    let xj = end2.gmp_xi;
    let r_en = ri * rj * (xi.sqrt() - xj.sqrt()) * (xi.sqrt() - xj.sqrt()) / (xi * ri + xj * rj);
    ri + rj + r_bo - r_en
}

/// RDKit `Utils::calcBondForceConstant`.
pub fn calc_bond_force_constant(rest_length: f64, end1: &AtomicParams, end2: &AtomicParams) -> f64 {
    2.0 * G * end1.z1 * end2.z1 / (rest_length * rest_length * rest_length)
}

/// Harmonic bond stretch `0.5·k·(r - r0)²` (RDKit `BondStretchContrib`).
#[derive(Debug, Clone)]
pub struct BondStretchContrib {
    pub idx1: usize,
    pub idx2: usize,
    pub rest_len: f64,
    pub force_constant: f64,
}

impl BondStretchContrib {
    pub fn new(idx1: usize, idx2: usize, bond_order: f64, end1: &AtomicParams, end2: &AtomicParams) -> Self {
        let rest_len = calc_bond_rest_length(bond_order, end1, end2);
        let force_constant = calc_bond_force_constant(rest_len, end1, end2);
        BondStretchContrib { idx1, idx2, rest_len, force_constant }
    }
}

impl Contrib for BondStretchContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        let dist_term = rd_distance(pos, self.idx1, self.idx2) - self.rest_len;
        0.5 * self.force_constant * dist_term * dist_term
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let dist = rd_distance(pos, self.idx1, self.idx2);
        let pre_factor = self.force_constant * (dist - self.rest_len);
        let (a, b) = (3 * self.idx1, 3 * self.idx2);
        for i in 0..3 {
            let d_grad = if dist > 0.0 {
                pre_factor * (pos[a + i] - pos[b + i]) / dist
            } else {
                self.force_constant * 0.01
            };
            grad[a + i] += d_grad;
            grad[b + i] -= d_grad;
        }
    }
    fn label(&self) -> &'static str {
        "BondStretch"
    }
}
