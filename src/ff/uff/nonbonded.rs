//! UFF van der Waals (Lennard-Jones 12-6) term.
//!
//! Port of RDKit `Code/ForceField/UFF/Nonbonded.{h,cpp}`.
//!
//! Copyright (C) 2004-2008 Greg Landrum and Rational Discovery LLC.
//! Distributed under the BSD 3-Clause license (see `LICENSE-RDKIT.txt`).

use super::params::AtomicParams;
use super::{int_pow, rd_distance};
use crate::ff::Contrib;

/// RDKit `Utils::calcNonbondedMinimum`.
pub fn calc_nonbonded_minimum(at1: &AtomicParams, at2: &AtomicParams) -> f64 {
    (at1.x1 * at2.x1).sqrt()
}

/// RDKit `Utils::calcNonbondedDepth`.
pub fn calc_nonbonded_depth(at1: &AtomicParams, at2: &AtomicParams) -> f64 {
    (at1.d1 * at2.d1).sqrt()
}

/// Lennard-Jones term `D·((x/r)¹² - 2(x/r)⁶)`, zero beyond
/// `thresh_multiplier·x` (RDKit `vdWContrib`).
#[derive(Debug, Clone)]
pub struct VdwContrib {
    pub idx1: usize,
    pub idx2: usize,
    pub xij: f64,
    pub well_depth: f64,
    pub thresh: f64,
}

impl VdwContrib {
    pub fn new(idx1: usize, idx2: usize, at1: &AtomicParams, at2: &AtomicParams, thresh_multiplier: f64) -> Self {
        let xij = calc_nonbonded_minimum(at1, at2);
        let well_depth = calc_nonbonded_depth(at1, at2);
        VdwContrib { idx1, idx2, xij, well_depth, thresh: thresh_multiplier * xij }
    }
}

impl Contrib for VdwContrib {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        let dist = rd_distance(pos, self.idx1, self.idx2);
        if dist > self.thresh || dist <= 0.0 {
            return 0.0;
        }
        let r = self.xij / dist;
        let r6 = int_pow::<6>(r);
        let r12 = r6 * r6;
        self.well_depth * (r12 - 2.0 * r6)
    }

    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let dist = rd_distance(pos, self.idx1, self.idx2);
        if dist > self.thresh {
            return;
        }
        let (a, b) = (3 * self.idx1, 3 * self.idx2);
        if dist <= 0.0 {
            for i in 0..3 {
                grad[a + i] += 100.0;
                grad[b + i] -= 100.0;
            }
            return;
        }
        let r = self.xij / dist;
        let r7 = int_pow::<7>(r);
        let r13 = int_pow::<13>(r);
        let pre_factor = 12. * self.well_depth / self.xij * (r7 - r13);
        for i in 0..3 {
            let d_grad = pre_factor * (pos[a + i] - pos[b + i]) / dist;
            grad[a + i] += d_grad;
            grad[b + i] -= d_grad;
        }
    }

    fn label(&self) -> &'static str {
        "VdW"
    }
}
