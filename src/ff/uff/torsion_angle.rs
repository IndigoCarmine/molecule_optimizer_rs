//! UFF torsion term.
//!
//! Port of RDKit `Code/ForceField/UFF/TorsionAngle.{h,cpp}`.
//!
//! Copyright (C) 2004-2006 Rational Discovery LLC.
//! Distributed under the BSD 3-Clause license (see `LICENSE-RDKIT.txt`).

use super::is_double_zero;
use super::params::AtomicParams;
use crate::core::Hybridization;
use crate::core::geometry::V3;
use crate::ff::Contrib;
use crate::ff::utils::Torsion;

/// RDKit `Utils::calculateCosTorsion`.
pub fn calculate_cos_torsion(p1: V3, p2: V3, p3_: V3, p4: V3) -> f64 {
    Torsion::new(p1, p2, p3_, p4).cos_torsion()
}

/// RDKit `Utils::isInGroup6`.
pub fn is_in_group6(num: u8) -> bool {
    matches!(num, 8 | 16 | 34 | 52 | 84)
}

/// RDKit `Utils::equation17` (sp2-sp2 barrier).
pub fn equation17(bond_order23: f64, at2: &AtomicParams, at3: &AtomicParams) -> f64 {
    5. * (at2.u1 * at3.u1).sqrt() * (1. + 4.18 * bond_order23.ln())
}

/// UFF torsion `V/2·(1 - cos(n·phi0)·cos(n·phi))` (RDKit `TorsionAngleContrib`).
#[derive(Debug, Clone)]
pub struct TorsionAngleContrib {
    pub idx: [usize; 4],
    pub force_constant: f64,
    pub order: u32,
    pub cos_term: f64,
}

impl TorsionAngleContrib {
    /// `hyb2`/`hyb3` must be SP2 or SP3.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        idx: [usize; 4],
        bond_order23: f64,
        at_num2: u8,
        at_num3: u8,
        hyb2: Hybridization,
        hyb3: Hybridization,
        at2: &AtomicParams,
        at3: &AtomicParams,
        end_atom_is_sp2: bool,
    ) -> Self {
        use Hybridization::{SP2, SP3};
        debug_assert!(matches!(hyb2, SP2 | SP3) && matches!(hyb3, SP2 | SP3), "bad hybridizations");
        let (force_constant, order, cos_term);
        if hyb2 == SP3 && hyb3 == SP3 {
            if bond_order23 == 1.0 && is_in_group6(at_num2) && is_in_group6(at_num3) {
                let v2 = if at_num2 == 8 { 2.0 } else { 6.8 };
                let v3 = if at_num3 == 8 { 2.0 } else { 6.8 };
                force_constant = f64::sqrt(v2 * v3);
                order = 2;
                cos_term = -1.0;
            } else {
                force_constant = (at2.v1 * at3.v1).sqrt();
                order = 3;
                cos_term = -1.0;
            }
        } else if hyb2 == SP2 && hyb3 == SP2 {
            force_constant = equation17(bond_order23, at2, at3);
            order = 2;
            cos_term = 1.0;
        } else {
            let mut k = 1.0;
            let mut o = 6;
            let mut ct = 1.0;
            if bond_order23 == 1.0 {
                if (hyb2 == SP3 && is_in_group6(at_num2) && !is_in_group6(at_num3))
                    || (hyb3 == SP3 && is_in_group6(at_num3) && !is_in_group6(at_num2))
                {
                    k = equation17(bond_order23, at2, at3);
                    o = 2;
                    ct = -1.0;
                } else if end_atom_is_sp2 {
                    k = 2.0;
                    o = 3;
                    ct = -1.0;
                }
            }
            force_constant = k;
            order = o;
            cos_term = ct;
        }
        TorsionAngleContrib { idx, force_constant, order, cos_term }
    }

    /// RDKit `scaleForceConstant`.
    pub fn scale_force_constant(&mut self, count: usize) {
        self.force_constant /= count as f64;
    }

    fn theta_deriv(&self, cos_theta: f64, sin_theta: f64) -> f64 {
        let sin_theta_sq = sin_theta * sin_theta;
        let mut res = match self.order {
            2 => 2.0 * sin_theta * cos_theta,
            3 => sin_theta * (3.0 - 4.0 * sin_theta_sq),
            6 => cos_theta * sin_theta * (32.0 * sin_theta_sq * (sin_theta_sq - 1.0) + 6.0),
            _ => 0.0,
        };
        // RDKit: `res *= d_forceConstant / 2.0 * d_cosTerm * -1 * d_order` (negation is exact)
        res *= -(self.force_constant / 2.0 * self.cos_term) * self.order as f64;
        res
    }

    #[inline(always)]
    fn energy_of(&self, g: &Torsion) -> f64 {
        let cos_phi = g.cos_torsion();
        let sin_phi_sq = 1.0 - cos_phi * cos_phi;
        let cos_n_phi = match self.order {
            2 => 1.0 - 2.0 * sin_phi_sq,
            3 => cos_phi * (cos_phi * cos_phi - 3. * sin_phi_sq),
            6 => 1.0 + sin_phi_sq * (-32. * sin_phi_sq * sin_phi_sq + 48. * sin_phi_sq - 18.),
            _ => 0.0,
        };
        self.force_constant / 2.0 * (1. - self.cos_term * cos_n_phi)
    }

    #[inline(always)]
    fn grad_of(&self, g: &Torsion, grad: &mut [f64]) {
        let cos_phi = g.cos_phi;
        let sin_phi_sq = 1.0 - cos_phi * cos_phi;
        let sin_phi = if sin_phi_sq > 0.0 { sin_phi_sq.sqrt() } else { 0.0 };
        let de_dphi = self.theta_deriv(cos_phi, sin_phi);
        let sin_term = de_dphi * (if is_double_zero(sin_phi) { 1.0 / cos_phi } else { 1.0 / sin_phi });
        g.add_grad(grad, self.idx, sin_term);
    }
}

impl Contrib for TorsionAngleContrib {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        self.energy_of(&Torsion::at(pos, self.idx))
    }

    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        self.grad_of(&Torsion::at(pos, self.idx), grad);
    }

    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> f64 {
        let g = Torsion::at(pos, self.idx);
        self.grad_of(&g, grad);
        self.energy_of(&g)
    }

    fn label(&self) -> &'static str {
        "Torsion"
    }
}
