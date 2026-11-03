//! UFF torsion term.
//!
//! Port of RDKit `Code/ForceField/UFF/TorsionAngle.{h,cpp}`.
//!
//! Copyright (C) 2004-2006 Rational Discovery LLC.
//! Distributed under the BSD 3-Clause license (see `LICENSE-RDKIT.txt`).

use super::is_double_zero;
use super::params::AtomicParams;
use crate::core::Hybridization;
use crate::core::geometry::{V3, cross, dot, norm, sub};
use crate::ff::Contrib;
use crate::ff::utils::{compute_dihedral, p3};

/// RDKit `Utils::calculateCosTorsion`.
pub fn calculate_cos_torsion(p1: V3, p2: V3, p3_: V3, p4: V3) -> f64 {
    let r1 = sub(p1, p2);
    let r2 = sub(p3_, p2);
    let r3 = sub(p2, p3_);
    let r4 = sub(p4, p3_);
    let t1 = cross(r1, r2);
    let t2 = cross(r3, r4);
    let d1 = norm(t1);
    let d2 = norm(t2);
    if is_double_zero(d1) || is_double_zero(d2) {
        return 0.0;
    }
    (dot(t1, t2) / (d1 * d2)).clamp(-1.0, 1.0)
}

/// RDKit `Utils::isInGroup6`.
pub fn is_in_group6(num: u8) -> bool {
    matches!(num, 8 | 16 | 34 | 52 | 84)
}

/// RDKit `Utils::equation17` (sp2-sp2 barrier).
pub fn equation17(bond_order23: f64, at2: &AtomicParams, at3: &AtomicParams) -> f64 {
    5. * (at2.u1 * at3.u1).sqrt() * (1. + 4.18 * bond_order23.ln())
}

/// RDKit `Utils::calcTorsionGrad`.
pub(crate) fn calc_torsion_grad(
    r: &[V3; 4],
    t: &[V3; 2],
    d: &[f64; 2],
    g: [usize; 4],
    grad: &mut [f64],
    sin_term: f64,
    cos_phi: f64,
) {
    let dcos_dt = [
        1.0 / d[0] * (t[1][0] - cos_phi * t[0][0]),
        1.0 / d[0] * (t[1][1] - cos_phi * t[0][1]),
        1.0 / d[0] * (t[1][2] - cos_phi * t[0][2]),
        1.0 / d[1] * (t[0][0] - cos_phi * t[1][0]),
        1.0 / d[1] * (t[0][1] - cos_phi * t[1][1]),
        1.0 / d[1] * (t[0][2] - cos_phi * t[1][2]),
    ];
    let (x, y, z) = (0, 1, 2);
    let a = 3 * g[0];
    grad[a] += sin_term * (dcos_dt[2] * r[1][y] - dcos_dt[1] * r[1][z]);
    grad[a + 1] += sin_term * (dcos_dt[0] * r[1][z] - dcos_dt[2] * r[1][x]);
    grad[a + 2] += sin_term * (dcos_dt[1] * r[1][x] - dcos_dt[0] * r[1][y]);

    let b = 3 * g[1];
    grad[b] += sin_term
        * (dcos_dt[1] * (r[1][z] - r[0][z])
            + dcos_dt[2] * (r[0][y] - r[1][y])
            + dcos_dt[4] * (-r[3][z])
            + dcos_dt[5] * (r[3][y]));
    grad[b + 1] += sin_term
        * (dcos_dt[0] * (r[0][z] - r[1][z])
            + dcos_dt[2] * (r[1][x] - r[0][x])
            + dcos_dt[3] * (r[3][z])
            + dcos_dt[5] * (-r[3][x]));
    grad[b + 2] += sin_term
        * (dcos_dt[0] * (r[1][y] - r[0][y])
            + dcos_dt[1] * (r[0][x] - r[1][x])
            + dcos_dt[3] * (-r[3][y])
            + dcos_dt[4] * (r[3][x]));

    let c = 3 * g[2];
    grad[c] += sin_term
        * (dcos_dt[1] * (r[0][z])
            + dcos_dt[2] * (-r[0][y])
            + dcos_dt[4] * (r[3][z] - r[2][z])
            + dcos_dt[5] * (r[2][y] - r[3][y]));
    grad[c + 1] += sin_term
        * (dcos_dt[0] * (-r[0][z])
            + dcos_dt[2] * (r[0][x])
            + dcos_dt[3] * (r[2][z] - r[3][z])
            + dcos_dt[5] * (r[3][x] - r[2][x]));
    grad[c + 2] += sin_term
        * (dcos_dt[0] * (r[0][y])
            + dcos_dt[1] * (-r[0][x])
            + dcos_dt[3] * (r[3][y] - r[2][y])
            + dcos_dt[4] * (r[2][x] - r[3][x]));

    let e = 3 * g[3];
    grad[e] += sin_term * (dcos_dt[4] * r[2][z] - dcos_dt[5] * r[2][y]);
    grad[e + 1] += sin_term * (dcos_dt[5] * r[2][x] - dcos_dt[3] * r[2][z]);
    grad[e + 2] += sin_term * (dcos_dt[3] * r[2][y] - dcos_dt[4] * r[2][x]);
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
}

impl Contrib for TorsionAngleContrib {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        let [i1, i2, i3, i4] = self.idx;
        let cos_phi = calculate_cos_torsion(p3(pos, i1), p3(pos, i2), p3(pos, i3), p3(pos, i4));
        let sin_phi_sq = 1.0 - cos_phi * cos_phi;
        let cos_n_phi = match self.order {
            2 => 1.0 - 2.0 * sin_phi_sq,
            3 => cos_phi * (cos_phi * cos_phi - 3. * sin_phi_sq),
            6 => 1.0 + sin_phi_sq * (-32. * sin_phi_sq * sin_phi_sq + 48. * sin_phi_sq - 18.),
            _ => 0.0,
        };
        self.force_constant / 2.0 * (1. - self.cos_term * cos_n_phi)
    }

    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let [i1, i2, i3, i4] = self.idx;
        let dd = compute_dihedral(pos, i1, i2, i3, i4);
        let cos_phi = dd.cos_phi;
        let sin_phi_sq = 1.0 - cos_phi * cos_phi;
        let sin_phi = if sin_phi_sq > 0.0 { sin_phi_sq.sqrt() } else { 0.0 };
        let de_dphi = self.theta_deriv(cos_phi, sin_phi);
        let sin_term = de_dphi * (if is_double_zero(sin_phi) { 1.0 / cos_phi } else { 1.0 / sin_phi });
        calc_torsion_grad(&dd.r, &dd.t, &dd.d, self.idx, grad, sin_term, cos_phi);
    }

    fn label(&self) -> &'static str {
        "Torsion"
    }
}
