//! Geometric constraints (flat-bottom harmonic restraints).
//!
//! Port of RDKit `Code/ForceField/{Distance,Angle,Torsion,Position}Constraint.cpp`
//! (BSD-3-Clause, Copyright (C) 2013-2020 Paolo Tosco and other RDKit contributors).

use super::utils::{RAD2DEG, add_grad, compute_dihedral, distance, normalize_angle_deg, p3};
use super::{Contrib, ContribForceField};
use crate::core::geometry::{cross, dot, norm, scale, sub};

/// A geometric constraint applied on top of a force field.
///
/// Energies follow RDKit: distance and position restraints are
/// `0.5·k·Δ²` (Å), angle and torsion restraints are `k·Δ²` with Δ in degrees.
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    /// Keep `|r_i - r_j|` within `[min, max]` Å.
    Distance { i: usize, j: usize, min: f64, max: f64, k: f64 },
    /// Keep the angle i-j-k within `[min, max]` degrees.
    Angle { i: usize, j: usize, k: usize, min: f64, max: f64, force: f64 },
    /// Keep the dihedral i-j-k-l within `[min, max]` degrees (may wrap).
    Dihedral { i: usize, j: usize, k: usize, l: usize, min: f64, max: f64, force: f64 },
    /// Keep atom `i` within `max_displ` Å of `pos` (`None` = its current position).
    Position { i: usize, pos: Option<[f64; 3]>, max_displ: f64, k: f64 },
    /// Freeze atom `i` completely.
    Fixed { i: usize },
}

impl Constraint {
    /// Distance restrained to an exact value.
    pub fn distance(i: usize, j: usize, value: f64, k: f64) -> Self {
        Constraint::Distance { i, j, min: value, max: value, k }
    }
    /// Angle restrained to an exact value (degrees).
    pub fn angle(i: usize, j: usize, k: usize, value_deg: f64, force: f64) -> Self {
        Constraint::Angle { i, j, k, min: value_deg, max: value_deg, force }
    }
    /// Dihedral restrained to an exact value (degrees).
    pub fn dihedral(i: usize, j: usize, k: usize, l: usize, value_deg: f64, force: f64) -> Self {
        let v = normalize_angle_deg(value_deg);
        Constraint::Dihedral { i, j, k, l, min: v, max: v, force }
    }

    /// Adds the corresponding contrib(s) to `ff`.
    pub fn apply(&self, ff: &mut ContribForceField, coords: &[[f64; 3]]) {
        match *self {
            Constraint::Distance { i, j, min, max, k } => {
                ff.add_contrib(Box::new(DistanceConstraintContrib { i, j, min, max, k }))
            }
            Constraint::Angle { i, j, k, min, max, force } => {
                ff.add_contrib(Box::new(AngleConstraintContrib { i, j, k, min, max, force }))
            }
            Constraint::Dihedral { i, j, k, l, min, max, force } => ff.add_contrib(Box::new(TorsionConstraintContrib {
                idx: [i, j, k, l],
                min: normalize_angle_deg(min),
                max: normalize_angle_deg(max),
                force,
            })),
            Constraint::Position { i, pos, max_displ, k } => ff.add_contrib(Box::new(PositionConstraintContrib {
                i,
                pos0: pos.unwrap_or(coords[i]),
                max_displ,
                k,
            })),
            Constraint::Fixed { i } => ff.fix_point(i),
        }
    }

    /// Atom indices involved.
    pub fn atoms(&self) -> Vec<usize> {
        match *self {
            Constraint::Distance { i, j, .. } => vec![i, j],
            Constraint::Angle { i, j, k, .. } => vec![i, j, k],
            Constraint::Dihedral { i, j, k, l, .. } => vec![i, j, k, l],
            Constraint::Position { i, .. } | Constraint::Fixed { i } => vec![i],
        }
    }
}

#[derive(Debug, Clone)]
pub struct DistanceConstraintContrib {
    pub i: usize,
    pub j: usize,
    pub min: f64,
    pub max: f64,
    pub k: f64,
}

impl Contrib for DistanceConstraintContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        let d = distance(pos, self.i, self.j);
        let t = if d < self.min {
            self.min - d
        } else if d > self.max {
            d - self.max
        } else {
            0.0
        };
        0.5 * self.k * t * t
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let d = distance(pos, self.i, self.j);
        let pre = if d < self.min {
            d - self.min
        } else if d > self.max {
            d - self.max
        } else {
            return;
        } * self.k;
        let a = p3(pos, self.i);
        let b = p3(pos, self.j);
        let g = scale(sub(a, b), pre / d.max(1e-8));
        add_grad(grad, self.i, g);
        add_grad(grad, self.j, scale(g, -1.0));
    }
    fn label(&self) -> &'static str {
        "Constraint"
    }
}

#[derive(Debug, Clone)]
pub struct AngleConstraintContrib {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub min: f64,
    pub max: f64,
    pub force: f64,
}

impl AngleConstraintContrib {
    fn term(&self, angle: f64) -> f64 {
        if angle < self.min {
            angle - self.min
        } else if angle > self.max {
            angle - self.max
        } else {
            0.0
        }
    }
}

impl Contrib for AngleConstraintContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        let r0 = sub(p3(pos, self.i), p3(pos, self.j));
        let r1 = sub(p3(pos, self.k), p3(pos, self.j));
        let l0 = dot(r0, r0).max(1e-5);
        let l1 = dot(r1, r1).max(1e-5);
        let c = (dot(r0, r1) / (l0 * l1).sqrt()).clamp(-1.0, 1.0);
        let t = self.term(RAD2DEG * c.acos());
        self.force * t * t
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let r0 = sub(p3(pos, self.i), p3(pos, self.j));
        let r1 = sub(p3(pos, self.k), p3(pos, self.j));
        let l0 = dot(r0, r0).max(1e-5);
        let l1 = dot(r1, r1).max(1e-5);
        let c = (dot(r0, r1) / (l0 * l1).sqrt()).clamp(-1.0, 1.0);
        let t = self.term(RAD2DEG * c.acos());
        let de = 2.0 * RAD2DEG * self.force * t;
        let rp = cross(r1, r0);
        let pre = de / norm(rp).max(1e-5);
        let t0 = -pre / l0;
        let t1 = pre / l1;
        let d0 = scale(cross(r0, rp), t0);
        let d2 = scale(cross(r1, rp), t1);
        let d1 = scale([d0[0] + d2[0], d0[1] + d2[1], d0[2] + d2[2]], -1.0);
        add_grad(grad, self.i, d0);
        add_grad(grad, self.j, d1);
        add_grad(grad, self.k, d2);
    }
    fn label(&self) -> &'static str {
        "Constraint"
    }
}

#[derive(Debug, Clone)]
pub struct TorsionConstraintContrib {
    pub idx: [usize; 4],
    pub min: f64,
    pub max: f64,
    pub force: f64,
}

impl TorsionConstraintContrib {
    fn term(&self, dih: f64) -> f64 {
        let (mn, mx) = (self.min, self.max);
        let mut target = dih;
        if !(dih > mn && dih < mx) && !(dih > mn && mn > mx) && !(dih < mx && mn > mx) {
            let a = normalize_angle_deg(dih - mn);
            let b = normalize_angle_deg(dih - mx);
            target = if a.abs() < b.abs() { mn } else { mx };
        }
        normalize_angle_deg(dih - target)
    }
}

impl Contrib for TorsionConstraintContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        let [a, b, c, d] = self.idx;
        let dd = compute_dihedral(pos, a, b, c, d);
        let t = self.term(dd.dihedral * RAD2DEG);
        self.force * t * t
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let [a, b, c, d] = self.idx;
        let dd = compute_dihedral(pos, a, b, c, d);
        let t = self.term(dd.dihedral * RAD2DEG);
        let de = 2.0 * RAD2DEG * self.force * t;
        torsion_grad(pos, self.idx, &dd, de, grad);
    }
    fn label(&self) -> &'static str {
        "Constraint"
    }
}

/// Distributes `dE/dφ` onto the four atoms (RDKit torsion gradient code).
pub fn torsion_grad(pos: &[f64], idx: [usize; 4], dd: &super::utils::DihedralData, de_dphi: f64, grad: &mut [f64]) {
    let [a, b, c, d] = idx;
    let r = dd.r;
    let d23 = distance(pos, b, c);
    let r31 = sub(p3(pos, c), p3(pos, a));
    let r42 = sub(p3(pos, d), p3(pos, b));
    let pre = de_dphi / d23;
    let tt0 = cross(r[0], r[1]);
    let tt1 = cross(r[2], r[3]);
    let l0 = dot(tt0, tt0).max(1e-12);
    let l1 = dot(tt1, tt1).max(1e-12);
    let dedt0 = scale(cross(tt0, r[2]), pre / l0);
    let dedt1 = scale(cross(tt1, r[1]), pre / l1);
    let g0 = cross(r[2], dedt0);
    let g1 = sub(cross(r31, dedt0), cross(r[3], dedt1));
    let g2 = crate::core::geometry::add(cross(r[0], dedt0), cross(r42, dedt1));
    let g3 = cross(r[2], dedt1);
    add_grad(grad, a, g0);
    add_grad(grad, b, g1);
    add_grad(grad, c, g2);
    add_grad(grad, d, g3);
}

#[derive(Debug, Clone)]
pub struct PositionConstraintContrib {
    pub i: usize,
    pub pos0: [f64; 3],
    pub max_displ: f64,
    pub k: f64,
}

impl Contrib for PositionConstraintContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        let d = norm(sub(p3(pos, self.i), self.pos0));
        let t = (d - self.max_displ).max(0.0);
        0.5 * self.k * t * t
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let p = p3(pos, self.i);
        let d = norm(sub(p, self.pos0));
        if d <= self.max_displ {
            return;
        }
        let pre = (d - self.max_displ) * self.k / d.max(1e-8);
        add_grad(grad, self.i, scale(sub(p, self.pos0), pre));
    }
    fn label(&self) -> &'static str {
        "Constraint"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::{ForceField, gradient_error};

    #[test]
    fn constraint_gradients() {
        let coords = [[0.1, 0.0, 0.0], [1.5, 0.2, -0.1], [2.0, 1.4, 0.3], [3.1, 1.6, 1.2]];
        let pos: Vec<f64> = coords.iter().flatten().copied().collect();
        let mut ff = ContribForceField::new(4, 3);
        ff.add_constraints(
            &coords,
            &[
                Constraint::distance(0, 3, 2.0, 10.0),
                Constraint::angle(0, 1, 2, 100.0, 0.5),
                Constraint::dihedral(0, 1, 2, 3, 170.0, 0.3),
                Constraint::Position { i: 2, pos: Some([2.5, 1.0, 0.0]), max_displ: 0.1, k: 5.0 },
            ],
        );
        assert!(ff.energy(&pos) > 0.0);
        assert!(gradient_error(&ff, &pos) < 1e-4);
    }
}
