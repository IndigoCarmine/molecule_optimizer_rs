//! Construction of the distance-geometry and ETKDG force fields.
//!
//! Port of RDKit `Code/DistGeom/DistGeomUtils.cpp` (constructForceField,
//! construct3DForceField, constructPlain3DForceField,
//! construct3DImproperForceField and their helpers) and
//! `Code/DistGeom/ChiralSet.h`.
//! RDKit: Copyright (C) 2004-2026 Greg Landrum and other RDKit contributors,
//! BSD-3-Clause.

use super::bounds::BoundsMatrix;
use super::contribs::*;
use super::point::P3;
use super::torsion_prefs::EtkdgDetails;
use crate::ff::ContribForceField;

const KNOWN_DIST_TOL: f64 = 0.01;
const KNOWN_DIST_FORCE_CONSTANT: f64 = 100.0;

/// RDKit `DistGeom::ChiralSet`: center `idx0` and four points whose signed
/// volume must lie in `[lower, upper]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChiralSet {
    pub idx0: usize,
    pub idx: [usize; 4],
    pub lower: f64,
    pub upper: f64,
    /// RDKit `ChiralSetStructureFlags::IN_FUSED_SMALL_RINGS`.
    pub in_fused_small_rings: bool,
}

fn new_ff(n: usize, dim: usize) -> ContribForceField {
    let mut ff = ContribForceField::new(n, dim);
    ff.parallel_threshold = 0;
    ff
}

/// RDKit `DistGeom::constructForceField` (bounds violation + chirality +
/// optional 4th-dimension terms).
pub fn construct_force_field(
    mmat: &BoundsMatrix,
    dim: usize,
    csets: &[ChiralSet],
    weight_distance: f64,
    weight_chiral: f64,
    weight_fourth_dim: f64,
    basin_size_tol: f64,
) -> ContribForceField {
    let n = mmat.num_rows();
    let mut ff = new_ff(n, dim);
    let mut dv = DistViolationContribs::new(dim);
    for i in 1..n {
        for j in 0..i {
            let l = mmat.lower(i, j);
            let u = mmat.upper(i, j);
            if u - l <= basin_size_tol {
                dv.add(i, j, u, l, weight_distance);
            }
        }
    }
    if !dv.is_empty() {
        ff.add_contrib(Box::new(dv));
    }
    if weight_chiral > 1.0e-8 {
        let mut cv = ChiralViolationContribs::new(dim);
        for c in csets {
            cv.add(c.idx, c.lower, c.upper, weight_chiral);
        }
        if !cv.is_empty() {
            ff.add_contrib(Box::new(cv));
        }
    }
    if dim == 4 && weight_fourth_dim > 1.0e-8 {
        let mut fd = FourthDimContribs::new();
        for i in 0..n {
            fd.add(i, weight_fourth_dim);
        }
        if !fd.is_empty() {
            ff.add_contrib(Box::new(fd));
        }
    }
    ff
}

fn add_improper_terms(ff: &mut ContribForceField, scaling: f64, impropers: &[[usize; 6]], constrained: &mut [bool]) {
    let mut inv = InversionContribs::new(3);
    for ia in impropers {
        for n in [[0, 1, 2, 3], [0, 1, 3, 2], [2, 1, 3, 0]] {
            inv.add([ia[n[0]], ia[n[1]], ia[n[2]], ia[n[3]]], ia[4] as i32, ia[5] != 0, scaling);
            constrained[ia[n[1]]] = true;
        }
    }
    if !inv.is_empty() {
        ff.add_contrib(Box::new(inv));
    }
}

fn add_exp_torsion_terms(ff: &mut ContribForceField, d: &EtkdgDetails, atom_pairs: &mut [bool], n: usize) {
    let mut done = vec![false; n * n];
    let mut tc = TorsionAngleContribs::new(3);
    for t in &d.exp_torsions {
        let [i, j, k, l] = t.atoms;
        let idx = if i < l { i * n + l } else { l * n + i };
        let bidx = if j < k { j * n + k } else { k * n + j };
        if done[bidx] {
            continue;
        }
        atom_pairs[idx] = true;
        tc.add(t.atoms, t.v, t.signs);
        done[bidx] = true;
    }
    if !tc.is_empty() {
        ff.add_contrib(Box::new(tc));
    }
}

fn p3(pos: &[f64], i: usize) -> P3 {
    P3::from_pos(pos, i, 3)
}

fn add12_terms(ff: &mut ContribForceField, d: &EtkdgDetails, atom_pairs: &mut [bool], pos: &[f64], n: usize) {
    let mut dc = DistanceConstraintContribs::new(3);
    for &(i, j) in &d.bonds {
        if i < j {
            atom_pairs[i * n + j] = true;
        } else {
            atom_pairs[j * n + i] = true;
        }
        let dist = (p3(pos, i) - p3(pos, j)).length();
        dc.add(i, j, dist - KNOWN_DIST_TOL, dist + KNOWN_DIST_TOL, KNOWN_DIST_FORCE_CONSTANT);
    }
    if !dc.is_empty() {
        ff.add_contrib(Box::new(dc));
    }
}

#[allow(clippy::too_many_arguments)]
fn add13_terms(
    ff: &mut ContribForceField,
    d: &EtkdgDetails,
    atom_pairs: &mut [bool],
    pos: &[f64],
    improper_constrained: &[bool],
    use_basic_knowledge: bool,
    mmat: &BoundsMatrix,
    n: usize,
) {
    let mut dc = DistanceConstraintContribs::new(3);
    let mut ac = AngleConstraintContribs::new(3);
    for a in &d.angles {
        let (i, j, k) = (a[0], a[1], a[2]);
        if i < k {
            atom_pairs[i * n + k] = true;
        } else {
            atom_pairs[k * n + i] = true;
        }
        if use_basic_knowledge && a[3] != 0 {
            ac.add(i, j, k, 179.0, 180.0, d.force_consts.k_term_angle);
        } else if improper_constrained[j] {
            dc.add(i, k, mmat.lower(i, k), mmat.upper(i, k), KNOWN_DIST_FORCE_CONSTANT);
        } else {
            let dist = (p3(pos, i) - p3(pos, k)).length();
            dc.add(i, k, dist - KNOWN_DIST_TOL, dist + KNOWN_DIST_TOL, KNOWN_DIST_FORCE_CONSTANT);
        }
    }
    if !ac.is_empty() {
        ff.add_contrib(Box::new(ac));
    }
    if !dc.is_empty() {
        ff.add_contrib(Box::new(dc));
    }
}

fn add_long_range(ff: &mut ContribForceField, d: &EtkdgDetails, atom_pairs: &[bool], mmat: &BoundsMatrix, n: usize) {
    let mut dc = DistanceConstraintContribs::new(3);
    for i in 1..n {
        for j in 0..i {
            if !atom_pairs[j * n + i] {
                let fdist = d.bounds_mat_force_scaling * 10.0;
                dc.add(i, j, mmat.lower(i, j), mmat.upper(i, j), fdist);
            }
        }
    }
    if !dc.is_empty() {
        ff.add_contrib(Box::new(dc));
    }
}

/// RDKit `construct3DForceField` (ETKDG / KDG). `pos` is 3D.
pub fn construct_3d_force_field(mmat: &BoundsMatrix, pos: &[f64], d: &EtkdgDetails) -> ContribForceField {
    let n = mmat.num_rows();
    let mut ff = new_ff(n, 3);
    let mut atom_pairs = vec![false; n * n];
    let mut improper_constrained = vec![false; n];
    add_exp_torsion_terms(&mut ff, d, &mut atom_pairs, n);
    add_improper_terms(&mut ff, d.force_consts.k_term_improper, &d.improper_atoms, &mut improper_constrained);
    add12_terms(&mut ff, d, &mut atom_pairs, pos, n);
    add13_terms(&mut ff, d, &mut atom_pairs, pos, &improper_constrained, true, mmat, n);
    add_long_range(&mut ff, d, &atom_pairs, mmat, n);
    ff
}

/// RDKit `constructPlain3DForceField` (ETDG).
pub fn construct_plain_3d_force_field(mmat: &BoundsMatrix, pos: &[f64], d: &EtkdgDetails) -> ContribForceField {
    let n = mmat.num_rows();
    let mut ff = new_ff(n, 3);
    let mut atom_pairs = vec![false; n * n];
    let improper_constrained = vec![false; n];
    add_exp_torsion_terms(&mut ff, d, &mut atom_pairs, n);
    add12_terms(&mut ff, d, &mut atom_pairs, pos, n);
    add13_terms(&mut ff, d, &mut atom_pairs, pos, &improper_constrained, false, mmat, n);
    add_long_range(&mut ff, d, &atom_pairs, mmat, n);
    ff
}

/// RDKit `construct3DImproperForceField` (planarity / linearity check).
pub fn construct_3d_improper_force_field(n: usize, d: &EtkdgDetails) -> ContribForceField {
    let mut ff = new_ff(n, 3);
    const OOB: f64 = 10.0;
    let mut constrained = vec![false; n];
    add_improper_terms(&mut ff, OOB, &d.improper_atoms, &mut constrained);
    let mut ac = AngleConstraintContribs::new(3);
    for a in &d.angles {
        if a[3] != 0 {
            ac.add(a[0], a[1], a[2], 179.0, 180.0, OOB);
        }
    }
    if !ac.is_empty() {
        ff.add_contrib(Box::new(ac));
    }
    ff
}
