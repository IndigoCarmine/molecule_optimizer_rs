//! Construction of the MMFF94 force field for a molecule.
//!
//! Port of RDKit `Code/GraphMol/ForceFieldHelpers/MMFF/Builder.cpp`
//! (`MMFF::Tools::add*`, `buildNeighborMatrix`, `constructForceField`).
//! Terms are added in the same order as RDKit (bonds, angles,
//! stretch-bends, out-of-plane, torsions, non-bonded), which also fixes the
//! floating-point summation order.
//!
//! RDKit: Copyright (C) 2013-2025 Paolo Tosco, Greg Landrum and other RDKit
//! contributors. BSD 3-Clause license (see `LICENSE-RDKIT.txt`).

use super::MmffBuilder;
use super::terms::*;
use super::typer::{MmffMolProperties, is_double_zero};
use crate::core::{BondOrder, Hybridization, Molecule};
use crate::ff::ContribForceField;
use crate::ff::utils::BondDistances;

const RELATION_1_2: u8 = 0;
const RELATION_1_3: u8 = 1;
const RELATION_1_4: u8 = 2;
const RELATION_1_X: u8 = 3;

/// Bonds matched by RDKit's default torsion SMARTS
/// `[!$(*#*)&!D1]~[!$(*#*)&!D1]`, in the order RDKit's substructure matcher
/// reports them (first atom ascending, second atom in neighbor order,
/// uniquified).
fn torsion_bonds(mol: &Molecule) -> Vec<(usize, usize)> {
    let n = mol.num_atoms();
    let ok: Vec<bool> = (0..n)
        .map(|i| mol.degree(i) != 1 && !mol.atom_bonds(i).iter().any(|&(_, b)| mol.bond(b).order == BondOrder::Triple))
        .collect();
    let mut seen = vec![false; mol.num_bonds()];
    let mut out = Vec::new();
    for a in 0..n {
        if !ok[a] {
            continue;
        }
        for &(b, bi) in mol.atom_bonds(a) {
            if ok[b] && !seen[bi] {
                seen[bi] = true;
                out.push((a, b));
            }
        }
    }
    out
}

/// Builds the MMFF contributions for `mol` (coordinates taken from
/// `coords`) with the given molecule properties.
pub(crate) fn add_mmff_terms(ff: &mut ContribForceField, props: &MmffMolProperties, coords: &[[f64; 3]], opts: &MmffBuilder) {
    let mol = &props.tm.mol;
    let n = mol.num_atoms();
    let params = props.params;

    // bonds
    let mut bonds = BondStretchContrib::default();
    for b in mol.bonds() {
        if let Some((_, p)) = props.bond_stretch_params(b.begin, b.end) {
            bonds.add_term(b.begin, b.end, p.r0, p.kb);
        }
    }
    if !bonds.is_empty() {
        ff.add_contrib(Box::new(bonds));
    }

    // angles
    let mut angles = AngleBendContrib::default();
    for j in 0..n {
        if mol.degree(j) == 1 {
            continue;
        }
        let central = params.prop.get(props.atom_type(j)).expect("MMFF properties");
        let nb: Vec<usize> = mol.neighbors(j).collect();
        for a in 0..nb.len() {
            for c in (a + 1)..nb.len() {
                if let Some((_, p)) = props.angle_bend_params(nb[a], j, nb[c]) {
                    angles.add_term(nb[a], j, nb[c], p.theta0, p.ka, central.linh > 0);
                }
            }
        }
    }
    if !angles.is_empty() {
        ff.add_contrib(Box::new(angles));
    }

    // stretch-bend
    let mut stbn = StretchBendContrib::default();
    for j in 0..n {
        if mol.degree(j) == 1 {
            continue;
        }
        let central = params.prop.get(props.atom_type(j)).expect("MMFF properties");
        if central.linh != 0 {
            continue;
        }
        let nb: Vec<usize> = mol.neighbors(j).collect();
        for a in 0..nb.len() {
            for c in (a + 1)..nb.len() {
                if let Some((_, s, b, ang)) = props.stretch_bend_params(nb[a], j, nb[c]) {
                    stbn.add_term(nb[a], j, nb[c], b[0].r0, b[1].r0, ang.theta0, s.kba_ijk, s.kba_kji);
                }
            }
        }
    }
    if !stbn.is_empty() {
        ff.add_contrib(Box::new(stbn));
    }

    // out-of-plane
    let mut oop = OopBendContrib::default();
    for j in 0..n {
        if mol.degree(j) != 3 {
            continue;
        }
        let nb: Vec<usize> = mol.neighbors(j).collect();
        let idx = [nb[0], j, nb[1], nb[2]];
        let Some(p) = props.oop_bend_params(idx[0], idx[1], idx[2], idx[3]) else {
            // no parameters: the term is excluded (SURDOX02)
            continue;
        };
        for perm in [[0usize, 1, 2, 3], [0, 1, 3, 2], [2, 1, 3, 0]] {
            oop.add_term(idx[perm[0]], idx[perm[1]], idx[perm[2]], idx[perm[3]], p.koop);
        }
    }
    if !oop.is_empty() {
        ff.add_contrib(Box::new(oop));
    }

    // torsions
    let mut tors = TorsionAngleContrib::default();
    for (i2, i3) in torsion_bonds(mol) {
        let hj = mol.atom(i2).hybridization;
        let hk = mol.atom(i3).hybridization;
        let ok = |h: Hybridization| h == Hybridization::SP2 || h == Hybridization::SP3;
        if !(ok(hj) && ok(hk)) {
            continue;
        }
        let bond = mol.bond_between(i2, i3).expect("torsion bond");
        for &(i1, b1) in mol.atom_bonds(i2) {
            if b1 == bond {
                continue;
            }
            for &(i4, b2) in mol.atom_bonds(i3) {
                if b2 == bond || b2 == b1 || i4 == i1 {
                    continue;
                }
                if let Some((_, p)) = props.torsion_params(i1, i2, i3, i4) {
                    tors.add_term(i1, i2, i3, i4, p.v1, p.v2, p.v3);
                }
            }
        }
    }
    if !tors.is_empty() {
        ff.add_contrib(Box::new(tors));
    }

    // non-bonded
    {
        // RDKit `buildNeighborMatrix`, one row at a time
        let mut rel = BondDistances::new(n, 3);
        let mut frag = vec![0usize; n];
        if opts.ignore_interfrag_interactions {
            for (k, f) in mol.fragments().iter().enumerate() {
                for &a in f {
                    frag[a] = k;
                }
            }
        }
        let q = props.partial_charges();
        let mut nb = NonbondedContrib::default();
        for i in 0..n {
            rel.compute(mol, i);
            for j in (i + 1)..n {
                if opts.ignore_interfrag_interactions && frag[i] != frag[j] {
                    continue;
                }
                let cell = match rel.get(j) {
                    1 => RELATION_1_2,
                    2 => RELATION_1_3,
                    3 => RELATION_1_4,
                    _ => RELATION_1_X,
                };
                if cell < RELATION_1_4 {
                    continue;
                }
                let is_1_4 = cell == RELATION_1_4;
                let d = [coords[i][0] - coords[j][0], coords[i][1] - coords[j][1], coords[i][2] - coords[j][2]];
                let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if dist > opts.nonbonded_thresh {
                    continue;
                }
                let vdw = props.vdw_params(i, j).map(|v| (v.r_ij_star, v.epsilon));
                let ele = if !is_double_zero(q[i]) && !is_double_zero(q[j]) {
                    Some((q[i] * q[j] / opts.dielectric_constant, opts.dielectric_model, is_1_4))
                } else {
                    None
                };
                nb.add_term(i, j, vdw, ele);
            }
        }
        if !nb.is_empty() {
            ff.add_contrib(Box::new(nb));
        }
    }
}
