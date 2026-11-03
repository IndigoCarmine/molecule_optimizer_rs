//! UFF force-field construction.
//!
//! Port of RDKit `Code/GraphMol/ForceFieldHelpers/UFF/Builder.cpp`
//! (`addBonds`, `addAngles`, `addAngleSpecialCases`,
//! `addTrigonalBipyramidAngles`, `buildNeighborMatrix`, `addNonbonded`,
//! `addTorsions`, `addInversions`, `constructForceField`). Contribs are
//! added in exactly the same order as RDKit so that energies are summed
//! identically.
//!
//! Copyright (C) 2004-2018 Greg Landrum and Rational Discovery LLC.
//! Distributed under the BSD 3-Clause license (see `LICENSE-RDKIT.txt`).

use super::angle_bend::AngleBendContrib;
use super::bond_stretch::BondStretchContrib;
use super::inversion::InversionContrib;
use super::nonbonded::{VdwContrib, calc_nonbonded_minimum};
use super::params::AtomicParams;
use super::torsion_angle::TorsionAngleContrib;
use crate::core::geometry::{dot, norm, sub};
use crate::core::{BondOrder, Hybridization, Molecule};
use crate::error::{Error, Result};
use crate::ff::utils::BondDistances;
use crate::ff::{Contrib, ContribForceField, TermList};
use crate::perception::rings::RingInfo;

type Params = [Option<&'static AtomicParams>];

/// RDKit `Bond::getBondTypeAsDouble`.
pub(crate) fn bond_type_as_double(order: BondOrder) -> f64 {
    match order {
        BondOrder::Zero => 0.0,
        BondOrder::Single | BondOrder::Dative => 1.0,
        BondOrder::Double => 2.0,
        BondOrder::Triple => 3.0,
        BondOrder::Aromatic => 1.5,
    }
}

fn bond_order_between(mol: &Molecule, a: usize, b: usize) -> Result<f64> {
    let bi = mol
        .bond_between(a, b)
        .ok_or_else(|| Error::ForceField(format!("no bond between atoms {a} and {b}")))?;
    let bo = bond_type_as_double(mol.bond(bi).order);
    if bo <= 0.0 {
        return Err(Error::ForceField(format!("UFF: bad bond order for bond {a}-{b}")));
    }
    Ok(bo)
}

fn add_bonds(mol: &Molecule, params: &Params, out: &mut Vec<BondStretchContrib>) -> Result<()> {
    for b in mol.bonds() {
        if let (Some(p1), Some(p2)) = (params[b.begin], params[b.end]) {
            let bo = bond_type_as_double(b.order);
            if bo <= 0.0 {
                return Err(Error::ForceField(format!("UFF: bad bond order for bond {}-{}", b.begin, b.end)));
            }
            out.push(BondStretchContrib::new(b.begin, b.end, bo, p1, p2));
        }
    }
    Ok(())
}

fn add_angles(mol: &Molecule, rings: &RingInfo, params: &Params, out: &mut Vec<AngleBendContrib>) -> Result<()> {
    for j in 0..mol.num_atoms() {
        let Some(pj) = params[j] else { continue };
        if mol.degree(j) == 1 {
            continue;
        }
        let hyb = mol.atom(j).hybridization;
        let nbrs: Vec<usize> = mol.neighbors(j).collect();
        for (n1, &i) in nbrs.iter().enumerate() {
            let Some(pi) = params[i] else { continue };
            for &k in &nbrs[n1 + 1..] {
                let Some(pk) = params[k] else { continue };
                if hyb == Hybridization::SP3D && mol.degree(j) == 5 {
                    continue;
                }
                let order = match hyb {
                    Hybridization::SP => 1,
                    Hybridization::SP2 => {
                        let mut order = 3;
                        if rings.is_atom_in_ring_of_size(j, 3) {
                            let (i3, k3) = (rings.is_atom_in_ring_of_size(i, 3), rings.is_atom_in_ring_of_size(k, 3));
                            if i3 != k3 {
                                order = 30;
                            } else if i3 && k3 {
                                order = 35;
                            }
                        } else if rings.is_atom_in_ring_of_size(j, 4) {
                            let (i4, k4) = (rings.is_atom_in_ring_of_size(i, 4), rings.is_atom_in_ring_of_size(k, 4));
                            if i4 != k4 {
                                order = 40;
                            } else if i4 && k4 {
                                order = 45;
                            }
                        }
                        order
                    }
                    Hybridization::SP3D2 => 4,
                    _ => 0,
                };
                let b1 = bond_order_between(mol, i, j)?;
                let b2 = bond_order_between(mol, k, j)?;
                out.push(AngleBendContrib::new(i, j, k, b1, b2, pi, pj, pk, order));
            }
        }
    }
    Ok(())
}

fn direction_vector(from: [f64; 3], to: [f64; 3]) -> [f64; 3] {
    let r = sub(to, from);
    let l = norm(r);
    [r[0] / l, r[1] / l, r[2] / l]
}

fn add_trigonal_bipyramid_angles(
    mol: &Molecule,
    aid: usize,
    params: &Params,
    out: &mut Vec<AngleBendContrib>,
) -> Result<()> {
    let bonds = mol.atom_bonds(aid);
    let mut ax: Option<(usize, usize)> = None; // (bond idx, other atom) pairs
    let mut ax_other: Option<(usize, usize)> = None;
    let mut most_neg = 100.0;
    let pa = mol.coords[aid];
    for &(o1, b1) in bonds {
        let v1 = direction_vector(pa, mol.coords[o1]);
        for &(o2, b2) in bonds {
            if b2 > b1 {
                let v2 = direction_vector(pa, mol.coords[o2]);
                let d = dot(v1, v2);
                if d < most_neg {
                    most_neg = d;
                    ax = Some((b1, o1));
                    ax_other = Some((b2, o2));
                }
            }
        }
    }
    let (Some((ax1b, ax1)), Some((ax2b, ax2))) = (ax, ax_other) else {
        return Err(Error::ForceField("UFF: axial bond not found".into()));
    };
    let eq: Vec<(usize, usize)> = bonds.iter().filter(|&&(_, b)| b != ax1b && b != ax2b).map(|&(o, b)| (b, o)).collect();
    if eq.len() < 3 {
        return Err(Error::ForceField("UFF: equatorial bond not found".into()));
    }
    let center = params[aid].ok_or_else(|| Error::MissingParameters(format!("UFF: no parameters for atom {aid}")))?;
    let bo = |b: usize| bond_type_as_double(mol.bond(b).order);
    let mut add = |(bi, i): (usize, usize), (bj, j): (usize, usize), order: u32| -> Result<()> {
        if let (Some(pi), Some(pj)) = (params[i], params[j]) {
            let (o1, o2) = (bo(bi), bo(bj));
            if o1 <= 0.0 || o2 <= 0.0 {
                return Err(Error::ForceField(format!("UFF: bad bond order at atom {aid}")));
            }
            out.push(AngleBendContrib::new(i, aid, j, o1, o2, pi, center, pj, order));
        }
        Ok(())
    };
    let a1 = (ax1b, ax1);
    let a2 = (ax2b, ax2);
    let (e1, e2, e3) = (eq[0], eq[1], eq[2]);
    add(a1, a2, 2)?;
    add(e1, e2, 3)?;
    add(e1, e3, 3)?;
    add(e2, e3, 3)?;
    add(a1, e1, 0)?;
    add(a1, e2, 0)?;
    add(a1, e3, 0)?;
    add(a2, e1, 0)?;
    add(a2, e2, 0)?;
    add(a2, e3, 0)?;
    Ok(())
}

fn add_angle_special_cases(mol: &Molecule, params: &Params, out: &mut Vec<AngleBendContrib>) -> Result<()> {
    for i in 0..mol.num_atoms() {
        if mol.atom(i).hybridization == Hybridization::SP3D && mol.degree(i) == 5 {
            add_trigonal_bipyramid_angles(mol, i, params, out)?;
        }
    }
    Ok(())
}

fn add_nonbonded(
    mol: &Molecule,
    params: &Params,
    vdw_thresh: f64,
    ignore_interfrag: bool,
    out: &mut Vec<VdwContrib>,
) {
    let n = mol.num_atoms();
    let mut frag = vec![0usize; n];
    if ignore_interfrag {
        for (f, members) in mol.fragments().iter().enumerate() {
            for &a in members {
                frag[a] = f;
            }
        }
    }
    // 1-2 and 1-3 related pairs are excluded (RDKit `buildNeighborMatrix`
    // relation < `RELATION_1_4`), one row at a time
    let mut close = BondDistances::new(n, 2);
    for i in 0..n {
        let Some(pi) = params[i] else { continue };
        close.compute(mol, i);
        for j in i + 1..n {
            let Some(pj) = params[j] else { continue };
            if ignore_interfrag && frag[i] != frag[j] {
                continue;
            }
            if close.get(j) <= 2 {
                continue;
            }
            let dist = norm(sub(mol.coords[i], mol.coords[j]));
            if dist < vdw_thresh * calc_nonbonded_minimum(pi, pj) {
                out.push(VdwContrib::new(i, j, pi, pj, 10.0));
            }
        }
    }
}

/// Atom matches `[!$(*#*)&!D1]`.
fn torsion_atom_ok(mol: &Molecule, i: usize) -> bool {
    mol.degree(i) != 1 && !mol.atom_bonds(i).iter().any(|&(_, b)| mol.bond(b).order == BondOrder::Triple)
}

fn add_torsions(mol: &Molecule, params: &Params, out: &mut Vec<TorsionAngleContrib>) {
    use Hybridization::{SP2, SP3};
    let n = mol.num_atoms();
    // Matches of the default torsion SMARTS `[!$(*#*)&!D1]~[!$(*#*)&!D1]`
    // in the order RDKit's uniquified substructure search reports them: the
    // first atom ascending, the second in neighbour (bond insertion) order,
    // keeping only the orientation with idx1 < idx2. (This reproduces
    // RDKit's energies and gradients bit-for-bit on the reference set.)
    let ok: Vec<bool> = (0..n).map(|i| torsion_atom_ok(mol, i)).collect();
    for idx1 in 0..n {
        if !ok[idx1] {
            continue;
        }
        let partners: Vec<usize> = mol.neighbors(idx1).filter(|&b| b > idx1 && ok[b]).collect();
        for idx2 in partners {
            let (Some(p1), Some(p2)) = (params[idx1], params[idx2]) else { continue };
            let bond = mol.bond_between(idx1, idx2).expect("bond");
            let a1 = mol.atom(idx1);
            let a2 = mol.atom(idx2);
            let mut here: Vec<TorsionAngleContrib> = Vec::new();
            if matches!(a1.hybridization, SP2 | SP3) && matches!(a2.hybridization, SP2 | SP3) {
                let bo = bond_type_as_double(mol.bond(bond).order);
                for &(b_idx, t1) in mol.atom_bonds(idx1) {
                    if t1 == bond {
                        continue;
                    }
                    for &(e_idx, t2) in mol.atom_bonds(idx2) {
                        if t2 == bond || t2 == t1 || e_idx == b_idx {
                            continue;
                        }
                        let has_sp2 = mol.atom(b_idx).hybridization == SP2 || mol.atom(e_idx).hybridization == SP2;
                        here.push(TorsionAngleContrib::new(
                            [b_idx, idx1, idx2, e_idx],
                            bo,
                            a1.atomic_num,
                            a2.atomic_num,
                            a1.hybridization,
                            a2.hybridization,
                            p1,
                            p2,
                            has_sp2,
                        ));
                    }
                }
            }
            let count = here.len();
            for mut t in here {
                t.scale_force_constant(count);
                out.push(t);
            }
        }
    }
}

fn add_inversions(mol: &Molecule, out: &mut Vec<InversionContrib>) {
    for center in 0..mol.num_atoms() {
        let at2 = mol.atom(center).atomic_num;
        if !matches!(at2, 6 | 7 | 8 | 15 | 33 | 51 | 83) || mol.degree(center) != 3 {
            continue;
        }
        if matches!(at2, 6..=8) && mol.atom(center).hybridization != Hybridization::SP2 {
            continue;
        }
        let nb: Vec<usize> = mol.neighbors(center).collect();
        let is_bound_to_sp2_o = at2 == 6
            && nb.iter().any(|&a| mol.atom(a).atomic_num == 8 && mol.atom(a).hybridization == Hybridization::SP2);
        // RDKit stores the neighbours at idx[0], idx[2], idx[3]
        let idx = [nb[0], center, nb[1], nb[2]];
        for perm in [[0, 1, 2, 3], [0, 1, 3, 2], [2, 1, 3, 0]] {
            out.push(InversionContrib::new(
                [idx[perm[0]], idx[perm[1]], idx[perm[2]], idx[perm[3]]],
                at2,
                is_bound_to_sp2_o,
                1.0,
            ));
        }
    }
}

/// Builds all UFF contribs for `mol` with the given per-atom parameters
/// (RDKit `constructForceField`), in RDKit's order: one [`TermList`] per
/// term kind (bond stretch, angle bend, vdW, torsion, inversion; empty
/// kinds are left out).
pub fn build_contribs(
    mol: &Molecule,
    params: &Params,
    vdw_thresh: f64,
    ignore_interfrag_interactions: bool,
) -> Result<Vec<Box<dyn Contrib>>> {
    if params.len() != mol.num_atoms() {
        return Err(Error::InvalidArgument("bad parameters".into()));
    }
    let computed;
    let rings = match mol.ring_info() {
        Some(r) => r,
        None => {
            computed = RingInfo::compute(mol);
            &computed
        }
    };
    // one term list per kind (in RDKit's order; summed term by term)
    let (mut bonds, mut angles, mut vdw, mut torsions, mut inversions) = (vec![], vec![], vec![], vec![], vec![]);
    add_bonds(mol, params, &mut bonds)?;
    add_angles(mol, rings, params, &mut angles)?;
    add_angle_special_cases(mol, params, &mut angles)?;
    add_nonbonded(mol, params, vdw_thresh, ignore_interfrag_interactions, &mut vdw);
    add_torsions(mol, params, &mut torsions);
    add_inversions(mol, &mut inversions);
    Ok([
        TermList::boxed(bonds),
        TermList::boxed(angles),
        TermList::boxed(vdw),
        TermList::boxed(torsions),
        TermList::boxed(inversions),
    ]
    .into_iter()
    .flatten()
    .collect())
}

/// Builds a [`ContribForceField`] for `mol` (no constraints).
pub fn construct_force_field(
    mol: &Molecule,
    params: &Params,
    vdw_thresh: f64,
    ignore_interfrag_interactions: bool,
) -> Result<ContribForceField> {
    let mut ff = ContribForceField::new(mol.num_atoms(), 3);
    ff.add_contribs(build_contribs(mol, params, vdw_thresh, ignore_interfrag_interactions)?);
    Ok(ff)
}
