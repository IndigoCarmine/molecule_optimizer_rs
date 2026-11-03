//! Conjugation and hybridization.
//!
//! Port of RDKit `Code/GraphMol/ConjugHybrid.cpp`
//! (BSD-3-Clause, Copyright (C) 2001-2006 Rational Discovery LLC and others).

use super::aromaticity::count_atom_elec;
use crate::core::{BondOrder, ChiralTag, Hybridization, Molecule, element};

fn is_atom_conjug_cand(mol: &Molecule, i: usize) -> bool {
    let at = mol.atom(i);
    let vals = &element(at.atomic_num).valences;
    if at.formal_charge == 0 {
        if let Some(&v0) = vals.first() {
            if v0 >= 0 && mol.total_valence(i) > v0 {
                return false;
            }
        }
    }
    let nouter = element(at.atomic_num).n_outer;
    ((at.atomic_num <= 10) || (nouter != 5 && nouter != 6) || (nouter == 6 && mol.total_degree(i) < 2))
        && count_atom_elec(mol, i) > 0
}

/// RDKit `MolOps::setConjugation`.
pub fn set_conjugation(mol: &mut Molecule) {
    let n = mol.num_atoms();
    for b in mol.bonds_mut() {
        b.is_conjugated = b.is_aromatic;
    }
    let mut cand = vec![false; n];
    let mut nsub = vec![0usize; n];
    for i in 0..n {
        cand[i] = is_atom_conjug_cand(mol, i);
        nsub[i] = if cand[i] { mol.degree(i) + mol.atom(i).total_hidden_hs() as usize } else { 0 };
    }
    let mut to_mark = Vec::new();
    for i in 0..n {
        if !cand[i] || nsub[i] < 2 || nsub[i] > 3 {
            continue;
        }
        let bonds = mol.atom_bonds(i).to_vec();
        for &(o1, b1) in &bonds {
            if mol.bond(b1).valence_contrib(i) < 1.5 || !cand[o1] {
                continue;
            }
            for &(o2, b2) in &bonds {
                if b1 == b2 {
                    continue;
                }
                if nsub[o2] > 3 {
                    continue;
                }
                if cand[o2] {
                    to_mark.push(b1);
                    to_mark.push(b2);
                }
            }
        }
    }
    for b in to_mark {
        mol.bond_mut(b).is_conjugated = true;
    }
}

fn num_bonds_plus_lone_pairs(mol: &Molecule, i: usize) -> i32 {
    let at = mol.atom(i);
    let mut deg = mol.total_degree(i) as i32;
    for &(_, bi) in mol.atom_bonds(i) {
        let b = mol.bond(bi);
        if b.order == BondOrder::Zero || (b.order == BondOrder::Dative && i != b.end) {
            deg -= 1;
        }
    }
    if at.atomic_num <= 1 {
        return deg;
    }
    let nouter = element(at.atomic_num).n_outer as i32;
    let total_valence = mol.total_valence(i);
    let chg = at.formal_charge as i32;
    let num_free = nouter - (total_valence + chg);
    if total_valence + nouter - chg < 8 {
        let nrad = at.num_radical_electrons as i32;
        let nlp = (num_free - nrad) / 2;
        deg + nlp + nrad
    } else {
        let nlp = num_free / 2;
        deg + nlp
    }
}

fn atom_has_conjugated_bond(mol: &Molecule, i: usize) -> bool {
    mol.atom_bonds(i).iter().any(|&(_, b)| mol.bond(b).is_conjugated)
}

/// RDKit `MolOps::setHybridization`.
pub fn set_hybridization(mol: &mut Molecule) {
    for i in 0..mol.num_atoms() {
        let at = mol.atom(i);
        if at.atomic_num == 0 {
            mol.atom_mut(i).hybridization = Hybridization::Unspecified;
            continue;
        }
        if at.chiral_tag != ChiralTag::Unspecified && mol.total_degree(i) == 4 {
            mol.atom_mut(i).hybridization = Hybridization::SP3;
            continue;
        }
        let norbs = if at.atomic_num < 89 { num_bonds_plus_lone_pairs(mol, i) } else { mol.total_degree(i) as i32 };
        let h = match norbs {
            0 | 1 => Hybridization::S,
            2 => Hybridization::SP,
            3 => Hybridization::SP2,
            4 => {
                if mol.total_degree(i) > 3 || !atom_has_conjugated_bond(mol, i) {
                    Hybridization::SP3
                } else {
                    Hybridization::SP2
                }
            }
            5 => Hybridization::SP3D,
            6 => Hybridization::SP3D2,
            _ => Hybridization::Unspecified,
        };
        mol.atom_mut(i).hybridization = h;
    }
}
