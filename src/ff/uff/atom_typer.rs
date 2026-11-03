//! UFF atom typing.
//!
//! Port of RDKit `Code/GraphMol/ForceFieldHelpers/UFF/AtomTyper.cpp`
//! (`getAtomLabel`, `addAtomChargeFlags` with `tolerateChargeMismatch =
//! true`, `getAtomTypes`).
//!
//! Copyright (C) 2004-2021 Greg Landrum and other RDKit contributors.
//! Distributed under the BSD 3-Clause license (see `LICENSE-RDKIT.txt`).

use super::params::{AtomicParams, params};
use crate::core::{Hybridization, Molecule, element};

fn atom_has_conjugated_bond(mol: &Molecule, i: usize) -> bool {
    mol.atom_bonds(i).iter().any(|&(_, b)| mol.bond(b).is_conjugated)
}

/// RDKit `Tools::addAtomChargeFlags` (with `tolerateChargeMismatch = true`,
/// the default used by the builder).
fn add_atom_charge_flags(mol: &Molecule, i: usize, key: &mut String) {
    let atom = mol.atom(i);
    let total_valence = mol.total_valence(i);
    let anum = atom.atomic_num;
    // With tolerateChargeMismatch the fixed-charge metals always get their
    // single charge suffix.
    let fixed = match anum {
        29 | 47 => Some(1),
        4 | 20 | 25 | 26 | 28 | 46 | 78 => Some(2),
        21 | 24 | 27 | 79 | 89 | 96..=103 => Some(3),
        2 | 18 | 22 | 36 | 54 | 90..=95 => Some(4),
        23 | 41 | 43 | 73 => Some(5),
        42 => Some(6),
        _ => None,
    };
    if let Some(c) = fixed {
        key.push_str(&format!("+{c}"));
    } else {
        match anum {
            12 => key.push_str("+2"),
            13 | 14 => {}
            15 => key.push_str(if total_valence == 3 { "+3" } else { "+5" }),
            16 => {
                if atom.hybridization != Hybridization::SP2 {
                    key.push_str(match total_valence {
                        2 => "+2",
                        4 => "+4",
                        _ => "+6",
                    });
                }
            }
            30 | 34 | 48 | 52 | 80 | 84 => key.push_str("+2"),
            31 | 33 | 49 | 51 | 81 | 82 | 83 => key.push_str("+3"),
            75 => {
                if key == "Re6" {
                    *key = "Re6+5".to_string();
                } else if key == "Re3" {
                    *key = "Re3+7".to_string();
                }
            }
            _ => {}
        }
    }
    if (57..=71).contains(&anum) {
        key.push_str("+3");
    }
}

/// RDKit `Tools::getAtomLabel`: the UFF atom type label of atom `i`.
pub fn atom_label(mol: &Molecule, i: usize) -> String {
    let atom = mol.atom(i);
    let anum = atom.atomic_num;
    let el = element(anum);
    let mut key = el.symbol.to_string();
    if key.len() == 1 {
        key.push('_');
    }
    if anum != 0 && (el.default_valence() == -1 || (el.n_outer != 1 && el.n_outer != 7)) {
        match anum {
            12 | 13 | 14 | 15 | 50 | 51 | 52 | 81 | 82 | 83 | 84 => key.push('3'),
            80 => key.push('1'),
            _ => match atom.hybridization {
                Hybridization::S => {}
                Hybridization::SP => key.push('1'),
                Hybridization::SP2 => {
                    if (atom.is_aromatic || atom_has_conjugated_bond(mol, i)) && matches!(anum, 6 | 7 | 8 | 16) {
                        key.push('R');
                    } else {
                        key.push('2');
                    }
                }
                Hybridization::SP3 => key.push('3'),
                Hybridization::SP2D => key.push('4'),
                Hybridization::SP3D => key.push('5'),
                Hybridization::SP3D2 => key.push('6'),
                Hybridization::Unspecified | Hybridization::Other => {}
            },
        }
    }
    add_atom_charge_flags(mol, i, &mut key);
    key
}

/// Atom labels plus parameters (`None` where the label is unknown) and a
/// flag telling whether all atoms were typed (RDKit `getAtomTypes`).
pub fn atom_types_and_params(mol: &Molecule) -> (Vec<String>, Vec<Option<&'static AtomicParams>>, bool) {
    let labels: Vec<String> = (0..mol.num_atoms()).map(|i| atom_label(mol, i)).collect();
    let ps: Vec<_> = labels.iter().map(|l| params(l)).collect();
    let all = ps.iter().all(|p| p.is_some());
    (labels, ps, all)
}
