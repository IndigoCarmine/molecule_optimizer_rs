//! Valence model and implicit hydrogen counting (simplified port of RDKit's
//! valence rules: charged atoms use the valence list of the isoelectronic
//! neutral element).

use crate::core::{Molecule, element};

/// Allowed valences for an atom with the given charge. Empty = unrestricted.
pub fn allowed_valences(atomic_num: u8, charge: i8) -> Vec<i32> {
    let base = element(atomic_num);
    if base.valences.first() == Some(&-1) || base.valences.is_empty() {
        return Vec::new();
    }
    if charge == 0 {
        return base.valences.clone();
    }
    let eff = atomic_num as i32 - charge as i32;
    if eff < 1 || eff as usize >= crate::core::element::num_elements() {
        return Vec::new();
    }
    let iso = element(eff as u8);
    // closed-shell ions (Na+, Cl-, Mg2+ ...): no bonds
    if iso.valences == [0] {
        return vec![0];
    }
    // only use isoelectronic rules within the same row for main-group atoms
    if iso.row != base.row || iso.valences.first() == Some(&-1) {
        return base.valences.clone();
    }
    iso.valences.clone()
}

/// Default (first) valence for the charged atom, `-1` if unrestricted.
pub fn default_valence(atomic_num: u8, charge: i8) -> i32 {
    allowed_valences(atomic_num, charge).first().copied().unwrap_or(-1)
}

/// Computes `implicit_hs` for every atom.
pub fn compute_implicit_hs(mol: &mut Molecule) {
    for i in 0..mol.num_atoms() {
        let a = mol.atom(i);
        if a.no_implicit || a.atomic_num == 0 {
            mol.atom_mut(i).implicit_hs = 0;
            continue;
        }
        let vals = allowed_valences(a.atomic_num, a.formal_charge);
        if vals.is_empty() {
            mol.atom_mut(i).implicit_hs = 0;
            continue;
        }
        let ev = mol.explicit_valence(i) + a.num_radical_electrons as i32;
        let target = vals.iter().copied().find(|&v| v >= ev);
        let nh = match target {
            Some(v) => (v - ev).max(0),
            None => 0,
        };
        mol.atom_mut(i).implicit_hs = nh as u8;
    }
}

/// Marks all atoms as having no implicit hydrogens (all H explicit).
pub fn set_all_hydrogens_explicit(mol: &mut Molecule) {
    for a in mol.atoms_mut() {
        a.no_implicit = true;
        a.implicit_hs = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn isoelectronic() {
        assert_eq!(allowed_valences(7, 1), vec![4]);
        assert_eq!(allowed_valences(8, -1), vec![1]);
        assert_eq!(allowed_valences(6, -1), vec![3]);
    }
}
