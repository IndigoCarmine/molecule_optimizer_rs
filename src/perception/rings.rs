//! Ring perception.
//!
//! Rings are the *relevant cycles* of the molecular graph (the union of all
//! minimum cycle bases), in exactly the set, order and per-ring atom order of
//! RDKit's default ring perception (`MolOps::symmetrizeSSSR` as called by
//! `sanitizeMol`, i.e. RingDecomposerLib's relevant cycles); see
//! [`super::ring_decomposer`].

use super::ring_decomposer::rdkit_atom_rings;
use crate::core::Molecule;

/// Ring information.
#[derive(Debug, Clone, Default)]
pub struct RingInfo {
    /// Each ring as an ordered cycle of atom indices.
    pub atom_rings: Vec<Vec<usize>>,
    /// Each ring as bond indices (same order as `atom_rings`, bond k joins
    /// atoms k and k+1).
    pub bond_rings: Vec<Vec<usize>>,
    atom_membership: Vec<Vec<usize>>,
    bond_membership: Vec<Vec<usize>>,
}

impl RingInfo {
    pub fn compute(mol: &Molecule) -> RingInfo {
        let rings = rdkit_atom_rings(mol);
        let mut info = RingInfo {
            atom_membership: vec![Vec::new(); mol.num_atoms()],
            bond_membership: vec![Vec::new(); mol.num_bonds()],
            ..Default::default()
        };
        for ring in rings {
            let mut bonds = Vec::with_capacity(ring.len());
            for k in 0..ring.len() {
                let a = ring[k];
                let b = ring[(k + 1) % ring.len()];
                bonds.push(mol.bond_between(a, b).expect("ring bond"));
            }
            let id = info.atom_rings.len();
            for &a in &ring {
                info.atom_membership[a].push(id);
            }
            for &b in &bonds {
                info.bond_membership[b].push(id);
            }
            info.atom_rings.push(ring);
            info.bond_rings.push(bonds);
        }
        info
    }

    pub fn num_rings(&self) -> usize {
        self.atom_rings.len()
    }
    pub fn num_atom_rings(&self, a: usize) -> usize {
        self.atom_membership[a].len()
    }
    pub fn num_bond_rings(&self, b: usize) -> usize {
        self.bond_membership[b].len()
    }
    pub fn atom_ring_ids(&self, a: usize) -> &[usize] {
        &self.atom_membership[a]
    }
    pub fn bond_ring_ids(&self, b: usize) -> &[usize] {
        &self.bond_membership[b]
    }
    pub fn is_atom_in_ring(&self, a: usize) -> bool {
        !self.atom_membership[a].is_empty()
    }
    pub fn is_bond_in_ring(&self, b: usize) -> bool {
        !self.bond_membership[b].is_empty()
    }
    pub fn is_atom_in_ring_of_size(&self, a: usize, size: usize) -> bool {
        self.atom_membership[a].iter().any(|&r| self.atom_rings[r].len() == size)
    }
    pub fn is_bond_in_ring_of_size(&self, b: usize, size: usize) -> bool {
        self.bond_membership[b].iter().any(|&r| self.bond_rings[r].len() == size)
    }
    /// Size of the smallest ring containing atom `a` (0 if acyclic).
    pub fn min_atom_ring_size(&self, a: usize) -> usize {
        self.atom_membership[a].iter().map(|&r| self.atom_rings[r].len()).min().unwrap_or(0)
    }
    pub fn min_bond_ring_size(&self, b: usize) -> usize {
        self.bond_membership[b].iter().map(|&r| self.bond_rings[r].len()).min().unwrap_or(0)
    }
    /// `true` if atoms `a` and `b` share a ring.
    pub fn are_atoms_in_same_ring(&self, a: usize, b: usize) -> bool {
        self.atom_membership[a].iter().any(|r| self.atom_membership[b].contains(r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, BondOrder};

    fn ring_mol(sizes: &[(usize, usize)], n: usize) -> Molecule {
        let mut m = Molecule::new();
        for _ in 0..n {
            m.add_atom(Atom::new(6), [0.0; 3]);
        }
        for &(a, b) in sizes {
            m.add_bond(a, b, BondOrder::Single);
        }
        m
    }

    #[test]
    fn naphthalene_rings() {
        // two fused 6-rings sharing bond 0-5
        let bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)];
        let m = ring_mol(&bonds, 10);
        let ri = RingInfo::compute(&m);
        assert_eq!(ri.num_rings(), 2);
        assert!(ri.atom_rings.iter().all(|r| r.len() == 6));
        assert_eq!(ri.num_atom_rings(0), 2);
    }

    #[test]
    fn cubane_rings() {
        let bonds = [
            (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7),
        ];
        let m = ring_mol(&bonds, 8);
        let ri = RingInfo::compute(&m);
        assert_eq!(ri.num_rings(), 6);
    }
}
