//! Ring perception.
//!
//! Rings are the *relevant cycles* of the molecular graph (the union of all
//! minimum cycle bases), in exactly the set, order and per-ring atom order of
//! RDKit's default ring perception (`MolOps::symmetrizeSSSR` as called by
//! `sanitizeMol`, i.e. RingDecomposerLib's relevant cycles); see
//! [`super::ring_decomposer`].

use super::ring_decomposer::{rdkit_atom_rings, rdkit_atom_rings_with};
use crate::core::Molecule;

/// Ring information.
#[derive(Debug, Clone, Default)]
pub struct RingInfo {
    /// Each ring as an ordered cycle of atom indices.
    pub atom_rings: Vec<Vec<usize>>,
    /// Each ring as bond indices (same order as `atom_rings`, bond k joins
    /// atoms k and k+1).
    pub bond_rings: Vec<Vec<usize>>,
    /// ring ids of every atom, ascending (atom `a` owns
    /// `atom_membership[atom_start[a]..atom_start[a + 1]]`)
    atom_start: Vec<usize>,
    atom_membership: Vec<usize>,
    /// ring ids of every bond, laid out like the atom memberships
    bond_start: Vec<usize>,
    bond_membership: Vec<usize>,
}

/// Flat (CSR) membership lists: for each of `n` items the ids of the
/// `rings` (lists of item indices) containing it, in ring order.
fn membership(n: usize, rings: &[Vec<usize>]) -> (Vec<usize>, Vec<usize>) {
    let mut start = vec![0usize; n + 1];
    for r in rings {
        for &x in r {
            start[x + 1] += 1;
        }
    }
    for i in 0..n {
        start[i + 1] += start[i];
    }
    let mut fill = start[..n].to_vec();
    let mut ids = vec![0usize; start[n]];
    for (id, r) in rings.iter().enumerate() {
        for &x in r {
            ids[fill[x]] = id;
            fill[x] += 1;
        }
    }
    (start, ids)
}

impl RingInfo {
    pub fn compute(mol: &Molecule) -> RingInfo {
        Self::from_atom_rings(mol, rdkit_atom_rings(mol))
    }

    /// Like [`RingInfo::compute`], but dative bonds can be ring bonds (RDKit
    /// `symmetrizeSSSR(mol, includeDativeBonds = true)`).
    pub fn compute_including_dative(mol: &Molecule) -> RingInfo {
        Self::from_atom_rings(mol, rdkit_atom_rings_with(mol, true))
    }

    fn from_atom_rings(mol: &Molecule, atom_rings: Vec<Vec<usize>>) -> RingInfo {
        let bond_rings: Vec<Vec<usize>> = atom_rings
            .iter()
            .map(|ring| (0..ring.len()).map(|k| mol.bond_between(ring[k], ring[(k + 1) % ring.len()]).expect("ring bond")).collect())
            .collect();
        let (atom_start, atom_membership) = membership(mol.num_atoms(), &atom_rings);
        let (bond_start, bond_membership) = membership(mol.num_bonds(), &bond_rings);
        RingInfo { atom_rings, bond_rings, atom_start, atom_membership, bond_start, bond_membership }
    }

    pub fn num_rings(&self) -> usize {
        self.atom_rings.len()
    }
    pub fn num_atom_rings(&self, a: usize) -> usize {
        self.atom_start[a + 1] - self.atom_start[a]
    }
    pub fn num_bond_rings(&self, b: usize) -> usize {
        self.bond_start[b + 1] - self.bond_start[b]
    }
    pub fn atom_ring_ids(&self, a: usize) -> &[usize] {
        &self.atom_membership[self.atom_start[a]..self.atom_start[a + 1]]
    }
    pub fn bond_ring_ids(&self, b: usize) -> &[usize] {
        &self.bond_membership[self.bond_start[b]..self.bond_start[b + 1]]
    }
    pub fn is_atom_in_ring(&self, a: usize) -> bool {
        self.num_atom_rings(a) > 0
    }
    pub fn is_bond_in_ring(&self, b: usize) -> bool {
        self.num_bond_rings(b) > 0
    }
    pub fn is_atom_in_ring_of_size(&self, a: usize, size: usize) -> bool {
        self.atom_ring_ids(a).iter().any(|&r| self.atom_rings[r].len() == size)
    }
    pub fn is_bond_in_ring_of_size(&self, b: usize, size: usize) -> bool {
        self.bond_ring_ids(b).iter().any(|&r| self.bond_rings[r].len() == size)
    }
    /// Size of the smallest ring containing atom `a` (0 if acyclic).
    pub fn min_atom_ring_size(&self, a: usize) -> usize {
        self.atom_ring_ids(a).iter().map(|&r| self.atom_rings[r].len()).min().unwrap_or(0)
    }
    pub fn min_bond_ring_size(&self, b: usize) -> usize {
        self.bond_ring_ids(b).iter().map(|&r| self.bond_rings[r].len()).min().unwrap_or(0)
    }
    /// `true` if atoms `a` and `b` share a ring.
    pub fn are_atoms_in_same_ring(&self, a: usize, b: usize) -> bool {
        let rb = self.atom_ring_ids(b);
        self.atom_ring_ids(a).iter().any(|r| rb.contains(r))
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
