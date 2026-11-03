//! Ring perception.
//!
//! Rings are the *relevant cycles* of the molecular graph (the union of all
//! minimum cycle bases), computed from Horton's candidate set with Gaussian
//! elimination over GF(2).  For ordinary molecules this coincides with
//! RDKit's symmetrized SSSR.

use crate::core::Molecule;
use std::collections::{HashSet, VecDeque};

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
        let rings = relevant_cycles(mol);
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

type BitSet = Vec<u64>;

fn bs_new(n: usize) -> BitSet {
    vec![0u64; n.div_ceil(64).max(1)]
}
fn bs_set(b: &mut BitSet, i: usize) {
    b[i / 64] |= 1u64 << (i % 64);
}
fn bs_get(b: &BitSet, i: usize) -> bool {
    b[i / 64] >> (i % 64) & 1 == 1
}
fn bs_xor(a: &mut BitSet, b: &BitSet) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= *y;
    }
}
fn bs_is_zero(a: &BitSet) -> bool {
    a.iter().all(|&x| x == 0)
}
fn bs_first(a: &BitSet) -> Option<usize> {
    for (k, &w) in a.iter().enumerate() {
        if w != 0 {
            return Some(k * 64 + w.trailing_zeros() as usize);
        }
    }
    None
}

/// GF(2) elimination basis with pivot tracking.
struct Gf2Basis {
    rows: Vec<(usize, BitSet)>, // (pivot, row)
}
impl Gf2Basis {
    fn new() -> Self {
        Gf2Basis { rows: Vec::new() }
    }
    fn reduce(&self, v: &BitSet) -> BitSet {
        let mut r = v.clone();
        for (p, row) in &self.rows {
            if bs_get(&r, *p) {
                bs_xor(&mut r, row);
            }
        }
        r
    }
    fn is_independent(&self, v: &BitSet) -> bool {
        !bs_is_zero(&self.reduce(v))
    }
    fn insert(&mut self, v: &BitSet) -> bool {
        let r = self.reduce(v);
        if let Some(p) = bs_first(&r) {
            // keep rows reduced w.r.t. the new pivot
            for (_, row) in self.rows.iter_mut() {
                if bs_get(row, p) {
                    bs_xor(row, &r);
                }
            }
            self.rows.push((p, r));
            true
        } else {
            false
        }
    }
}

fn bfs_tree(mol: &Molecule, root: usize) -> (Vec<usize>, Vec<usize>) {
    let n = mol.num_atoms();
    let mut dist = vec![usize::MAX; n];
    let mut parent = vec![usize::MAX; n];
    let mut q = VecDeque::new();
    dist[root] = 0;
    q.push_back(root);
    while let Some(u) = q.pop_front() {
        // deterministic order
        let mut nb: Vec<usize> = mol.neighbors(u).collect();
        nb.sort_unstable();
        for v in nb {
            if dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                parent[v] = u;
                q.push_back(v);
            }
        }
    }
    (dist, parent)
}

fn path_to_root(parent: &[usize], mut v: usize) -> Vec<usize> {
    let mut p = vec![v];
    while parent[v] != usize::MAX {
        v = parent[v];
        p.push(v);
    }
    p
}

fn relevant_cycles(mol: &Molecule) -> Vec<Vec<usize>> {
    let n = mol.num_atoms();
    let m = mol.num_bonds();
    if m == 0 {
        return Vec::new();
    }
    // Restrict to the 2-core (atoms in cycles) to save work.
    let mut deg: Vec<usize> = (0..n).map(|i| mol.degree(i)).collect();
    let mut removed = vec![false; n];
    let mut stack: Vec<usize> = (0..n).filter(|&i| deg[i] < 2).collect();
    while let Some(u) = stack.pop() {
        if removed[u] {
            continue;
        }
        removed[u] = true;
        for v in mol.neighbors(u) {
            if !removed[v] {
                deg[v] -= 1;
                if deg[v] < 2 {
                    stack.push(v);
                }
            }
        }
    }
    let core: Vec<usize> = (0..n).filter(|&i| !removed[i]).collect();
    if core.is_empty() {
        return Vec::new();
    }
    let core_bonds: Vec<usize> = (0..m)
        .filter(|&b| !removed[mol.bond(b).begin] && !removed[mol.bond(b).end])
        .collect();

    // Horton candidates
    let mut candidates: Vec<(usize, BitSet, Vec<usize>)> = Vec::new();
    let mut seen: HashSet<BitSet> = HashSet::new();
    for &root in &core {
        let (dist, parent) = bfs_tree(mol, root);
        for &b in &core_bonds {
            let x = mol.bond(b).begin;
            let y = mol.bond(b).end;
            if dist[x] == usize::MAX || dist[y] == usize::MAX {
                continue;
            }
            if parent[x] == y || parent[y] == x {
                continue;
            }
            let px = path_to_root(&parent, x);
            let py = path_to_root(&parent, y);
            // paths must only share the root
            let sx: HashSet<usize> = px.iter().copied().collect();
            if py.iter().filter(|v| sx.contains(v)).count() != 1 {
                continue;
            }
            let mut cyc: Vec<usize> = px.clone();
            cyc.reverse(); // root ... x
            cyc.extend(py.iter().take(py.len() - 1).copied()); // y ... (child of root)
            let len = cyc.len();
            if len < 3 {
                continue;
            }
            let mut bits = bs_new(m);
            let mut ok = true;
            for k in 0..len {
                match mol.bond_between(cyc[k], cyc[(k + 1) % len]) {
                    Some(bi) => bs_set(&mut bits, bi),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && seen.insert(bits.clone()) {
                candidates.push((len, bits, cyc));
            }
        }
    }
    candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(&b.2)));

    // A cycle is relevant iff it is not a GF(2) sum of strictly shorter cycles.
    let mut basis_shorter = Gf2Basis::new();
    let mut result = Vec::new();
    let mut i = 0;
    while i < candidates.len() {
        let len = candidates[i].0;
        let mut j = i;
        while j < candidates.len() && candidates[j].0 == len {
            j += 1;
        }
        let mut accepted = Vec::new();
        for c in &candidates[i..j] {
            if basis_shorter.is_independent(&c.1) {
                accepted.push(c);
            }
        }
        for c in &accepted {
            basis_shorter.insert(&c.1);
            result.push(c.2.clone());
        }
        i = j;
    }
    result
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
