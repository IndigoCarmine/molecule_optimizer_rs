//! RDKit *legacy* stereo perception: CIP ranking and legitimacy of
//! tetrahedral stereocenters and double-bond stereo.
//!
//! Ported from RDKit 2026.09.1 `Code/GraphMol/Chirality.cpp`
//! (`buildCIPInvariants`, `iterateCIPRanks`, `assignAtomCIPRanks`,
//! `rerankAtoms`, `isAtomPotentialChiralCenter`,
//! `atomIsCandidateForRingStereochem`, `findChiralAtomSpecialCases`,
//! `assignAtomChiralCodes`, `assignBondStereoCodes`,
//! `findAtomNeighborDirHelper`, `legacyStereoPerception`,
//! `assignChiralTypesFrom3D`, `setDoubleBondNeighborDirections`,
//! `updateDoubleBondNeighbors`, `controllingBondFromAtom`), from
//! `Code/GraphMol/QueryOps.cpp` (`queryIsAtomBridgehead`) and from the flow of
//! `Code/GraphMol/FileParsers/MolFileParser.cpp` (`finishMolProcessing`).
//!
//! Copyright (C) 2004-2026 Greg Landrum and other RDKit contributors,
//! BSD-3-Clause.
//!
//! The goal is to end up with exactly the chiral tags and double-bond stereo
//! that `Chem.SDMolSupplier(path, removeHs=False)` produces for a V2000
//! molblock with 3D coordinates when the legacy stereo perception is active
//! (RDKit's default): `assignChiralTypesFrom3D` → sanitize →
//! `detectBondStereochemistry` → `assignStereochemistry(cleanIt=true,
//! force=true, flagPossibleStereoCenters=true)`.

use crate::core::{BondDir, BondOrder, BondStereo, ChiralTag, Molecule};
use crate::perception::rings::RingInfo;

/// Most common isotope per atomic number (RDKit periodic table,
/// `getMostCommonIsotope`), indices 0..=118.
const MOST_COMMON_ISOTOPE: [u16; 119] = [
    0, 1, 4, 7, 9, 11, 12, 14, 16, 19, 20, 23, 24, 27, 28, 31, 32, 35, 40, 39, 40, 45, 48, 51, 52, 55, 56, 59, 58,
    63, 64, 69, 74, 75, 80, 79, 84, 85, 88, 89, 90, 93, 98, 97, 102, 103, 106, 107, 114, 115, 120, 121, 130, 127,
    132, 133, 138, 139, 140, 141, 142, 145, 152, 153, 158, 159, 164, 165, 166, 169, 174, 175, 180, 181, 184, 187,
    192, 193, 195, 197, 202, 205, 208, 209, 209, 210, 222, 223, 226, 227, 232, 231, 238, 236, 238, 241, 243, 247,
    249, 252, 257, 258, 259, 262, 267, 268, 271, 270, 269, 278, 281, 281, 285, 284, 289, 288, 293, 292, 294,
];

/// `Chirality::minRingSizeForDoubleBondStereo`.
const MIN_RING_SIZE_FOR_DOUBLE_BOND_STEREO: usize = 8;

/// RDKit `Bond::BondDir` subset relevant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    None,
    EndUpRight,
    EndDownRight,
}

fn flip(d: Dir) -> Dir {
    if d == Dir::EndUpRight { Dir::EndDownRight } else { Dir::EndUpRight }
}

fn is_directional(d: Dir) -> bool {
    matches!(d, Dir::EndUpRight | Dir::EndDownRight)
}

/// RDKit `Bond::BondStereo` subset relevant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum St {
    None,
    Any,
    E,
    Z,
}

fn twice_bond_type(o: BondOrder) -> u32 {
    match o {
        BondOrder::Zero => 0,
        BondOrder::Single | BondOrder::Dative => 2,
        BondOrder::Double => 4,
        BondOrder::Triple => 6,
        BondOrder::Aromatic => 3,
    }
}

/// `Chirality::detail::bondAffectsAtomChirality`.
fn bond_affects_atom_chirality(mol: &Molecule, b: usize, atom: usize) -> bool {
    let bond = mol.bond(b);
    !(bond.order == BondOrder::Zero || (bond.order == BondOrder::Dative && bond.begin == atom))
}

/// `Chirality::detail::getAtomNonzeroDegree`.
fn nonzero_degree(mol: &Molecule, atom: usize) -> usize {
    mol.atom_bonds(atom).iter().filter(|&&(_, b)| bond_affects_atom_chirality(mol, b, atom)).count()
}

/// `Atom::getTotalNumHs()` (without neighbors): hidden hydrogens.
fn hidden_hs(mol: &Molecule, atom: usize) -> usize {
    mol.atom(atom).total_hidden_hs() as usize
}

fn should_detect_double_bond_stereo(ri: &RingInfo, b: usize) -> bool {
    ri.num_bond_rings(b) == 0 || ri.min_bond_ring_size(b) >= MIN_RING_SIZE_FOR_DOUBLE_BOND_STEREO
}

/// `queryIsAtomBridgehead`.
fn is_atom_bridgehead(mol: &Molecule, ri: &RingInfo, atom: usize) -> bool {
    if mol.degree(atom) < 3 {
        return false;
    }
    let mut atom_ring_bonds = vec![false; mol.num_bonds()];
    let mut count = 0;
    for &(_, b) in mol.atom_bonds(atom) {
        if ri.num_bond_rings(b) > 0 {
            atom_ring_bonds[b] = true;
            count += 1;
        }
    }
    if count < 3 {
        return false;
    }
    let nr = ri.bond_rings.len();
    let mut rings_overlap = vec![false; nr];
    let mut in_ring_i = vec![false; mol.num_bonds()];
    for i in 0..nr {
        in_ring_i.iter_mut().for_each(|x| *x = false);
        let mut atom_in_ring_i = false;
        for &b in &ri.bond_rings[i] {
            in_ring_i[b] = true;
            if atom_ring_bonds[b] {
                atom_in_ring_i = true;
            }
        }
        if !atom_in_ring_i {
            continue;
        }
        for j in (i + 1)..nr {
            let mut overlap = 0;
            let mut atom_in_ring_j = false;
            for &b in &ri.bond_rings[j] {
                if atom_ring_bonds[b] {
                    atom_in_ring_j = true;
                }
                if in_ring_i[b] {
                    overlap += 1;
                }
                if overlap >= 2 && atom_in_ring_j {
                    rings_overlap[i] = true;
                    rings_overlap[j] = true;
                    break;
                }
            }
        }
        if !rings_overlap[i] {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// CIP ranking
// ---------------------------------------------------------------------------

/// `buildCIPInvariants`.
fn build_cip_invariants(mol: &Molecule) -> Vec<i64> {
    const N_MASS_BITS: i64 = 10;
    const MAX_MASS: i64 = 1 << N_MASS_BITS;
    mol.atoms()
        .iter()
        .map(|a| {
            let num = (a.atomic_num as i64) % 128;
            let mut mass: i64 = 0;
            if a.isotope != 0 {
                let common = MOST_COMMON_ISOTOPE.get(a.atomic_num as usize).copied().unwrap_or(0) as i64;
                mass = a.isotope as i64 - common;
                if mass >= 0 {
                    mass += 1;
                }
            }
            mass += MAX_MASS / 2;
            if mass < 0 {
                mass = 0;
            } else {
                mass %= MAX_MASS;
            }
            let mut inv = num;
            inv = (inv << N_MASS_BITS) | mass;
            // no atom-map numbers: (-1 + 1) % 1024 == 0
            inv <<= 10;
            inv
        })
        .collect()
}

/// `findSegmentsToResort` + `recomputeRanks`: walks the sorted entries,
/// assigns ranks and returns the tied regions and the number of ranks.
fn find_segments(order: &[usize], cip: &[Vec<i64>], ranks: &mut [u32]) -> (Vec<(usize, usize)>, usize) {
    let mut res: Vec<(usize, usize)> = Vec::new();
    let mut num_independent = order.len();
    let mut current = 0usize;
    let mut running: u32 = 0;
    ranks[order[0]] = 0;
    let mut in_equal = false;
    for i in 1..order.len() {
        if cip[order[current]] == cip[order[i]] {
            ranks[order[i]] = running;
            num_independent -= 1;
            if !in_equal {
                in_equal = true;
                res.push((i - 1, 0));
            }
        } else {
            running += 1;
            ranks[order[i]] = running;
            current = i;
            if in_equal {
                res.last_mut().unwrap().1 = i;
                in_equal = false;
            }
        }
    }
    if in_equal {
        res.last_mut().unwrap().1 = order.len() - 1;
    }
    (res, num_independent)
}

/// `iterateCIPRanks`.
fn iterate_cip_ranks(mol: &Molecule, invars: &[i64], seed_with_invars: bool) -> Vec<u32> {
    let n = mol.num_atoms();
    let mut ranks = vec![0u32; n];
    if n == 0 {
        return ranks;
    }
    let mut cip: Vec<Vec<i64>> = invars.iter().map(|&v| vec![v as i32 as i64]).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| cip[a].cmp(&cip[b]));
    let (mut needs_sorting, mut num_ranks) = find_segments(&order, &cip, &mut ranks);

    for i in 0..n {
        if seed_with_invars {
            cip[i][0] = invars[i] as i32 as i64;
        } else {
            cip[i][0] = mol.atom(i).atomic_num as i64;
            cip[i].push(ranks[i] as i64);
        }
    }
    let cip_rank_index = if seed_with_invars { 1 } else { 2 };

    // computeBondFeatures: (count, neighbor) per bond of each atom.
    let features: Vec<Vec<(u32, usize)>> = (0..n)
        .map(|a| {
            mol.atom_bonds(a)
                .iter()
                .map(|&(nbr, b)| {
                    let bond = mol.bond(b);
                    let p_special = bond.order == BondOrder::Double
                        && mol.atom(nbr).atomic_num == 15
                        && matches!(mol.degree(nbr), 3 | 4);
                    (if p_special { 1 } else { twice_bond_type(bond.order) }, nbr)
                })
                .collect()
        })
        .collect();

    let max_its = n / 2 + 1;
    let mut num_its = 0;
    let mut last_num_ranks: i64 = -1;
    let mut nb: Vec<(u32, usize)> = Vec::new();
    while !needs_sorting.is_empty() && num_its < max_its && (last_num_ranks < 0 || (last_num_ranks as usize) < num_ranks) {
        for index in 0..n {
            nb.clear();
            nb.extend_from_slice(&features[index]);
            // neighbors in decreasing rank order; ties are irrelevant since
            // equal ranks produce identical runs of values
            nb.sort_by(|x, y| ranks[y.1].cmp(&ranks[x.1]));
            let entry = &mut cip[index];
            for &(count, idx) in &nb {
                for _ in 0..count {
                    entry.push(ranks[idx] as i64 + 1);
                }
            }
            for _ in 0..hidden_hs(mol, index) {
                entry.push(0);
            }
        }
        last_num_ranks = num_ranks as i64;
        for &(first, last) in &needs_sorting {
            order[first..=last].sort_by(|&a, &b| cip[a].cmp(&cip[b]));
        }
        let (ns, nr) = find_segments(&order, &cip, &mut ranks);
        needs_sorting = ns;
        num_ranks = nr;
        if last_num_ranks as usize != num_ranks {
            for i in 0..n {
                cip[i].truncate(cip_rank_index + 1);
                cip[i].resize(cip_rank_index + 1, 0);
                cip[i][cip_rank_index] = ranks[i] as i64;
            }
        }
        num_its += 1;
    }
    ranks
}

/// RDKit legacy CIP ranks (`Chirality::assignAtomCIPRanks`).
pub fn cip_ranks(mol: &Molecule) -> Vec<u32> {
    let invars = build_cip_invariants(mol);
    iterate_cip_ranks(mol, &invars, false)
}

// ---------------------------------------------------------------------------
// Perception state
// ---------------------------------------------------------------------------

struct State<'a> {
    mol: &'a Molecule,
    ri: &'a RingInfo,
    tags: Vec<ChiralTag>,
    cip_code: Vec<Option<char>>,
    dirs: Vec<Dir>,
    stereo: Vec<St>,
    stereo_atoms: Vec<Option<(usize, usize)>>,
    /// Double-bond stereo already known (instead of being derived from bond
    /// directions): used when the molecule's stereo was assigned earlier.
    preset: Option<Vec<St>>,
}

/// `Atom::getPerturbationOrder` parity: number of swaps (mod 2 is what
/// matters) to turn the atom's bond order into `probe`.
fn perturbation_order(mol: &Molecule, atom: usize, probe: &[usize]) -> usize {
    let reference: Vec<usize> = mol.atom_bonds(atom).iter().map(|&(_, b)| b).collect();
    let mut p: Vec<usize> = probe.to_vec();
    let mut swaps = 0;
    for i in 0..reference.len().min(p.len()) {
        if p[i] != reference[i]
            && let Some(j) = (i + 1..p.len()).find(|&j| p[j] == reference[i])
        {
            p.swap(i, j);
            swaps += 1;
        }
    }
    swaps
}

impl<'a> State<'a> {
    /// `isAtomPotentialChiralCenter`. Returns (legalCenter, hasDupes, nbrs).
    fn is_potential_chiral_center(&self, atom: usize, ranks: Option<&[u32]>) -> (bool, bool, Vec<(u32, usize)>) {
        let mol = self.mol;
        let mut legal = true;
        let mut dupes = false;
        let mut nbrs = Vec::new();
        let nz = nonzero_degree(mol, atom);
        let nh = hidden_hs(mol, atom);
        let tnz = nz + nh;
        let anum = mol.atom(atom).atomic_num;
        if tnz > 4 {
            legal = false;
        } else {
            if tnz < 3 || (nz < 3 && anum != 15 && anum != 33) {
                // less than three neighbors is never stereogenic unless it is
                // a phosphine/arsine with implicit H
                legal = false;
            } else if nz == 3 {
                if nh == 1 {
                    let protium = mol.neighbors(atom).any(|x| mol.atom(x).atomic_num == 1 && mol.atom(x).isotope == 0);
                    if protium {
                        legal = false;
                    }
                } else {
                    legal = false;
                    if anum == 7 {
                        let conj = mol.atom_bonds(atom).iter().any(|&(_, b)| mol.bond(b).is_conjugated);
                        if mol.atom(atom).hybridization == crate::core::Hybridization::SP3
                            && !conj
                            && (self.ri.is_atom_in_ring_of_size(atom, 3) || is_atom_bridgehead(mol, self.ri, atom))
                        {
                            legal = true;
                        }
                    } else if anum == 15 || anum == 33 {
                        legal = true;
                    } else if anum == 16 || anum == 34 {
                        let ev = mol.explicit_valence(atom);
                        if ev == 4 || (ev == 3 && mol.atom(atom).formal_charge == 1) {
                            legal = true;
                        }
                    }
                }
            }
            if legal && let Some(ranks) = ranks {
                // ranks seen so far (at most 4 neighbors here: tnz <= 4)
                let mut seen = [0u32; 4];
                let mut n_seen = 0;
                for &(other, b) in mol.atom_bonds(atom) {
                    nbrs.push((ranks[other], b));
                    if !bond_affects_atom_chirality(mol, b, atom) {
                        continue;
                    }
                    let r = ranks[other];
                    if seen[..n_seen].contains(&r) {
                        dupes = true;
                        break;
                    }
                    seen[n_seen] = r;
                    n_seen += 1;
                }
            }
        }
        (legal, dupes, nbrs)
    }

    /// `assignAtomChiralCodes` with `flagPossibleStereoCenters = true`.
    /// Returns (unassigned atoms remain, any assigned).
    fn assign_atom_chiral_codes(&mut self, ranks: &mut Option<Vec<u32>>) -> (bool, bool) {
        let mol = self.mol;
        let mut changed = false;
        let mut unassigned: i64 = 0;
        for atom in 0..mol.num_atoms() {
            let mut tag = self.tags[atom];
            if self.cip_code[atom].is_some() {
                continue;
            }
            if ranks.is_none() {
                *ranks = Some(cip_ranks(mol));
            }
            let (legal, dupes, mut nbrs) = self.is_potential_chiral_center(atom, ranks.as_deref());
            if legal {
                unassigned += 1;
            }
            if legal && !dupes && tag != ChiralTag::Unspecified {
                changed = true;
                unassigned -= 1;
                nbrs.sort_by_key(|x| x.0);
                let idx: Vec<usize> = nbrs.iter().map(|x| x.1).collect();
                let mut swaps = perturbation_order(mol, atom, &idx);
                if idx.len() == 3 && hidden_hs(mol, atom) == 1 {
                    swaps += 1;
                }
                if swaps % 2 == 1 {
                    tag = if tag == ChiralTag::Ccw { ChiralTag::Cw } else { ChiralTag::Ccw };
                }
                self.cip_code[atom] = Some(if tag == ChiralTag::Ccw { 'S' } else { 'R' });
            }
        }
        (unassigned > 0, changed)
    }

    /// `findAtomNeighborDirHelper`: (neighbor, dir) pairs seen from `atom`.
    fn neighbor_dirs(&self, atom: usize, ref_bond: usize, ranks: &[u32]) -> Vec<(usize, Dir)> {
        let mol = self.mol;
        let mut seen_dir = false;
        let mut out = Vec::new();
        for &(nbr, b) in mol.atom_bonds(atom) {
            if b == ref_bond {
                continue;
            }
            let mut dir = self.dirs[b];
            if is_directional(dir) {
                seen_dir = true;
                if atom != mol.bond(b).begin {
                    dir = flip(dir);
                }
            }
            out.push((nbr, dir));
        }
        if !seen_dir || (out.len() == 2 && ranks[out[0].0] == ranks[out[1].0]) {
            // no direction, or two identical substituents
            out.clear();
        } else if !is_directional(out[0].1) {
            if out.len() > 1 {
                out[0].1 = flip(out[1].1);
            }
        } else if out.len() > 1 && !is_directional(out[1].1) {
            out[1].1 = flip(out[0].1);
        }
        out
    }

    /// `assignBondStereoCodes`. Returns (unassigned bonds remain, any assigned).
    fn assign_bond_stereo_codes(&mut self, ranks: &mut Option<Vec<u32>>) -> (bool, bool) {
        let mol = self.mol;
        let mut assigned = false;
        let mut unassigned: i64 = 0;
        let mut to_clear = vec![false; mol.num_bonds()];
        for b in 0..mol.num_bonds() {
            if mol.bond(b).order != BondOrder::Double || self.stereo[b] != St::None {
                continue;
            }
            if ranks.is_none() {
                *ranks = Some(cip_ranks(mol));
            }
            let rk = ranks.as_deref().unwrap();
            self.stereo_atoms[b] = None;
            if !should_detect_double_bond_stereo(self.ri, b) {
                continue;
            }
            let (beg, end) = (mol.bond(b).begin, mol.bond(b).end);
            if !matches!(mol.degree(beg), 2 | 3) || !matches!(mol.degree(end), 2 | 3) {
                continue;
            }
            unassigned += 1;
            if let Some(preset) = &self.preset {
                if matches!(preset[b], St::E | St::Z) && mol.bond(b).stereo_atoms.is_some() {
                    self.stereo[b] = preset[b];
                    self.stereo_atoms[b] = mol.bond(b).stereo_atoms;
                    assigned = true;
                    unassigned -= 1;
                }
                continue;
            }
            let bn = self.neighbor_dirs(beg, b, rk);
            let en = self.neighbor_dirs(end, b, rk);
            if bn.is_empty() || en.is_empty() {
                continue;
            }
            let pick = |v: &[(usize, Dir)]| if v.len() == 1 || rk[v[0].0] > rk[v[1].0] { v[0] } else { v[1] };
            let (beg_nbr, beg_dir) = pick(&bn);
            let (end_nbr, end_dir) = pick(&en);
            let conf_beg = bn.len() == 2 && bn[0].1 == bn[1].1;
            let conf_end = en.len() == 2 && en[0].1 == en[1].1;
            if conf_beg || conf_end {
                self.stereo[b] = St::None;
                assigned = true;
                if conf_beg {
                    to_clear[mol.bond_between(bn[0].0, beg).unwrap()] = true;
                    to_clear[mol.bond_between(bn[1].0, beg).unwrap()] = true;
                }
                if conf_end {
                    to_clear[mol.bond_between(en[0].0, end).unwrap()] = true;
                    to_clear[mol.bond_between(en[1].0, end).unwrap()] = true;
                }
            } else {
                self.stereo_atoms[b] = Some((beg_nbr, end_nbr));
                self.stereo[b] = if beg_dir == end_dir { St::Z } else { St::E };
                assigned = true;
            }
            unassigned -= 1;
        }
        for (b, &c) in to_clear.iter().enumerate() {
            if c {
                self.dirs[b] = Dir::None;
            }
        }
        (unassigned > 0, assigned)
    }

    /// `rerankAtoms`.
    fn rerank_atoms(&self, ranks: &mut [u32]) {
        let mol = self.mol;
        let n = mol.num_atoms();
        let mut factor: i64 = 100;
        while factor < n as i64 {
            factor *= 10;
        }
        let mut invars = vec![0i64; n];
        for i in 0..n {
            invars[i] = ranks[i] as i64 * factor;
            match self.cip_code[i] {
                Some('S') => invars[i] += 10,
                Some('R') => invars[i] += 20,
                _ => {}
            }
            for &(_, b) in mol.atom_bonds(i) {
                if mol.bond(b).order == BondOrder::Double {
                    match self.stereo[b] {
                        St::E => invars[i] += 1,
                        St::Z => invars[i] += 2,
                        _ => {}
                    }
                }
            }
        }
        let new = iterate_cip_ranks(mol, &invars, true);
        ranks.copy_from_slice(&new);
    }

    /// `atomIsCandidateForRingStereochem`.
    fn is_ring_stereo_candidate(&self, atom: usize, ranks: &[u32]) -> bool {
        let mol = self.mol;
        let ri = self.ri;
        if ri.num_atom_rings(atom) == 0 {
            return false;
        }
        if mol.atom(atom).atomic_num == 7
            && mol.total_degree(atom) == 3
            && !ri.is_atom_in_ring_of_size(atom, 3)
            && !is_atom_bridgehead(mol, ri, atom)
        {
            return false;
        }
        let mut non_ring = Vec::new();
        let mut ring_nbrs = 0usize;
        let mut ring_ranks = std::collections::BTreeSet::new();
        for &(nbr, b) in mol.atom_bonds(atom) {
            if ri.num_bond_rings(b) == 0 {
                non_ring.push(nbr);
            } else {
                ring_nbrs += 1;
                ring_ranks.insert(ranks[nbr]);
            }
        }
        match non_ring.len() {
            2 => ranks[non_ring[0]] != ranks[non_ring[1]] && ring_nbrs != ring_ranks.len(),
            1 => ring_nbrs > ring_ranks.len(),
            0 => (ring_nbrs == 4 && ring_ranks.len() == 3) || (ring_nbrs == 3 && ring_ranks.len() == 2),
            _ => false,
        }
    }

    /// `findChiralAtomSpecialCases`. Returns (possibleSpecialCases,
    /// _ringStereoAtoms per atom).
    fn find_chiral_atom_special_cases(&self, ranks: &[u32]) -> (Vec<bool>, Vec<Option<Vec<i64>>>) {
        let mol = self.mol;
        let ri = self.ri;
        let n = mol.num_atoms();
        let mut possible = vec![false; n];
        let mut rsa: Vec<Option<Vec<i64>>> = vec![None; n];
        let mut seen = vec![false; n];
        let mut used = vec![false; n];
        let mut bonds_seen = vec![false; mol.num_bonds()];
        let cand: Vec<bool> = (0..n).map(|a| self.is_ring_stereo_candidate(a, ranks)).collect();

        for atom in 0..n {
            if seen[atom] {
                continue;
            }
            if self.tags[atom] == ChiralTag::Unspecified
                || self.cip_code[atom].is_some()
                || ri.num_atom_rings(atom) == 0
                || !cand[atom]
            {
                continue;
            }
            let mut next: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
            for &(o, b) in mol.atom_bonds(atom) {
                if !bonds_seen[b] {
                    bonds_seen[b] = true;
                    if ri.num_bond_rings(b) > 0 && !seen[o] {
                        next.push_back(o);
                        used[o] = true;
                    }
                }
            }
            let mut ring_stereo_atoms: Vec<i64> = Vec::new();
            if !next.is_empty()
                && let Some(v) = &rsa[atom]
            {
                ring_stereo_atoms = v.clone();
            }
            while let Some(ratom) = next.pop_front() {
                seen[ratom] = true;
                if self.tags[ratom] != ChiralTag::Unspecified && self.cip_code[ratom].is_none() && cand[ratom] {
                    let same: i64 = if self.tags[ratom] == self.tags[atom] { 1 } else { -1 };
                    ring_stereo_atoms.push(same * (ratom as i64 + 1));
                    rsa[ratom].get_or_insert_with(Vec::new).push(same * (atom as i64 + 1));
                    possible[ratom] = true;
                    possible[atom] = true;
                }
                for &(o, b) in mol.atom_bonds(ratom) {
                    if !bonds_seen[b] {
                        bonds_seen[b] = true;
                        if ri.num_bond_rings(b) > 0 && !seen[o] && !used[o] {
                            next.push_back(o);
                            used[o] = true;
                        }
                    }
                }
            }
            if !ring_stereo_atoms.is_empty() {
                rsa[atom] = Some(ring_stereo_atoms.clone());
                let idx_of = |e: i64| if e < 0 { (-e - 1) as usize } else { (e - 1) as usize };
                for (k, &e) in ring_stereo_atoms.iter().enumerate() {
                    let ai = idx_of(e);
                    let mut l = rsa[ai].clone().unwrap_or_default();
                    for &oe in &ring_stereo_atoms[k + 1..] {
                        let oi = idx_of(oe);
                        let diff = (e < 0) ^ (oe < 0);
                        l.push(if diff { -(oi as i64 + 1) } else { oi as i64 + 1 });
                        let mut ol = rsa[oi].clone().unwrap_or_default();
                        ol.push(if diff { -(ai as i64 + 1) } else { ai as i64 + 1 });
                        rsa[oi] = Some(ol);
                    }
                    rsa[ai] = Some(l);
                }
            } else {
                possible[atom] = false;
            }
            seen[atom] = true;
        }
        (possible, rsa)
    }

    // -----------------------------------------------------------------------
    // detectBondStereochemistry (bond directions from the 3D conformer)
    // -----------------------------------------------------------------------

    fn is_bond_candidate_for_stereo(&self, b: usize) -> bool {
        let mol = self.mol;
        let bond = mol.bond(b);
        bond.order == BondOrder::Double
            && self.stereo[b] != St::Any
            && mol.degree(bond.begin) > 1
            && mol.degree(bond.end) > 1
            && should_detect_double_bond_stereo(self.ri, b)
    }

    /// `Chirality::detail::setStereoForBond` (used for STEREOANY).
    fn set_stereo_for_bond(&mut self, b: usize, st: St) {
        let mol = self.mol;
        let bond = mol.bond(b);
        let (mut beg, mut end) = (bond.begin, bond.end);
        if beg > end {
            std::mem::swap(&mut beg, &mut end);
        }
        if mol.degree(beg) > 1 && mol.degree(end) > 1 {
            let mut bc = mol.num_atoms();
            for n in mol.neighbors(beg) {
                if n != end {
                    bc = bc.min(n);
                }
            }
            let mut ec = 0;
            for n in mol.neighbors(end) {
                if n != beg {
                    ec = ec.max(n);
                }
            }
            if beg != bond.begin {
                std::mem::swap(&mut bc, &mut ec);
            }
            self.stereo_atoms[b] = Some((bc, ec));
            self.stereo[b] = st;
        }
    }

    fn set_bond_dir_relative_to_atom(&mut self, b: usize, atom: usize, mut dir: Dir, mut reverse: bool) {
        if self.mol.bond(b).begin != atom {
            reverse = !reverse;
        }
        if reverse {
            dir = flip(dir);
        }
        self.dirs[b] = dir;
    }

    /// `controllingBondFromAtom`: (bond, obond).
    fn controlling_bond_from_atom(
        &self,
        needs_dir: &[bool],
        single_bond_counts: &[u32],
        dbl: usize,
        atom: usize,
    ) -> (Option<usize>, Option<usize>) {
        let mol = self.mol;
        let mut bond: Option<usize> = None;
        let mut obond: Option<usize> = None;
        for &(_, t) in mol.atom_bonds(atom) {
            if t == dbl {
                continue;
            }
            let o = mol.bond(t).order;
            if matches!(o, BondOrder::Single | BondOrder::Aromatic) {
                match bond {
                    None => bond = Some(t),
                    Some(cur) => {
                        if needs_dir[t] {
                            if single_bond_counts[t] > single_bond_counts[cur] {
                                obond = Some(cur);
                                bond = Some(t);
                            } else {
                                obond = Some(t);
                            }
                        } else {
                            obond = Some(cur);
                            bond = Some(t);
                        }
                    }
                }
            }
        }
        (bond, obond)
    }

    /// `updateDoubleBondNeighbors` (with a conformer).
    fn update_double_bond_neighbors(
        &mut self,
        dbl: usize,
        needs_dir: &mut [bool],
        single_bond_counts: &[u32],
        single_bond_nbrs: &[Vec<usize>],
    ) {
        if !needs_dir[dbl] {
            return;
        }
        needs_dir[dbl] = false;
        let mol = self.mol;
        let atom1 = mol.bond(dbl).begin;
        let atom2 = mol.bond(dbl).end;
        let (b1, ob1) = self.controlling_bond_from_atom(needs_dir, single_bond_counts, dbl, atom1);
        let Some(mut bond1) = b1 else { return };
        let mut obond1 = ob1;
        let (b2, ob2) = self.controlling_bond_from_atom(needs_dir, single_bond_counts, dbl, atom2);
        let Some(mut bond2) = b2 else { return };
        let mut obond2 = ob2;

        let begin_p = mol.pos(atom1);
        let end_p = mol.pos(atom2);
        let mut bond1_p = mol.pos(mol.bond(bond1).other(atom1));
        let mut bond2_p = mol.pos(mol.bond(bond2).other(atom2));
        let mut linear = false;
        let p2 = sub(end_p, begin_p);
        if is_linear_arrangement(sub(bond1_p, begin_p), p2) {
            if let Some(o) = obond1 {
                obond1 = Some(bond1);
                bond1 = o;
                bond1_p = mol.pos(mol.bond(bond1).other(atom1));
                if is_linear_arrangement(sub(bond1_p, begin_p), p2) {
                    linear = true;
                }
            } else {
                linear = true;
            }
        }
        if !linear {
            let p2 = sub(begin_p, end_p);
            if is_linear_arrangement(sub(bond2_p, end_p), p2) {
                if let Some(o) = obond2 {
                    obond2 = Some(bond2);
                    bond2 = o;
                    bond2_p = mol.pos(mol.bond(bond2).other(atom2));
                    // NB: RDKit uses beginP here (sic)
                    if is_linear_arrangement(sub(bond2_p, begin_p), p2) {
                        linear = true;
                    }
                } else {
                    linear = true;
                }
            }
        }
        if linear {
            self.set_stereo_for_bond(dbl, St::Any);
            return;
        }
        let ang = dihedral_angle(bond1_p, begin_p, end_p, bond2_p);
        let same_torsion_dir = ang >= std::f64::consts::FRAC_PI_2;
        let mut reverse = same_torsion_dir;

        let mut followup = Vec::new();
        if needs_dir[bond1] {
            for &bi in &single_bond_nbrs[bond1] {
                if needs_dir[bi] {
                    followup.push(bi);
                }
            }
        }
        if needs_dir[bond2] {
            for &bi in &single_bond_nbrs[bond2] {
                if needs_dir[bi] {
                    followup.push(bi);
                }
            }
        }
        if !needs_dir[bond1] {
            if needs_dir[bond2] {
                if mol.bond(bond1).begin != atom1 {
                    reverse = !reverse;
                }
                let d = self.dirs[bond1];
                self.set_bond_dir_relative_to_atom(bond2, atom2, d, reverse);
            }
        } else if !needs_dir[bond2] {
            if mol.bond(bond2).begin != atom2 {
                reverse = !reverse;
            }
            let d = self.dirs[bond2];
            self.set_bond_dir_relative_to_atom(bond1, atom1, d, reverse);
        } else {
            self.set_bond_dir_relative_to_atom(bond1, atom1, Dir::EndDownRight, false);
            self.set_bond_dir_relative_to_atom(bond2, atom2, Dir::EndDownRight, reverse);
        }
        needs_dir[bond1] = false;
        needs_dir[bond2] = false;
        if let Some(o) = obond1
            && needs_dir[o]
        {
            let d = self.dirs[bond1];
            let r = mol.bond(bond1).begin == atom1;
            self.set_bond_dir_relative_to_atom(o, atom1, d, r);
            needs_dir[o] = false;
        }
        if let Some(o) = obond2
            && needs_dir[o]
        {
            let d = self.dirs[bond2];
            let r = mol.bond(bond2).begin == atom2;
            self.set_bond_dir_relative_to_atom(o, atom2, d, r);
            needs_dir[o] = false;
        }
        for d in followup {
            self.update_double_bond_neighbors(d, needs_dir, single_bond_counts, single_bond_nbrs);
        }
    }

    /// `setDoubleBondNeighborDirections` (with a conformer).
    fn set_double_bond_neighbor_directions(&mut self) {
        let mol = self.mol;
        let nb = mol.num_bonds();
        let mut single_bond_counts = vec![0u32; nb];
        let mut in_play = Vec::new();
        let mut dbl_bond_nbrs: Vec<Vec<usize>> = vec![Vec::new(); nb];
        let mut single_bond_nbrs: Vec<Vec<usize>> = vec![Vec::new(); nb];
        let mut needs_dir = vec![false; nb];
        for b in 0..nb {
            if !self.is_bond_candidate_for_stereo(b) {
                continue;
            }
            for atom in [mol.bond(b).begin, mol.bond(b).end] {
                for &(_, nbond) in mol.atom_bonds(atom) {
                    if matches!(mol.bond(nbond).order, BondOrder::Single | BondOrder::Aromatic) {
                        single_bond_counts[nbond] += 1;
                        needs_dir[b] = true;
                        // all directions are NONE / ENDUPRIGHT / ENDDOWNRIGHT here
                        needs_dir[nbond] = true;
                        dbl_bond_nbrs[b].push(nbond);
                        if !single_bond_nbrs[nbond].contains(&b) {
                            single_bond_nbrs[nbond].push(b);
                        }
                    }
                }
            }
            in_play.push(b);
        }
        if in_play.is_empty() {
            return;
        }
        let mut ordered: Vec<(usize, usize)> = in_play
            .iter()
            .map(|&b| {
                // RDKit sums the neighbor bond *indices* here
                let mut c: usize = dbl_bond_nbrs[b].iter().sum();
                if self.ri.num_bond_rings(b) == 0 {
                    c *= 10;
                }
                (c, b)
            })
            .collect();
        ordered.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for (_, b) in ordered {
            self.update_double_bond_neighbors(b, &mut needs_dir, &single_bond_counts, &single_bond_nbrs);
        }
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// `isLinearArrangement`.
fn is_linear_arrangement(v1: [f64; 3], v2: [f64; 3]) -> bool {
    let lsq = dot(v1, v1) * dot(v2, v2);
    if lsq < 1.0e-6 {
        return true;
    }
    let cos178 = -0.999388;
    dot(v1, v2) < cos178 * lsq.sqrt()
}

/// `RDGeom::computeDihedralAngle` (unsigned, in [0, π]).
fn dihedral_angle(p1: [f64; 3], p2: [f64; 3], p3: [f64; 3], p4: [f64; 3]) -> f64 {
    let beg_end = sub(p3, p2);
    let crs1 = cross(sub(p1, p2), beg_end);
    let crs2 = cross(sub(p4, p3), beg_end);
    let lsq = dot(crs1, crs1) * dot(crs2, crs2);
    let d = dot(crs1, crs2) / lsq.sqrt();
    if d <= -1.0 {
        std::f64::consts::PI
    } else if d >= 1.0 {
        0.0
    } else {
        d.acos()
    }
}

/// `MolOps::assignChiralTypesFrom3D` (tetrahedral part,
/// `replaceExistingTags = true`).
fn chiral_tags_from_3d(mol: &Molecule) -> Vec<ChiralTag> {
    const ZERO_VOLUME_TOL: f64 = 0.1;
    let mut tags = vec![ChiralTag::Unspecified; mol.num_atoms()];
    for (atom, tag) in tags.iter_mut().enumerate() {
        let nz = nonzero_degree(mol, atom);
        let tnz = nz + hidden_hs(mol, atom);
        if nz < 3 || tnz > 6 || tnz > 4 {
            continue;
        }
        let anum = mol.atom(atom).atomic_num;
        if anum != 16 && anum != 34 && tnz != 4 {
            continue;
        }
        let p0 = mol.pos(atom);
        let nbrs: Vec<[f64; 3]> = mol
            .atom_bonds(atom)
            .iter()
            .filter(|&&(_, b)| bond_affects_atom_chirality(mol, b, atom))
            .map(|&(n, _)| sub(mol.pos(n), p0))
            .collect();
        let (v1, v2, v3) = (nbrs[0], nbrs[1], nbrs[2]);
        let mut vol = dot(v1, cross(v2, v3));
        if vol.abs() <= ZERO_VOLUME_TOL || vol.is_nan() {
            if nbrs.len() == 4 {
                vol = -dot(v1, cross(v2, nbrs[3]));
            } else {
                continue;
            }
        }
        *tag = if vol < -ZERO_VOLUME_TOL {
            ChiralTag::Cw
        } else if vol > ZERO_VOLUME_TOL {
            ChiralTag::Ccw
        } else {
            ChiralTag::Unspecified
        };
    }
    tags
}

/// Reproduces the stereo RDKit ends up with after reading a molblock with 3D
/// coordinates (legacy perception, `cleanIt = true`): keeps chiral tags only
/// on legitimate stereocenters (including RDKit's ring-stereo special cases),
/// recomputes the tag from the 3D coordinates, and sets double-bond stereo
/// only for legitimate stereo bonds, with `stereo_atoms` = RDKit's stereo
/// atoms (CIP-highest neighbour on each side, as RDKit's STEREOE/STEREOZ use)
/// and Cis/Trans relative to those atoms (Z → Cis, E → Trans).
///
/// The coordinates are always treated as 3D. Existing chiral tags and bond
/// stereo are replaced.
pub fn assign_stereo_legacy_from_3d(mol: &mut Molecule) {
    assign_stereo_legacy_from_3d_with_ranks(mol);
}

/// [`assign_stereo_legacy_from_3d`], also returning the `_CIPRank` values
/// RDKit leaves on the atoms (`None` when RDKit's perception never needed
/// to rank the atoms, i.e. there was nothing that could be stereogenic).
pub fn assign_stereo_legacy_from_3d_with_ranks(mol: &mut Molecule) -> Option<Vec<u32>> {
    let ri_owned;
    let ri: &RingInfo = match mol.ring_info() {
        Some(r) => r,
        None => {
            ri_owned = RingInfo::compute(mol);
            &ri_owned
        }
    };
    let n = mol.num_atoms();
    let nb = mol.num_bonds();
    let mut st = State {
        mol,
        ri,
        tags: chiral_tags_from_3d(mol),
        cip_code: vec![None; n],
        dirs: vec![Dir::None; nb],
        stereo: either_double_stereo(mol),
        stereo_atoms: vec![None; nb],
        preset: None,
    };

    // detectBondStereochemistry
    st.set_double_bond_neighbor_directions();

    let ranks = run_legacy_perception(&mut st);

    // cleanIt: drop chiral tags on atoms that are not stereocenters
    let final_ranks = ranks.clone().unwrap_or_default();
    let mut new_tags = st.tags.clone();
    if !final_ranks.is_empty() || n == 0 {
        let (possible, rsa) = st.find_chiral_atom_special_cases(&final_ranks);
        for atom in 0..n {
            if st.tags[atom] != ChiralTag::Unspecified
                && st.cip_code[atom].is_none()
                && (!possible[atom] || rsa[atom].is_none())
            {
                new_tags[atom] = ChiralTag::Unspecified;
            }
        }
    } else {
        // no ranking was needed: nothing could be a stereocenter
        new_tags.iter_mut().for_each(|t| *t = ChiralTag::Unspecified);
    }
    // cleanIt: STEREOANY on double bonds to degree-one atoms is cleared (any
    // STEREOANY is reported as no stereo anyway)
    let stereo = st.stereo.clone();
    let stereo_atoms = st.stereo_atoms.clone();

    for (atom, t) in new_tags.into_iter().enumerate() {
        mol.atom_mut(atom).chiral_tag = t;
    }
    for b in 0..nb {
        let bm = mol.bond_mut(b);
        match stereo[b] {
            St::E | St::Z => {
                bm.stereo = if stereo[b] == St::Z { BondStereo::Cis } else { BondStereo::Trans };
                bm.stereo_atoms = stereo_atoms[b];
            }
            St::None | St::Any => {
                bm.stereo = BondStereo::None;
                bm.stereo_atoms = None;
            }
        }
    }
    ranks
}

/// `_CIPRank` values RDKit's legacy `assignStereochemistry` leaves on the
/// atoms of a molecule whose stereo (chiral tags, double-bond stereo with
/// RDKit's CIP-highest stereo atoms) is already assigned, e.g. one parsed
/// from SMILES or from a 2D molblock; `None` when RDKit would not have ranked
/// the atoms.
pub fn legacy_cip_ranks_for_assigned_stereo(mol: &Molecule, ri: &RingInfo) -> Option<Vec<u32>> {
    let n = mol.num_atoms();
    let nb = mol.num_bonds();
    let preset: Vec<St> = mol
        .bonds()
        .iter()
        .map(|b| match (b.order, b.stereo) {
            (BondOrder::Double, BondStereo::Cis) => St::Z,
            (BondOrder::Double, BondStereo::Trans) => St::E,
            _ => St::None,
        })
        .collect();
    let mut st = State {
        mol,
        ri,
        tags: mol.atoms().iter().map(|a| a.chiral_tag).collect(),
        cip_code: vec![None; n],
        dirs: vec![Dir::None; nb],
        stereo: either_double_stereo(mol),
        stereo_atoms: vec![None; nb],
        preset: Some(preset),
    };
    run_legacy_perception(&mut st)
}

/// Initial bond stereo: `STEREOANY` on crossed ("either") double bonds (as
/// RDKit's molfile parser sets it), none elsewhere.
fn either_double_stereo(mol: &Molecule) -> Vec<St> {
    mol.bonds()
        .iter()
        .map(|b| if b.order == BondOrder::Double && b.dir == BondDir::Either { St::Any } else { St::None })
        .collect()
}

/// The ranking loop of `legacyStereoPerception(cleanIt = true,
/// flagPossibleStereoCenters = true)`; returns the final atom ranks, if the
/// atoms were ranked at all.
fn run_legacy_perception(st: &mut State) -> Option<Vec<u32>> {
    let mol = st.mol;
    let ri = st.ri;
    let n = mol.num_atoms();
    let nb = mol.num_bonds();
    let mut has_stereo_atoms = false;
    let mut has_potential_stereo_atoms = false;
    for atom in 0..n {
        if !has_stereo_atoms && st.tags[atom] != ChiralTag::Unspecified {
            has_stereo_atoms = true;
        } else if !has_potential_stereo_atoms {
            has_potential_stereo_atoms = st.is_potential_chiral_center(atom, None).0;
        }
    }
    let mut has_stereo_bonds = false;
    let mut has_potential_stereo_bonds = false;
    for b in 0..nb {
        let order = mol.bond(b).order;
        if matches!(order, BondOrder::Double | BondOrder::Aromatic) && !should_detect_double_bond_stereo(ri, b) {
            if st.stereo[b] != St::None {
                st.stereo[b] = St::None;
                st.stereo_atoms[b] = None;
            }
            continue;
        } else if order == BondOrder::Double && st.stereo[b] != St::Any {
            st.stereo[b] = St::None;
            st.stereo_atoms[b] = None;
        }
        if !has_stereo_bonds && order == BondOrder::Double {
            let bond = mol.bond(b);
            let specified = [bond.begin, bond.end]
                .iter()
                .any(|&a| mol.atom_bonds(a).iter().any(|&(_, nbd)| is_directional(st.dirs[nbd])))
                || st.preset.as_ref().is_some_and(|p| matches!(p[b], St::E | St::Z));
            if specified {
                has_stereo_bonds = true;
            }
            if !has_potential_stereo_bonds && !specified && should_detect_double_bond_stereo(ri, b) {
                has_potential_stereo_bonds = true;
            }
        }
    }

    let mut ranks: Option<Vec<u32>> = None;
    let mut keep_going = has_stereo_atoms || has_stereo_bonds;
    if !keep_going {
        keep_going = has_potential_stereo_atoms || has_potential_stereo_bonds;
    }
    while keep_going {
        let changed_atoms;
        let changed_bonds;
        if has_stereo_atoms || has_potential_stereo_atoms {
            (has_stereo_atoms, changed_atoms) = st.assign_atom_chiral_codes(&mut ranks);
        } else {
            changed_atoms = false;
        }
        if has_stereo_bonds || has_potential_stereo_bonds {
            (has_stereo_bonds, changed_bonds) = st.assign_bond_stereo_codes(&mut ranks);
        } else {
            changed_bonds = false;
        }
        keep_going = (has_stereo_atoms || has_stereo_bonds) && (changed_atoms || changed_bonds);
        if keep_going && let Some(r) = ranks.as_mut() {
            st.rerank_atoms(r);
        }
    }

    ranks
}
