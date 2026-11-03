//! RDKit canonical atom ranking of a whole molecule as a fragment
//! (`Canon::rankFragmentAtoms` with all atoms and bonds in play and the
//! default flags: break ties, include chirality, isotopes, atom maps and ring
//! stereo, no chiral presence).  Used by the canonical Kekulize port
//! ([`super::rdkit_kekulize`]).
//!
//! Ported from RDKit 2026.09.1 `Code/GraphMol/new_canon.h`,
//! `Code/GraphMol/new_canon.cpp` and `Code/RDGeneral/hanoiSort.h`,
//! Copyright (C) 2014-2026 Greg Landrum, NextMove Software and other RDKit
//! contributors; BSD 3-Clause license (see `LICENSE-RDKIT.txt`).
//!
//! Not modelled (treated as absent): stereo groups, atom-map numbers,
//! `_ringStereoAtoms` ring-stereo marks and atropisomer bond stereo.
//! Double-bond stereo is taken from [`Bond::stereo`]: `Cis`/`Trans` relative
//! to the CIP-ranked stereo atoms map to RDKit's `STEREOZ`/`STEREOE` (what
//! RDKit's legacy stereo perception assigns).

use crate::core::{Bond, BondOrder, BondStereo, ChiralTag, Molecule};
use crate::perception::rings::RingInfo;

/// RDKit `Bond::BondType` enum values.
fn rd_bond_type(b: &Bond) -> u32 {
    if b.is_aromatic {
        return 12;
    }
    match b.order {
        BondOrder::Single => 1,
        BondOrder::Double => 2,
        BondOrder::Triple => 3,
        BondOrder::Aromatic => 12,
        BondOrder::Dative => 17,
        BondOrder::Zero => 21,
    }
}

/// RDKit `Bond::BondStereo` enum values (STEREONONE = 0, STEREOZ = 2,
/// STEREOE = 3).
fn rd_bond_stereo(b: &Bond) -> u32 {
    match b.stereo {
        BondStereo::None => 0,
        BondStereo::Cis => 2,
        BondStereo::Trans => 3,
    }
}

#[derive(Clone, Debug)]
struct BondHolder {
    bond_type: u32,
    bond_stereo: u32,
    nbr_sym_class: u32,
    nbr_idx: usize,
}

impl BondHolder {
    /// `bondholder::compare` (div = 1).  E/Z labels compare directly, so
    /// `compareStereo` reduces to comparing the stereo values.
    fn compare(x: &BondHolder, y: &BondHolder) -> i32 {
        if x.bond_type != y.bond_type {
            return if x.bond_type < y.bond_type { -1 } else { 1 };
        }
        if x.bond_stereo != y.bond_stereo {
            return if x.bond_stereo < y.bond_stereo { -1 } else { 1 };
        }
        if x.nbr_sym_class != y.nbr_sym_class {
            return if x.nbr_sym_class < y.nbr_sym_class { -1 } else { 1 };
        }
        0
    }
    fn greater(x: &BondHolder, y: &BondHolder) -> bool {
        Self::compare(x, y) > 0
    }
}

#[derive(Clone, Debug, Default)]
struct CanonAtom {
    index: i32,
    degree: usize,
    total_num_hs: u32,
    nbr_ids: Vec<usize>,
    neighbor_num: Vec<i32>,
    revisited_neighbors: Vec<i32>,
    bonds: Vec<BondHolder>,
}

/// `countSwapsToInterconvert`.
fn count_swaps_to_interconvert(reference: &[usize], probe: &[usize]) -> u32 {
    let mut probe = probe.to_vec();
    let mut n = 0;
    for i in 0..reference.len() {
        if probe[i] != reference[i] {
            let j = (i..probe.len()).find(|&j| probe[j] == reference[i]).expect("probe element");
            probe.swap(i, j);
            n += 1;
        }
    }
    n
}

/// `updateAtomNeighborIndex`: refresh neighbor classes and re-sort
/// (descending) with an insertion sort.
fn update_atom_neighbor_index(classes: &[i32], bonds: &mut [BondHolder]) {
    for b in bonds.iter_mut() {
        b.nbr_sym_class = classes[b.nbr_idx] as u32;
    }
    for i in 1..bonds.len() {
        if !BondHolder::greater(&bonds[i], &bonds[i - 1]) {
            continue;
        }
        let value = bonds[i].clone();
        let mut j = i;
        loop {
            bonds[j] = bonds[j - 1].clone();
            j -= 1;
            if !(j > 0 && BondHolder::greater(&value, &bonds[j - 1])) {
                break;
            }
        }
        bonds[j] = value;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Functor {
    /// `AtomCompareFunctor` with `df_useNbrs = true`
    Atom,
    SpecialChirality,
    SpecialSymmetry,
}

struct Ranker<'a> {
    mol: &'a Molecule,
    ri: &'a RingInfo,
    atoms: Vec<CanonAtom>,
}

impl Ranker<'_> {
    fn classes(&self) -> Vec<i32> {
        self.atoms.iter().map(|a| a.index).collect()
    }

    fn refresh_bonds(&mut self, i: usize) {
        let classes = self.classes();
        update_atom_neighbor_index(&classes, &mut self.atoms[i].bonds);
    }

    /// `getChiralRank`.
    fn chiral_rank(&self, i: usize) -> u32 {
        let mut perm: Vec<usize> = Vec::new();
        for n in self.mol.neighbors(i) {
            let rnk = self.atoms[n].index as usize;
            if perm.contains(&rnk) {
                break;
            }
            perm.push(rnk);
        }
        let mut res = 0;
        if perm.len() == self.mol.degree(i) {
            let tag = self.mol.atom(i).chiral_tag;
            if matches!(tag, ChiralTag::Cw | ChiralTag::Ccw) {
                let mut sorted = perm.clone();
                sorted.sort_unstable();
                let nswaps = count_swaps_to_interconvert(&perm, &sorted);
                res = if tag == ChiralTag::Cw { 2 } else { 1 };
                if nswaps % 2 == 1 {
                    res = if res == 2 { 1 } else { 2 };
                }
            }
        }
        res
    }

    /// `AtomCompareFunctor::basecomp`.
    fn basecomp(&self, i: usize, j: usize) -> i32 {
        fn cmp<T: PartialOrd>(a: T, b: T) -> i32 {
            if a < b {
                -1
            } else if a > b {
                1
            } else {
                0
            }
        }
        let (ai, aj) = (&self.atoms[i], &self.atoms[j]);
        let c = cmp(ai.index, aj.index);
        if c != 0 {
            return c;
        }
        let c = cmp(ai.degree, aj.degree);
        if c != 0 {
            return c;
        }
        let (mi, mj) = (self.mol.atom(i), self.mol.atom(j));
        let c = cmp(mi.atomic_num, mj.atomic_num);
        if c != 0 {
            return c;
        }
        let c = cmp(mi.isotope, mj.isotope);
        if c != 0 {
            return c;
        }
        let c = cmp(ai.total_num_hs, aj.total_num_hs);
        if c != 0 {
            return c;
        }
        // RDKit stores the charge in an `unsigned int` before comparing, so
        // negative charges sort above positive ones
        let c = cmp(mi.formal_charge as i32 as u32, mj.formal_charge as i32 as u32);
        if c != 0 {
            return c;
        }
        // chirality (no stereo groups)
        let ti = mi.chiral_tag != ChiralTag::Unspecified;
        let tj = mj.chiral_tag != ChiralTag::Unspecified;
        let c = cmp(ti, tj);
        if c != 0 {
            return c;
        }
        if ti && tj {
            let c = cmp(self.chiral_rank(i), self.chiral_rank(j));
            if c != 0 {
                return c;
            }
        }
        // ring stereo neighbor code: no ring stereo atoms -> always 0
        0
    }

    fn compare_bond_lists(&self, i: usize, j: usize, check_size: bool) -> i32 {
        let (bi, bj) = (&self.atoms[i].bonds, &self.atoms[j].bonds);
        for (x, y) in bi.iter().zip(bj.iter()) {
            let c = BondHolder::compare(x, y);
            if c != 0 {
                return c;
            }
        }
        if check_size {
            if bi.len() < bj.len() {
                return -1;
            } else if bi.len() > bj.len() {
                return 1;
            }
        }
        0
    }

    /// `updateAtomNeighborNumSwaps`.
    fn neighbor_num_swaps(&self, atom_idx: usize) -> Vec<(u32, u32)> {
        let is_ring_atom = self.ri.num_atom_rings(atom_idx) > 0;
        let mut result = Vec::new();
        for nbr in &self.atoms[atom_idx].bonds {
            let nbr_idx = nbr.nbr_idx;
            let tag = self.mol.atom(nbr_idx).chiral_tag;
            if is_ring_atom && tag != ChiralTag::Unspecified {
                let na = &self.atoms[nbr_idx];
                let mut seen: Vec<i32> = Vec::new();
                let mut too_many = false;
                let mut reference = Vec::new();
                for &nn in &na.nbr_ids[..na.degree] {
                    reference.push(nn);
                    if nn != atom_idx {
                        let cls = self.atoms[nn].index;
                        if seen.contains(&cls) {
                            too_many = true;
                        } else {
                            seen.push(cls);
                        }
                    }
                }
                let mut probe = vec![atom_idx];
                for b in &na.bonds {
                    if b.nbr_idx != atom_idx {
                        probe.push(b.nbr_idx);
                    }
                }
                if too_many {
                    result.push((nbr.nbr_sym_class, 0));
                } else {
                    let odd = count_swaps_to_interconvert(&reference, &probe) % 2 == 1;
                    match tag {
                        ChiralTag::Cw => result.push((nbr.nbr_sym_class, if odd { 2 } else { 1 })),
                        ChiralTag::Ccw => result.push((nbr.nbr_sym_class, if odd { 1 } else { 2 })),
                        ChiralTag::Unspecified => {}
                    }
                }
            } else {
                result.push((nbr.nbr_sym_class, 0));
            }
        }
        result.sort_unstable();
        result
    }

    fn compare(&mut self, f: Functor, i: usize, j: usize) -> i32 {
        match f {
            Functor::Atom => {
                let v = self.basecomp(i, j);
                if v != 0 {
                    return v;
                }
                self.refresh_bonds(i);
                self.refresh_bonds(j);
                self.compare_bond_lists(i, j, true)
            }
            Functor::SpecialChirality => {
                self.refresh_bonds(i);
                self.refresh_bonds(j);
                let c = self.compare_bond_lists(i, j, false);
                if c != 0 {
                    return c;
                }
                let si = self.neighbor_num_swaps(i);
                let sj = self.neighbor_num_swaps(j);
                for (x, y) in si.iter().zip(sj.iter()) {
                    let c = x.1 as i32 - y.1 as i32;
                    if c != 0 {
                        return c;
                    }
                }
                0
            }
            Functor::SpecialSymmetry => {
                let (ai, aj) = (&self.atoms[i], &self.atoms[j]);
                if ai.neighbor_num != aj.neighbor_num {
                    return if ai.neighbor_num < aj.neighbor_num { -1 } else { 1 };
                }
                if ai.revisited_neighbors != aj.revisited_neighbors {
                    return if ai.revisited_neighbors < aj.revisited_neighbors { -1 } else { 1 };
                }
                self.refresh_bonds(i);
                self.refresh_bonds(j);
                self.compare_bond_lists(i, j, true)
            }
        }
    }

    /// `RDKit::detail::hanoi`: merge sort of `mem[base..base+nel]` using
    /// `mem[temp..temp+nel]` as scratch; returns `true` if the result is in
    /// the scratch area.
    #[allow(clippy::too_many_arguments)]
    fn hanoi(
        &mut self,
        f: Functor,
        mem: &mut [usize],
        base: usize,
        nel: usize,
        temp: usize,
        count: &mut [i32],
        changed: &[i32],
    ) -> bool {
        if nel == 1 {
            count[mem[base]] = 1;
            return false;
        } else if nel == 2 {
            let n1 = mem[base];
            let n2 = mem[base + 1];
            let stat = if changed[n1] != 0 || changed[n2] != 0 { self.compare(f, n1, n2) } else { 0 };
            if stat == 0 {
                count[n1] = 2;
                count[n2] = 0;
            } else if stat < 0 {
                count[n1] = 1;
                count[n2] = 1;
            } else {
                count[n1] = 1;
                count[n2] = 1;
                mem[base] = n2;
                mem[base + 1] = n1;
            }
            return false;
        }
        let mut n1 = nel / 2;
        let mut n2 = nel - n1;
        let b1 = base;
        let t1 = temp;
        let b2 = base + n1;
        let t2 = temp + n1;
        let (mut s1, mut s2, mut ptr, result);
        if self.hanoi(f, mem, b1, n1, t1, count, changed) {
            s2 = if self.hanoi(f, mem, b2, n2, t2, count, changed) { t2 } else { b2 };
            result = false;
            ptr = base;
            s1 = t1;
        } else {
            s2 = if self.hanoi(f, mem, b2, n2, t2, count, changed) { t2 } else { b2 };
            result = true;
            ptr = temp;
            s1 = b1;
        }
        // memmove with dst <= src (or disjoint): forward copy is safe
        fn mv(mem: &mut [usize], dst: usize, src: usize, len: usize) {
            if dst != src {
                for k in 0..len {
                    mem[dst + k] = mem[src + k];
                }
            }
        }
        loop {
            let (a, b) = (mem[s1], mem[s2]);
            let stat = if changed[a] != 0 || changed[b] != 0 { self.compare(f, a, b) } else { 0 };
            let len1 = count[a] as usize;
            let len2 = count[b] as usize;
            if stat == 0 {
                count[a] = (len1 + len2) as i32;
                count[b] = 0;
                mv(mem, ptr, s1, len1);
                ptr += len1;
                n1 -= len1;
                if n1 == 0 {
                    if ptr != s2 {
                        mv(mem, ptr, s2, n2);
                    }
                    return result;
                }
                s1 += len1;
                mv(mem, ptr, s2, len2);
                ptr += len2;
                n2 -= len2;
                if n2 == 0 {
                    mv(mem, ptr, s1, n1);
                    return result;
                }
                s2 += len2;
            } else if stat < 0 && len1 > 0 {
                mv(mem, ptr, s1, len1);
                ptr += len1;
                n1 -= len1;
                if n1 == 0 {
                    if ptr != s2 {
                        mv(mem, ptr, s2, n2);
                    }
                    return result;
                }
                s1 += len1;
            } else if stat > 0 && len2 > 0 {
                mv(mem, ptr, s2, len2);
                ptr += len2;
                n2 -= len2;
                if n2 == 0 {
                    mv(mem, ptr, s1, n1);
                    return result;
                }
                s2 += len2;
            } else {
                unreachable!("hanoi: empty partition");
            }
        }
    }
}

struct Partitions {
    order: Vec<usize>,
    count: Vec<i32>,
    next: Vec<i32>,
    changed: Vec<i32>,
    touched: Vec<bool>,
    activeset: i32,
}

impl Partitions {
    /// `ActivatePartitions`.
    fn activate(&mut self) {
        let n = self.order.len();
        self.activeset = -1;
        self.next.iter_mut().for_each(|x| *x = -2);
        let mut i = 0;
        while i < n {
            let j = self.order[i];
            if self.count[j] > 1 {
                self.next[j] = self.activeset;
                self.activeset = j as i32;
                i += self.count[j] as usize;
            } else {
                i += 1;
            }
        }
        self.changed.iter_mut().for_each(|x| *x = 1);
    }
}

/// `RefinePartitions` (mode = true).
fn refine(r: &mut Ranker, f: Functor, p: &mut Partitions) {
    let n = r.atoms.len();
    let mut symclass: i32 = 0;
    while p.activeset != -1 {
        let partition = p.activeset as usize;
        p.activeset = p.next[partition];
        p.next[partition] = -2;
        let len = p.count[partition] as usize;
        let offset = r.atoms[partition].index as usize;

        let mut mem: Vec<usize> = Vec::with_capacity(2 * len);
        mem.extend_from_slice(&p.order[offset..offset + len]);
        mem.resize(2 * len, 0);
        let in_temp = r.hanoi(f, &mut mem, 0, len, len, &mut p.count, &p.changed);
        let src = if in_temp { len } else { 0 };
        p.order[offset..offset + len].copy_from_slice(&mem[src..src + len]);

        for k in 0..len {
            p.changed[p.order[offset + k]] = 0;
        }
        let first = p.order[offset];
        let start_i = p.count[first] as usize;
        for i in start_i..len {
            let index = p.order[offset + i];
            if p.count[index] != 0 {
                symclass = (offset + i) as i32;
            }
            r.atoms[index].index = symclass;
            for j in 0..r.atoms[index].degree {
                let nb = r.atoms[index].nbr_ids[j];
                p.changed[nb] = 1;
            }
        }
        for i in start_i..len {
            let index = p.order[offset + i];
            for j in 0..r.atoms[index].degree {
                let nb = r.atoms[index].nbr_ids[j];
                p.touched[r.atoms[nb].index as usize] = true;
            }
        }
        for ii in 0..n {
            if p.touched[ii] {
                let part = p.order[ii];
                if p.count[part] > 1 && p.next[part] == -2 {
                    p.next[part] = p.activeset;
                    p.activeset = part as i32;
                }
                p.touched[ii] = false;
            }
        }
    }
}

/// `BreakTies` (mode = true).
fn break_ties(r: &mut Ranker, f: Functor, p: &mut Partitions) {
    let n = r.atoms.len() as isize;
    let mut i: isize = 0;
    while i < n {
        let partition = p.order[i as usize];
        let old_part = r.atoms[partition].index;
        while p.count[partition] > 1 {
            let len = p.count[partition];
            let offset = (r.atoms[partition].index + len - 1) as usize;
            let index = p.order[offset];
            r.atoms[index].index = offset as i32;
            p.count[partition] = len - 1;
            p.count[index] = 1;
            if r.atoms[index].degree < 1 {
                continue;
            }
            for j in 0..r.atoms[index].degree {
                let nb = r.atoms[index].nbr_ids[j];
                p.touched[r.atoms[nb].index as usize] = true;
                p.changed[nb] = 1;
            }
            for ii in 0..n as usize {
                if p.touched[ii] {
                    let npart = p.order[ii];
                    if p.count[npart] > 1 && p.next[npart] == -2 {
                        p.next[npart] = p.activeset;
                        p.activeset = npart as i32;
                    }
                    p.touched[ii] = false;
                }
            }
            refine(r, f, p);
        }
        if r.atoms[partition].index != old_part {
            i -= 1;
        }
        i += 1;
    }
}

/// `compareRingAtomsConcerningNumNeighbors`.
fn compare_ring_atoms_concerning_num_neighbors(r: &mut Ranker) {
    let n = r.atoms.len();
    let in_ring = |a: usize| r.ri.num_atom_rings(a) >= 1;
    let ring_flags: Vec<bool> = (0..n).map(in_ring).collect();
    let mut visited = vec![false; n];
    let mut last_level = vec![false; n];
    let mut current_level = vec![false; n];
    let mut revisited = vec![0i32; n];
    for idx in 0..n {
        if !ring_flags[idx] {
            continue;
        }
        visited.iter_mut().for_each(|x| *x = false);
        last_level.iter_mut().for_each(|x| *x = false);
        let mut neighbors: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        neighbors.push_back(idx);
        let mut neighbor_num: Vec<i32> = Vec::new();
        let mut rev_out: Vec<i32> = Vec::new();
        while !neighbors.is_empty() {
            let mut num_level_nbrs = 0;
            let mut next_level: Vec<usize> = Vec::new();
            while let Some(nidx) = neighbors.pop_front() {
                if !ring_flags[nidx] {
                    continue;
                }
                last_level[nidx] = true;
                visited[nidx] = true;
                for j in 0..r.atoms[nidx].degree {
                    let iidx = r.atoms[nidx].nbr_ids[j];
                    if !visited[iidx] {
                        current_level[iidx] = true;
                        num_level_nbrs += 1;
                        visited[iidx] = true;
                        next_level.push(iidx);
                    }
                }
            }
            let mut modified: Vec<usize> = Vec::new();
            for &i in &next_level {
                for k in 0..r.atoms[i].degree {
                    let jidx = r.atoms[i].nbr_ids[k];
                    if current_level[jidx] || last_level[jidx] {
                        if revisited[jidx] == 0 {
                            modified.push(jidx);
                        }
                        revisited[jidx] += 1;
                    }
                }
            }
            last_level.iter_mut().for_each(|x| *x = false);
            for &i in &next_level {
                last_level[i] = true;
            }
            for &i in &next_level {
                current_level[i] = false;
            }
            let mut tmp: Vec<i32> = modified.iter().map(|&i| revisited[i]).collect();
            tmp.sort_unstable();
            tmp.push(-1);
            rev_out.extend_from_slice(&tmp);
            for &i in &modified {
                revisited[i] = 0;
            }
            neighbor_num.push(num_level_nbrs);
            neighbor_num.push(-1);
            neighbors.extend(next_level.iter().copied());
        }
        r.atoms[idx].neighbor_num = neighbor_num;
        r.atoms[idx].revisited_neighbors = rev_out;
    }
}

/// Canonical atom ranks (`Canon::rankFragmentAtoms` over the whole
/// molecule).  `ri` must be the molecule's ring info.
pub fn rank_fragment_atoms(mol: &Molecule, ri: &RingInfo) -> Vec<usize> {
    let n = mol.num_atoms();
    if n == 0 {
        return Vec::new();
    }
    // initFragmentCanonAtoms
    let mut atoms: Vec<CanonAtom> = (0..n)
        .map(|i| CanonAtom { index: i as i32, total_num_hs: mol.atom(i).total_hidden_hs(), ..Default::default() })
        .collect();
    for b in 0..mol.num_bonds() {
        let bond = mol.bond(b);
        let (s, e) = (bond.begin, bond.end);
        let bt = rd_bond_type(bond);
        let st = rd_bond_stereo(bond);
        atoms[s].nbr_ids.push(e);
        atoms[s].degree += 1;
        atoms[e].nbr_ids.push(s);
        atoms[e].degree += 1;
        atoms[s].bonds.push(BondHolder { bond_type: bt, bond_stereo: st, nbr_sym_class: 0, nbr_idx: e });
        atoms[e].bonds.push(BondHolder { bond_type: bt, bond_stereo: st, nbr_sym_class: 0, nbr_idx: s });
    }
    for a in atoms.iter_mut() {
        a.bonds.sort_by(|x, y| BondHolder::compare(y, x).cmp(&0));
    }
    let mut r = Ranker { mol, ri, atoms };

    // rankWithFunctor(ftor, breakTies = true, useSpecial = true,
    //                 useChirality = true, includeRingStereo = true)
    let mut p = Partitions {
        order: (0..n).collect(),
        count: vec![0; n],
        next: vec![0; n],
        changed: vec![1; n],
        touched: vec![false; n],
        activeset: -1,
    };
    for a in r.atoms.iter_mut() {
        a.index = 0;
    }
    p.count[0] = n as i32;
    p.activate();
    refine(&mut r, Functor::Atom, &mut p);
    let ties = p.count.contains(&0);
    if ties {
        p.activate();
        refine(&mut r, Functor::SpecialChirality, &mut p);
    }
    let mut ties = false;
    let mut sym_ring_atoms = 0usize;
    let mut ring_atoms = 0usize;
    let mut branching_ring_atom = false;
    for i in 0..n {
        let oi = p.order[i];
        let nr = ri.num_atom_rings(oi);
        if nr > 0 {
            if p.count[oi] > 2 {
                sym_ring_atoms += p.count[oi] as usize;
            }
            ring_atoms += 1;
            if nr > 1 && p.count[oi] > 1 {
                branching_ring_atom = true;
            }
        }
        if p.count[i] == 0 {
            ties = true;
        }
    }
    if ties && ring_atoms > 0 && (sym_ring_atoms as f32) / (ring_atoms as f32) > 0.5 && branching_ring_atom {
        compare_ring_atoms_concerning_num_neighbors(&mut r);
        p.activate();
        refine(&mut r, Functor::SpecialSymmetry, &mut p);
    }
    break_ties(&mut r, Functor::Atom, &mut p);

    let mut res = vec![0usize; n];
    for i in 0..n {
        let oi = p.order[i];
        res[oi] = r.atoms[oi].index as usize;
    }
    res
}
