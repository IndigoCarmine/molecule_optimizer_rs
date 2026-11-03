//! VF2 subgraph-isomorphism search.
//!
//! Faithful port of RDKit `Code/GraphMol/Substruct/vf2.hpp` (Greg Landrum's
//! modification of vflib-2.0 by P. Foggia, including the "VF2 Plus"
//! neighbour iterator) and the driver logic of
//! `Code/GraphMol/Substruct/SubstructMatch.cpp` (`SubstructMatch`,
//! `RecursiveMatcher`, `MatchSubqueries`).  Candidate pairs are generated in
//! exactly the same order as RDKit, so the list of matches comes out in the
//! same order.
//!
//! Copyright (C) 2001-2026 Greg Landrum and other RDKit contributors,
//! BSD-3-Clause (see `LICENSE-RDKIT.txt`).

use super::parser::QueryMol;
use super::query::Target;

const NULL_NODE: u32 = u32::MAX;

/// Candidate pair with the optional VF2-Plus neighbour iterator.
struct Pair {
    n1: u32,
    n2: u32,
    hasiter: bool,
    /// target atom whose adjacency list is iterated
    nbr_atom: usize,
    /// current position in that adjacency list
    nbr_pos: usize,
}

impl Pair {
    fn new() -> Pair {
        Pair {
            n1: NULL_NODE,
            n2: NULL_NODE,
            hasiter: false,
            nbr_atom: 0,
            nbr_pos: 0,
        }
    }
}

/// `VF2SubState` (non-shared, mutated in place with backtracking).
struct State<'a> {
    q: &'a QueryMol,
    t: &'a Target<'a>,
    rec: &'a [Vec<bool>],
    n1: u32,
    n2: u32,
    core_len: u32,
    t1_len: u32,
    t2_len: u32,
    core_1: Vec<u32>,
    core_2: Vec<u32>,
    term_1: Vec<u32>,
    term_2: Vec<u32>,
    /// cache of the (pure) atom compatibility test: 0 unknown, 1 yes, 2 no
    vc_cache: Vec<u8>,
}

/// What to do with complete matches.
enum Sink<'s> {
    /// collect all matches (query index -> target index)
    All(&'s mut Vec<Vec<usize>>),
    /// stop at the first match
    First(&'s mut bool),
}

impl<'a> State<'a> {
    fn new(q: &'a QueryMol, t: &'a Target<'a>, rec: &'a [Vec<bool>]) -> State<'a> {
        let n1 = q.atoms.len();
        let n2 = t.mol.num_atoms();
        State {
            q,
            t,
            rec,
            n1: n1 as u32,
            n2: n2 as u32,
            core_len: 0,
            t1_len: 0,
            t2_len: 0,
            core_1: vec![NULL_NODE; n1],
            core_2: vec![NULL_NODE; n2],
            term_1: vec![0; n1],
            term_2: vec![0; n2],
            vc_cache: vec![0; n1 * n2],
        }
    }

    fn is_goal(&self) -> bool {
        self.core_len == self.n1
    }
    fn is_dead(&self) -> bool {
        self.n1 > self.n2 || self.t1_len > self.t2_len
    }

    fn next_pair(&self, pair: &mut Pair) -> bool {
        if pair.n1 == NULL_NODE {
            pair.n1 = 0;
        }
        if pair.n2 == NULL_NODE {
            pair.n2 = 0;
        } else {
            pair.n2 += 1;
        }
        let terminal = self.t1_len > self.core_len && self.t2_len > self.core_len;
        if terminal {
            while pair.n1 < self.n1
                && (self.core_1[pair.n1 as usize] != NULL_NODE
                    || self.term_1[pair.n1 as usize] == 0)
            {
                pair.n1 += 1;
                pair.n2 = 0;
            }
            if !pair.hasiter {
                // VF2 Plus: candidates are neighbours of the image of the
                // first already-mapped query neighbour of n1
                let mapped = self.q.adj[pair.n1 as usize]
                    .iter()
                    .map(|&(nb, _)| nb)
                    .find(|&nb| self.core_1[nb] != NULL_NODE)
                    .expect("terminal query node without mapped neighbour");
                pair.nbr_atom = self.core_1[mapped] as usize;
                pair.nbr_pos = 0;
                pair.hasiter = true;
            }
        } else {
            while pair.n1 < self.n1 && self.core_1[pair.n1 as usize] != NULL_NODE {
                pair.n1 += 1;
                pair.n2 = 0;
            }
        }

        if pair.hasiter {
            let nbrs = self.t.mol.atom_bonds(pair.nbr_atom);
            while pair.nbr_pos < nbrs.len() && self.core_2[nbrs[pair.nbr_pos].0] != NULL_NODE {
                pair.nbr_pos += 1;
            }
            if pair.nbr_pos < nbrs.len() {
                pair.n2 = nbrs[pair.nbr_pos].0 as u32;
                pair.nbr_pos += 1;
            } else {
                pair.n2 = self.n2;
            }
        } else if terminal {
            while pair.n2 < self.n2
                && (self.core_2[pair.n2 as usize] != NULL_NODE
                    || self.term_2[pair.n2 as usize] == 0)
            {
                pair.n2 += 1;
            }
        } else {
            while pair.n2 < self.n2 && self.core_2[pair.n2 as usize] != NULL_NODE {
                pair.n2 += 1;
            }
        }
        pair.n1 < self.n1 && pair.n2 < self.n2
    }

    fn atom_compat(&mut self, node1: usize, node2: usize) -> bool {
        let k = node1 * self.n2 as usize + node2;
        match self.vc_cache[k] {
            1 => true,
            2 => false,
            _ => {
                let r = self.q.atoms[node1].expr.matches(self.t, node2, self.rec);
                self.vc_cache[k] = if r { 1 } else { 2 };
                r
            }
        }
    }

    fn is_feasible_pair(&mut self, node1: u32, node2: u32) -> bool {
        let (node1, node2) = (node1 as usize, node2 as usize);
        if self.q.adj[node1].len() > self.t.mol.degree(node2) {
            return false;
        }
        if !self.atom_compat(node1, node2) {
            return false;
        }
        for &(other1, qbond) in &self.q.adj[node1] {
            let other2 = self.core_1[other1];
            if other2 != NULL_NODE {
                match self.t.mol.bond_between(node2, other2 as usize) {
                    Some(tbond) if self.q.bonds[qbond].expr.matches(self.t, tbond) => {}
                    _ => return false,
                }
            }
        }
        true
    }

    fn add_pair(&mut self, node1: u32, node2: u32) {
        self.core_len += 1;
        let cl = self.core_len;
        let (node1, node2) = (node1 as usize, node2 as usize);
        if self.term_1[node1] == 0 {
            self.term_1[node1] = cl;
            self.t1_len += 1;
        }
        if self.term_2[node2] == 0 {
            self.term_2[node2] = cl;
            self.t2_len += 1;
        }
        self.core_1[node1] = node2 as u32;
        self.core_2[node2] = node1 as u32;
        for &(other, _) in &self.q.adj[node1] {
            if self.term_1[other] == 0 {
                self.term_1[other] = cl;
                self.t1_len += 1;
            }
        }
        for &(other, _) in self.t.mol.atom_bonds(node2) {
            if self.term_2[other] == 0 {
                self.term_2[other] = cl;
                self.t2_len += 1;
            }
        }
    }

    fn backtrack(&mut self, node1: u32, node2: u32) {
        let cl = self.core_len;
        let (node1, node2) = (node1 as usize, node2 as usize);
        if self.term_1[node1] == cl {
            self.term_1[node1] = 0;
            self.t1_len -= 1;
        }
        for &(other, _) in &self.q.adj[node1] {
            if self.term_1[other] == cl {
                self.term_1[other] = 0;
                self.t1_len -= 1;
            }
        }
        if self.term_2[node2] == cl {
            self.term_2[node2] = 0;
            self.t2_len -= 1;
        }
        for &(other, _) in self.t.mol.atom_bonds(node2) {
            if self.term_2[other] == cl {
                self.term_2[other] = 0;
                self.t2_len -= 1;
            }
        }
        self.core_1[node1] = NULL_NODE;
        self.core_2[node2] = NULL_NODE;
        self.core_len -= 1;
    }

    /// `MatchAll` / `Match`; returns `true` if the search must stop.
    fn match_all(&mut self, sink: &mut Sink) -> bool {
        if self.is_goal() {
            // GetCoreSet + MatchChecks (always true: no chirality, the
            // uniquify filter is applied by the caller in match order)
            match sink {
                Sink::All(v) => {
                    v.push(self.core_1.iter().map(|&x| x as usize).collect());
                    return false;
                }
                Sink::First(found) => {
                    **found = true;
                    return true;
                }
            }
        }
        if self.is_dead() {
            return false;
        }
        let mut pair = Pair::new();
        while self.next_pair(&mut pair) {
            if self.is_feasible_pair(pair.n1, pair.n2) {
                self.add_pair(pair.n1, pair.n2);
                if self.match_all(sink) {
                    return true;
                }
                self.backtrack(pair.n1, pair.n2);
            }
        }
        false
    }
}

/// Per-atom results of every recursive sub-pattern (`MatchSubqueries`):
/// `sets[id][atom]` is true if sub-pattern `id` matches with its first atom
/// on `atom`.  `recursive` must be in post order.
pub(crate) fn recursive_sets(recursive: &[QueryMol], t: &Target) -> Vec<Vec<bool>> {
    let n = t.mol.num_atoms();
    let mut sets: Vec<Vec<bool>> = Vec::with_capacity(recursive.len());
    for q in recursive {
        let mut set = vec![false; n];
        if !q.atoms.is_empty() && q.atoms.len() <= n {
            // RecursiveMatcher records the target atom of query atom 0 of
            // every match; the root pair is always (0, j), so the set equals
            // the roots j whose subtree contains at least one match.
            let mut st = State::new(q, t, &sets);
            for j in 0..n as u32 {
                if st.is_feasible_pair(0, j) {
                    st.add_pair(0, j);
                    let mut found = false;
                    st.match_all(&mut Sink::First(&mut found));
                    if found {
                        set[j as usize] = true;
                        // state is left mid-search; rebuild for the next root
                        st = State::new(q, t, &sets);
                    } else {
                        st.backtrack(0, j);
                    }
                }
            }
        }
        sets.push(set);
    }
    sets
}

/// All matches in RDKit order (`SubstructMatch` with `uniquify=false`).
pub(crate) fn all_matches(q: &QueryMol, t: &Target, rec: &[Vec<bool>]) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if q.atoms.is_empty() || q.atoms.len() > t.mol.num_atoms() {
        return out;
    }
    let mut st = State::new(q, t, rec);
    st.match_all(&mut Sink::All(&mut out));
    out
}

/// `true` if there is at least one match.
pub(crate) fn any_match(q: &QueryMol, t: &Target, rec: &[Vec<bool>]) -> bool {
    if q.atoms.is_empty() || q.atoms.len() > t.mol.num_atoms() {
        return false;
    }
    let mut st = State::new(q, t, rec);
    let mut found = false;
    st.match_all(&mut Sink::First(&mut found));
    found
}
