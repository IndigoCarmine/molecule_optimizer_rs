//! RDKit's canonical kekulization (`MolOps::Kekulize(mol, markAtomsBonds =
//! true, canonical = true)`), as called by RDKit's MMFF typer on aromatic
//! molecules.  The chosen Kekulé structure depends on the ring order, the
//! canonical atom ranks ([`super::canon`]) and the neighbor order, and it
//! decides RDKit's MMFF aromaticity for systems such as fullerenes.
//!
//! Ported from RDKit 2026.09.1 `Code/GraphMol/Kekulize.cpp`
//! (`KekulizeFragment`, `kekulizeFused`, `markDbondCands`, `kekulizeWorker`,
//! `backTrack`, `permuteDummiesAndKekulize`), `Code/GraphMol/Aromaticity.cpp`
//! (`RingUtils::pickFusedRings`, `RingUtils::makeRingNeighborMap`) and
//! `Code/GraphMol/Atom.cpp` (`isEarlyAtom`),
//! Copyright (C) 2001-2026 Greg Landrum and other RDKit contributors;
//! BSD 3-Clause license (see `LICENSE-RDKIT.txt`).
//!
//! Bond wedging (which RDKit's molfile parser clears before sanitization) is
//! not modelled.

use super::canon::rank_fragment_atoms;
use crate::core::{BondOrder, Molecule, element};
use crate::error::{Error, Result};
use crate::perception::rings::RingInfo;
use std::collections::{HashMap, VecDeque};

/// `isEarlyAtom` (`Code/GraphMol/Atom.cpp`).
fn is_early_atom(anum: u8) -> bool {
    matches!(
        anum,
        3 | 4 | 5 | 11 | 12 | 13 | 19..=22 | 30..=32 | 37..=41 | 48..=51 | 55..=61 | 72 | 73 | 80..=83 | 87..=93 | 104..=118
    )
}

/// `isAromaticAtom`: the atom flag or any aromatic bond.
fn is_aromatic_atom(mol: &Molecule, i: usize) -> bool {
    mol.atom(i).is_aromatic
        || mol.atom_bonds(i).iter().any(|&(_, b)| mol.bond(b).is_aromatic || mol.bond(b).order == BondOrder::Aromatic)
}

/// Atom properties RDKit has cached before kekulization starts.
struct Cached {
    total_valence: Vec<i32>,
    total_num_hs: Vec<i32>,
    implicit: Vec<i32>,
}

/// `RingUtils::pickFusedRings` (preorder DFS over the ring neighbor map).
fn pick_fused_rings(curr: usize, neigh: &[Vec<usize>], res: &mut Vec<usize>, done: &mut [bool]) {
    done[curr] = true;
    res.push(curr);
    let mut stack: Vec<(usize, usize)> = vec![(curr, 0)];
    while let Some(top) = stack.last_mut() {
        let (ring, next) = *top;
        if next == neigh[ring].len() {
            stack.pop();
            continue;
        }
        top.1 += 1;
        let nb = neigh[ring][next];
        if done[nb] {
            continue;
        }
        done[nb] = true;
        res.push(nb);
        stack.push((nb, 0));
    }
}

/// `markDbondCands`.
fn mark_dbond_cands(
    mol: &mut Molecule,
    ri: &RingInfo,
    cache: &Cached,
    all_atms: &[usize],
    dbnd_cands: &mut [bool],
    questions: &mut Vec<usize>,
    done: &mut Vec<usize>,
) {
    if !all_atms.iter().any(|&a| mol.atom(a).atomic_num == 0 || is_aromatic_atom(mol, a)) {
        return;
    }
    let ring_not_cand: Vec<bool> = ri
        .atom_rings
        .iter()
        .map(|ring| !ring.iter().any(|&ai| is_aromatic_atom(mol, ai) && ri.num_atom_rings(ai) == 1))
        .collect();
    let mut make_single = Vec::new();
    let mut in_all = vec![false; mol.num_atoms()];
    for &a in all_atms {
        in_all[a] = true;
    }
    for &a in all_atms {
        let anum = mol.atom(a).atomic_num;
        if anum != 0 && !is_aromatic_atom(mol, a) {
            done.push(a);
            continue;
        }
        let mut sbo: i32 = 0;
        let mut n_to_ignore: i32 = 0;
        let mut non_ar_non_dummy_nbr = 0usize;
        for &(other, b) in mol.atom_bonds(a) {
            if mol.atom(other).atomic_num != 0 && !mol.atom(other).is_aromatic && in_all[other] {
                non_ar_non_dummy_nbr += 1;
            }
            let bond = mol.bond(b);
            if bond.is_aromatic && matches!(bond.order, BondOrder::Single | BondOrder::Double | BondOrder::Aromatic) {
                sbo += 1;
                make_single.push(b);
            } else {
                let contrib = bond.valence_contrib(a).round() as i32;
                sbo += contrib;
                if contrib == 0 {
                    n_to_ignore += 1;
                }
            }
        }
        let num_atom_rings = ri.num_atom_rings(a);
        let num_non_cand = ri.atom_ring_ids(a).iter().filter(|&&r| ring_not_cand[r]).count();
        if anum == 0 && non_ar_non_dummy_nbr < num_atom_rings && num_non_cand < num_atom_rings {
            dbnd_cands[a] = true;
            questions.push(a);
            continue;
        }
        sbo += cache.total_num_hs[a];
        let vals = &element(anum).valences;
        let mut dv = vals.first().copied().unwrap_or(-1);
        let mut chrg = mol.atom(a).formal_charge as i32;
        if is_early_atom(anum) {
            chrg = -chrg;
        }
        if anum == 6 && chrg > 0 {
            chrg = -chrg;
        }
        dv += chrg;
        let tbo = cache.total_valence[a];
        let n_radicals = mol.atom(a).num_radical_electrons as i32;
        let total_degree = mol.degree(a) as i32 + cache.implicit[a] - n_to_ignore;
        let mut vi = 1;
        while tbo > dv && vi < vals.len() && vals[vi] > 0 {
            dv = vals[vi] + chrg;
            vi += 1;
        }
        if tbo == 5
            && sbo == 4
            && dv == 3
            && total_degree == 3
            && n_radicals == 0
            && chrg == 0
            && cache.total_num_hs[a] == 0
            && matches!(anum, 7 | 15 | 33)
        {
            dv = 5;
        }
        if total_degree + n_radicals >= dv {
            continue;
        }
        if dv == sbo + 1 + n_radicals || (n_radicals == 0 && mol.atom(a).no_implicit && dv == sbo + 2) {
            dbnd_cands[a] = true;
        }
    }
    for b in make_single {
        mol.bond_mut(b).order = BondOrder::Single;
    }
}

/// `backTrack`.
fn back_track(
    mol: &mut Molecule,
    last_opt: usize,
    done: &mut Vec<usize>,
    aqueue: &mut VecDeque<usize>,
    dbnd_cands: &mut [bool],
    dbnd_adds: &mut [bool],
) {
    let first = done.iter().position(|&x| x == last_opt).unwrap_or(done.len());
    let tdone: Vec<usize> = done[..first].to_vec();
    if let Some(last) = done.iter().rposition(|&x| x == last_opt) {
        for k in (last..done.len()).rev() {
            aqueue.push_front(done[k]);
        }
    }
    for bi in 0..mol.num_bonds() {
        if dbnd_adds[bi] {
            let (a1, a2) = (mol.bond(bi).begin, mol.bond(bi).end);
            if !tdone.contains(&a1) && !tdone.contains(&a2) {
                dbnd_adds[bi] = false;
                mol.bond_mut(bi).order = BondOrder::Single;
                dbnd_cands[a1] = true;
                dbnd_cands[a2] = true;
            }
        }
    }
    *done = tdone;
}

/// `kekulizeWorker`.
fn kekulize_worker(
    mol: &mut Molecule,
    all_atms: &[usize],
    mut dbnd_cands: Vec<bool>,
    mut dbnd_adds: Vec<bool>,
    mut done: Vec<usize>,
    ranks: &[usize],
    max_back_tracks: u32,
) -> bool {
    let mut astack: VecDeque<usize> = VecDeque::new();
    let mut options: HashMap<usize, VecDeque<usize>> = HashMap::new();
    let mut last_opt: Option<usize> = None;
    let mut local_bonds_added = vec![false; mol.num_bonds()];
    let mut in_all = vec![false; mol.num_atoms()];
    for &a in all_atms {
        in_all[a] = true;
    }
    let less_key = |a: usize| (ranks[a], a);
    let mut sorted = all_atms.to_vec();
    sorted.sort_by_key(|&a| less_key(a));

    let mut btmoves: VecDeque<usize> = VecDeque::new();
    let mut num_bt = 0u32;
    while done.len() < sorted.len() || !astack.is_empty() {
        let curr = match astack.pop_front() {
            Some(c) => c,
            None => match sorted.iter().copied().find(|a| !done.contains(a)) {
                Some(c) => c,
                None => return false,
            },
        };
        done.push(curr);

        let mut opts: VecDeque<usize>;
        let c_cand = dbnd_cands[curr];
        let revisited = options.contains_key(&curr);
        if let Some(o) = options.get(&curr) {
            opts = o.clone();
        } else {
            opts = VecDeque::new();
            let mut nbrs: Vec<usize> =
                mol.neighbors(curr).filter(|&nb| in_all[nb] && !done.contains(&nb)).collect();
            nbrs.sort_by_key(|&a| less_key(a));
            let mut lstack = Vec::new();
            for nb in nbrs {
                let b = mol.bond_between(curr, nb).expect("bond");
                if !astack.contains(&nb) {
                    lstack.push(nb);
                }
                if c_cand
                    && dbnd_cands[nb]
                    && (mol.bond(b).is_aromatic || mol.atom(curr).atomic_num == 0 || mol.atom(nb).atomic_num == 0)
                {
                    opts.push_back(nb);
                }
            }
            astack.extend(lstack);
        }
        if c_cand {
            if let Some(ncnd) = opts.pop_front() {
                let b = mol.bond_between(curr, ncnd).expect("bond");
                mol.bond_mut(b).order = BondOrder::Double;
                dbnd_cands[curr] = false;
                dbnd_cands[ncnd] = false;
                dbnd_adds[b] = true;
                local_bonds_added[b] = true;
                if revisited {
                    if opts.is_empty() {
                        options.remove(&curr);
                        btmoves.pop_back();
                        last_opt = btmoves.back().copied();
                    } else {
                        options.insert(curr, opts);
                    }
                } else if !opts.is_empty() {
                    last_opt = Some(curr);
                    btmoves.push_back(curr);
                    options.insert(curr, opts);
                }
            } else if mol.atom(curr).atomic_num != 0 {
                match last_opt {
                    Some(lo) if num_bt < max_back_tracks => {
                        back_track(mol, lo, &mut done, &mut astack, &mut dbnd_cands, &mut dbnd_adds);
                        num_bt += 1;
                    }
                    _ => {
                        for (bi, &added) in local_bonds_added.iter().enumerate() {
                            if added {
                                mol.bond_mut(bi).order = BondOrder::Single;
                            }
                        }
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// `kekulizeFused` (including `permuteDummiesAndKekulize`).
fn kekulize_fused(
    mol: &mut Molecule,
    ri: &RingInfo,
    cache: &Cached,
    arings: &[&Vec<usize>],
    ranks: &[usize],
    max_back_tracks: u32,
) -> Result<()> {
    let mut all_atms: Vec<usize> = Vec::new();
    for ring in arings {
        for &a in ring.iter() {
            if !all_atms.contains(&a) {
                all_atms.push(a);
            }
        }
    }
    let mut done = Vec::new();
    let mut questions = Vec::new();
    let mut dbnd_cands = vec![false; mol.num_atoms()];
    mark_dbond_cands(mol, ri, cache, &all_atms, &mut dbnd_cands, &mut questions, &mut done);
    let nb = mol.num_bonds();
    let mut kekulized =
        kekulize_worker(mol, &all_atms, dbnd_cands.clone(), vec![false; nb], done, ranks, max_back_tracks);
    if !kekulized && !questions.is_empty() {
        // permuteDummiesAndKekulize: enumerate subsets of the questionable
        // (dummy) atoms to switch off, starting with {questions[0]}
        let mut in_play = vec![false; mol.num_atoms()];
        for &a in &all_atms {
            in_play[a] = true;
        }
        let nq = questions.len();
        let mut state = vec![false; nq];
        state[0] = true;
        let mut enum_done = false;
        while !kekulized {
            for b in 0..nb {
                let bond = mol.bond(b);
                if bond.is_aromatic && bond.order != BondOrder::Single && in_play[bond.begin] && in_play[bond.end] {
                    mol.bond_mut(b).order = BondOrder::Single;
                }
            }
            if enum_done {
                break;
            }
            let switch_off: Vec<usize> = (0..nq).filter(|&i| state[i]).map(|i| questions[i]).collect();
            let mut pos = 0;
            while pos < nq && state[pos] {
                state[pos] = false;
                pos += 1;
            }
            if pos == nq {
                enum_done = true;
            } else {
                state[pos] = true;
            }
            if switch_off.is_empty() {
                break;
            }
            let mut t = dbnd_cands.clone();
            for &a in &switch_off {
                t[a] = false;
            }
            kekulized = kekulize_worker(mol, &all_atms, t, vec![false; nb], Vec::new(), ranks, max_back_tracks);
        }
    }
    if !kekulized {
        let problems: Vec<usize> = (0..mol.num_atoms()).filter(|&i| dbnd_cands[i]).collect();
        return Err(Error::Perception(format!("Can't kekulize mol. Unkekulized atoms: {problems:?}")));
    }
    Ok(())
}

/// Kekulizes `mol` in place exactly like RDKit's `MolOps::Kekulize` with
/// `markAtomsBonds = true, canonical = true`.  `ranks` are the canonical atom
/// ranks; if `None` they are computed from `mol` itself.  Requires ring info
/// (`mol.rings()`) and aromatic bonds flagged (`is_aromatic`).
pub fn kekulize_canonical(mol: &mut Molecule, ranks: Option<&[usize]>) -> Result<()> {
    let n = mol.num_atoms();
    let any_arom_bond = mol.bonds().iter().any(|b| b.is_aromatic);
    let any_arom_atom = (0..n).any(|i| is_aromatic_atom(mol, i));
    if !any_arom_bond && !any_arom_atom {
        return Ok(());
    }
    // kekulization changes only bond orders and atom flags: the ring info is
    // moved out for the duration (no copy) and put back afterwards
    mol.rings();
    let ri = mol.ring_info.take().expect("ring info");
    let res = kekulize_canonical_with(mol, &ri, ranks);
    mol.ring_info = Some(ri);
    res
}

fn kekulize_canonical_with(mol: &mut Molecule, ri: &RingInfo, ranks: Option<&[usize]>) -> Result<()> {
    let n = mol.num_atoms();
    let owned;
    let ranks: &[usize] = match ranks {
        Some(r) => r,
        None => {
            owned = rank_fragment_atoms(mol, ri);
            &owned
        }
    };
    let cache = Cached {
        total_valence: (0..n).map(|i| mol.total_valence(i)).collect(),
        total_num_hs: (0..n).map(|i| mol.atom(i).total_hidden_hs() as i32).collect(),
        implicit: (0..n).map(|i| mol.atom(i).implicit_hs as i32).collect(),
    };

    // rings with at least one non-dummy atom, in ring-info order
    let ring_ids: Vec<usize> = (0..ri.atom_rings.len())
        .filter(|&r| ri.atom_rings[r].iter().any(|&a| mol.atom(a).atomic_num != 0))
        .collect();
    let brings: Vec<&Vec<usize>> = ring_ids.iter().map(|&r| &ri.bond_rings[r]).collect();
    let nr = ring_ids.len();
    // makeRingNeighborMap
    let mut neigh: Vec<Vec<usize>> = vec![Vec::new(); nr];
    for i in 0..nr {
        for j in i + 1..nr {
            if brings[i].iter().any(|b| brings[j].contains(b)) {
                neigh[i].push(j);
                neigh[j].push(i);
            }
        }
    }
    let mut fus_done = vec![false; nr];
    let mut curr = 0;
    while curr < nr {
        let mut fused = Vec::new();
        pick_fused_rings(curr, &neigh, &mut fused, &mut fus_done);
        let frings: Vec<&Vec<usize>> = fused.iter().map(|&k| &ri.atom_rings[ring_ids[k]]).collect();
        kekulize_fused(mol, ri, &cache, &frings, ranks, 100)?;
        match (0..nr).find(|&k| !fus_done[k]) {
            Some(k) => curr = k,
            None => break,
        }
    }

    // markAtomsBonds
    for b in mol.bonds_mut() {
        b.is_aromatic = false;
    }
    for i in 0..n {
        if !mol.atom(i).is_aromatic {
            continue;
        }
        if ri.num_atom_rings(i) == 0 {
            return Err(Error::Perception(format!("non-ring atom {i} marked aromatic")));
        }
        let a = mol.atom_mut(i);
        a.is_aromatic = false;
        // GitHub #141: "explicit" Hs on pyrrole-like N/P become implicit
        if (a.atomic_num == 7 || a.atomic_num == 15) && a.formal_charge == 0 && a.num_explicit_hs == 1 {
            a.no_implicit = false;
            a.num_explicit_hs = 0;
            a.implicit_hs += 1;
        }
    }
    Ok(())
}
