//! Aromaticity perception: port of the RDKit default aromaticity model
//! (`Code/GraphMol/Aromaticity.cpp`, BSD-3-Clause,
//! Copyright (C) 2003-2021 Greg Landrum and other RDKit contributors).

use super::rings::RingInfo;
use crate::core::{BondOrder, Molecule, element};
use crate::core::element::more_electronegative;
use std::collections::{BTreeMap, BTreeSet, HashSet};

const MAX_FUSED_AROMATIC_RING_SIZE: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum Donor {
    Vacant,
    One,
    Two,
    OneOrTwo,
    Any,
    No,
}

/// RDKit `MolOps::countAtomElec`.
pub fn count_atom_elec(mol: &Molecule, i: usize) -> i32 {
    let at = mol.atom(i);
    let dv = element(at.atomic_num).default_valence();
    if dv <= 1 {
        return -1;
    }
    let mut degree = (mol.degree(i) + at.total_hidden_hs() as usize) as i32;
    for &(_, bi) in mol.atom_bonds(i) {
        if mol.bond(bi).valence_contrib(i) == 0.0 {
            degree -= 1;
        }
    }
    if degree > 3 {
        return -1;
    }
    let mut nlp = element(at.atomic_num).n_outer as i32 - dv;
    nlp = (nlp - at.formal_charge as i32).max(0);
    let nrad = at.num_radical_electrons as i32;
    let mut res = (dv - degree) + nlp - nrad;
    if res > 1 {
        let nunsat = mol.explicit_valence(i) - mol.degree(i) as i32;
        if nunsat > 1 {
            res = 1;
        }
    }
    res
}

fn incident_noncyclic_multiple_bond(mol: &Molecule, ri: &RingInfo, i: usize) -> Option<usize> {
    for &(o, bi) in mol.atom_bonds(i) {
        if ri.num_bond_rings(bi) == 0 && mol.bond(bi).valence_contrib(i) >= 2.0 {
            return Some(o);
        }
    }
    None
}

fn incident_cyclic_multiple_bond(mol: &Molecule, ri: &RingInfo, i: usize) -> bool {
    mol.atom_bonds(i)
        .iter()
        .any(|&(_, bi)| ri.num_bond_rings(bi) > 0 && mol.bond(bi).valence_contrib(i) >= 2.0)
}

fn incident_multiple_bond(mol: &Molecule, i: usize) -> bool {
    let mut deg = (mol.degree(i) + mol.atom(i).num_explicit_hs as usize) as i32;
    for &(_, bi) in mol.atom_bonds(i) {
        if mol.bond(bi).valence_contrib(i).round() == 0.0 {
            deg -= 1;
        }
    }
    mol.explicit_valence(i) != deg
}

fn donor_type(mol: &Molecule, ri: &RingInfo, i: usize) -> Donor {
    let at = mol.atom(i);
    if at.atomic_num == 0 {
        return if incident_cyclic_multiple_bond(mol, ri, i) { Donor::One } else { Donor::Any };
    }
    let mut nelec = count_atom_elec(mol, i);
    if nelec < 0 {
        Donor::No
    } else if nelec == 0 {
        if incident_noncyclic_multiple_bond(mol, ri, i).is_some() {
            Donor::Vacant
        } else if incident_cyclic_multiple_bond(mol, ri, i) {
            Donor::One
        } else {
            Donor::No
        }
    } else if nelec == 1 {
        if let Some(who) = incident_noncyclic_multiple_bond(mol, ri, i) {
            if more_electronegative(mol.atom(who).atomic_num, at.atomic_num) {
                Donor::Vacant
            } else {
                Donor::One
            }
        } else if incident_multiple_bond(mol, i) {
            Donor::One
        } else if at.formal_charge == 1 {
            Donor::Vacant
        } else {
            Donor::No
        }
    } else {
        if let Some(who) = incident_noncyclic_multiple_bond(mol, ri, i) {
            if more_electronegative(mol.atom(who).atomic_num, at.atomic_num) {
                nelec -= 1;
            }
        }
        if nelec % 2 == 1 { Donor::One } else { Donor::Two }
    }
}

fn is_atom_cand_for_arom(mol: &Molecule, ri: &RingInfo, i: usize, edon: Donor) -> bool {
    let _ = ri;
    let at = mol.atom(i);
    if at.atomic_num > 18 && at.atomic_num != 34 && at.atomic_num != 52 {
        return false;
    }
    match edon {
        Donor::Vacant | Donor::One | Donor::Two | Donor::OneOrTwo | Donor::Any => {}
        Donor::No => return false,
    }
    let def_val = element(at.atomic_num).default_valence();
    if def_val > 0 {
        let eff = at.atomic_num as i32 - at.formal_charge as i32;
        if eff >= 0 && (eff as usize) < crate::core::element::num_elements() {
            let dv2 = element(eff as u8).default_valence();
            if mol.total_valence(i) > dv2 {
                return false;
            }
        }
    }
    if at.num_radical_electrons > 0 && (at.atomic_num != 6 || at.formal_charge != 0) {
        return false;
    }
    let nunsat = mol.explicit_valence(i) - mol.degree(i) as i32;
    if nunsat > 1 {
        let mut nmult = 0;
        for &(_, bi) in mol.atom_bonds(i) {
            match mol.bond(bi).order {
                BondOrder::Double | BondOrder::Triple => nmult += 1,
                _ => {}
            }
            if nmult > 1 {
                break;
            }
        }
        if nmult > 1 {
            return false;
        }
    }
    true
}

fn min_max_elecs(d: Donor) -> (i32, i32) {
    match d {
        Donor::Any | Donor::OneOrTwo => (1, 2),
        Donor::One => (1, 1),
        Donor::Two => (2, 2),
        Donor::No | Donor::Vacant => (0, 0),
    }
}

fn apply_huckel(ring: &[usize], edon: &[Donor], min_ring_size: usize) -> bool {
    if ring.len() < min_ring_size {
        return false;
    }
    let (mut rlw, mut rup) = (0, 0);
    let mut nany = 0;
    for &idx in ring {
        if edon[idx] == Donor::Any {
            nany += 1;
            if nany > 1 {
                return false;
            }
        }
        let (lw, up) = min_max_elecs(edon[idx]);
        rlw += lw;
        rup += up;
    }
    if rup >= 6 {
        (rlw..=rup).any(|rie| (rie - 2) % 4 == 0)
    } else {
        rup == 2
    }
}

fn pick_fused_rings(curr: usize, neigh: &BTreeMap<usize, Vec<usize>>, res: &mut Vec<usize>, done: &mut [bool]) {
    done[curr] = true;
    res.push(curr);
    let mut stack = vec![(curr, 0usize)];
    while let Some(top) = stack.last_mut() {
        let neighs = &neigh[&top.0];
        if top.1 == neighs.len() {
            stack.pop();
            continue;
        }
        let nb = neighs[top.1];
        top.1 += 1;
        if done[nb] {
            continue;
        }
        done[nb] = true;
        res.push(nb);
        stack.push((nb, 0));
    }
}

fn check_fused(rids: &[usize], neigh: &BTreeMap<usize, Vec<usize>>) -> bool {
    let n = neigh.len();
    let mut done = vec![false; n];
    for &r in neigh.keys() {
        if !rids.contains(&r) {
            done[r] = true;
        }
    }
    let mut fused = Vec::new();
    pick_fused_rings(rids[0], neigh, &mut fused, &mut done);
    fused.len() == rids.len()
}

fn make_ring_neighbor_map(brings: &[Vec<usize>], max_size: usize, max_overlap: usize) -> BTreeMap<usize, Vec<usize>> {
    let mut m: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..brings.len() {
        m.entry(i).or_default();
        if max_size > 0 && brings[i].len() > max_size {
            continue;
        }
        let s1: HashSet<usize> = brings[i].iter().copied().collect();
        for j in i + 1..brings.len() {
            if max_size > 0 && brings[j].len() > max_size {
                continue;
            }
            let inter = brings[j].iter().filter(|b| s1.contains(b)).count();
            if inter > 0 && (max_overlap == 0 || inter <= max_overlap) {
                m.entry(i).or_default().push(j);
                m.entry(j).or_default().push(i);
            }
        }
    }
    m
}

/// Calls `f` for every k-combination of 0..n (lexicographic order).
fn for_each_combination(n: usize, k: usize, mut f: impl FnMut(&[usize])) {
    if k == 0 || k > n {
        return;
    }
    let mut c: Vec<usize> = (0..k).collect();
    loop {
        f(&c);
        let mut i = k;
        loop {
            if i == 0 {
                return;
            }
            i -= 1;
            if c[i] != i + n - k {
                break;
            }
            if i == 0 && c[0] == n - k {
                return;
            }
        }
        c[i] += 1;
        for j in i + 1..k {
            c[j] = c[j - 1] + 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_huckel_to_fused(
    mol: &mut Molecule,
    srings: &[Vec<usize>],
    brings: &[Vec<usize>],
    fused: &[usize],
    edon: &[Donor],
    ring_neighs: &BTreeMap<usize, Vec<usize>>,
    max_num_fused: usize,
    min_ring_size: usize,
) {
    let nrings = fused.len();
    let mut fused_bonds: HashSet<usize> = HashSet::new();
    for &r in fused {
        fused_bonds.extend(brings[r].iter().copied());
    }
    let n_ring_bonds = fused_bonds.len();
    let mut done_bonds: BTreeSet<usize> = BTreeSet::new();
    let natoms = mol.num_atoms();
    let mut cur_size = 0;
    loop {
        if cur_size == 2 && nrings > 300 {
            break;
        }
        cur_size += 1;
        if cur_size > nrings.min(max_num_fused) || done_bonds.len() >= n_ring_bonds {
            break;
        }
        let mut to_mark: Vec<Vec<usize>> = Vec::new();
        for_each_combination(nrings, cur_size, |comb| {
            let cur_rs: Vec<usize> = comb.iter().map(|&i| fused[i]).collect();
            if !ring_neighs.is_empty() && !check_fused(&cur_rs, ring_neighs) {
                return;
            }
            let mut ats = vec![0usize; natoms];
            for &r in &cur_rs {
                for &a in &srings[r] {
                    ats[a] += 1;
                }
            }
            let unon: Vec<usize> = (0..natoms).filter(|&i| ats[i] == 1 || ats[i] == 2).collect();
            if apply_huckel(&unon, edon, min_ring_size) {
                to_mark.push(cur_rs);
            }
        });
        for cur_rs in to_mark {
            mark_atoms_bonds_arom(mol, brings, &cur_rs, &mut done_bonds);
        }
    }
}

fn mark_atoms_bonds_arom(mol: &mut Molecule, brings: &[Vec<usize>], ring_ids: &[usize], done: &mut BTreeSet<usize>) {
    let mut cnt: BTreeMap<usize, usize> = BTreeMap::new();
    for &r in ring_ids {
        for &b in &brings[r] {
            *cnt.entry(b).or_default() += 1;
        }
    }
    for (b, c) in cnt {
        if c == 1 {
            let bond = mol.bond_mut(b);
            bond.is_aromatic = true;
            if matches!(bond.order, BondOrder::Single | BondOrder::Double) {
                bond.order = BondOrder::Aromatic;
                let (x, y) = (bond.begin, bond.end);
                mol.atom_mut(x).is_aromatic = true;
                mol.atom_mut(y).is_aromatic = true;
            }
            done.insert(b);
        }
    }
}

/// RDKit default aromaticity model. Expects a kekulized molecule with ring
/// info and implicit H counts computed. Returns the number of aromatic rings
/// systems processed.
pub fn set_aromaticity(mol: &mut Molecule) {
    let ri = mol.rings().clone();
    let natoms = mol.num_atoms();
    let mut acands = vec![false; natoms];
    let mut aseen = vec![false; natoms];
    let mut edon = vec![Donor::No; natoms];
    let mut crings: Vec<Vec<usize>> = Vec::new();
    let mut cbrings: Vec<Vec<usize>> = Vec::new();
    for (rid, sring) in ri.atom_rings.iter().enumerate() {
        let ring_sz = sring.len();
        let mut all_arom = true;
        let mut all_dummy = true;
        for &a in sring {
            let at = mol.atom(a);
            if at.atomic_num != 0 {
                all_dummy = false;
            }
            if aseen[a] {
                if !acands[a] {
                    all_arom = false;
                }
                continue;
            }
            aseen[a] = true;
            let mut d = donor_type(mol, &ri, a);
            if d == Donor::Two && ring_sz >= 9 {
                let at = mol.atom(a);
                if (at.atomic_num == 8 || at.atomic_num == 16) && mol.degree(a) == 2 && at.formal_charge == 0 {
                    let has_mult = mol
                        .atom_bonds(a)
                        .iter()
                        .any(|&(_, b)| matches!(mol.bond(b).order, BondOrder::Double | BondOrder::Triple));
                    if !has_mult {
                        d = Donor::No;
                    }
                }
            }
            edon[a] = d;
            acands[a] = is_atom_cand_for_arom(mol, &ri, a, d);
            if !acands[a] {
                all_arom = false;
            }
        }
        if all_arom && !all_dummy {
            crings.push(sring.clone());
            cbrings.push(ri.bond_rings[rid].clone());
        }
    }
    let neigh = make_ring_neighbor_map(&cbrings, MAX_FUSED_AROMATIC_RING_SIZE, 1);
    let cnrs = crings.len();
    let mut fus_done = vec![false; cnrs];
    let mut curr = 0;
    while curr < cnrs {
        let mut fused = Vec::new();
        pick_fused_rings(curr, &neigh, &mut fused, &mut fus_done);
        apply_huckel_to_fused(mol, &crings, &cbrings, &fused, &edon, &neigh, 6, 0);
        match (0..cnrs).find(|&r| !fus_done[r]) {
            Some(r) => curr = r,
            None => break,
        }
    }
}

/// Removes aromatic flags from atoms (bonds must already be kekulized).
pub fn clear_aromatic_flags(mol: &mut Molecule) {
    for a in mol.atoms_mut() {
        a.is_aromatic = false;
    }
    for b in mol.bonds_mut() {
        b.is_aromatic = false;
    }
}
