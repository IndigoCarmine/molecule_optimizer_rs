//! Kekulization of aromatic bonds and bond-order assignment for
//! connectivity-only input (XYZ / PDB).

use super::matching::max_matching;
use super::valence::allowed_valences;
use crate::core::{BondOrder, Molecule};
use crate::error::{Error, Result};

/// Number of extra bond orders atom `i` needs (smallest allowed valence that
/// is ≥ current valence, counting each `counted_as_single` bond as 1).
fn unsaturation(mol: &Molecule, i: usize, single_bonds: &[bool]) -> i32 {
    let at = mol.atom(i);
    let vals = allowed_valences(at.atomic_num, at.formal_charge);
    if vals.is_empty() {
        return 0;
    }
    let mut v = at.num_explicit_hs as i32 + at.implicit_hs as i32 + at.num_radical_electrons as i32;
    for &(_, bi) in mol.atom_bonds(i) {
        v += if single_bonds[bi] { 1 } else { mol.bond(bi).valence_contrib(i).round() as i32 };
    }
    match vals.iter().copied().find(|&x| x >= v) {
        Some(x) => x - v,
        None => 0,
    }
}

/// Replaces aromatic bonds by alternating single/double bonds.
/// Atom/bond aromatic flags are kept (they are recomputed by
/// [`crate::perception::sanitize`]).
pub fn kekulize(mol: &mut Molecule) -> Result<()> {
    let nb = mol.num_bonds();
    let arom: Vec<bool> = (0..nb).map(|b| mol.bond(b).order == BondOrder::Aromatic).collect();
    if !arom.iter().any(|&x| x) {
        return Ok(());
    }
    let n = mol.num_atoms();
    // Which atoms need a double bond?
    let mut needs = vec![false; n];
    for i in 0..n {
        let has_arom = mol.atom_bonds(i).iter().any(|&(_, b)| arom[b]);
        if !has_arom {
            continue;
        }
        // atoms with explicit implicit-H data: implicit Hs computed with
        // aromatic contributions are unreliable; recompute assuming single.
        needs[i] = unsaturation(mol, i, &arom) >= 1;
    }
    let mut adj = vec![Vec::new(); n];
    for b in 0..nb {
        if arom[b] {
            let (x, y) = (mol.bond(b).begin, mol.bond(b).end);
            if needs[x] && needs[y] {
                adj[x].push(y);
                adj[y].push(x);
            }
        }
    }
    let mate = max_matching(&adj);
    let mut unmatched = Vec::new();
    for i in 0..n {
        if needs[i] && mate[i] == usize::MAX {
            unmatched.push(i);
        }
    }
    for b in 0..nb {
        if arom[b] {
            let (x, y) = (mol.bond(b).begin, mol.bond(b).end);
            let ord = if mate[x] == y { BondOrder::Double } else { BondOrder::Single };
            mol.bond_mut(b).order = ord;
        }
    }
    if !unmatched.is_empty() {
        // Common case: aromatic N written without its H ("n" for "[nH]").
        // Give such atoms an implicit H if that makes them saturated.
        let mut ok = true;
        for &i in &unmatched {
            let a = mol.atom(i);
            if a.atomic_num == 7 && a.formal_charge == 0 && mol.degree(i) == 2 && !a.no_implicit {
                mol.atom_mut(i).implicit_hs += 1;
            } else {
                ok = false;
            }
        }
        if !ok {
            return Err(Error::Perception(format!("can't kekulize atoms {:?}", unmatched)));
        }
    }
    Ok(())
}

/// Assigns bond orders for a molecule whose bonds are all single
/// (connectivity from distances) and whose hydrogens are all explicit.
/// Also assigns simple formal charges (ammonium, nitro, carboxylate-like
/// leftovers). This is a heuristic.
pub fn assign_bond_orders(mol: &mut Molecule) {
    let n = mol.num_atoms();
    for a in mol.atoms_mut() {
        a.no_implicit = true;
        a.implicit_hs = 0;
    }
    let terminal_o = |mol: &Molecule, i: usize| -> Vec<(usize, usize)> {
        mol.atom_bonds(i).iter().copied().filter(|&(o, _)| mol.atom(o).atomic_num == 8 && mol.degree(o) == 1).collect()
    };
    for i in 0..n {
        let z = mol.atom(i).atomic_num;
        let deg = mol.degree(i);
        let to = terminal_o(mol, i);
        if z == 7 && deg == 4 {
            // ammonium / amine oxide (the oxide O becomes O- below)
            mol.atom_mut(i).formal_charge = 1;
        } else if z == 7 && deg == 3 && to.len() >= 2 {
            // nitro: N+(=O)O-
            mol.atom_mut(i).formal_charge = 1;
            mol.atom_mut(to[0].0).formal_charge = -1;
        } else if z == 7 && deg == 3 && to.len() == 1 {
            // N-oxide (e.g. pyridine N-oxide): N+ O-
            mol.atom_mut(i).formal_charge = 1;
            mol.atom_mut(to[0].0).formal_charge = -1;
        } else if matches!(z, 16 | 34 | 15 | 33) && deg >= 3 && !to.is_empty() {
            // hypervalent oxo groups: S(=O), S(=O)2, P(=O), Se(=O)...
            let max_val = if matches!(z, 16 | 34) { 6 } else { 5 };
            let n_double = (max_val - deg as i32).max(0) as usize;
            for &(_, b) in to.iter().take(n_double) {
                mol.bond_mut(b).order = BondOrder::Double;
            }
        }
    }
    let base: Vec<BondOrder> = mol.bonds().iter().map(|b| b.order).collect();
    let mut leftover = match_unsaturations(mol, &base);
    // Cations without charge information (pyridinium, guanidinium, iminium):
    // an unpaired unsaturated atom next to a neutral trivalent N → N+.
    for _ in 0..n {
        if leftover.is_empty() {
            break;
        }
        let mut cands: Vec<usize> = Vec::new();
        for &u in &leftover {
            for v in mol.neighbors(u) {
                let a = mol.atom(v);
                if a.atomic_num == 7 && a.formal_charge == 0 && mol.degree(v) == 3 && !cands.contains(&v) {
                    cands.push(v);
                }
            }
        }
        // also search one bond further (ring positions)
        if cands.is_empty() {
            for &u in &leftover {
                for w in mol.neighbors(u) {
                    for v in mol.neighbors(w) {
                        let a = mol.atom(v);
                        if a.atomic_num == 7 && a.formal_charge == 0 && mol.degree(v) == 3 && !cands.contains(&v) {
                            cands.push(v);
                        }
                    }
                }
            }
        }
        let mut best: Option<(usize, Vec<BondOrder>, Vec<usize>)> = None;
        for &c in &cands {
            let saved: Vec<BondOrder> = mol.bonds().iter().map(|b| b.order).collect();
            mol.atom_mut(c).formal_charge = 1;
            let lo = match_unsaturations(mol, &base);
            if lo.len() < leftover.len() && best.as_ref().is_none_or(|b| lo.len() < b.2.len()) {
                best = Some((c, mol.bonds().iter().map(|b| b.order).collect(), lo));
            }
            mol.atom_mut(c).formal_charge = 0;
            for (b, o) in saved.into_iter().enumerate() {
                mol.bond_mut(b).order = o;
            }
        }
        match best {
            Some((c, orders, lo)) => {
                mol.atom_mut(c).formal_charge = 1;
                for (b, o) in orders.into_iter().enumerate() {
                    mol.bond_mut(b).order = o;
                }
                leftover = lo;
            }
            None => break,
        }
    }
    // leftover terminal O/S with one missing bond -> anion
    let flags = vec![false; mol.num_bonds()];
    for i in 0..n {
        let a = mol.atom(i);
        if (a.atomic_num == 8 || a.atomic_num == 16) && mol.degree(i) == 1 && a.formal_charge == 0 && unsaturation(mol, i, &flags) == 1 {
            mol.atom_mut(i).formal_charge = -1;
        }
    }
}

/// Resets bond orders to `base` and raises orders by repeated maximum
/// matchings of unsaturated atoms. Returns the atoms left unsaturated
/// (excluding terminal O/S, which become anions).
fn match_unsaturations(mol: &mut Molecule, base: &[BondOrder]) -> Vec<usize> {
    let n = mol.num_atoms();
    let nb = mol.num_bonds();
    for (b, &o) in base.iter().enumerate() {
        mol.bond_mut(b).order = o;
    }
    let flags = vec![false; nb];
    for _round in 0..3 {
        let unsat: Vec<i32> = (0..n).map(|i| unsaturation(mol, i, &flags)).collect();
        let mut adj = vec![Vec::new(); n];
        let mut any = false;
        for b in 0..nb {
            let (x, y) = (mol.bond(b).begin, mol.bond(b).end);
            if unsat[x] > 0 && unsat[y] > 0 {
                adj[x].push(y);
                adj[y].push(x);
                any = true;
            }
        }
        if !any {
            break;
        }
        // prefer neighbours with fewer options first (helps conjugated chains)
        for l in adj.iter_mut() {
            l.sort_by_key(|&v| (unsat[v], v));
        }
        let mate = max_matching(&adj);
        let mut changed = false;
        for b in 0..nb {
            let (x, y) = (mol.bond(b).begin, mol.bond(b).end);
            if mate[x] == y {
                let o = match mol.bond(b).order {
                    BondOrder::Single => BondOrder::Double,
                    BondOrder::Double => BondOrder::Triple,
                    o => o,
                };
                if o != mol.bond(b).order {
                    mol.bond_mut(b).order = o;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    (0..n)
        .filter(|&i| {
            let a = mol.atom(i);
            unsaturation(mol, i, &flags) > 0 && !((a.atomic_num == 8 || a.atomic_num == 16) && mol.degree(i) == 1)
        })
        .collect()
}
