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
    let nb = mol.num_bonds();
    for a in mol.atoms_mut() {
        a.no_implicit = true;
        a.implicit_hs = 0;
    }
    // ammonium
    for i in 0..n {
        let a = mol.atom(i);
        if a.atomic_num == 7 && mol.degree(i) == 4 {
            mol.atom_mut(i).formal_charge = 1;
        }
    }
    // nitro-like: N(deg 3) bonded to two terminal O
    for i in 0..n {
        if mol.atom(i).atomic_num != 7 || mol.degree(i) != 3 {
            continue;
        }
        let term_o: Vec<usize> = mol
            .neighbors(i)
            .filter(|&o| mol.atom(o).atomic_num == 8 && mol.degree(o) == 1)
            .collect();
        if term_o.len() >= 2 {
            mol.atom_mut(i).formal_charge = 1;
            mol.atom_mut(term_o[0]).formal_charge = -1;
        }
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
    // leftover terminal O/S with one missing bond -> anion
    for i in 0..n {
        let a = mol.atom(i);
        if (a.atomic_num == 8 || a.atomic_num == 16) && mol.degree(i) == 1 && a.formal_charge == 0 && unsaturation(mol, i, &flags) == 1 {
            mol.atom_mut(i).formal_charge = -1;
        }
    }
}
