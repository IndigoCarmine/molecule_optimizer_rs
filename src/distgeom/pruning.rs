//! RMS pruning of embedded conformers.
//!
//! Port of RDKit `Code/GraphMol/DistGeomHelpers/Embedder.cpp`
//! (`getMolSelfMatches`, `_isConfFarFromRest`).
//! RDKit: Copyright (C) 2004-2026 Greg Landrum and other RDKit contributors,
//! BSD-3-Clause.
//!
//! Differences: the heavy-atom self matches are enumerated with a simple
//! backtracking graph-automorphism search (atomic number, formal charge,
//! heavy degree, bond order; bonds to conjugated terminal heavy atoms match
//! any bond order, mimicking `symmetrizeTerminalAtoms`), capped at 1000
//! matches like RDKit; the optimal superposition uses Kabsch instead of
//! RDKit's quaternion method (identical up to round-off).

use super::EmbedParams;
use crate::core::Molecule;
use crate::core::geometry::kabsch;

const MAX_MATCHES: usize = 1000;

/// Atom maps (heavy atoms) used to compare conformers.
pub fn self_matches(mol: &Molecule, p: &EmbedParams) -> Vec<Vec<usize>> {
    let heavy: Vec<usize> = (0..mol.num_atoms()).filter(|&i| mol.atom(i).atomic_num != 1).collect();
    if p.use_symmetry_for_pruning && !heavy.is_empty() {
        return automorphisms(mol, &heavy);
    }
    if p.only_heavy_atoms_for_rms {
        vec![heavy]
    } else {
        vec![(0..mol.num_atoms()).collect()]
    }
}

fn automorphisms(mol: &Molecule, heavy: &[usize]) -> Vec<Vec<usize>> {
    let n = mol.num_atoms();
    let is_heavy = |a: usize| mol.atom(a).atomic_num != 1;
    let hdeg: Vec<usize> = (0..n).map(|a| mol.neighbors(a).filter(|&b| is_heavy(b)).count()).collect();
    let terminal_conj = |a: usize| -> bool {
        hdeg[a] == 1
            && matches!(mol.atom(a).atomic_num, 7 | 8 | 16)
            && mol.atom_bonds(a).iter().any(|&(nb, b)| is_heavy(nb) && mol.bond(b).is_conjugated)
    };
    let tc: Vec<bool> = (0..n).map(terminal_conj).collect();
    let atom_ok = |q: usize, t: usize| {
        let (aq, at) = (mol.atom(q), mol.atom(t));
        aq.atomic_num == at.atomic_num && aq.formal_charge == at.formal_charge && hdeg[q] == hdeg[t]
    };
    let bond_ok = |q1: usize, q2: usize, t1: usize, t2: usize| -> bool {
        match (mol.bond_between(q1, q2), mol.bond_between(t1, t2)) {
            (Some(bq), Some(bt)) => {
                if tc[q1] || tc[q2] || tc[t1] || tc[t2] {
                    true
                } else {
                    mol.bond(bq).order == mol.bond(bt).order
                }
            }
            (None, None) => true,
            _ => false,
        }
    };
    let mut out = Vec::new();
    let mut map: Vec<usize> = Vec::with_capacity(heavy.len());
    let mut used = vec![false; n];
    #[allow(clippy::too_many_arguments)]
    fn rec(
        k: usize,
        heavy: &[usize],
        map: &mut Vec<usize>,
        used: &mut [bool],
        out: &mut Vec<Vec<usize>>,
        atom_ok: &dyn Fn(usize, usize) -> bool,
        bond_ok: &dyn Fn(usize, usize, usize, usize) -> bool,
    ) {
        if out.len() >= MAX_MATCHES {
            return;
        }
        if k == heavy.len() {
            out.push(map.clone());
            return;
        }
        let q = heavy[k];
        for &t in heavy {
            if used[t] || !atom_ok(q, t) {
                continue;
            }
            if (0..k).all(|j| bond_ok(heavy[j], q, map[j], t)) {
                used[t] = true;
                map.push(t);
                rec(k + 1, heavy, map, used, out, atom_ok, bond_ok);
                map.pop();
                used[t] = false;
            }
        }
    }
    rec(0, heavy, &mut map, &mut used, &mut out, &atom_ok, &bond_ok);
    if out.is_empty() {
        out.push(heavy.to_vec());
    }
    out
}

/// RDKit `_isConfFarFromRest`.
pub fn is_far_from_rest(conf: &[[f64; 3]], kept: &[Vec<[f64; 3]>], threshold: f64, matches: &[Vec<usize>]) -> bool {
    let reference: Vec<[f64; 3]> = matches[0].iter().map(|&i| conf[i]).collect();
    let ssr_thresh = matches[0].len() as f64 * threshold * threshold;
    for m in matches {
        for other in kept {
            let probe: Vec<[f64; 3]> = m.iter().map(|&i| other[i]).collect();
            let rmsd = kabsch(&reference, &probe).0;
            if rmsd * rmsd * (reference.len() as f64) < ssr_thresh {
                return false;
            }
        }
    }
    true
}
