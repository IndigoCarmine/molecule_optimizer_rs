//! Connectivity from interatomic distances.

use crate::core::{BondOrder, Molecule, element, geometry::dist};

/// Tolerance added to the sum of covalent radii.
pub const BOND_TOLERANCE: f64 = 0.45;

/// Adds single bonds between atoms closer than `r_cov(a)+r_cov(b)+tol`.
/// Existing bonds are kept. Uses a cell list, so it is O(N).
pub fn perceive_connectivity(mol: &mut Molecule, tol: f64) {
    let n = mol.num_atoms();
    if n < 2 {
        return;
    }
    let max_r = (0..n).map(|i| element(mol.atom(i).atomic_num).r_cov).fold(0.0, f64::max);
    let cell = (2.0 * max_r + tol).max(1.0);
    let mut mins = [f64::INFINITY; 3];
    for p in &mol.coords {
        for k in 0..3 {
            mins[k] = mins[k].min(p[k]);
        }
    }
    let key = |p: &[f64; 3]| -> (i64, i64, i64) {
        (
            ((p[0] - mins[0]) / cell).floor() as i64,
            ((p[1] - mins[1]) / cell).floor() as i64,
            ((p[2] - mins[2]) / cell).floor() as i64,
        )
    };
    let mut grid: std::collections::HashMap<(i64, i64, i64), Vec<usize>> = Default::default();
    for i in 0..n {
        grid.entry(key(&mol.coords[i])).or_default().push(i);
    }
    let mut new_bonds = Vec::new();
    for i in 0..n {
        let (cx, cy, cz) = key(&mol.coords[i]);
        let ri = element(mol.atom(i).atomic_num).r_cov;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(list) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in list {
                            if j <= i {
                                continue;
                            }
                            let ai = mol.atom(i).atomic_num;
                            let aj = mol.atom(j).atomic_num;
                            if ai == 1 && aj == 1 {
                                // only bond H-H if very close (H2)
                                if dist(mol.coords[i], mol.coords[j]) < 0.9 {
                                    new_bonds.push((i, j));
                                }
                                continue;
                            }
                            let rj = element(aj).r_cov;
                            let d = dist(mol.coords[i], mol.coords[j]);
                            if d > 0.4 && d < ri + rj + tol {
                                new_bonds.push((i, j));
                            }
                        }
                    }
                }
            }
        }
    }
    for (i, j) in new_bonds {
        mol.add_bond(i, j, BondOrder::Single);
    }
    // Hydrogens should have exactly one bond: keep only the shortest.
    prune_hydrogen_bonds(mol);
}

fn prune_hydrogen_bonds(mol: &mut Molecule) {
    let n = mol.num_atoms();
    let mut keep = vec![true; mol.num_bonds()];
    let mut need = false;
    for h in 0..n {
        if mol.atom(h).atomic_num != 1 || mol.degree(h) <= 1 {
            continue;
        }
        let best = mol
            .atom_bonds(h)
            .iter()
            .min_by(|a, b| {
                dist(mol.coords[h], mol.coords[a.0]).partial_cmp(&dist(mol.coords[h], mol.coords[b.0])).unwrap()
            })
            .map(|x| x.1)
            .unwrap();
        for &(_, bi) in mol.atom_bonds(h) {
            if bi != best {
                keep[bi] = false;
                need = true;
            }
        }
    }
    if !need {
        return;
    }
    let bonds: Vec<_> = mol.bonds().iter().cloned().enumerate().filter(|(i, _)| keep[*i]).map(|x| x.1).collect();
    mol.clear_bonds();
    for b in bonds {
        let bi = mol.add_bond(b.begin, b.end, b.order);
        *mol.bond_mut(bi) = b;
    }
}
