//! Stereochemistry from 3D coordinates.

use crate::core::geometry::{cross, dot, sub};
use crate::core::{BondOrder, BondStereo, ChiralTag, Molecule};

const ZERO_VOLUME_TOL: f64 = 0.1;

/// Sets tetrahedral chiral tags from the coordinates for every atom with
/// three or four neighbors that is sp3-like (degree 4, or degree 3 for
/// non-nitrogen with a lone pair such as S/P). Uses the RDKit convention
/// (`assignChiralTypesFrom3D`): with neighbors n1,n2,n3 in bond order,
/// `V = (n1-c)·((n2-c)×(n3-c))`; `V < 0` → CW, `V > 0` → CCW.
///
/// Note: unlike RDKit, centers are not filtered by CIP ranking; every
/// tetrahedral center keeps the handedness of the input, which preserves the
/// input stereoisomer during embedding.
pub fn assign_chiral_tags_from_3d(mol: &mut Molecule) {
    for i in 0..mol.num_atoms() {
        let deg = mol.degree(i);
        let anum = mol.atom(i).atomic_num;
        let eligible = (deg == 4 && mol.atom(i).total_hidden_hs() == 0)
            || (deg == 3 && mol.atom(i).total_hidden_hs() == 0 && matches!(anum, 15 | 16 | 33 | 34));
        let tetra = mol
            .atom_bonds(i)
            .iter()
            .all(|&(_, b)| matches!(mol.bond(b).order, BondOrder::Single | BondOrder::Dative));
        if !eligible || !tetra || mol.atom(i).is_aromatic {
            mol.atom_mut(i).chiral_tag = ChiralTag::Unspecified;
            continue;
        }
        let nbrs: Vec<usize> = mol.neighbors(i).collect();
        let p0 = mol.pos(i);
        let v1 = sub(mol.pos(nbrs[0]), p0);
        let v2 = sub(mol.pos(nbrs[1]), p0);
        let v3 = sub(mol.pos(nbrs[2]), p0);
        let mut vol = dot(v1, cross(v2, v3));
        if vol.abs() <= ZERO_VOLUME_TOL && nbrs.len() == 4 {
            let v4 = sub(mol.pos(nbrs[3]), p0);
            vol = -dot(v1, cross(v2, v4));
        }
        mol.atom_mut(i).chiral_tag = if vol < -ZERO_VOLUME_TOL {
            ChiralTag::Cw
        } else if vol > ZERO_VOLUME_TOL {
            ChiralTag::Ccw
        } else {
            ChiralTag::Unspecified
        };
    }
}

/// Sets cis/trans stereo of non-ring (or ≥8-membered ring) double bonds from
/// the coordinates. Reference atoms are the lowest-index heavy neighbor
/// (or any neighbor) on each side.
pub fn assign_double_bond_stereo_from_3d(mol: &mut Molecule) {
    let ri = mol.rings().clone();
    for b in 0..mol.num_bonds() {
        let bond = mol.bond(b).clone();
        if bond.order != BondOrder::Double {
            continue;
        }
        if ri.is_bond_in_ring(b) && ri.min_bond_ring_size(b) < 8 {
            continue;
        }
        let pick = |a: usize, other: usize| -> Option<usize> {
            let mut c: Vec<usize> = mol.neighbors(a).filter(|&x| x != other).collect();
            if c.is_empty() {
                return None;
            }
            c.sort_by_key(|&x| (mol.atom(x).atomic_num == 1, x));
            Some(c[0])
        };
        let (Some(ra), Some(rb)) = (pick(bond.begin, bond.end), pick(bond.end, bond.begin)) else {
            continue;
        };
        if mol.degree(bond.begin) < 2 || mol.degree(bond.end) < 2 {
            continue;
        }
        let d = crate::core::geometry::dihedral(mol.pos(ra), mol.pos(bond.begin), mol.pos(bond.end), mol.pos(rb));
        let st = if d.abs() < std::f64::consts::FRAC_PI_2 { BondStereo::Cis } else { BondStereo::Trans };
        let bm = mol.bond_mut(b);
        bm.stereo = st;
        bm.stereo_atoms = Some((ra, rb));
    }
}

/// Non-tetrahedral coordination geometry perceived from 3D
/// (RDKit `assignNontetrahedralChiralTypeFrom3D`, tolerance 0.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonTetrahedralGeometry {
    SquarePlanar,
    TrigonalBipyramidal,
    Octahedral,
}

fn direction(from: [f64; 3], to: [f64; 3]) -> [f64; 3] {
    let r = sub(to, from);
    let l = dot(r, r).sqrt();
    [r[0] / l, r[1] / l, r[2] / l]
}

/// Port of the geometry classification in RDKit
/// `assignNontetrahedralChiralTypeFrom3D` (only the chiral *type*, not the
/// permutation), including the pre-checks done by `assignChiralTypesFrom3D`
/// (≥3 neighbours that affect chirality, ≤6 including hydrogens, Z ≥ 15).
pub fn nontetrahedral_geometry_from_3d(mol: &Molecule, i: usize) -> Option<NonTetrahedralGeometry> {
    use NonTetrahedralGeometry::*;
    let atom = mol.atom(i);
    let nz_degree = mol
        .atom_bonds(i)
        .iter()
        .filter(|&&(_, b)| {
            let bd = mol.bond(b);
            !(bd.order == BondOrder::Zero || (bd.order == BondOrder::Dative && bd.begin == i))
        })
        .count();
    let tnz = nz_degree + atom.total_hidden_hs() as usize;
    if nz_degree < 3 || tnz > 6 || atom.atomic_num < 15 {
        return None;
    }
    let cen = mol.pos(i);
    let v: Vec<[f64; 3]> = mol.neighbors(i).map(|n| direction(cen, mol.pos(n))).collect();
    let count = v.len();
    if !(3..=6).contains(&count) {
        return None;
    }
    let tolerance = 0.1;
    let mut pair = [0usize; 6];
    let mut pairs = 0;
    for a in 0..count {
        for b in a + 1..count {
            if dot(v[a], v[b]) < -(1.0 - tolerance) {
                if pair[a] != 0 || pair[b] != 0 {
                    return None;
                }
                pair[a] = b + 1;
                pair[b] = a + 1;
                pairs += 1;
            }
        }
    }
    let angle = |a: usize, b: usize| dot(v[a], v[b]).clamp(-1.0, 1.0).acos();
    match (pairs, count) {
        (1, 3) => Some(SquarePlanar),
        (1, 4) => {
            // see-saw: octahedral if the two unpaired bonds are < 100° apart
            let free: Vec<usize> = (0..4).filter(|&k| pair[k] == 0).collect();
            if angle(free[0], free[1]) < 100.0 * std::f64::consts::PI / 180.0 {
                Some(Octahedral)
            } else {
                Some(TrigonalBipyramidal)
            }
        }
        (1, 5) => Some(TrigonalBipyramidal),
        (2, 4) => Some(SquarePlanar),
        (2, 5) => Some(Octahedral),
        (3, 6) => Some(Octahedral),
        _ => None,
    }
}

/// Applies RDKit's hybridization rule for atoms carrying a non-tetrahedral
/// chiral tag perceived from 3D (`setHybridization`: square planar → SP2D,
/// trigonal bipyramidal → SP3D, octahedral → SP3D2, when the total degree
/// is in range). RDKit's molfile parser assigns these tags before
/// sanitization, so they determine the hybridization.
pub fn apply_nontetrahedral_hybridization_from_3d(mol: &mut Molecule) {
    use crate::core::Hybridization;
    for i in 0..mol.num_atoms() {
        let Some(g) = nontetrahedral_geometry_from_3d(mol, i) else { continue };
        let td = mol.total_degree(i);
        let h = match g {
            NonTetrahedralGeometry::SquarePlanar if (2..=4).contains(&td) => Hybridization::SP2D,
            NonTetrahedralGeometry::TrigonalBipyramidal if (2..=5).contains(&td) => Hybridization::SP3D,
            NonTetrahedralGeometry::Octahedral if (2..=6).contains(&td) => Hybridization::SP3D2,
            _ => continue,
        };
        mol.atom_mut(i).chiral_tag = ChiralTag::Unspecified;
        mol.atom_mut(i).hybridization = h;
    }
}
