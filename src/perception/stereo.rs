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
