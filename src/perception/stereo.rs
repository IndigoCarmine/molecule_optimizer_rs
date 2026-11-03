//! Stereochemistry from 3D coordinates, or from the wedges of a 2D drawing.

use crate::core::geometry::{cross, dot, sub};
use crate::core::{BondDir, BondOrder, BondStereo, ChiralTag, Molecule};

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

/// Sets the stereo a 2D drawing shows: tetrahedral chiral tags from its
/// wedges ([`crate::core::Bond::dir`]) and cis/trans from the drawn double
/// bonds, in the same convention as [`assign_chiral_tags_from_3d`] and
/// [`assign_double_bond_stereo_from_3d`], so the embedder reproduces the
/// stereoisomer drawn.
///
/// A centre counts only when a wedge starts at it (its narrow end, the
/// bond's `begin`); centres without one, and those with a wavy bond, are
/// left unspecified. Hydrogens may be implicit or explicit: an explicit,
/// unwedged hydrogen on a centre with three other neighbours is taken to
/// point away from them, wherever
/// [`crate::perception::hydrogens::add_hydrogens`] placed it in the plane.
/// Double bonds drawn crossed, or next to a wavy bond, are left unspecified.
pub fn assign_stereo_from_2d(mol: &mut Molecule) {
    for i in 0..mol.num_atoms() {
        let tag = wedge_chirality(mol, i);
        mol.atom_mut(i).chiral_tag = tag;
    }
    assign_double_bond_stereo_from_3d(mol);
    for b in 0..mol.num_bonds() {
        let bond = mol.bond(b);
        if bond.stereo == BondStereo::None {
            continue;
        }
        let wavy = bond.dir == BondDir::Either
            || [bond.begin, bond.end].iter().any(|&a| {
                mol.atom_bonds(a).iter().any(|&(_, ob)| ob != b && mol.bond(ob).dir == BondDir::Either)
            });
        if wavy {
            let bm = mol.bond_mut(b);
            bm.stereo = BondStereo::None;
            bm.stereo_atoms = None;
        }
    }
}

/// The chiral tag the wedges at atom `i` give it (see [`assign_stereo_from_2d`]).
fn wedge_chirality(mol: &Molecule, i: usize) -> ChiralTag {
    let bonds = mol.atom_bonds(i);
    let deg = bonds.len();
    let atom = mol.atom(i);
    if !(3..=4).contains(&deg) || atom.is_aromatic || atom.atomic_num == 1 {
        return ChiralTag::Unspecified;
    }
    if deg + atom.total_hidden_hs() as usize > 4 {
        return ChiralTag::Unspecified;
    }
    if !bonds.iter().all(|&(_, b)| matches!(mol.bond(b).order, BondOrder::Single | BondOrder::Dative)) {
        return ChiralTag::Unspecified;
    }
    // The wedge's height for each neighbour: only wedges starting at `i` count.
    let mut heights = Vec::with_capacity(deg);
    for &(_, b) in bonds {
        let bond = mol.bond(b);
        let h = if bond.begin != i {
            0.0
        } else {
            match bond.dir {
                BondDir::Up => 1.0,
                BondDir::Down => -1.0,
                BondDir::None => 0.0,
                BondDir::Either => return ChiralTag::Unspecified,
            }
        };
        heights.push(h);
    }
    if heights.iter().all(|&h| h == 0.0) {
        return ChiralTag::Unspecified;
    }
    let c = mol.pos(i);
    let mut v: Vec<[f64; 3]> = bonds
        .iter()
        .zip(&heights)
        .map(|(&(n, _), &h)| {
            let p = mol.pos(n);
            let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
            let l = (dx * dx + dy * dy).sqrt().max(1e-8);
            [dx / l, dy / l, h]
        })
        .collect();
    if deg == 4 {
        // A single unwedged explicit H: it points away from the three others.
        let hs: Vec<usize> = (0..4).filter(|&k| mol.atom(bonds[k].0).atomic_num == 1 && heights[k] == 0.0).collect();
        if let [k] = hs[..] {
            let mut o = [0.0; 3];
            for (m, w) in v.iter().enumerate() {
                if m != k {
                    for d in 0..3 {
                        o[d] -= w[d];
                    }
                }
            }
            v[k] = o;
        }
    }
    let mut vol = dot(v[0], cross(v[1], v[2]));
    if vol.abs() <= 1e-3 && deg == 4 {
        vol = -dot(v[0], cross(v[1], v[3]));
    }
    if vol > 1e-3 {
        ChiralTag::Ccw
    } else if vol < -1e-3 {
        ChiralTag::Cw
    } else {
        ChiralTag::Unspecified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Atom;

    /// CHFClBr drawn flat with the Br bond wedged `dir`; the H implicit.
    fn drawn(dir: BondDir) -> Molecule {
        let mut m = Molecule::new();
        let c = m.add_atom(Atom::new(6), [0.0, 0.0, 0.0]);
        let f = m.add_atom(Atom::new(9), [1.0, 0.0, 0.0]);
        let cl = m.add_atom(Atom::new(17), [-0.5, 0.87, 0.0]);
        let br = m.add_atom(Atom::new(35), [-0.5, -0.87, 0.0]);
        m.add_bond(c, f, BondOrder::Single);
        m.add_bond(c, cl, BondOrder::Single);
        let b = m.add_bond(c, br, BondOrder::Single);
        m.bond_mut(b).dir = dir;
        crate::perception::sanitize(&mut m).unwrap();
        m
    }

    #[test]
    fn up_and_down_wedges_give_opposite_tags() {
        let mut up = drawn(BondDir::Up);
        let mut down = drawn(BondDir::Down);
        assign_stereo_from_2d(&mut up);
        assign_stereo_from_2d(&mut down);
        assert_ne!(up.atom(0).chiral_tag, ChiralTag::Unspecified);
        assert_ne!(down.atom(0).chiral_tag, ChiralTag::Unspecified);
        assert_ne!(up.atom(0).chiral_tag, down.atom(0).chiral_tag);
    }

    #[test]
    fn unwedged_or_wavy_centres_stay_unspecified() {
        for dir in [BondDir::None, BondDir::Either] {
            let mut m = drawn(dir);
            assign_stereo_from_2d(&mut m);
            assert_eq!(m.atom(0).chiral_tag, ChiralTag::Unspecified, "{dir:?}");
        }
    }

    /// The tag from the wedges is the one the matching 3D geometry gives,
    /// with the hydrogen implicit or added as an atom.
    #[test]
    fn wedge_tag_matches_the_3d_tag() {
        for dir in [BondDir::Up, BondDir::Down] {
            let z = if dir == BondDir::Up { 0.9 } else { -0.9 };

            let mut with_h = drawn(dir);
            crate::perception::hydrogens::add_hydrogens(&mut with_h);
            assign_stereo_from_2d(&mut with_h);
            // Br lifted (or lowered), the H on the other side.
            let mut real = with_h.clone();
            real.coords[3][2] = z;
            real.coords[4] = [0.0, 0.0, -z];
            assign_chiral_tags_from_3d(&mut real);
            assert_eq!(with_h.atom(0).chiral_tag, real.atom(0).chiral_tag, "{dir:?}");

            // Three neighbours: the embedder takes the centre as the fourth point.
            let mut flat = drawn(dir);
            assign_stereo_from_2d(&mut flat);
            let mut p = flat.coords.clone();
            p[3][2] = z;
            let v = |k: usize| sub(p[k], p[0]);
            let vol = dot(v(1), cross(v(2), v(3)));
            let want = if vol > 0.0 { ChiralTag::Ccw } else { ChiralTag::Cw };
            assert_eq!(flat.atom(0).chiral_tag, want, "{dir:?}");
        }
    }

    #[test]
    fn drawn_double_bonds_get_cis_trans_unless_crossed() {
        // 2-butene drawn trans
        let mut m = Molecule::new();
        let a = m.add_atom(Atom::new(6), [-1.0, 0.8, 0.0]);
        let b = m.add_atom(Atom::new(6), [-0.5, 0.0, 0.0]);
        let c = m.add_atom(Atom::new(6), [0.5, 0.0, 0.0]);
        let d = m.add_atom(Atom::new(6), [1.0, -0.8, 0.0]);
        m.add_bond(a, b, BondOrder::Single);
        let db = m.add_bond(b, c, BondOrder::Double);
        m.add_bond(c, d, BondOrder::Single);
        crate::perception::sanitize(&mut m).unwrap();
        let mut trans = m.clone();
        assign_stereo_from_2d(&mut trans);
        assert_eq!(trans.bond(db).stereo, BondStereo::Trans);
        let mut crossed = m.clone();
        crossed.bond_mut(db).dir = BondDir::Either;
        assign_stereo_from_2d(&mut crossed);
        assert_eq!(crossed.bond(db).stereo, BondStereo::None);
    }
}
