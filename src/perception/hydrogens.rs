//! Adding hidden hydrogens as explicit atoms with reasonable coordinates.

use crate::core::geometry::{V3, add, cross, dot, norm, normalize, scale, sub};
use crate::core::{Atom, BondOrder, Hybridization, Molecule};

fn xh_length(anum: u8) -> f64 {
    match anum {
        6 => 1.09,
        7 => 1.01,
        8 => 0.96,
        16 => 1.34,
        15 => 1.42,
        5 => 1.19,
        14 => 1.48,
        _ => crate::core::element(anum).r_cov + 0.31,
    }
}

fn any_perp(u: V3) -> V3 {
    let t = if u[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    normalize(cross(u, t))
}

fn rotate(v: V3, axis: V3, theta: f64) -> V3 {
    crate::core::geometry::rotate_about_axis(v, [0.0; 3], axis, theta)
}

/// Directions for `nh` new bonds on atom `i`.
fn new_directions(mol: &Molecule, i: usize, nh: usize) -> Vec<V3> {
    let c = mol.pos(i);
    let nbrs: Vec<usize> = mol.neighbors(i).collect();
    let us: Vec<V3> = nbrs.iter().map(|&n| normalize(sub(mol.pos(n), c))).collect();
    let total = nbrs.len() + nh;
    let hyb = match mol.atom(i).hybridization {
        Hybridization::SP => 2,
        Hybridization::SP2 => 3,
        Hybridization::SP3 => 4,
        _ => total.clamp(1, 4),
    }
    .max(total.min(4));
    let tet = 109.471_f64.to_radians();
    let mut out = Vec::new();
    match (hyb, us.len()) {
        (_, 0) => {
            let base = [[1.0, 0.0, 0.0], [-0.333, 0.943, 0.0], [-0.333, -0.471, 0.816], [-0.333, -0.471, -0.816]];
            let tri = [[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0]];
            for k in 0..nh {
                out.push(match hyb {
                    2 => scale([1.0, 0.0, 0.0], if k == 0 { 1.0 } else { -1.0 }),
                    3 => tri[k % 3],
                    _ => base[k % 4],
                });
            }
        }
        (2, _) => {
            out.push(scale(us[0], -1.0));
        }
        (3, 1) => {
            // plane from a neighbor's neighbor if present
            let n0 = nbrs[0];
            let ref_dir = mol
                .neighbors(n0)
                .find(|&x| x != i)
                .map(|x| normalize(sub(mol.pos(x), mol.pos(n0))))
                .unwrap_or_else(|| any_perp(us[0]));
            let mut perp = sub(ref_dir, scale(us[0], dot(ref_dir, us[0])));
            if norm(perp) < 1e-6 {
                perp = any_perp(us[0]);
            }
            let perp = normalize(perp);
            let a = 120f64.to_radians();
            // place trans to ref first
            out.push(normalize(add(scale(us[0], a.cos()), scale(perp, -a.sin()))));
            out.push(normalize(add(scale(us[0], a.cos()), scale(perp, a.sin()))));
        }
        (3, _) => {
            out.push(normalize(scale(add(us[0], us[1]), -1.0)));
        }
        (_, 1) => {
            let perp = any_perp(us[0]);
            let first = normalize(add(scale(us[0], tet.cos()), scale(perp, tet.sin())));
            for k in 0..3 {
                out.push(rotate(first, us[0], 2.0 * std::f64::consts::PI / 3.0 * k as f64));
            }
        }
        (_, 2) => {
            let b = normalize(scale(add(us[0], us[1]), -1.0));
            let mut nrm = cross(us[0], us[1]);
            if norm(nrm) < 1e-6 {
                nrm = any_perp(us[0]);
            }
            let nrm = normalize(nrm);
            let half = tet / 2.0;
            out.push(normalize(add(scale(b, half.cos()), scale(nrm, half.sin()))));
            out.push(normalize(add(scale(b, half.cos()), scale(nrm, -half.sin()))));
        }
        _ => {
            let s = us.iter().fold([0.0; 3], |a, &u| add(a, u));
            let d = if norm(s) < 1e-6 { any_perp(us[0]) } else { normalize(scale(s, -1.0)) };
            out.push(d);
        }
    }
    while out.len() < nh {
        // fall back: random-ish distinct directions
        let k = out.len() as f64;
        out.push(normalize([k.cos(), k.sin(), 0.3 * k]));
    }
    out.truncate(nh);
    out
}

/// Converts hidden hydrogens (implicit + explicit counts) into H atoms and
/// places them geometrically. Returns the number of hydrogens added.
/// Run [`crate::perception::sanitize`] first so hybridization is known.
pub fn add_hydrogens(mol: &mut Molecule) -> usize {
    let n0 = mol.num_atoms();
    let mut added = 0;
    for i in 0..n0 {
        let nh = mol.atom(i).total_hidden_hs() as usize;
        if nh == 0 {
            continue;
        }
        let dirs = new_directions(mol, i, nh);
        let l = xh_length(mol.atom(i).atomic_num);
        let c = mol.pos(i);
        {
            let a = mol.atom_mut(i);
            a.implicit_hs = 0;
            a.num_explicit_hs = 0;
            a.no_implicit = true;
        }
        for d in dirs {
            let mut h = Atom::new(1);
            h.no_implicit = true;
            h.hybridization = Hybridization::S;
            let hi = mol.add_atom(h, add(c, scale(d, l)));
            mol.add_bond(i, hi, BondOrder::Single);
            added += 1;
        }
    }
    for a in mol.atoms_mut() {
        a.no_implicit = true;
    }
    if added > 0 {
        mol.ring_info = None;
        mol.rings();
    }
    added
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn methane_from_c() {
        let mut m = Molecule::new();
        m.add_atom(Atom::new(6), [0.0; 3]);
        crate::perception::sanitize(&mut m).unwrap();
        assert_eq!(m.atom(0).implicit_hs, 4);
        assert_eq!(add_hydrogens(&mut m), 4);
        assert_eq!(m.num_atoms(), 5);
        for i in 1..5 {
            for j in i + 1..5 {
                let a = crate::core::geometry::angle(m.pos(i), m.pos(0), m.pos(j)).to_degrees();
                assert!((a - 109.47).abs() < 1.0, "{a}");
            }
        }
    }
}
