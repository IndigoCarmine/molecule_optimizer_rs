//! Periodic unit cells (orthorhombic and triclinic).
//!
//! A [`UnitCell`] is defined by its three box vectors `a`, `b`, `c` (Å).
//! Fractional coordinates `s` and Cartesian coordinates `r` are related by
//! `r = s_a a + s_b b + s_c c`.
//!
//! Conventions used by the periodic force-field paths ([`crate::fast`]):
//!
//! * Bonded terms are evaluated with the coordinates exactly as given, so
//!   the positions passed to a force field must keep every molecule
//!   *whole* (bonded atoms close to each other, not split across the box
//!   boundary). Use [`UnitCell::make_molecules_whole`] after reading a
//!   wrapped structure.
//! * Nonbonded interactions use the minimum-image convention, which is
//!   well defined for interaction cutoffs up to half the shortest
//!   perpendicular width of the cell ([`UnitCell::max_cutoff`]).

use super::molecule::Molecule;
use crate::error::{Error, Result};

type V3 = [f64; 3];

#[inline]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline]
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// A periodic simulation cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitCell {
    /// Box vectors `a`, `b`, `c` (Å).
    vectors: [V3; 3],
    /// Reciprocal vectors (rows of the inverse box matrix, without 2π):
    /// `s_k = recip[k] · r`.
    recip: [V3; 3],
    volume: f64,
}

impl UnitCell {
    /// Rectangular box with edge lengths `a`, `b`, `c` along x, y, z.
    pub fn orthorhombic(a: f64, b: f64, c: f64) -> Result<Self> {
        Self::from_vectors([a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c])
    }

    /// Cubic box with edge length `a`.
    pub fn cubic(a: f64) -> Result<Self> {
        Self::orthorhombic(a, a, a)
    }

    /// Cell from lengths (Å) and angles (degrees; `alpha` = ∠(b,c),
    /// `beta` = ∠(a,c), `gamma` = ∠(a,b)) in the standard (PDB) orientation:
    /// `a` along x, `b` in the xy plane.
    pub fn from_parameters(a: f64, b: f64, c: f64, alpha: f64, beta: f64, gamma: f64) -> Result<Self> {
        if !(a > 0.0 && b > 0.0 && c > 0.0) {
            return Err(Error::InvalidArgument(format!("cell lengths must be positive ({a}, {b}, {c})")));
        }
        let (ca, cb) = (alpha.to_radians().cos(), beta.to_radians().cos());
        let (cg, sg) = (gamma.to_radians().cos(), gamma.to_radians().sin());
        // exact zeros for right angles keep orthorhombic cells exactly diagonal
        let fix = |x: f64| if x.abs() < 1e-15 { 0.0 } else { x };
        let (ca, cb, cg) = (fix(ca), fix(cb), fix(cg));
        if sg.abs() < 1e-8 {
            return Err(Error::InvalidArgument("degenerate cell (gamma = 0 or 180 degrees)".into()));
        }
        let cy = (ca - cb * cg) / sg;
        let cz2 = 1.0 - cb * cb - cy * cy;
        if cz2 <= 1e-12 {
            return Err(Error::InvalidArgument(format!("invalid cell angles ({alpha}, {beta}, {gamma})")));
        }
        Self::from_vectors([a, 0.0, 0.0], [b * cg, b * sg, 0.0], [c * cb, c * cy, c * cz2.sqrt()])
    }

    /// Cell from box vectors (must be right-handed and non-degenerate).
    pub fn from_vectors(a: V3, b: V3, c: V3) -> Result<Self> {
        let volume = dot(a, cross(b, c));
        if !volume.is_finite() || volume <= 1e-9 {
            return Err(Error::InvalidArgument(format!(
                "box vectors must be right-handed and span a non-zero volume (det = {volume})"
            )));
        }
        let ra = cross(b, c).map(|x| x / volume);
        let rb = cross(c, a).map(|x| x / volume);
        let rc = cross(a, b).map(|x| x / volume);
        Ok(UnitCell { vectors: [a, b, c], recip: [ra, rb, rc], volume })
    }

    /// Box vectors `[a, b, c]`.
    pub fn vectors(&self) -> [V3; 3] {
        self.vectors
    }
    /// Reciprocal vectors `[a*, b*, c*]` (rows of the inverse box matrix,
    /// without the factor 2π): `a*·a = 1`, `a*·b = 0`, ...
    pub fn reciprocal(&self) -> [V3; 3] {
        self.recip
    }
    pub fn volume(&self) -> f64 {
        self.volume
    }
    /// Edge lengths `|a|, |b|, |c|`.
    pub fn lengths(&self) -> V3 {
        self.vectors.map(norm)
    }
    /// Angles `(alpha, beta, gamma)` in degrees.
    pub fn angles(&self) -> V3 {
        let [a, b, c] = self.vectors;
        let ang = |u: V3, v: V3| (dot(u, v) / (norm(u) * norm(v))).clamp(-1.0, 1.0).acos().to_degrees();
        [ang(b, c), ang(a, c), ang(a, b)]
    }
    /// `true` if the box vectors are along x, y and z.
    pub fn is_orthorhombic(&self) -> bool {
        let [a, b, c] = self.vectors;
        a[1] == 0.0 && a[2] == 0.0 && b[0] == 0.0 && b[2] == 0.0 && c[0] == 0.0 && c[1] == 0.0
    }
    /// Distances between opposite faces (`V/|b×c|`, `V/|c×a|`, `V/|a×b|`).
    pub fn perpendicular_widths(&self) -> V3 {
        self.recip.map(|r| 1.0 / norm(r))
    }
    /// Shortest perpendicular width.
    pub fn min_perpendicular_width(&self) -> f64 {
        let w = self.perpendicular_widths();
        w[0].min(w[1]).min(w[2])
    }
    /// Largest cutoff for which the minimum-image convention is unique
    /// (half the shortest perpendicular width).
    pub fn max_cutoff(&self) -> f64 {
        0.5 * self.min_perpendicular_width()
    }

    /// Fractional coordinates of a Cartesian point.
    #[inline]
    pub fn to_fractional(&self, r: V3) -> V3 {
        [dot(self.recip[0], r), dot(self.recip[1], r), dot(self.recip[2], r)]
    }
    /// Cartesian coordinates of a fractional point.
    #[inline]
    pub fn to_cartesian(&self, s: V3) -> V3 {
        let [a, b, c] = self.vectors;
        [
            s[0] * a[0] + s[1] * b[0] + s[2] * c[0],
            s[0] * a[1] + s[1] * b[1] + s[2] * c[1],
            s[0] * a[2] + s[1] * b[2] + s[2] * c[2],
        ]
    }

    /// The image of `r` inside the home cell (fractional coordinates in `[0, 1)`).
    pub fn wrap_point(&self, r: V3) -> V3 {
        let s = self.to_fractional(r);
        let t = self.to_cartesian(s.map(|x| -x.floor()));
        [r[0] + t[0], r[1] + t[1], r[2] + t[2]]
    }

    /// Wraps every point into the home cell (molecules may be split).
    pub fn wrap_coords(&self, coords: &mut [V3]) {
        for p in coords {
            *p = self.wrap_point(*p);
        }
    }

    /// Minimum-image version of the difference vector `d`: the shortest
    /// vector `d + n_a a + n_b b + n_c c` over integer `n`.
    pub fn min_image(&self, d: V3) -> V3 {
        let s = self.to_fractional(d);
        let base = self.to_cartesian(s.map(|x| x - x.round()));
        // Any vector shorter than half the shortest perpendicular width is
        // the unique minimum image (another image differs by a lattice
        // vector, whose length is at least that width).
        let wmin = self.min_perpendicular_width();
        let l2 = dot(base, base);
        if 4.0 * l2 < wmin * wmin {
            return base;
        }
        // Skewed cells: search the neighboring images.
        let [a, b, c] = self.vectors;
        let mut best = base;
        let mut best2 = l2;
        for i in -1i32..=1 {
            for j in -1i32..=1 {
                for k in -1i32..=1 {
                    let (fi, fj, fk) = (i as f64, j as f64, k as f64);
                    let v = [
                        base[0] + fi * a[0] + fj * b[0] + fk * c[0],
                        base[1] + fi * a[1] + fj * b[1] + fk * c[1],
                        base[2] + fi * a[2] + fj * b[2] + fk * c[2],
                    ];
                    let v2 = dot(v, v);
                    if v2 < best2 {
                        best2 = v2;
                        best = v;
                    }
                }
            }
        }
        best
    }

    /// Minimum-image distance between two points.
    pub fn distance(&self, p: V3, q: V3) -> f64 {
        norm(self.min_image([p[0] - q[0], p[1] - q[1], p[2] - q[2]]))
    }

    /// Makes every molecule (connected component of `mol`'s bond graph)
    /// whole: starting from the lowest-index atom of each fragment, every
    /// bonded neighbor is moved to the image closest to the atom it is
    /// reached from (breadth-first traversal). The first atom of each
    /// fragment keeps its position.
    pub fn make_molecules_whole(&self, mol: &Molecule, coords: &mut [V3]) {
        let n = mol.num_atoms();
        assert_eq!(coords.len(), n, "one position per atom expected");
        let mut seen = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        for s in 0..n {
            if seen[s] {
                continue;
            }
            seen[s] = true;
            queue.push_back(s);
            while let Some(u) = queue.pop_front() {
                for v in mol.neighbors(u) {
                    if !seen[v] {
                        seen[v] = true;
                        let (pu, pv) = (coords[u], coords[v]);
                        let d = self.min_image([pv[0] - pu[0], pv[1] - pu[1], pv[2] - pu[2]]);
                        coords[v] = [pu[0] + d[0], pu[1] + d[1], pu[2] + d[2]];
                        queue.push_back(v);
                    }
                }
            }
        }
    }

    /// Makes molecules whole and translates each one (by a lattice vector)
    /// so that its geometric center lies in the home cell.
    pub fn wrap_molecules(&self, mol: &Molecule, coords: &mut [V3]) {
        self.make_molecules_whole(mol, coords);
        for frag in mol.fragments() {
            let mut c = [0.0; 3];
            for &i in &frag {
                for d in 0..3 {
                    c[d] += coords[i][d];
                }
            }
            let inv = 1.0 / frag.len() as f64;
            let c = c.map(|x| x * inv);
            let w = self.wrap_point(c);
            let t = [w[0] - c[0], w[1] - c[1], w[2] - c[2]];
            for &i in &frag {
                for d in 0..3 {
                    coords[i][d] += t[d];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, BondOrder};

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn parameters_round_trip() {
        let c = UnitCell::from_parameters(10.0, 12.0, 14.0, 80.0, 95.0, 105.0).unwrap();
        let l = c.lengths();
        let a = c.angles();
        for (x, y) in l.iter().zip([10.0, 12.0, 14.0]) {
            assert!(close(*x, y, 1e-12));
        }
        for (x, y) in a.iter().zip([80.0, 95.0, 105.0]) {
            assert!(close(*x, y, 1e-10), "{x} vs {y}");
        }
        let o = UnitCell::from_parameters(10.0, 12.0, 14.0, 90.0, 90.0, 90.0).unwrap();
        assert!(o.is_orthorhombic());
        assert!(close(o.volume(), 1680.0, 1e-12));
        assert_eq!(o.perpendicular_widths(), [10.0, 12.0, 14.0]);
        assert!(UnitCell::from_vectors([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]).is_err());
    }

    #[test]
    fn fractional_and_wrapping() {
        let c = UnitCell::from_parameters(9.0, 11.0, 13.0, 70.0, 100.0, 115.0).unwrap();
        let p = [23.1, -17.2, 5.5];
        let s = c.to_fractional(p);
        let q = c.to_cartesian(s);
        for d in 0..3 {
            assert!(close(p[d], q[d], 1e-12));
        }
        let w = c.wrap_point(p);
        let sw = c.to_fractional(w);
        assert!(sw.iter().all(|&x| (-1e-12..1.0 + 1e-12).contains(&x)));
        for d in 0..3 {
            let k = s[d] - sw[d];
            assert!((k - k.round()).abs() < 1e-9);
        }
    }

    #[test]
    fn minimum_image_is_shortest() {
        let c = UnitCell::from_parameters(10.0, 10.0, 10.0, 60.0, 60.0, 60.0).unwrap();
        let [a, b, cc] = c.vectors();
        let mut rng = 12345u64;
        let mut rnd = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..200 {
            let d = [30.0 * (rnd() - 0.5), 30.0 * (rnd() - 0.5), 30.0 * (rnd() - 0.5)];
            let m = c.min_image(d);
            let ml = norm(m);
            // brute force over a wide range of images
            let mut best = f64::INFINITY;
            for i in -4..=4 {
                for j in -4..=4 {
                    for k in -4..=4 {
                        let (i, j, k) = (i as f64, j as f64, k as f64);
                        let v = [
                            d[0] + i * a[0] + j * b[0] + k * cc[0],
                            d[1] + i * a[1] + j * b[1] + k * cc[1],
                            d[2] + i * a[2] + j * b[2] + k * cc[2],
                        ];
                        best = best.min(norm(v));
                    }
                }
            }
            assert!(close(ml, best, 1e-12), "{ml} vs {best}");
        }
    }

    #[test]
    fn whole_molecules() {
        let c = UnitCell::cubic(10.0).unwrap();
        let mut m = Molecule::new();
        let o = m.add_atom(Atom::new(8), [9.8, 5.0, 9.9]);
        let h1 = m.add_atom(Atom::new(1), [0.6, 5.0, 9.9]);
        let h2 = m.add_atom(Atom::new(1), [9.6, 5.9, 0.1]);
        m.add_bond(o, h1, BondOrder::Single);
        m.add_bond(o, h2, BondOrder::Single);
        let mut x = m.coords.clone();
        c.make_molecules_whole(&m, &mut x);
        assert!(close(x[1][0], 10.6, 1e-12));
        assert!(close(x[2][2], 10.1, 1e-12));
        c.wrap_molecules(&m, &mut x);
        let mut cen = [0.0; 3];
        for p in &x {
            for d in 0..3 {
                cen[d] += p[d] / 3.0;
            }
        }
        assert!(cen.iter().all(|&v| (0.0..10.0).contains(&v)));
        let d01 = norm([x[0][0] - x[1][0], x[0][1] - x[1][1], x[0][2] - x[1][2]]);
        assert!(close(d01, 0.8, 1e-12));
    }
}
