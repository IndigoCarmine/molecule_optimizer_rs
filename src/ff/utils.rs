//! Shared helpers for force-field terms (RDKit `ForceFieldsHelper`).

use crate::core::geometry::{V3, cross, dot, norm, scale, sub};

pub const RAD2DEG: f64 = 180.0 / std::f64::consts::PI;
pub const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

#[inline]
pub fn p3(pos: &[f64], i: usize) -> V3 {
    [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]
}

#[inline]
pub fn add_grad(grad: &mut [f64], i: usize, g: V3) {
    grad[3 * i] += g[0];
    grad[3 * i + 1] += g[1];
    grad[3 * i + 2] += g[2];
}

/// RDKit `ForceField::distance` (indices ordered, squares summed x, y, z).
#[inline]
pub(crate) fn rd_distance(pos: &[f64], i: usize, j: usize) -> f64 {
    let (i, j) = if j < i { (j, i) } else { (i, j) };
    let mut res = 0.0;
    for k in 0..3 {
        let t = pos[3 * i + k] - pos[3 * j + k];
        res += t * t;
    }
    res.sqrt()
}

#[inline]
pub fn distance(pos: &[f64], i: usize, j: usize) -> f64 {
    norm(sub(p3(pos, i), p3(pos, j)))
}

/// Releases the unused capacity of `v` if it is at least 4 KiB (smaller
/// waste is not worth the reallocation): term storage is built by pushing,
/// which leaves up to half of a large vector unused.
pub fn shrink_vec<T>(v: &mut Vec<T>) {
    if (v.capacity() - v.len()) * std::mem::size_of::<T>() >= 4096 {
        v.shrink_to_fit();
    }
}

/// Bond-path (topological) distances from one atom to the atoms within
/// `depth` bonds (breadth-first search), recomputed row by row: the
/// builders classify atom pairs as 1-2 / 1-3 / 1-4 with O(n) memory
/// instead of an n×n matrix.
pub(crate) struct BondDistances {
    depth: u8,
    dist: Vec<u8>,
    touched: Vec<usize>,
    frontier: Vec<usize>,
    next: Vec<usize>,
}

impl BondDistances {
    pub(crate) fn new(n: usize, depth: u8) -> Self {
        assert!(depth < u8::MAX);
        BondDistances { depth, dist: vec![u8::MAX; n], touched: Vec::new(), frontier: Vec::new(), next: Vec::new() }
    }
    /// Computes the distances from atom `s` (replacing the previous row).
    pub(crate) fn compute(&mut self, mol: &crate::core::Molecule, s: usize) {
        for &v in &self.touched {
            self.dist[v] = u8::MAX;
        }
        self.touched.clear();
        self.dist[s] = 0;
        self.touched.push(s);
        self.frontier.clear();
        self.frontier.push(s);
        for d in 1..=self.depth {
            self.next.clear();
            for &u in &self.frontier {
                for v in mol.neighbors(u) {
                    if self.dist[v] == u8::MAX {
                        self.dist[v] = d;
                        self.touched.push(v);
                        self.next.push(v);
                    }
                }
            }
            std::mem::swap(&mut self.frontier, &mut self.next);
        }
    }
    /// Distance of atom `j` from the current atom (`u8::MAX` = more than
    /// `depth` bonds or not connected).
    #[inline]
    pub(crate) fn get(&self, j: usize) -> u8 {
        self.dist[j]
    }
    /// The atoms within `depth` bonds, the current atom first, in
    /// breadth-first order.
    pub(crate) fn within(&self) -> &[usize] {
        &self.touched
    }
}

/// RDKit `normalizeAngleDeg`: maps into [-180, 180].
pub fn normalize_angle_deg(mut a: f64) -> f64 {
    a %= 360.0;
    if a < -180.0 {
        a += 360.0;
    } else if a > 180.0 {
        a -= 360.0;
    }
    a
}

/// Result of [`compute_dihedral`]: `r`, `t`, `d` as in RDKit.
#[derive(Debug, Clone, Copy)]
pub struct DihedralData {
    pub dihedral: f64,
    pub cos_phi: f64,
    pub r: [V3; 4],
    pub t: [V3; 2],
    pub d: [f64; 2],
}

/// RDKit `ForceFieldsHelper::computeDihedral` (radians).
pub fn compute_dihedral(pos: &[f64], i1: usize, i2: usize, i3: usize, i4: usize) -> DihedralData {
    Torsion::at(pos, [i1, i2, i3, i4]).data()
}

pub fn compute_dihedral_pts(p1: V3, p2: V3, p3_: V3, p4: V3) -> DihedralData {
    Torsion::new(p1, p2, p3_, p4).data()
}

/// RDKit `isDoubleZero`.
#[inline]
pub(crate) fn is_double_zero(x: f64) -> bool {
    x < 1.0e-10 && x > -1.0e-10
}

/// Geometry of the torsion p1-p2-p3-p4, computed once and shared by the
/// energy and the gradient of every torsion-like term (MMFF and UFF
/// torsions, dihedral constraints). It is RDKit
/// `ForceFieldsHelper::computeDihedral` without the angle itself (see
/// [`Torsion::angle`]), plus the raw plane normals from which RDKit's
/// energy-side cosine (`calculateCosTorsion` / `calcTorsionCosPhi`) is
/// computed (see [`Torsion::cos_torsion`]).
#[derive(Debug, Clone, Copy)]
pub struct Torsion {
    /// `p1 - p2`, `p3 - p2`, `-(p3 - p2)`, `p4 - p3`
    pub r: [V3; 4],
    /// raw plane normals `r0 x r1`, `r2 x r3`
    pub n: [V3; 2],
    /// `|n|`
    pub n_len: [f64; 2],
    /// unit normals `n / d`
    pub t: [V3; 2],
    /// `max(|n|, 1e-5)`
    pub d: [f64; 2],
    /// `clamp(t0 . t1)`
    pub cos_phi: f64,
}

impl Torsion {
    #[inline(always)]
    pub fn new(p1: V3, p2: V3, p3_: V3, p4: V3) -> Self {
        let r0 = sub(p1, p2);
        let r1 = sub(p3_, p2);
        let r2 = scale(r1, -1.0);
        let r3 = sub(p4, p3_);
        let n0 = cross(r0, r1);
        let n1 = cross(r2, r3);
        let l0 = norm(n0);
        let l1 = norm(n1);
        // RDKit divides component-wise (`Point3D::operator/=`); multiplying by
        // the reciprocal would differ in the last bit.
        let d0 = l0.max(1.0e-5);
        let t0 = [n0[0] / d0, n0[1] / d0, n0[2] / d0];
        let d1 = l1.max(1.0e-5);
        let t1 = [n1[0] / d1, n1[1] / d1, n1[2] / d1];
        let cos_phi = dot(t0, t1).clamp(-1.0, 1.0);
        Torsion { r: [r0, r1, r2, r3], n: [n0, n1], n_len: [l0, l1], t: [t0, t1], d: [d0, d1], cos_phi }
    }

    #[inline(always)]
    pub fn at(pos: &[f64], idx: [usize; 4]) -> Self {
        Torsion::new(p3(pos, idx[0]), p3(pos, idx[1]), p3(pos, idx[2]), p3(pos, idx[3]))
    }

    /// Cosine as RDKit's torsion energies compute it (UFF
    /// `calculateCosTorsion`, MMFF `calcTorsionCosPhi`): `n0 . n1 / (|n0| |n1|)`
    /// clamped, 0 for a degenerate torsion. (RDKit's second normal is
    /// `(p2 - p3) x r3`, which differs from `n1` at most in the signs of zero
    /// components; that can only flip the sign of a zero cosine, and the
    /// torsion energies are the same for `+0` and `-0`.)
    #[inline(always)]
    pub fn cos_torsion(&self) -> f64 {
        let [l0, l1] = self.n_len;
        if is_double_zero(l0) || is_double_zero(l1) {
            return 0.0;
        }
        (dot(self.n[0], self.n[1]) / (l0 * l1)).clamp(-1.0, 1.0)
    }

    /// The dihedral angle in radians (the rest of RDKit `computeDihedral`).
    #[inline]
    pub fn angle(&self) -> f64 {
        let m = cross(self.t[0], self.r[1]);
        let ml = norm(m).max(1.0e-5);
        -(dot(m, self.t[1]) / ml).atan2(self.cos_phi)
    }

    /// [`DihedralData`] (with the angle).
    #[inline]
    pub fn data(&self) -> DihedralData {
        DihedralData { dihedral: self.angle(), cos_phi: self.cos_phi, r: self.r, t: self.t, d: self.d }
    }

    /// RDKit `calcTorsionGrad` (UFF and MMFF): adds `sin_term * dcos(phi)/dx`
    /// to the gradients of the atoms `idx`.
    #[inline(always)]
    pub fn add_grad(&self, grad: &mut [f64], idx: [usize; 4], sin_term: f64) {
        let (r, t, d, cos_phi) = (&self.r, &self.t, &self.d, self.cos_phi);
        let dcos_dt = [
            1.0 / d[0] * (t[1][0] - cos_phi * t[0][0]),
            1.0 / d[0] * (t[1][1] - cos_phi * t[0][1]),
            1.0 / d[0] * (t[1][2] - cos_phi * t[0][2]),
            1.0 / d[1] * (t[0][0] - cos_phi * t[1][0]),
            1.0 / d[1] * (t[0][1] - cos_phi * t[1][1]),
            1.0 / d[1] * (t[0][2] - cos_phi * t[1][2]),
        ];
        let (x, y, z) = (0, 1, 2);
        let a = 3 * idx[0];
        grad[a] += sin_term * (dcos_dt[2] * r[1][y] - dcos_dt[1] * r[1][z]);
        grad[a + 1] += sin_term * (dcos_dt[0] * r[1][z] - dcos_dt[2] * r[1][x]);
        grad[a + 2] += sin_term * (dcos_dt[1] * r[1][x] - dcos_dt[0] * r[1][y]);

        let b = 3 * idx[1];
        grad[b] += sin_term
            * (dcos_dt[1] * (r[1][z] - r[0][z])
                + dcos_dt[2] * (r[0][y] - r[1][y])
                + dcos_dt[4] * (-r[3][z])
                + dcos_dt[5] * (r[3][y]));
        grad[b + 1] += sin_term
            * (dcos_dt[0] * (r[0][z] - r[1][z])
                + dcos_dt[2] * (r[1][x] - r[0][x])
                + dcos_dt[3] * (r[3][z])
                + dcos_dt[5] * (-r[3][x]));
        grad[b + 2] += sin_term
            * (dcos_dt[0] * (r[1][y] - r[0][y])
                + dcos_dt[1] * (r[0][x] - r[1][x])
                + dcos_dt[3] * (-r[3][y])
                + dcos_dt[4] * (r[3][x]));

        let c = 3 * idx[2];
        grad[c] += sin_term
            * (dcos_dt[1] * (r[0][z])
                + dcos_dt[2] * (-r[0][y])
                + dcos_dt[4] * (r[3][z] - r[2][z])
                + dcos_dt[5] * (r[2][y] - r[3][y]));
        grad[c + 1] += sin_term
            * (dcos_dt[0] * (-r[0][z])
                + dcos_dt[2] * (r[0][x])
                + dcos_dt[3] * (r[2][z] - r[3][z])
                + dcos_dt[5] * (r[3][x] - r[2][x]));
        grad[c + 2] += sin_term
            * (dcos_dt[0] * (r[0][y])
                + dcos_dt[1] * (-r[0][x])
                + dcos_dt[3] * (r[3][y] - r[2][y])
                + dcos_dt[4] * (r[2][x] - r[3][x]));

        let e = 3 * idx[3];
        grad[e] += sin_term * (dcos_dt[4] * r[2][z] - dcos_dt[5] * r[2][y]);
        grad[e + 1] += sin_term * (dcos_dt[5] * r[2][x] - dcos_dt[3] * r[2][z]);
        grad[e + 2] += sin_term * (dcos_dt[3] * r[2][y] - dcos_dt[4] * r[2][x]);
    }
}

/// Out-of-plane geometry of the centre `j` with neighbours `i`, `k`, `l`,
/// computed once and shared by the energy and the gradient of MMFF
/// out-of-plane bending and UFF inversion: the bond vectors `i - j`,
/// `k - j`, `l - j`, their squared lengths and their lengths.
#[derive(Debug, Clone, Copy)]
pub struct OutOfPlane {
    pub r: [V3; 3],
    pub l2: [f64; 3],
    pub d: [f64; 3],
}

impl OutOfPlane {
    #[inline(always)]
    pub fn new(i: V3, j: V3, k: V3, l: V3) -> Self {
        let r = [sub(i, j), sub(k, j), sub(l, j)];
        let l2 = [dot(r[0], r[0]), dot(r[1], r[1]), dot(r[2], r[2])];
        OutOfPlane { r, l2, d: [l2[0].sqrt(), l2[1].sqrt(), l2[2].sqrt()] }
    }

    #[inline(always)]
    pub fn at(pos: &[f64], idx: [usize; 4]) -> Self {
        OutOfPlane::new(p3(pos, idx[0]), p3(pos, idx[1]), p3(pos, idx[2]), p3(pos, idx[3]))
    }

    /// Whether a bond length is (numerically) zero; RDKit's gradients then
    /// add nothing.
    #[inline(always)]
    pub fn is_degenerate(&self) -> bool {
        is_double_zero(self.d[0]) || is_double_zero(self.d[1]) || is_double_zero(self.d[2])
    }

    /// Unit bond vectors (component-wise division as in RDKit).
    #[inline(always)]
    pub fn unit(&self) -> [V3; 3] {
        let (r, d) = (&self.r, &self.d);
        [
            [r[0][0] / d[0], r[0][1] / d[0], r[0][2] / d[0]],
            [r[1][0] / d[1], r[1][1] / d[1], r[1][2] / d[1]],
            [r[2][0] / d[2], r[2][1] / d[2], r[2][2] / d[2]],
        ]
    }

    /// Gradient-side Wilson angle of RDKit's MMFF `OopBend` / UFF `Inversion`
    /// gradients, from the unit vectors `u`: `(n . u_jl, cos theta_ijk)`,
    /// both clamped to [-1, 1], with `n = (-u_ji) x u_jk` normalized.
    #[inline(always)]
    pub fn wilson(u: &[V3; 3]) -> (f64, f64) {
        let n = cross([-u[0][0], -u[0][1], -u[0][2]], u[1]);
        let nl = norm(n);
        let n = [n[0] / nl, n[1] / nl, n[2] / nl];
        (dot(n, u[2]).clamp(-1.0, 1.0), dot(u[0], u[1]).clamp(-1.0, 1.0))
    }

    /// Adds RDKit's out-of-plane gradient (MMFF `OopBendContrib::getGrad`,
    /// UFF `InversionContrib::getGrad`) for `de` = dE/d(angle), with
    /// `(sin_w, cos_w)` = (`sinChi`, `cosChi`) in MMFF and (`cosY`, `sinY`)
    /// in UFF.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn add_grad(&self, grad: &mut [f64], idx: [usize; 4], u: &[V3; 3], de: f64, sin_w: f64, cos_w: f64, cos_theta: f64, sin_theta: f64, sin_theta_sq: f64) {
        let [rji, rjk, rjl] = *u;
        let [dji, djk, djl] = self.d;
        let t1 = cross(rjl, rjk);
        let t2 = cross(rji, rjl);
        let t3 = cross(rjk, rji);
        let term1 = cos_w * sin_theta;
        let term2 = sin_w / (cos_w * sin_theta_sq);
        let mut tg1 = [0.0; 3];
        let mut tg3 = [0.0; 3];
        let mut tg4 = [0.0; 3];
        for k in 0..3 {
            tg1[k] = (t1[k] / term1 - (rji[k] - rjk[k] * cos_theta) * term2) / dji;
            tg3[k] = (t2[k] / term1 - (rjk[k] - rji[k] * cos_theta) * term2) / djk;
            tg4[k] = (t3[k] / term1 - rjl[k] * sin_w / cos_w) / djl;
        }
        let [i1, i2, i3, i4] = idx;
        for k in 0..3 {
            grad[3 * i1 + k] += de * tg1[k];
            grad[3 * i2 + k] += -de * (tg1[k] + tg3[k] + tg4[k]);
            grad[3 * i3 + k] += de * tg3[k];
            grad[3 * i4 + k] += de * tg4[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rdkit_dihedral_matches_iupac() {
        let pts = [[1.0, 0.2, 0.1], [0.0, 0.0, 0.0], [0.1, -0.2, 1.4], [-0.7, 0.9, 1.9]];
        let pos: Vec<f64> = pts.iter().flatten().copied().collect();
        let a = compute_dihedral(&pos, 0, 1, 2, 3).dihedral;
        let b = crate::core::geometry::dihedral(pts[0], pts[1], pts[2], pts[3]);
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
}
