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

#[inline]
pub fn distance(pos: &[f64], i: usize, j: usize) -> f64 {
    norm(sub(p3(pos, i), p3(pos, j)))
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
    compute_dihedral_pts(p3(pos, i1), p3(pos, i2), p3(pos, i3), p3(pos, i4))
}

pub fn compute_dihedral_pts(p1: V3, p2: V3, p3_: V3, p4: V3) -> DihedralData {
    let r0 = sub(p1, p2);
    let r1 = sub(p3_, p2);
    let r2 = scale(r1, -1.0);
    let r3 = sub(p4, p3_);
    // RDKit divides component-wise (`Point3D::operator/=`); multiplying by
    // the reciprocal would differ in the last bit.
    let t0 = cross(r0, r1);
    let d0 = norm(t0).max(1.0e-5);
    let t0 = [t0[0] / d0, t0[1] / d0, t0[2] / d0];
    let t1 = cross(r2, r3);
    let d1 = norm(t1).max(1.0e-5);
    let t1 = [t1[0] / d1, t1[1] / d1, t1[2] / d1];
    let cos_phi = dot(t0, t1).clamp(-1.0, 1.0);
    let m = cross(t0, r1);
    let ml = norm(m).max(1.0e-5);
    let dihedral = -(dot(m, t1) / ml).atan2(cos_phi);
    DihedralData { dihedral, cos_phi, r: [r0, r1, r2, r3], t: [t0, t1], d: [d0, d1] }
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
