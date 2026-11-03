//! Minimal 3D point type whose arithmetic follows RDKit's `RDGeom::Point3D`
//! operation by operation (same evaluation order, division instead of
//! multiplication by a reciprocal), so that results are bit-identical.
//!
//! Port of parts of RDKit `Code/Geometry/point.{h,cpp}`.
//! RDKit: Copyright (C) 2003-2026 Greg Landrum and other RDKit contributors,
//! BSD-3-Clause.

use std::ops::{Add, Div, DivAssign, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct P3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl P3 {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        P3 { x, y, z }
    }
    /// Point `i` of a flat coordinate array with dimension `dim` (first three
    /// components only).
    #[inline]
    pub fn from_pos(pos: &[f64], i: usize, dim: usize) -> Self {
        P3 { x: pos[dim * i], y: pos[dim * i + 1], z: pos[dim * i + 2] }
    }
    #[inline]
    pub fn length_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    #[inline]
    pub fn length(&self) -> f64 {
        self.length_sq().sqrt()
    }
    #[inline]
    pub fn dot(&self, o: &P3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    #[inline]
    pub fn cross(&self, o: &P3) -> P3 {
        P3 { x: self.y * o.z - self.z * o.y, y: -self.x * o.z + self.z * o.x, z: self.x * o.y - self.y * o.x }
    }
    /// RDKit `normalize` (no-op on zero vectors instead of throwing).
    #[inline]
    pub fn normalize(&mut self) {
        let l = self.length();
        if l < 1e-16 {
            return;
        }
        self.x /= l;
        self.y /= l;
        self.z /= l;
    }
    /// RDKit `angleTo`.
    pub fn angle_to(&self, o: &P3) -> f64 {
        let lsq = self.length_sq() * o.length_sq();
        let mut dp = self.dot(o);
        dp /= lsq.sqrt();
        if dp <= -1.0 {
            return std::f64::consts::PI;
        }
        if dp >= 1.0 {
            return 0.0;
        }
        dp.acos()
    }
}

impl Add for P3 {
    type Output = P3;
    #[inline]
    fn add(self, o: P3) -> P3 {
        P3 { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }
}
impl Sub for P3 {
    type Output = P3;
    #[inline]
    fn sub(self, o: P3) -> P3 {
        P3 { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z }
    }
}
impl Neg for P3 {
    type Output = P3;
    #[inline]
    fn neg(self) -> P3 {
        P3 { x: -self.x, y: -self.y, z: -self.z }
    }
}
impl Mul<f64> for P3 {
    type Output = P3;
    #[inline]
    fn mul(self, s: f64) -> P3 {
        P3 { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}
impl Div<f64> for P3 {
    type Output = P3;
    #[inline]
    fn div(self, s: f64) -> P3 {
        P3 { x: self.x / s, y: self.y / s, z: self.z / s }
    }
}
impl DivAssign<f64> for P3 {
    #[inline]
    fn div_assign(&mut self, s: f64) {
        self.x /= s;
        self.y /= s;
        self.z /= s;
    }
}

/// RDKit `RDGeom::computeDihedralAngle` (unsigned, in `[0, π]`).
pub fn compute_dihedral_angle(p1: P3, p2: P3, p3: P3, p4: P3) -> f64 {
    let beg_end = p3 - p2;
    let beg_nbr = p1 - p2;
    let crs1 = beg_nbr.cross(&beg_end);
    let end_nbr = p4 - p3;
    let crs2 = end_nbr.cross(&beg_end);
    crs1.angle_to(&crs2)
}
