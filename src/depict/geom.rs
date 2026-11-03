//! 2D points and affine transforms with the floating-point behaviour of
//! RDKit's `RDGeom::Point2D` / `RDGeom::Transform2D`
//! (`Code/Geometry/point.h`, `Code/Geometry/Transform2D.cpp`).

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub const fn new(x: f64, y: f64) -> Self {
        Point2D { x, y }
    }
    pub fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    pub fn length_sq(self) -> f64 {
        self.x * self.x + self.y * self.y
    }
    pub fn dot(self, o: Point2D) -> f64 {
        self.x * o.x + self.y * o.y
    }
    /// `Point2D::normalize`; RDKit throws for zero-length vectors, here the
    /// vector is left unchanged.
    pub fn normalize(&mut self) {
        let l = self.length();
        if l < 1.0e-16 {
            return;
        }
        self.x /= l;
        self.y /= l;
    }
    pub fn rotate90(&mut self) {
        let t = self.x;
        self.x = -self.y;
        self.y = t;
    }
    /// `Point2D::angleTo` (unsigned, in [0, π]).
    pub fn angle_to(self, other: Point2D) -> f64 {
        let mut t1 = self;
        let mut t2 = other;
        t1.normalize();
        t2.normalize();
        t1.dot(t2).clamp(-1.0, 1.0).acos()
    }
}

impl Add for Point2D {
    type Output = Point2D;
    fn add(self, o: Point2D) -> Point2D {
        Point2D::new(self.x + o.x, self.y + o.y)
    }
}
impl Sub for Point2D {
    type Output = Point2D;
    fn sub(self, o: Point2D) -> Point2D {
        Point2D::new(self.x - o.x, self.y - o.y)
    }
}
impl AddAssign for Point2D {
    fn add_assign(&mut self, o: Point2D) {
        self.x += o.x;
        self.y += o.y;
    }
}
impl SubAssign for Point2D {
    fn sub_assign(&mut self, o: Point2D) {
        self.x -= o.x;
        self.y -= o.y;
    }
}
impl MulAssign<f64> for Point2D {
    fn mul_assign(&mut self, s: f64) {
        self.x *= s;
        self.y *= s;
    }
}
impl Mul<f64> for Point2D {
    type Output = Point2D;
    fn mul(self, s: f64) -> Point2D {
        Point2D::new(self.x * s, self.y * s)
    }
}
impl Neg for Point2D {
    type Output = Point2D;
    fn neg(self) -> Point2D {
        Point2D::new(-self.x, -self.y)
    }
}

/// 3×3 homogeneous 2D transform, row major.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2D {
    pub d: [f64; 9],
}

impl Default for Transform2D {
    fn default() -> Self {
        Self::identity()
    }
}

impl Transform2D {
    pub const fn identity() -> Self {
        Transform2D { d: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] }
    }

    fn translation(p: Point2D) -> Self {
        let mut t = Self::identity();
        t.d[2] = p.x;
        t.d[5] = p.y;
        t
    }

    /// Matrix product `a * b` (`RDNumeric::multiply`).
    fn multiply(a: &Transform2D, b: &Transform2D) -> Transform2D {
        let mut r = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += a.d[i * 3 + k] * b.d[k * 3 + j];
                }
                r[i * 3 + j] = s;
            }
        }
        Transform2D { d: r }
    }

    /// Rotation by `angle` about `pt` (`SetTransform(pt, angle)`).
    pub fn rotation_about(pt: Point2D, angle: f64) -> Self {
        let mut rot = Self::identity();
        let (s, c) = (angle.sin(), angle.cos());
        rot.d[0] = c;
        rot.d[1] = -s;
        rot.d[3] = s;
        rot.d[4] = c;
        // this *= T(-pt); then T(pt) * this
        let this = Self::multiply(&rot, &Self::translation(-pt));
        Self::multiply(&Self::translation(pt), &this)
    }

    /// Transform taking `pt1` onto `ref1` and the direction `pt1→pt2` onto
    /// `ref1→ref2` (`SetTransform(ref1, ref2, pt1, pt2)`).
    pub fn align(ref1: Point2D, ref2: Point2D, pt1: Point2D, pt2: Point2D) -> Self {
        let rvec = ref2 - ref1;
        let pvec = pt2 - pt1;
        let dp = rvec.dot(pvec);
        let lp = rvec.length() * pvec.length();
        if lp <= 0.0 {
            return Self::identity();
        }
        let cval = (dp / lp).clamp(-1.0, 1.0);
        let mut ang = cval.acos();
        let cross = pvec.x * rvec.y - pvec.y * rvec.x;
        if cross < 0.0 {
            ang *= -1.0;
        }
        let mut t = Self::identity();
        let (s, c) = (ang.sin(), ang.cos());
        t.d[0] = c;
        t.d[1] = -s;
        t.d[3] = s;
        t.d[4] = c;
        let npt1 = t.apply(pt1);
        t.d[2] = ref1.x - npt1.x;
        t.d[5] = ref1.y - npt1.y;
        t
    }

    pub fn apply(&self, p: Point2D) -> Point2D {
        let d = &self.d;
        Point2D::new(d[0] * p.x + d[1] * p.y + d[2], d[3] * p.x + d[4] * p.y + d[5])
    }

    pub fn transform_point(&self, p: &mut Point2D) {
        *p = self.apply(*p);
    }
}
