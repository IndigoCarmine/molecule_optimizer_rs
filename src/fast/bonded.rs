//! Bonded term tables and their kernels.
//!
//! Every term type of the RDKit-compatible UFF / MMFF94 contribs is stored as
//! a structure-of-arrays table. Each kernel is written once, generically over
//! a [`Lane`] type: `f64` gives the scalar reference kernel, [`V4`] (four
//! `f64` lanes, crate `wide`) the SIMD kernel that evaluates four terms at
//! once from gathered coordinates. Both instantiations perform exactly the
//! same IEEE operations per term in the same order as the reference contribs
//! (transcendental functions are evaluated per lane with the platform `libm`),
//! so per-term energies and gradients are bit-identical to the
//! [`crate::ff::ContribForceField`] path.
//!
//! Summation and scatter order: block energies are added to the running
//! total lane by lane and gradients are scattered lane by lane, i.e. in term
//! order, exactly as evaluating the contribs one after the other; serial
//! evaluation therefore reproduces `ContribForceField` bit for bit. Parallel
//! evaluation (see [`super::FastForceField`]) gives every fixed chunk its own
//! gradient buffer and reduces the buffers in chunk order (deterministic for
//! any thread count, but a different summation order than the serial path).
//!
//! `wide` chooses its instruction set at compile time: on the default
//! x86-64 target [`V4`] runs as two 128-bit halves. [`run_simd`] therefore
//! instantiates the kernels with the 256-bit AVX lanes of
//! [`super::avx::A4`] when the CPU supports AVX (checked at run time),
//! with identical results.

use std::ops::{Add, BitAnd, BitOr, Div, Mul, Neg, Not, Range, Sub};
use wide::{CmpEq, CmpGt, CmpLt, f64x4};

use crate::ff::mmff::{DEG2RAD as MMFF_DEG2RAD, MDYNE_A_TO_KCAL_MOL, RAD2DEG as MMFF_RAD2DEG};

// ---------------------------------------------------------------------------
// lanes
// ---------------------------------------------------------------------------

/// A pack of `N` f64 values processed together (scalar: `N = 1`).
pub trait Lane:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + Send
    + Sync
{
    /// Lane mask.
    type M: Copy + BitAnd<Output = Self::M> + BitOr<Output = Self::M> + Not<Output = Self::M>;
    const N: usize;
    fn splat(x: f64) -> Self;
    fn from_fn(f: impl FnMut(usize) -> f64) -> Self;
    fn sqrt(self) -> Self;
    /// `f64::max` semantics (non-NaN operand wins).
    fn max(self, o: f64) -> Self;
    fn min(self, o: f64) -> Self;
    fn gt(self, o: f64) -> Self::M;
    /// Lane-wise `self > o`.
    fn gtv(self, o: Self) -> Self::M;
    fn lt(self, o: f64) -> Self::M;
    fn le(self, o: f64) -> Self::M;
    fn eq(self, o: f64) -> Self::M;
    fn select(m: Self::M, a: Self, b: Self) -> Self;
    fn any(m: Self::M) -> bool;
    /// Applies a scalar function to every lane.
    fn map(self, f: impl Fn(f64) -> f64) -> Self;
    /// Horizontal sum `((l0 + l1) + l2) + l3`.
    fn hsum(self) -> f64;
    /// Writes the lanes to `out[..N]`.
    fn write(self, out: &mut [f64; 4]);
}

impl Lane for f64 {
    type M = bool;
    const N: usize = 1;
    #[inline(always)]
    fn splat(x: f64) -> Self {
        x
    }
    #[inline(always)]
    fn from_fn(mut f: impl FnMut(usize) -> f64) -> Self {
        f(0)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline(always)]
    fn max(self, o: f64) -> Self {
        f64::max(self, o)
    }
    #[inline(always)]
    fn min(self, o: f64) -> Self {
        f64::min(self, o)
    }
    #[inline(always)]
    fn gt(self, o: f64) -> bool {
        self > o
    }
    #[inline(always)]
    fn gtv(self, o: f64) -> bool {
        self > o
    }
    #[inline(always)]
    fn lt(self, o: f64) -> bool {
        self < o
    }
    #[inline(always)]
    fn le(self, o: f64) -> bool {
        self <= o
    }
    #[inline(always)]
    fn eq(self, o: f64) -> bool {
        self == o
    }
    #[inline(always)]
    fn select(m: bool, a: Self, b: Self) -> Self {
        if m { a } else { b }
    }
    #[inline(always)]
    fn any(m: bool) -> bool {
        m
    }
    #[inline(always)]
    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        f(self)
    }
    #[inline(always)]
    fn hsum(self) -> f64 {
        self
    }
    #[inline(always)]
    fn write(self, out: &mut [f64; 4]) {
        out[0] = self;
    }
}

/// Four f64 lanes (`wide::f64x4`).
#[derive(Clone, Copy, Debug)]
pub struct V4(pub f64x4);

/// Lane mask of [`V4`] (all-ones / all-zeros per lane).
#[derive(Clone, Copy, Debug)]
pub struct M4(f64x4);

impl BitAnd for M4 {
    type Output = M4;
    #[inline(always)]
    fn bitand(self, o: M4) -> M4 {
        M4(self.0 & o.0)
    }
}
impl BitOr for M4 {
    type Output = M4;
    #[inline(always)]
    fn bitor(self, o: M4) -> M4 {
        M4(self.0 | o.0)
    }
}
impl Not for M4 {
    type Output = M4;
    #[inline(always)]
    fn not(self) -> M4 {
        M4(!self.0)
    }
}

macro_rules! v4_ops {
    ($tr:ident, $f:ident, $op:tt) => {
        impl $tr for V4 {
            type Output = V4;
            #[inline(always)]
            fn $f(self, o: V4) -> V4 {
                V4(self.0 $op o.0)
            }
        }
        impl $tr<f64> for V4 {
            type Output = V4;
            #[inline(always)]
            fn $f(self, o: f64) -> V4 {
                V4(self.0 $op f64x4::splat(o))
            }
        }
    };
}
v4_ops!(Add, add, +);
v4_ops!(Sub, sub, -);
v4_ops!(Mul, mul, *);
v4_ops!(Div, div, /);

impl Neg for V4 {
    type Output = V4;
    #[inline(always)]
    fn neg(self) -> V4 {
        // sign-bit flip, identical to scalar negation (also for zeros)
        V4(self.0 ^ f64x4::splat(-0.0))
    }
}

impl Lane for V4 {
    type M = M4;
    const N: usize = 4;
    #[inline(always)]
    fn splat(x: f64) -> Self {
        V4(f64x4::splat(x))
    }
    #[inline(always)]
    fn from_fn(mut f: impl FnMut(usize) -> f64) -> Self {
        V4(f64x4::from([f(0), f(1), f(2), f(3)]))
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        V4(self.0.sqrt())
    }
    #[inline(always)]
    fn max(self, o: f64) -> Self {
        V4(self.0.max(f64x4::splat(o)))
    }
    #[inline(always)]
    fn min(self, o: f64) -> Self {
        V4(self.0.min(f64x4::splat(o)))
    }
    #[inline(always)]
    fn gt(self, o: f64) -> M4 {
        M4(self.0.cmp_gt(f64x4::splat(o)))
    }
    #[inline(always)]
    fn gtv(self, o: Self) -> M4 {
        M4(self.0.cmp_gt(o.0))
    }
    #[inline(always)]
    fn lt(self, o: f64) -> M4 {
        M4(self.0.cmp_lt(f64x4::splat(o)))
    }
    #[inline(always)]
    fn le(self, o: f64) -> M4 {
        M4(wide::CmpLe::cmp_le(self.0, f64x4::splat(o)))
    }
    #[inline(always)]
    fn eq(self, o: f64) -> M4 {
        M4(self.0.cmp_eq(f64x4::splat(o)))
    }
    #[inline(always)]
    fn select(m: M4, a: Self, b: Self) -> Self {
        V4(m.0.blend(a.0, b.0))
    }
    #[inline(always)]
    fn any(m: M4) -> bool {
        m.0.any()
    }
    #[inline(always)]
    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        let a = self.0.to_array();
        V4(f64x4::from([f(a[0]), f(a[1]), f(a[2]), f(a[3])]))
    }
    #[inline(always)]
    fn hsum(self) -> f64 {
        let a = self.0.to_array();
        ((a[0] + a[1]) + a[2]) + a[3]
    }
    #[inline(always)]
    fn write(self, out: &mut [f64; 4]) {
        *out = self.0.to_array();
    }
}

// ---------------------------------------------------------------------------
// vector helpers (same operation order as the reference contribs)
// ---------------------------------------------------------------------------

pub(crate) type P<L> = [L; 3];

#[inline(always)]
pub(crate) fn ld<L: Lane>(pos: &[f64], idx: &[u32], t: usize) -> P<L> {
    [
        L::from_fn(|k| pos[3 * idx[t + k] as usize]),
        L::from_fn(|k| pos[3 * idx[t + k] as usize + 1]),
        L::from_fn(|k| pos[3 * idx[t + k] as usize + 2]),
    ]
}
#[inline(always)]
pub(crate) fn lp<L: Lane>(v: &[f64], t: usize) -> L {
    L::from_fn(|k| v[t + k])
}
#[inline(always)]
pub(crate) fn sub3<L: Lane>(a: P<L>, b: P<L>) -> P<L> {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline(always)]
fn neg3<L: Lane>(a: P<L>) -> P<L> {
    [-a[0], -a[1], -a[2]]
}
#[inline(always)]
pub(crate) fn dot3<L: Lane>(a: P<L>, b: P<L>) -> L {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline(always)]
fn cross3<L: Lane>(a: P<L>, b: P<L>) -> P<L> {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline(always)]
fn divs<L: Lane>(a: P<L>, s: L) -> P<L> {
    [a[0] / s, a[1] / s, a[2] / s]
}
#[inline(always)]
fn len3<L: Lane>(a: P<L>) -> L {
    dot3(a, a).sqrt()
}
#[inline(always)]
fn clamp1<L: Lane>(x: L) -> L {
    x.max(-1.0).min(1.0)
}
/// RDKit `isDoubleZero`.
#[inline(always)]
fn is_dz<L: Lane>(x: L) -> L::M {
    x.lt(1.0e-10) & x.gt(-1.0e-10)
}
#[inline(always)]
fn one<L: Lane>() -> L {
    L::splat(1.0)
}
#[inline(always)]
fn zero<L: Lane>() -> L {
    L::splat(0.0)
}

/// Adds per-term atom gradients (`g[a]` belongs to atom `idx[a][t + lane]`)
/// lane by lane, i.e. in term order.
#[inline(always)]
pub(crate) fn scatter<L: Lane, const A: usize>(grad: &mut [f64], idx: [&[u32]; A], t: usize, g: [P<L>; A]) {
    let mut v = [[[0.0f64; 4]; 3]; A];
    for a in 0..A {
        for d in 0..3 {
            g[a][d].write(&mut v[a][d]);
        }
    }
    for k in 0..L::N {
        for a in 0..A {
            let i = 3 * idx[a][t + k] as usize;
            grad[i] += v[a][0][k];
            grad[i + 1] += v[a][1][k];
            grad[i + 2] += v[a][2][k];
        }
    }
}

// ---------------------------------------------------------------------------
// tables
// ---------------------------------------------------------------------------

/// A table of terms of one type with a block kernel.
pub trait Table: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Evaluates terms `t .. t + L::N`, returns their energies and (if `G`)
    /// adds their gradients to `grad`.
    fn block<L: Lane, const G: bool>(&self, pos: &[f64], grad: &mut [f64], t: usize) -> L;
}

/// Evaluates the terms in `r` (SIMD blocks of `L::N`, scalar tail) and adds
/// their energies to `acc` one term after the other (the same summation
/// order as evaluating the terms one by one).
#[inline(always)]
pub fn run<T: Table, L: Lane, const G: bool>(tab: &T, pos: &[f64], grad: &mut [f64], r: Range<usize>, acc: &mut f64) {
    let mut t = r.start;
    let mut lanes = [0.0f64; 4];
    while t + L::N <= r.end {
        tab.block::<L, G>(pos, grad, t).write(&mut lanes);
        for e in &lanes[..L::N] {
            *acc += *e;
        }
        t += L::N;
    }
    while t < r.end {
        *acc += tab.block::<f64, G>(pos, grad, t);
        t += 1;
    }
}

/// [`run`] with the widest SIMD lanes available: 256-bit AVX ([`super::avx::A4`])
/// when the CPU has it, [`V4`] otherwise (same results either way).
#[inline]
pub fn run_simd<T: Table, const G: bool>(tab: &T, pos: &[f64], grad: &mut [f64], r: Range<usize>, acc: &mut f64) {
    #[cfg(target_arch = "x86_64")]
    if super::avx::available() {
        #[target_feature(enable = "avx")]
        unsafe fn avx<T: Table, const G: bool>(tab: &T, pos: &[f64], grad: &mut [f64], r: Range<usize>, acc: &mut f64) {
            run::<T, super::avx::A4, G>(tab, pos, grad, r, acc)
        }
        // SAFETY: the CPU supports AVX (checked above).
        return unsafe { avx::<T, G>(tab, pos, grad, r, acc) };
    }
    run::<T, V4, G>(tab, pos, grad, r, acc)
}

/// UFF harmonic bond stretch.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UffBonds {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub r0: Vec<f64>,
    pub k: Vec<f64>,
}

impl Table for UffBonds {
    fn len(&self) -> usize {
        self.a.len()
    }
    #[inline(always)]
    fn block<L: Lane, const G: bool>(&self, pos: &[f64], grad: &mut [f64], t: usize) -> L {
        let pa = ld::<L>(pos, &self.a, t);
        let pb = ld::<L>(pos, &self.b, t);
        let k = lp::<L>(&self.k, t);
        let r0 = lp::<L>(&self.r0, t);
        let d = sub3(pa, pb);
        let dist = len3(d);
        let dt = dist - r0;
        let e = k * 0.5 * dt * dt;
        if G {
            let pre = k * (dist - r0);
            let pos_d = dist.gt(0.0);
            let fb = k * 0.01;
            let g: P<L> = [
                L::select(pos_d, pre * d[0] / dist, fb),
                L::select(pos_d, pre * d[1] / dist, fb),
                L::select(pos_d, pre * d[2] / dist, fb),
            ];
            scatter(grad, [&self.a, &self.b], t, [g, neg3(g)]);
        }
        e
    }
}

/// MMFF quartic bond stretch.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MmffBonds {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub r0: Vec<f64>,
    pub kb: Vec<f64>,
}

impl Table for MmffBonds {
    fn len(&self) -> usize {
        self.a.len()
    }
    #[inline(always)]
    fn block<L: Lane, const G: bool>(&self, pos: &[f64], grad: &mut [f64], t: usize) -> L {
        const C1: f64 = MDYNE_A_TO_KCAL_MOL;
        const CS: f64 = -2.0;
        const C3: f64 = 7.0 / 12.0;
        let pa = ld::<L>(pos, &self.a, t);
        let pb = ld::<L>(pos, &self.b, t);
        let kb = lp::<L>(&self.kb, t);
        let r0 = lp::<L>(&self.r0, t);
        let d = sub3(pa, pb);
        let dist = len3(d);
        let dt = dist - r0;
        let dt2 = dt * dt;
        // 0.5 * c1 * kb * dt2 * (1.0 + cs * dt + c3 * cs * cs * dt2)
        let e = kb * (0.5 * C1) * dt2 * (one::<L>() + dt * CS + dt2 * (C3 * CS * CS));
        if G {
            // C1 * kb * dt * (1.0 + 1.5 * CS * dt + 2.0 * C3 * CS * CS * dt * dt)
            let de_dr = kb * C1 * dt * (one::<L>() + dt * (1.5 * CS) + dt * (2.0 * C3 * CS * CS) * dt);
            let pos_d = dist.gt(0.0);
            let fb = kb * 0.01;
            let g: P<L> = [
                L::select(pos_d, de_dr * d[0] / dist, fb),
                L::select(pos_d, de_dr * d[1] / dist, fb),
                L::select(pos_d, de_dr * d[2] / dist, fb),
            ];
            scatter(grad, [&self.a, &self.b], t, [g, neg3(g)]);
        }
        e
    }
}

/// RDKit `calcAngleBendGrad` (UFF and MMFF share it). Returns the gradients
/// of the three atoms.
#[inline(always)]
fn angle_grad<L: Lane>(r: [P<L>; 2], dist: [L; 2], de: L, cos: L, sin: L) -> [P<L>; 3] {
    let i0 = one::<L>() / dist[0];
    let i1 = one::<L>() / dist[1];
    let dc0 = [i0 * (r[1][0] - cos * r[0][0]), i0 * (r[1][1] - cos * r[0][1]), i0 * (r[1][2] - cos * r[0][2])];
    let dc1 = [i1 * (r[0][0] - cos * r[1][0]), i1 * (r[0][1] - cos * r[1][1]), i1 * (r[0][2] - cos * r[1][2])];
    let ms = -sin;
    let g0 = [de * dc0[0] / ms, de * dc0[1] / ms, de * dc0[2] / ms];
    let g1 = [
        de * (-dc0[0] - dc1[0]) / ms,
        de * (-dc0[1] - dc1[1]) / ms,
        de * (-dc0[2] - dc1[2]) / ms,
    ];
    let g2 = [de * dc1[0] / ms, de * dc1[1] / ms, de * dc1[2] / ms];
    [g0, g1, g2]
}

const UFF_ANGLE_CORRECTION_THRESHOLD: f64 = 0.8660;

/// UFF angle bend (all orders).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UffAngles {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub c: Vec<u32>,
    pub fc: Vec<f64>,
    pub c0: Vec<f64>,
    pub c1: Vec<f64>,
    pub c2: Vec<f64>,
    pub theta0: Vec<f64>,
    /// order as f64 (0..=4)
    pub ord: Vec<f64>,
    /// `(order * order) as f64`
    pub ord_sq: Vec<f64>,
    /// `force_constant / order` (unused for order 0)
    pub fc_ord: Vec<f64>,
}

impl Table for UffAngles {
    fn len(&self) -> usize {
        self.a.len()
    }
    #[inline(always)]
    fn block<L: Lane, const G: bool>(&self, pos: &[f64], grad: &mut [f64], t: usize) -> L {
        let p1 = ld::<L>(pos, &self.a, t);
        let p2 = ld::<L>(pos, &self.b, t);
        let p3 = ld::<L>(pos, &self.c, t);
        let fc = lp::<L>(&self.fc, t);
        let c0 = lp::<L>(&self.c0, t);
        let c1 = lp::<L>(&self.c1, t);
        let c2 = lp::<L>(&self.c2, t);
        let th0 = lp::<L>(&self.theta0, t);
        let ord = lp::<L>(&self.ord, t);
        let (o0, o1, o2, o3, o4) = (ord.eq(0.0), ord.eq(1.0), ord.eq(2.0), ord.eq(3.0), ord.eq(4.0));
        let corr_ok = !o0 & ord.lt(5.0);
        let d0 = sub3(p1, p2);
        let d1 = sub3(p3, p2);
        let dist1 = len3(d0);
        let dist2 = len3(d1);
        // energy
        let cos = clamp1(dot3(d0, d1) / (dist1 * dist2));
        let s = one::<L>() - cos * cos;
        let cos2 = cos * cos - s;
        let e0 = c0 + c1 * cos + c2 * cos2;
        let r3 = cos * (cos * cos - s * 3.0);
        let r4 = (cos * cos) * (cos * cos) - cos * 6.0 * cos * s + s * s;
        let rk = L::select(o1, -cos, L::select(o2, cos2, L::select(o3, r3, L::select(o4, r4, zero()))));
        let ek = (one::<L>() - rk) / lp::<L>(&self.ord_sq, t);
        let mut e = fc * L::select(o0, e0, ek);
        let me = corr_ok & cos.gt(UFF_ANGLE_CORRECTION_THRESHOLD);
        if L::any(me) {
            let theta = cos.map(f64::acos);
            let corr = ((theta - th0 + 0.25) * -20.0).map(f64::exp);
            e = L::select(me, e + corr, e);
        }
        if G {
            let r = [divs(d0, dist1), divs(d1, dist2)];
            let cos = clamp1(dot3(r[0], r[1]));
            let ssq = one::<L>() - cos * cos;
            let sin = ssq.sqrt().max(1.0e-8);
            let sin2 = sin * 2.0 * cos;
            let de0 = -fc * (c1 * sin + c2 * 2.0 * sin2);
            let dk3 = sin * (L::splat(3.0) - sin * 4.0 * sin);
            let dk4 = cos * sin * (L::splat(4.0) - sin * 8.0 * sin);
            let dk = L::select(o1, -sin, L::select(o2, sin2, L::select(o3, dk3, L::select(o4, dk4, zero()))));
            let mut de = L::select(o0, de0, dk * lp::<L>(&self.fc_ord, t));
            let mg = corr_ok & cos.gt(UFF_ANGLE_CORRECTION_THRESHOLD);
            if L::any(mg) {
                let theta = cos.map(f64::acos);
                let corr = ((theta - th0 + 0.25) * -20.0).map(f64::exp) * -20.0;
                de = L::select(mg, de + corr, de);
            }
            let g = angle_grad(r, [dist1, dist2], de, cos, sin);
            scatter(grad, [&self.a, &self.b, &self.c], t, g);
        }
        e
    }
}

const ANGLE_CB: f64 = -0.006981317;
const ANGLE_C2: f64 = MDYNE_A_TO_KCAL_MOL * MMFF_DEG2RAD * MMFF_DEG2RAD;

/// MMFF angle bend.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MmffAngles {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub c: Vec<u32>,
    pub theta0: Vec<f64>,
    pub ka: Vec<f64>,
    /// 1.0 = linear
    pub lin: Vec<f64>,
}

impl Table for MmffAngles {
    fn len(&self) -> usize {
        self.a.len()
    }
    #[inline(always)]
    fn block<L: Lane, const G: bool>(&self, pos: &[f64], grad: &mut [f64], t: usize) -> L {
        let p1 = ld::<L>(pos, &self.a, t);
        let p2 = ld::<L>(pos, &self.b, t);
        let p3 = ld::<L>(pos, &self.c, t);
        let th0 = lp::<L>(&self.theta0, t);
        let ka = lp::<L>(&self.ka, t);
        let lin = lp::<L>(&self.lin, t).gt(0.5);
        let d0 = sub3(p1, p2);
        let d1 = sub3(p3, p2);
        let dist1 = len3(d0);
        let dist2 = len3(d1);
        let cos = clamp1(dot3(d0, d1) / (dist1 * dist2));
        let angle = cos.map(f64::acos) * MMFF_RAD2DEG - th0;
        let e_lin = ka * MDYNE_A_TO_KCAL_MOL * (one::<L>() + cos);
        let e_nl = ka * (0.5 * ANGLE_C2) * angle * angle * (one::<L>() + angle * ANGLE_CB);
        let e = L::select(lin, e_lin, e_nl);
        if G {
            let r = [divs(d0, dist1), divs(d1, dist2)];
            let cos = clamp1(dot3(r[0], r[1]));
            let ssq = one::<L>() - cos * cos;
            let sin = L::select(ssq.gt(0.0), ssq.sqrt(), zero()).max(1.0e-8);
            let at = cos.map(f64::acos) * MMFF_RAD2DEG - th0;
            let de_lin = ka * -MDYNE_A_TO_KCAL_MOL * sin;
            let de_nl = ka * (MMFF_RAD2DEG * ANGLE_C2) * at * (one::<L>() + at * (1.5 * ANGLE_CB));
            let de = L::select(lin, de_lin, de_nl);
            let g = angle_grad(r, [dist1, dist2], de, cos, sin);
            scatter(grad, [&self.a, &self.b, &self.c], t, g);
        }
        e
    }
}

/// MMFF stretch-bend.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MmffStretchBends {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub c: Vec<u32>,
    pub r1: Vec<f64>,
    pub r2: Vec<f64>,
    pub theta0: Vec<f64>,
    pub k1: Vec<f64>,
    pub k2: Vec<f64>,
}

impl Table for MmffStretchBends {
    fn len(&self) -> usize {
        self.a.len()
    }
    #[inline(always)]
    fn block<L: Lane, const G: bool>(&self, pos: &[f64], grad: &mut [f64], t: usize) -> L {
        const C5: f64 = MDYNE_A_TO_KCAL_MOL * MMFF_DEG2RAD;
        let p1 = ld::<L>(pos, &self.a, t);
        let p2 = ld::<L>(pos, &self.b, t);
        let p3 = ld::<L>(pos, &self.c, t);
        let th0 = lp::<L>(&self.theta0, t);
        let k1 = lp::<L>(&self.k1, t);
        let k2 = lp::<L>(&self.k2, t);
        let rl1 = lp::<L>(&self.r1, t);
        let rl2 = lp::<L>(&self.r2, t);
        let d0 = sub3(p1, p2);
        let d1 = sub3(p3, p2);
        let dist1 = len3(d0);
        let dist2 = len3(d1);
        let cos_e = clamp1(dot3(d0, d1) / (dist1 * dist2));
        let dtheta = cos_e.map(f64::acos) * MMFF_RAD2DEG - th0;
        let factor = dtheta * (MDYNE_A_TO_KCAL_MOL * MMFF_DEG2RAD);
        let e = factor * k1 * (dist1 - rl1) + factor * k2 * (dist2 - rl2);
        if G {
            let p12 = divs(d0, dist1);
            let p32 = divs(d1, dist2);
            let cos = clamp1(dot3(p12, p32));
            let ssq = one::<L>() - cos * cos;
            let sin = ssq.sqrt().max(1.0e-8);
            let at = cos.map(f64::acos) * MMFF_RAD2DEG - th0;
            let dist_term = (k1 * (dist1 - rl1) + k2 * (dist2 - rl2)) * MMFF_RAD2DEG;
            let i1 = one::<L>() / dist1;
            let i2 = one::<L>() / dist2;
            let ms = -sin;
            let mut g = [[zero::<L>(); 3]; 3];
            for d in 0..3 {
                let dc1 = i1 * (p32[d] - cos * p12[d]);
                let dc2 = i2 * (p12[d] - cos * p32[d]);
                g[0][d] = (p12[d] * k1 * at + dc1 / ms * dist_term) * C5;
                g[1][d] = ((-p12[d] * k1 - p32[d] * k2) * at + (-dc1 - dc2) / ms * dist_term) * C5;
                g[2][d] = (p32[d] * k2 * at + dc2 / ms * dist_term) * C5;
            }
            scatter(grad, [&self.a, &self.b, &self.c], t, g);
        }
        e
    }
}

/// MMFF out-of-plane bend.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MmffOops {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub c: Vec<u32>,
    pub d: Vec<u32>,
    pub koop: Vec<f64>,
}

impl Table for MmffOops {
    fn len(&self) -> usize {
        self.a.len()
    }
    #[inline(always)]
    fn block<L: Lane, const G: bool>(&self, pos: &[f64], grad: &mut [f64], t: usize) -> L {
        const C2: f64 = MDYNE_A_TO_KCAL_MOL * MMFF_DEG2RAD * MMFF_DEG2RAD;
        let ip = ld::<L>(pos, &self.a, t);
        let jp = ld::<L>(pos, &self.b, t);
        let kp = ld::<L>(pos, &self.c, t);
        let lpp = ld::<L>(pos, &self.d, t);
        let koop = lp::<L>(&self.koop, t);
        let rji0 = sub3(ip, jp);
        let rjk0 = sub3(kp, jp);
        let rjl0 = sub3(lpp, jp);
        let dji = len3(rji0);
        let djk = len3(rjk0);
        let djl = len3(rjl0);
        let rji = divs(rji0, dji);
        let rjk = divs(rjk0, djk);
        let rjl = divs(rjl0, djl);
        // energy (calcOopChi)
        let ne = cross3(rji, rjk);
        let ne = divs(ne, len3(ne));
        let chi_e = clamp1(dot3(ne, rjl)).map(f64::asin) * MMFF_RAD2DEG;
        let e = koop * (0.5 * C2) * chi_e * chi_e;
        if G {
            let skip = is_dz(dji) | is_dz(djk) | is_dz(djl);
            let n = cross3(neg3(rji), rjk);
            let n = divs(n, len3(n));
            let sin_chi = clamp1(dot3(rjl, n));
            let ccsq = one::<L>() - sin_chi * sin_chi;
            let cos_chi = L::select(ccsq.gt(0.0), ccsq.sqrt(), zero()).max(1.0e-8);
            let chi = sin_chi.map(f64::asin) * MMFF_RAD2DEG;
            let cos_theta = clamp1(dot3(rji, rjk));
            let stsq = (one::<L>() - cos_theta * cos_theta).max(1.0e-8);
            let sin_theta = L::select(stsq.gt(0.0), stsq.sqrt(), zero()).max(1.0e-8);
            let de = koop * (MMFF_RAD2DEG * C2) * chi;
            let de = L::select(skip, zero(), de);
            let t1 = cross3(rjl, rjk);
            let t2 = cross3(rji, rjl);
            let t3 = cross3(rjk, rji);
            let term1 = cos_chi * sin_theta;
            let term2 = sin_chi / (cos_chi * stsq);
            let mut g = [[zero::<L>(); 3]; 4];
            for d in 0..3 {
                let tg1 = (t1[d] / term1 - (rji[d] - rjk[d] * cos_theta) * term2) / dji;
                let tg3 = (t2[d] / term1 - (rjk[d] - rji[d] * cos_theta) * term2) / djk;
                let tg4 = (t3[d] / term1 - rjl[d] * sin_chi / cos_chi) / djl;
                g[0][d] = L::select(skip, zero(), de * tg1);
                g[1][d] = L::select(skip, zero(), -de * (tg1 + tg3 + tg4));
                g[2][d] = L::select(skip, zero(), de * tg3);
                g[3][d] = L::select(skip, zero(), de * tg4);
            }
            scatter(grad, [&self.a, &self.b, &self.c, &self.d], t, g);
        }
        e
    }
}

/// Dihedral geometry (RDKit `computeDihedral`): `r`, `t` (normalized),
/// `d` (clamped lengths), `cos_phi`, plus the raw cross-product lengths.
struct Dihedral<L: Lane> {
    r: [P<L>; 4],
    t: [P<L>; 2],
    d: [L; 2],
    cos_phi: L,
    /// cosine as computed by the energy functions
    /// (`calculateCosTorsion` / `calcTorsionCosPhi`)
    cos_energy: L,
}

#[inline(always)]
fn dihedral<L: Lane>(p1: P<L>, p2: P<L>, p3: P<L>, p4: P<L>) -> Dihedral<L> {
    let r0 = sub3(p1, p2);
    let r1 = sub3(p3, p2);
    let r2 = neg3(r1);
    let r3 = sub3(p4, p3);
    let t0r = cross3(r0, r1);
    let t1r = cross3(r2, r3);
    let l0 = len3(t0r);
    let l1 = len3(t1r);
    // energy path: t1 = (p1-p2)x(p3-p2), t2 = (p2-p3)x(p4-p3)
    let ce = clamp1(dot3(t0r, t1r) / (l0 * l1));
    let cos_energy = L::select(is_dz(l0) | is_dz(l1), zero(), ce);
    let d0 = l0.max(1.0e-5);
    let d1 = l1.max(1.0e-5);
    let t0 = divs(t0r, d0);
    let t1 = divs(t1r, d1);
    let cos_phi = clamp1(dot3(t0, t1));
    Dihedral { r: [r0, r1, r2, r3], t: [t0, t1], d: [d0, d1], cos_phi, cos_energy }
}

/// RDKit `calcTorsionGrad` (UFF and MMFF share it).
#[inline(always)]
fn torsion_grad<L: Lane>(dh: &Dihedral<L>, sin_term: L) -> [P<L>; 4] {
    let (r, t, d, c) = (&dh.r, &dh.t, &dh.d, dh.cos_phi);
    let i0 = one::<L>() / d[0];
    let i1 = one::<L>() / d[1];
    let dc = [
        i0 * (t[1][0] - c * t[0][0]),
        i0 * (t[1][1] - c * t[0][1]),
        i0 * (t[1][2] - c * t[0][2]),
        i1 * (t[0][0] - c * t[1][0]),
        i1 * (t[0][1] - c * t[1][1]),
        i1 * (t[0][2] - c * t[1][2]),
    ];
    let (x, y, z) = (0, 1, 2);
    let s = sin_term;
    let g0 = [
        s * (dc[2] * r[1][y] - dc[1] * r[1][z]),
        s * (dc[0] * r[1][z] - dc[2] * r[1][x]),
        s * (dc[1] * r[1][x] - dc[0] * r[1][y]),
    ];
    let g1 = [
        s * (dc[1] * (r[1][z] - r[0][z]) + dc[2] * (r[0][y] - r[1][y]) + dc[4] * (-r[3][z]) + dc[5] * (r[3][y])),
        s * (dc[0] * (r[0][z] - r[1][z]) + dc[2] * (r[1][x] - r[0][x]) + dc[3] * (r[3][z]) + dc[5] * (-r[3][x])),
        s * (dc[0] * (r[1][y] - r[0][y]) + dc[1] * (r[0][x] - r[1][x]) + dc[3] * (-r[3][y]) + dc[4] * (r[3][x])),
    ];
    let g2 = [
        s * (dc[1] * (r[0][z]) + dc[2] * (-r[0][y]) + dc[4] * (r[3][z] - r[2][z]) + dc[5] * (r[2][y] - r[3][y])),
        s * (dc[0] * (-r[0][z]) + dc[2] * (r[0][x]) + dc[3] * (r[2][z] - r[3][z]) + dc[5] * (r[3][x] - r[2][x])),
        s * (dc[0] * (r[0][y]) + dc[1] * (-r[0][x]) + dc[3] * (r[3][y] - r[2][y]) + dc[4] * (r[2][x] - r[3][x])),
    ];
    let g3 = [
        s * (dc[4] * r[2][z] - dc[5] * r[2][y]),
        s * (dc[5] * r[2][x] - dc[3] * r[2][z]),
        s * (dc[3] * r[2][y] - dc[4] * r[2][x]),
    ];
    [g0, g1, g2, g3]
}

/// UFF torsion.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UffTorsions {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub c: Vec<u32>,
    pub d: Vec<u32>,
    /// `force_constant / 2.0`
    pub half_fc: Vec<f64>,
    pub cos_term: Vec<f64>,
    /// order as f64 (2, 3, 6; anything else = 0 energy/gradient terms)
    pub ord: Vec<f64>,
    /// `-(force_constant / 2.0 * cos_term) * order`
    pub dfac: Vec<f64>,
}

impl Table for UffTorsions {
    fn len(&self) -> usize {
        self.a.len()
    }
    #[inline(always)]
    fn block<L: Lane, const G: bool>(&self, pos: &[f64], grad: &mut [f64], t: usize) -> L {
        let p1 = ld::<L>(pos, &self.a, t);
        let p2 = ld::<L>(pos, &self.b, t);
        let p3 = ld::<L>(pos, &self.c, t);
        let p4 = ld::<L>(pos, &self.d, t);
        let ord = lp::<L>(&self.ord, t);
        let (o2, o3, o6) = (ord.eq(2.0), ord.eq(3.0), ord.eq(6.0));
        let dh = dihedral(p1, p2, p3, p4);
        let c = dh.cos_energy;
        let s = one::<L>() - c * c;
        let cn2 = one::<L>() - s * 2.0;
        let cn3 = c * (c * c - s * 3.0);
        let cn6 = one::<L>() + s * (s * -32.0 * s + s * 48.0 - 18.0);
        let cnp = L::select(o2, cn2, L::select(o3, cn3, L::select(o6, cn6, zero())));
        let e = lp::<L>(&self.half_fc, t) * (one::<L>() - lp::<L>(&self.cos_term, t) * cnp);
        if G {
            let cos = dh.cos_phi;
            let ssq = one::<L>() - cos * cos;
            let sin = L::select(ssq.gt(0.0), ssq.sqrt(), zero());
            let s2 = sin * sin;
            let r2 = sin * 2.0 * cos;
            let r3 = sin * (L::splat(3.0) - s2 * 4.0);
            let r6 = cos * sin * (s2 * 32.0 * (s2 - 1.0) + 6.0);
            let res = L::select(o2, r2, L::select(o3, r3, L::select(o6, r6, zero())));
            let de = res * lp::<L>(&self.dfac, t);
            let inv = L::select(is_dz(sin), one::<L>() / cos, one::<L>() / sin);
            let g = torsion_grad(&dh, de * inv);
            scatter(grad, [&self.a, &self.b, &self.c, &self.d], t, g);
        }
        e
    }
}

/// MMFF torsion.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MmffTorsions {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub c: Vec<u32>,
    pub d: Vec<u32>,
    pub v1: Vec<f64>,
    pub v2: Vec<f64>,
    pub v3: Vec<f64>,
}

impl Table for MmffTorsions {
    fn len(&self) -> usize {
        self.a.len()
    }
    #[inline(always)]
    fn block<L: Lane, const G: bool>(&self, pos: &[f64], grad: &mut [f64], t: usize) -> L {
        let p1 = ld::<L>(pos, &self.a, t);
        let p2 = ld::<L>(pos, &self.b, t);
        let p3 = ld::<L>(pos, &self.c, t);
        let p4 = ld::<L>(pos, &self.d, t);
        let v1 = lp::<L>(&self.v1, t);
        let v2 = lp::<L>(&self.v2, t);
        let v3 = lp::<L>(&self.v3, t);
        let dh = dihedral(p1, p2, p3, p4);
        let c = dh.cos_energy;
        let cos2 = c * 2.0 * c - 1.0;
        let cos3 = c * (cos2 * 2.0 - 1.0);
        let e = (v1 * (one::<L>() + c) + v2 * (one::<L>() - cos2) + v3 * (one::<L>() + cos3)) * 0.5;
        if G {
            let cos = dh.cos_phi;
            let ssq = one::<L>() - cos * cos;
            let sin = L::select(ssq.gt(0.0), ssq.sqrt(), zero());
            let sin2 = sin * 2.0 * cos;
            let sin3 = sin * 3.0 - sin * 4.0 * ssq;
            let de = (-v1 * sin + v2 * 2.0 * sin2 - v3 * 3.0 * sin3) * 0.5;
            let inv = L::select(is_dz(sin), one::<L>() / cos, one::<L>() / sin);
            let g = torsion_grad(&dh, -de * inv);
            scatter(grad, [&self.a, &self.b, &self.c, &self.d], t, g);
        }
        e
    }
}

/// UFF inversion (atom `b` is the centre).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UffInversions {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub c: Vec<u32>,
    pub d: Vec<u32>,
    pub fc: Vec<f64>,
    pub c0: Vec<f64>,
    pub c1: Vec<f64>,
    pub c2: Vec<f64>,
}

impl Table for UffInversions {
    fn len(&self) -> usize {
        self.a.len()
    }
    #[inline(always)]
    fn block<L: Lane, const G: bool>(&self, pos: &[f64], grad: &mut [f64], t: usize) -> L {
        const ZERO_TOL: f64 = 1.0e-16;
        let p1 = ld::<L>(pos, &self.a, t);
        let p2 = ld::<L>(pos, &self.b, t);
        let p3 = ld::<L>(pos, &self.c, t);
        let p4 = ld::<L>(pos, &self.d, t);
        let fc = lp::<L>(&self.fc, t);
        let c0 = lp::<L>(&self.c0, t);
        let c1 = lp::<L>(&self.c1, t);
        let c2 = lp::<L>(&self.c2, t);
        let r_ji = sub3(p1, p2);
        let r_jk = sub3(p3, p2);
        let r_jl = sub3(p4, p2);
        let l2_ji = dot3(r_ji, r_ji);
        let l2_jk = dot3(r_jk, r_jk);
        let l2_jl = dot3(r_jl, r_jl);
        let d_ji = l2_ji.sqrt();
        let d_jk = l2_jk.sqrt();
        let d_jl = l2_jl.sqrt();
        // energy (calculateCosY)
        let ne = divs(cross3(r_ji, r_jk), d_ji * d_jk);
        let l2n = dot3(ne, ne);
        let cy = dot3(ne, r_jl) / (d_jl * l2n.sqrt());
        let degenerate = l2_ji.lt(ZERO_TOL) | l2_jk.lt(ZERO_TOL) | l2_jl.lt(ZERO_TOL) | l2n.lt(ZERO_TOL);
        let cy = L::select(degenerate, zero(), cy);
        let sysq = one::<L>() - cy * cy;
        let sy = L::select(sysq.gt(0.0), sysq.sqrt(), zero());
        let cos2w = sy * 2.0 * sy - 1.0;
        let e = fc * (c0 + c1 * sy + c2 * cos2w);
        if G {
            let skip = is_dz(d_ji) | is_dz(d_jk) | is_dz(d_jl);
            let rji = divs(r_ji, d_ji);
            let rjk = divs(r_jk, d_jk);
            let rjl = divs(r_jl, d_jl);
            let n = cross3(neg3(rji), rjk);
            let n = divs(n, len3(n));
            let cos_y = clamp1(dot3(n, rjl));
            let sin_y = (one::<L>() - cos_y * cos_y).sqrt().max(1.0e-8);
            let cos_theta = clamp1(dot3(rji, rjk));
            let stsq = one::<L>() - cos_theta * cos_theta;
            let sin_theta = stsq.sqrt().max(1.0e-8);
            let de = -fc * (c1 * cos_y + c2 * 4.0 * cos_y * sin_y);
            let de = L::select(skip, zero(), de);
            let t1 = cross3(rjl, rjk);
            let t2 = cross3(rji, rjl);
            let t3 = cross3(rjk, rji);
            let term1 = sin_y * sin_theta;
            let term2 = cos_y / (sin_y * stsq);
            let mut g = [[zero::<L>(); 3]; 4];
            for d in 0..3 {
                let tg1 = (t1[d] / term1 - (rji[d] - rjk[d] * cos_theta) * term2) / d_ji;
                let tg3 = (t2[d] / term1 - (rjk[d] - rji[d] * cos_theta) * term2) / d_jk;
                let tg4 = (t3[d] / term1 - rjl[d] * cos_y / sin_y) / d_jl;
                g[0][d] = L::select(skip, zero(), de * tg1);
                g[1][d] = L::select(skip, zero(), -de * (tg1 + tg3 + tg4));
                g[2][d] = L::select(skip, zero(), de * tg3);
                g[3][d] = L::select(skip, zero(), de * tg4);
            }
            scatter(grad, [&self.a, &self.b, &self.c, &self.d], t, g);
        }
        e
    }
}

// ---------------------------------------------------------------------------
// the bonded part of a fast force field
// ---------------------------------------------------------------------------

/// Kinds of bonded tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableKind {
    UffBond,
    UffAngle,
    UffTorsion,
    UffInversion,
    MmffBond,
    MmffAngle,
    MmffStretchBend,
    MmffOop,
    MmffTorsion,
}

impl TableKind {
    pub const ALL: [TableKind; 9] = [
        TableKind::UffBond,
        TableKind::UffAngle,
        TableKind::UffTorsion,
        TableKind::UffInversion,
        TableKind::MmffBond,
        TableKind::MmffAngle,
        TableKind::MmffStretchBend,
        TableKind::MmffOop,
        TableKind::MmffTorsion,
    ];
    /// Breakdown label (same as the corresponding contrib).
    pub fn label(self) -> &'static str {
        match self {
            TableKind::UffBond | TableKind::MmffBond => "BondStretch",
            TableKind::UffAngle | TableKind::MmffAngle => "AngleBend",
            TableKind::UffTorsion | TableKind::MmffTorsion => "Torsion",
            TableKind::UffInversion => "Inversion",
            TableKind::MmffStretchBend => "StretchBend",
            TableKind::MmffOop => "OopBend",
        }
    }
    /// `true` if the reference contrib holds all terms of the table (MMFF:
    /// the energy is summed per table, then added to the total); `false`
    /// if every term is a contrib of its own (UFF).
    pub fn grouped(self) -> bool {
        matches!(
            self,
            TableKind::MmffBond | TableKind::MmffAngle | TableKind::MmffStretchBend | TableKind::MmffOop | TableKind::MmffTorsion
        )
    }
    /// Relative cost of one term (for chunk balancing).
    fn cost(self) -> usize {
        match self {
            TableKind::UffBond | TableKind::MmffBond => 1,
            TableKind::UffAngle | TableKind::MmffAngle | TableKind::MmffStretchBend => 2,
            _ => 3,
        }
    }
}

/// All bonded tables of a molecule plus the fixed parallel chunk plan.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bonded {
    pub uff_bonds: UffBonds,
    pub uff_angles: UffAngles,
    pub uff_torsions: UffTorsions,
    pub uff_inversions: UffInversions,
    pub mmff_bonds: MmffBonds,
    pub mmff_angles: MmffAngles,
    pub mmff_stbn: MmffStretchBends,
    pub mmff_oops: MmffOops,
    pub mmff_torsions: MmffTorsions,
    /// Parallel chunks: each is a list of `(table, term range)`; empty =
    /// serial evaluation. Depends only on the table sizes and the grain.
    pub chunks: Vec<Vec<(TableKind, Range<usize>)>>,
}

/// Maximum number of parallel chunks (bounded so that per-chunk gradient
/// buffers stay small; independent of the thread count).
pub const MAX_CHUNKS: usize = 64;

impl Bonded {
    pub fn len_of(&self, k: TableKind) -> usize {
        match k {
            TableKind::UffBond => self.uff_bonds.len(),
            TableKind::UffAngle => self.uff_angles.len(),
            TableKind::UffTorsion => self.uff_torsions.len(),
            TableKind::UffInversion => self.uff_inversions.len(),
            TableKind::MmffBond => self.mmff_bonds.len(),
            TableKind::MmffAngle => self.mmff_angles.len(),
            TableKind::MmffStretchBend => self.mmff_stbn.len(),
            TableKind::MmffOop => self.mmff_oops.len(),
            TableKind::MmffTorsion => self.mmff_torsions.len(),
        }
    }

    pub fn num_terms(&self) -> usize {
        TableKind::ALL.iter().map(|&k| self.len_of(k)).sum()
    }

    /// Evaluates `r` of table `k`, adding the energies to `acc` term by term.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn eval<const G: bool>(&self, k: TableKind, pos: &[f64], grad: &mut [f64], r: Range<usize>, simd: bool, acc: &mut f64) {
        macro_rules! go {
            ($t:expr) => {
                if simd { run_simd::<_, G>($t, pos, grad, r, acc) } else { run::<_, f64, G>($t, pos, grad, r, acc) }
            };
        }
        match k {
            TableKind::UffBond => go!(&self.uff_bonds),
            TableKind::UffAngle => go!(&self.uff_angles),
            TableKind::UffTorsion => go!(&self.uff_torsions),
            TableKind::UffInversion => go!(&self.uff_inversions),
            TableKind::MmffBond => go!(&self.mmff_bonds),
            TableKind::MmffAngle => go!(&self.mmff_angles),
            TableKind::MmffStretchBend => go!(&self.mmff_stbn),
            TableKind::MmffOop => go!(&self.mmff_oops),
            TableKind::MmffTorsion => go!(&self.mmff_torsions),
        }
    }

    /// Energy of a whole table.
    pub fn table_energy(&self, k: TableKind, pos: &[f64], simd: bool) -> f64 {
        let mut e = 0.0;
        self.eval::<false>(k, pos, &mut [], 0..self.len_of(k), simd, &mut e);
        e
    }

    /// Evaluates a whole table into the running total `acc` with the
    /// reference contribs' summation order (see [`TableKind::grouped`]).
    #[inline]
    pub fn eval_table<const G: bool>(&self, k: TableKind, pos: &[f64], grad: &mut [f64], simd: bool, acc: &mut f64) {
        let n = self.len_of(k);
        if n == 0 {
            return;
        }
        if k.grouped() {
            let mut local = 0.0;
            self.eval::<G>(k, pos, grad, 0..n, simd, &mut local);
            *acc += local;
        } else {
            self.eval::<G>(k, pos, grad, 0..n, simd, acc);
        }
    }

    /// Computes the parallel chunk plan for the given grain (minimum
    /// weighted number of terms per chunk).
    pub fn plan(&mut self, grain: usize) {
        let grain = grain.max(1);
        let total: usize = TableKind::ALL.iter().map(|&k| self.len_of(k) * k.cost()).sum();
        let n_chunks = (total / grain).clamp(1, MAX_CHUNKS);
        self.chunks.clear();
        if n_chunks <= 1 {
            return;
        }
        let per = total.div_ceil(n_chunks);
        let mut cur: Vec<(TableKind, Range<usize>)> = Vec::new();
        let mut cur_cost = 0;
        for k in TableKind::ALL {
            let n = self.len_of(k);
            let c = k.cost();
            let mut s = 0;
            while s < n {
                let room = (per - cur_cost).div_ceil(c).max(1);
                // keep SIMD blocks aligned
                let mut e = (s + room).min(n);
                if e < n {
                    e = (s + (e - s).div_ceil(4) * 4).min(n);
                }
                cur.push((k, s..e));
                cur_cost += (e - s) * c;
                s = e;
                if cur_cost >= per {
                    self.chunks.push(std::mem::take(&mut cur));
                    cur_cost = 0;
                }
            }
        }
        if !cur.is_empty() {
            self.chunks.push(cur);
        }
        if self.chunks.len() <= 1 {
            self.chunks.clear();
        }
    }
}
