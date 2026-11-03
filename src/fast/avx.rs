//! 256-bit AVX lanes selected at run time.
//!
//! [`super::bonded::V4`] (crate `wide`) picks its instruction set at compile
//! time, so on the default x86-64 target it runs as two 128-bit halves.
//! [`A4`] implements the same [`Lane`] operations with AVX intrinsics; the
//! kernels are instantiated with it inside `#[target_feature(enable = "avx")]`
//! entry points that are called only when the CPU supports AVX. Plain AVX
//! adds, multiplies, divisions, square roots, comparisons and blends are the
//! same IEEE operations as the SSE2 ones (no FMA contraction), and `max` /
//! `min` keep `f64::max` / `f64::min` semantics, so results are identical
//! to the [`super::bonded::V4`] and scalar kernels.

#![cfg(target_arch = "x86_64")]

use super::bonded::Lane;
use std::arch::x86_64::*;
use std::ops::{Add, BitAnd, BitOr, Div, Mul, Neg, Not, Sub};

/// `true` if the AVX kernels can be used on this CPU (cached by `std`).
#[inline]
pub fn available() -> bool {
    std::arch::is_x86_feature_detected!("avx")
}

/// Four f64 lanes in one AVX register. Only constructed inside AVX-enabled
/// code (see [`available`]).
#[derive(Clone, Copy, Debug)]
pub struct A4(__m256d);

/// Lane mask of [`A4`] (all-ones / all-zeros per lane).
#[derive(Clone, Copy, Debug)]
pub struct MA4(__m256d);

// SAFETY (all `unsafe` blocks below): `A4` / `MA4` values only exist in code
// reached through an AVX entry point guarded by [`available`].

impl BitAnd for MA4 {
    type Output = MA4;
    #[inline(always)]
    fn bitand(self, o: MA4) -> MA4 {
        MA4(unsafe { _mm256_and_pd(self.0, o.0) })
    }
}
impl BitOr for MA4 {
    type Output = MA4;
    #[inline(always)]
    fn bitor(self, o: MA4) -> MA4 {
        MA4(unsafe { _mm256_or_pd(self.0, o.0) })
    }
}
impl Not for MA4 {
    type Output = MA4;
    #[inline(always)]
    fn not(self) -> MA4 {
        MA4(unsafe { _mm256_xor_pd(self.0, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))) })
    }
}

macro_rules! a4_ops {
    ($tr:ident, $f:ident, $intr:ident) => {
        impl $tr for A4 {
            type Output = A4;
            #[inline(always)]
            fn $f(self, o: A4) -> A4 {
                A4(unsafe { $intr(self.0, o.0) })
            }
        }
        impl $tr<f64> for A4 {
            type Output = A4;
            #[inline(always)]
            fn $f(self, o: f64) -> A4 {
                A4(unsafe { $intr(self.0, _mm256_set1_pd(o)) })
            }
        }
    };
}
a4_ops!(Add, add, _mm256_add_pd);
a4_ops!(Sub, sub, _mm256_sub_pd);
a4_ops!(Mul, mul, _mm256_mul_pd);
a4_ops!(Div, div, _mm256_div_pd);

impl Neg for A4 {
    type Output = A4;
    #[inline(always)]
    fn neg(self) -> A4 {
        // sign-bit flip, identical to scalar negation (also for zeros)
        A4(unsafe { _mm256_xor_pd(self.0, _mm256_set1_pd(-0.0)) })
    }
}

impl A4 {
    #[inline(always)]
    fn to_array(self) -> [f64; 4] {
        let mut a = [0.0; 4];
        unsafe { _mm256_storeu_pd(a.as_mut_ptr(), self.0) };
        a
    }
    #[inline(always)]
    fn cmp<const OP: i32>(self, o: f64) -> MA4 {
        MA4(unsafe { _mm256_cmp_pd::<OP>(self.0, _mm256_set1_pd(o)) })
    }
}

impl Lane for A4 {
    type M = MA4;
    const N: usize = 4;
    #[inline(always)]
    fn splat(x: f64) -> Self {
        A4(unsafe { _mm256_set1_pd(x) })
    }
    #[inline(always)]
    fn from_fn(mut f: impl FnMut(usize) -> f64) -> Self {
        let a = [f(0), f(1), f(2), f(3)];
        A4(unsafe { _mm256_loadu_pd(a.as_ptr()) })
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        A4(unsafe { _mm256_sqrt_pd(self.0) })
    }
    #[inline(always)]
    fn max(self, o: f64) -> Self {
        // `maxpd(a, b)` returns `b` if either is NaN: pick `a` where `b` is
        // NaN, giving `f64::max` semantics.
        unsafe {
            let b = _mm256_set1_pd(o);
            A4(_mm256_blendv_pd(_mm256_max_pd(self.0, b), self.0, _mm256_cmp_pd::<_CMP_UNORD_Q>(b, b)))
        }
    }
    #[inline(always)]
    fn min(self, o: f64) -> Self {
        unsafe {
            let b = _mm256_set1_pd(o);
            A4(_mm256_blendv_pd(_mm256_min_pd(self.0, b), self.0, _mm256_cmp_pd::<_CMP_UNORD_Q>(b, b)))
        }
    }
    #[inline(always)]
    fn gt(self, o: f64) -> MA4 {
        self.cmp::<_CMP_GT_OQ>(o)
    }
    #[inline(always)]
    fn gtv(self, o: Self) -> MA4 {
        MA4(unsafe { _mm256_cmp_pd::<_CMP_GT_OQ>(self.0, o.0) })
    }
    #[inline(always)]
    fn lt(self, o: f64) -> MA4 {
        self.cmp::<_CMP_LT_OQ>(o)
    }
    #[inline(always)]
    fn le(self, o: f64) -> MA4 {
        self.cmp::<_CMP_LE_OQ>(o)
    }
    #[inline(always)]
    fn eq(self, o: f64) -> MA4 {
        self.cmp::<_CMP_EQ_OQ>(o)
    }
    #[inline(always)]
    fn select(m: MA4, a: Self, b: Self) -> Self {
        A4(unsafe { _mm256_blendv_pd(b.0, a.0, m.0) })
    }
    #[inline(always)]
    fn any(m: MA4) -> bool {
        unsafe { _mm256_movemask_pd(m.0) != 0 }
    }
    #[inline(always)]
    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        let a = self.to_array();
        let r = [f(a[0]), f(a[1]), f(a[2]), f(a[3])];
        A4(unsafe { _mm256_loadu_pd(r.as_ptr()) })
    }
    #[inline(always)]
    fn hsum(self) -> f64 {
        let a = self.to_array();
        ((a[0] + a[1]) + a[2]) + a[3]
    }
    #[inline(always)]
    fn write(self, out: &mut [f64; 4]) {
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), self.0) };
    }
    #[inline(always)]
    fn transpose4(r: [Self; 4]) -> [Self; 4] {
        unsafe {
            let t0 = _mm256_unpacklo_pd(r[0].0, r[1].0);
            let t1 = _mm256_unpackhi_pd(r[0].0, r[1].0);
            let t2 = _mm256_unpacklo_pd(r[2].0, r[3].0);
            let t3 = _mm256_unpackhi_pd(r[2].0, r[3].0);
            [
                A4(_mm256_permute2f128_pd(t0, t2, 0x20)),
                A4(_mm256_permute2f128_pd(t1, t3, 0x20)),
                A4(_mm256_permute2f128_pd(t0, t2, 0x31)),
                A4(_mm256_permute2f128_pd(t1, t3, 0x31)),
            ]
        }
    }
}
