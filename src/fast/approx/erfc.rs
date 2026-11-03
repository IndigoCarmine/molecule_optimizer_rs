//! Branch-free complementary error function for scalar and SIMD code.
//!
//! `erfc(x) = exp(-x²) · erfcx(x)` for `x ≥ 0`. The scaled function
//! `h(x) = (x + K) · erfcx(x)` (`K = 2`) is smooth on `[0, ∞)` and is
//! approximated by a degree-20 polynomial in the variable
//! `t = (x - K) / (x + K)`, obtained from a Chebyshev least-squares fit on
//! `x ∈ [0, 8]` (`t ∈ [-1, 0.6]`) against 110-digit reference values
//! (`tools/fit_erfc.py`). The maximum relative error of the polynomial is
//! ~7e-15 (Horner evaluation in double precision included); together with
//! the exponential the function stays below ~1e-14 relative error on
//! `[0, 8]` (checked against the high-precision table
//! `tests/data/erfc_reference.rs`). Beyond `x = 8` (`erfc < 1.2e-29`) the
//! polynomial argument is clamped, which keeps the result positive and
//! tiny.
//!
//! The same code runs on `f64` and on `wide::f64x4` (one division, one
//! exponential and 21 multiply-adds in Estrin form; no branches).

use std::ops::{Add, Div, Mul, Neg, Sub};
use wide::f64x4;

/// Minimal arithmetic needed by the kernels (implemented for `f64` and
/// `f64x4`).
pub(crate) trait Real:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self>
{
    fn splat(x: f64) -> Self;
    /// `self * m + a`
    fn mul_add(self, m: Self, a: Self) -> Self;
    fn exp(self) -> Self;
    fn min(self, o: Self) -> Self;
}

impl Real for f64 {
    #[inline(always)]
    fn splat(x: f64) -> Self {
        x
    }
    #[inline(always)]
    fn mul_add(self, m: Self, a: Self) -> Self {
        self * m + a
    }
    #[inline(always)]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline(always)]
    fn min(self, o: Self) -> Self {
        f64::min(self, o)
    }
}

impl Real for f64x4 {
    #[inline(always)]
    fn splat(x: f64) -> Self {
        f64x4::splat(x)
    }
    #[inline(always)]
    fn mul_add(self, m: Self, a: Self) -> Self {
        self * m + a
    }
    #[inline(always)]
    fn exp(self) -> Self {
        f64x4::exp(self)
    }
    #[inline(always)]
    fn min(self, o: Self) -> Self {
        f64x4::min(self, o)
    }
}

/// Shift `K` of the rational variable `t = (x - K)/(x + K)`.
pub(crate) const ERFC_K: f64 = 2.0;
const K: f64 = ERFC_K;
/// Coefficients of `h` as a polynomial in `u = (2t - (t0 + t1)) / (t1 - t0)`
/// (`t0 = -1`, `t1 = 0.6`, so `u = 1.25 t + 0.25 ∈ [-1, 1]`), obtained from
/// the Chebyshev fit by conversion to the monomial basis; the sum of their
/// magnitudes is 2.02, so Horner evaluation is well conditioned.
const A: [f64; 21] = [
    1.17044898009635,
    -0.6420709245167305,
    0.189270496393079,
    -0.008791835093310177,
    -0.010639323657391514,
    0.0007735334199295816,
    0.000927894079440755,
    3.2774558019340054e-05,
    -8.740409335757817e-05,
    -2.1218791460463667e-05,
    4.791704392405784e-06,
    3.825305951101819e-06,
    5.62630599285333e-07,
    -3.0617147936597934e-07,
    -1.8261953643157577e-07,
    -2.754334074391232e-08,
    1.4889779561398092e-08,
    1.089050051579349e-08,
    2.0690036903103812e-09,
    -1.053673166462374e-09,
    -4.879548831408739e-10,
];

/// Returns `(erfc(x), exp(-x²))` for `x ≥ 0`.
#[inline(always)]
pub(crate) fn erfc_exp<T: Real>(x: T) -> (T, T) {
    erfc_exp_inv(x, T::splat(1.0) / (x + T::splat(K)))
}

/// [`erfc_exp`] with a precomputed `inv = 1/(x + ERFC_K)` (lets SIMD
/// kernels share one division between several reciprocals).
#[inline(always)]
pub(crate) fn erfc_exp_inv<T: Real>(x: T, inv: T) -> (T, T) {
    let e = (-(x * x)).exp();
    // t = (x - K)/(x + K) = 1 - 2K/(x + K);  u = 1.25 t + 0.25
    let t = T::splat(1.0) - T::splat(2.0 * K) * inv;
    let u = t.mul_add(T::splat(1.25), T::splat(0.25)).min(T::splat(1.0));
    // Estrin's scheme (short dependency chains)
    let c = |k: usize| T::splat(A[k]);
    let u2 = u * u;
    let u4 = u2 * u2;
    let u8 = u4 * u4;
    let u16 = u8 * u8;
    let b = |k: usize| c(2 * k + 1).mul_add(u, c(2 * k));
    let (b0, b1, b2, b3, b4, b5, b6, b7, b8, b9) = (b(0), b(1), b(2), b(3), b(4), b(5), b(6), b(7), b(8), b(9));
    let c0 = b1.mul_add(u2, b0);
    let c1 = b3.mul_add(u2, b2);
    let c2 = b5.mul_add(u2, b4);
    let c3 = b7.mul_add(u2, b6);
    let c4 = b9.mul_add(u2, b8);
    let d0 = c1.mul_add(u4, c0);
    let d1 = c3.mul_add(u4, c2);
    let d2 = c(20).mul_add(u4, c4);
    let e0 = d1.mul_add(u8, d0);
    let h = d2.mul_add(u16, e0);
    (e * h * inv, e)
}

/// `erfc(x)` for `x ≥ 0`.
#[inline]
pub fn erfc(x: f64) -> f64 {
    erfc_exp(x.max(0.0)).0
}

#[cfg(test)]
mod tests {
    use super::*;
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/erfc_reference.rs"));

    #[test]
    fn scalar_and_simd_match_high_precision_reference() {
        let mut worst: f64 = 0.0;
        for &(x, r) in ERFC_REF {
            let v = erfc(x);
            let rel = ((v - r) / r).abs();
            worst = worst.max(rel);
            let (s, e) = erfc_exp(f64x4::splat(x));
            for l in s.to_array() {
                assert!(((l - r) / r).abs() < 1e-13, "simd x={x}: {l} vs {r}");
            }
            let ee = (-(x * x)).exp();
            assert!(e.to_array().iter().all(|&y| ((y - ee) / ee).abs() < 1e-14));
        }
        assert!(worst < 1e-13, "max relative error {worst}");
        println!("erfc max relative error vs reference: {worst:e}");
    }

    #[test]
    fn large_arguments_stay_tiny_and_positive() {
        for x in [8.0, 9.0, 12.0, 26.0, 40.0] {
            let v = erfc(x);
            assert!((0.0..1e-28).contains(&v), "{x}: {v}");
        }
    }
}
