//! Branch-free complementary error function for scalar and SIMD code.
//!
//! `erfc(x) = exp(-x²) · erfcx(x)` for `x ≥ 0`. The scaled function
//! `h(x) = (x + K) · erfcx(x)` (`K = 2`) is smooth on `[0, ∞)` and is
//! approximated by a degree-20 Chebyshev series in the variable
//! `t = (x - K) / (x + K)`, fitted on `x ∈ [0, 8]` (`t ∈ [-1, 0.6]`) against
//! 110-digit reference values (`tools/fit_erfc.py`). The maximum relative
//! error of the series is 2.4e-15; together with the exponential the
//! function stays below ~1e-14 relative error on `[0, 8]` (checked against
//! the high-precision table `tests/data/erfc_reference.rs`). Beyond `x = 8`
//! (`erfc < 1.2e-29`) the series argument is clamped, which keeps the
//! result positive and tiny.
//!
//! The same code runs on `f64` and on `wide::f64x4` (one division, one
//! exponential and a Clenshaw recurrence; no branches).

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

const K: f64 = 2.0;
/// Chebyshev coefficients of `h` in `u = (2t - (t0 + t1)) / (t1 - t0)`,
/// `t0 = -1`, `t1 = 0.6`.
const C: [f64; 21] = [
    1.2613618203624972,
    -0.6481722720349398,
    0.08971441914264203,
    -0.0019513067668896307,
    -0.0011738365250954647,
    4.9506406006450516e-05,
    2.399540986781604e-05,
    -5.116356114444789e-08,
    -5.78313112994305e-07,
    -4.8040309865920996e-08,
    1.0915188019145735e-08,
    2.6848181320554665e-09,
    2.552755769461299e-11,
    -8.126050367611704e-11,
    -1.366787924235073e-11,
    4.565523975612429e-13,
    5.617014138945596e-13,
    8.980641136317777e-14,
    -2.8287561446684936e-15,
    -4.019444146966453e-15,
    -9.307000792329291e-16,
];

/// Returns `(erfc(x), exp(-x²))` for `x ≥ 0`.
#[inline(always)]
pub(crate) fn erfc_exp<T: Real>(x: T) -> (T, T) {
    let e = (-(x * x)).exp();
    let inv = T::splat(1.0) / (x + T::splat(K));
    // t = (x - K)/(x + K) = 1 - 2K/(x + K);  u = 1.25 t + 0.25
    let t = T::splat(1.0) - T::splat(2.0 * K) * inv;
    let u = t.mul_add(T::splat(1.25), T::splat(0.25)).min(T::splat(1.0));
    let u2 = u + u;
    // Clenshaw recurrence
    let mut b1 = T::splat(0.0);
    let mut b2 = T::splat(0.0);
    for &c in C[1..].iter().rev() {
        let b0 = u2.mul_add(b1, T::splat(c)) - b2;
        b2 = b1;
        b1 = b0;
    }
    let h = u.mul_add(b1, T::splat(C[0])) - b2;
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
