//! Scalar math used by the expression evaluator (generic over `f32`/`f64`).

use std::ops::{Add, Div, Mul, Neg, Sub};

const FRAC_2_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

/// `erf(x)` to ~1e-15 relative accuracy.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let a = x.abs();
    let v = if a < 2.0 { erf_series(a) } else { 1.0 - erfc_cf(a) };
    if x < 0.0 { -v } else { v }
}

/// `erfc(x)` to ~1e-14 relative accuracy (no cancellation for large `x`).
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 2.0 { 1.0 - erf_series(x) } else { erfc_cf(x) }
}

/// `erf(x) = 2/sqrt(pi) exp(-x^2) sum_n (2x^2)^n x / (2n+1)!!` (all terms
/// positive, so no cancellation).
fn erf_series(x: f64) -> f64 {
    let x2 = 2.0 * x * x;
    let mut t = x;
    let mut s = x;
    let mut n = 0.0;
    while t > 1e-17 * s {
        n += 1.0;
        t *= x2 / (2.0 * n + 1.0);
        s += t;
    }
    FRAC_2_SQRT_PI * (-x * x).exp() * s
}

/// Continued fraction `erfc(x) = exp(-x^2)/sqrt(pi) / (x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))`
/// evaluated with the modified Lentz algorithm (`x >= 2`).
fn erfc_cf(x: f64) -> f64 {
    if x > 27.0 {
        return 0.0;
    }
    const TINY: f64 = 1e-300;
    let mut f = x;
    let mut c = x;
    let mut d = 0.0;
    let mut n = 1.0;
    loop {
        let a = 0.5 * n;
        d = x + a * d;
        if d.abs() < TINY {
            d = TINY;
        }
        d = 1.0 / d;
        c = x + a / c;
        if c.abs() < TINY {
            c = TINY;
        }
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-16 || n > 500.0 {
            break;
        }
        n += 1.0;
    }
    (-x * x).exp() * (0.5 * FRAC_2_SQRT_PI) / f
}

/// Floating-point type the evaluator runs on.
pub trait Real:
    Copy
    + Send
    + Sync
    + PartialOrd
    + Default
    + std::fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn acos(self) -> Self;
    fn asin(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, x: Self) -> Self;
    fn abs(self) -> Self;
    fn powf(self, e: Self) -> Self;
    fn cosh(self) -> Self;
    fn sinh(self) -> Self;
    fn tanh(self) -> Self;
    fn erf(self) -> Self;
    fn erfc(self) -> Self;
}

macro_rules! impl_real {
    ($t:ty) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            #[inline]
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            #[inline]
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            #[inline]
            fn ln(self) -> Self {
                <$t>::ln(self)
            }
            #[inline]
            fn sin(self) -> Self {
                <$t>::sin(self)
            }
            #[inline]
            fn cos(self) -> Self {
                <$t>::cos(self)
            }
            #[inline]
            fn tan(self) -> Self {
                <$t>::tan(self)
            }
            #[inline]
            fn acos(self) -> Self {
                <$t>::acos(self)
            }
            #[inline]
            fn asin(self) -> Self {
                <$t>::asin(self)
            }
            #[inline]
            fn atan(self) -> Self {
                <$t>::atan(self)
            }
            #[inline]
            fn atan2(self, x: Self) -> Self {
                <$t>::atan2(self, x)
            }
            #[inline]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            #[inline]
            fn powf(self, e: Self) -> Self {
                <$t>::powf(self, e)
            }
            #[inline]
            fn cosh(self) -> Self {
                <$t>::cosh(self)
            }
            #[inline]
            fn sinh(self) -> Self {
                <$t>::sinh(self)
            }
            #[inline]
            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }
            #[inline]
            fn erf(self) -> Self {
                erf(self as f64) as $t
            }
            #[inline]
            fn erfc(self) -> Self {
                erfc(self as f64) as $t
            }
        }
    };
}
impl_real!(f64);
impl_real!(f32);

/// `x^n` by repeated squaring (deterministic across platforms).
#[inline]
pub fn powi<T: Real>(x: T, n: i32) -> T {
    match n {
        0 => T::ONE,
        1 => x,
        2 => x * x,
        3 => x * x * x,
        4 => {
            let x2 = x * x;
            x2 * x2
        }
        -1 => T::ONE / x,
        -2 => T::ONE / (x * x),
        _ => {
            let mut e = n.unsigned_abs();
            let mut b = x;
            let mut r = T::ONE;
            while e > 0 {
                if e & 1 == 1 {
                    r = r * b;
                }
                b = b * b;
                e >>= 1;
            }
            if n < 0 { T::ONE / r } else { r }
        }
    }
}

#[inline]
pub fn step<T: Real>(x: T) -> T {
    if x >= T::ZERO { T::ONE } else { T::ZERO }
}

#[inline]
pub fn sign<T: Real>(x: T) -> T {
    if x > T::ZERO {
        T::ONE
    } else if x < T::ZERO {
        -T::ONE
    } else {
        T::ZERO
    }
}

#[inline]
pub fn min<T: Real>(a: T, b: T) -> T {
    if b < a { b } else { a }
}

#[inline]
pub fn max<T: Real>(a: T, b: T) -> T {
    if b > a { b } else { a }
}

#[inline]
pub fn select<T: Real>(c: T, a: T, b: T) -> T {
    if c != T::ZERO { a } else { b }
}
