//! BFGS / L-BFGS with the RDKit (Numerical Recipes) backtracking line search.
//!
//! Port of RDKit `Code/Numerics/Optimizer/BFGSOpt.h` and the gradient
//! functor in `Code/ForceField/ForceField.cpp` (BSD-3-Clause,
//! Copyright (C) 2004-2008 Greg Landrum and Rational Discovery LLC).

use super::{IterationInfo, MinimizeOptions, MinimizeResult, MinimizeStatus, Observer};
use crate::ff::ForceField;
use std::ops::ControlFlow;

const FUNCTOL: f64 = 1e-4;
const MOVETOL: f64 = 1e-7;
const EPS: f64 = 3e-8;
const TOLX: f64 = 4.0 * EPS;
const MAXSTEP: f64 = 100.0;
const MAX_ITER_LINEAR_SEARCH: usize = 1000;

/// RDKit `ForceFieldsHelper::calcGradient`: computes the gradient, scales it
/// by 0.1 and halves further while any component exceeds 10. Returns the
/// scale factor applied.
fn calc_grad(ff: &dyn ForceField, pos: &[f64], grad: &mut [f64], scaling: bool) -> f64 {
    ff.energy_grad(pos, grad);
    if !scaling {
        return 1.0;
    }
    let mut scale = 0.1;
    let mut max_grad: f64 = -1e8;
    for g in grad.iter_mut() {
        *g *= scale;
        max_grad = max_grad.max(g.abs());
    }
    if max_grad > 10.0 {
        while max_grad * scale > 10.0 {
            scale *= 0.5;
        }
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
    scale
}

/// Returns `(resCode, newVal)`; `new_pt` receives the new point.
/// resCode: 0 success, 1 step too small, -1 bad direction.
fn linear_search(
    ff: &dyn ForceField,
    old_pt: &[f64],
    old_val: f64,
    grad: &[f64],
    dir: &mut [f64],
    new_pt: &mut [f64],
    max_step: f64,
) -> (i32, f64) {
    let dim = old_pt.len();
    let mut new_val = old_val;
    let sum = dir.iter().map(|d| d * d).sum::<f64>().sqrt();
    if sum > max_step {
        for d in dir.iter_mut() {
            *d *= max_step / sum;
        }
    }
    let slope: f64 = dir.iter().zip(grad).map(|(d, g)| d * g).sum();
    if slope >= 0.0 {
        new_pt.copy_from_slice(old_pt);
        return (-1, old_val);
    }
    let mut test: f64 = 0.0;
    for i in 0..dim {
        let t = dir[i].abs() / old_pt[i].abs().max(1.0);
        if t > test {
            test = t;
        }
    }
    let lambda_min = MOVETOL / test;
    let mut lambda = 1.0;
    let mut lambda2 = 0.0;
    let mut val2 = 0.0;
    let mut it = 0;
    while it < MAX_ITER_LINEAR_SEARCH {
        if lambda < lambda_min {
            new_pt.copy_from_slice(old_pt);
            return (1, old_val);
        }
        for i in 0..dim {
            new_pt[i] = old_pt[i] + lambda * dir[i];
        }
        new_val = ff.energy(new_pt);
        if new_val - old_val <= FUNCTOL * lambda * slope {
            return (0, new_val);
        }
        let tmp_lambda = if it == 0 {
            -slope / (2.0 * (new_val - old_val - slope))
        } else {
            let rhs1 = new_val - old_val - lambda * slope;
            let rhs2 = val2 - old_val - lambda2 * slope;
            let a = (rhs1 / (lambda * lambda) - rhs2 / (lambda2 * lambda2)) / (lambda - lambda2);
            let b = (-lambda2 * rhs1 / (lambda * lambda) + lambda * rhs2 / (lambda2 * lambda2)) / (lambda - lambda2);
            let mut t = if a == 0.0 {
                -slope / (2.0 * b)
            } else {
                let disc = b * b - 3.0 * a * slope;
                if disc < 0.0 {
                    0.5 * lambda
                } else if b <= 0.0 {
                    (-b + disc.sqrt()) / (3.0 * a)
                } else {
                    -slope / (b + disc.sqrt())
                }
            };
            if t > 0.5 * lambda {
                t = 0.5 * lambda;
            }
            t
        };
        lambda2 = lambda;
        val2 = new_val;
        // NaN-safe: fall back to halving
        lambda = if tmp_lambda.is_finite() { tmp_lambda.max(0.1 * lambda) } else { 0.5 * lambda };
        it += 1;
    }
    new_pt.copy_from_slice(old_pt);
    let _ = new_val;
    (1, old_val)
}

fn grad_converged(grad: &[f64], pos: &[f64], func_val: f64, grad_scale: f64, grad_tol: f64) -> bool {
    let term = (func_val.abs() * grad_scale).max(1.0);
    let mut test: f64 = 0.0;
    for i in 0..grad.len() {
        test = test.max(grad[i].abs() * pos[i].abs().max(1.0));
    }
    test / term < grad_tol
}

fn notify(observer: &mut dyn Observer, iteration: usize, pos: &[f64], energy: f64, grad: &[f64]) -> bool {
    matches!(observer.on_iteration(&IterationInfo { iteration, pos, energy, grad }), ControlFlow::Break(()))
}

/// Endless mode ([`MinimizeOptions::is_endless`]): the line search could not
/// move (bad direction or no measurable step) and the minimizer restarts
/// from steepest descent. `stalled_before` = the previous iteration was such
/// a restart too; then the point is stationary to machine precision, and a
/// short sleep keeps the loop from spinning hot while it stays responsive
/// to the observer (cancellation, publishing). Returns `true` to stop.
fn endless_restart(observer: &mut dyn Observer, iteration: usize, pos: &[f64], energy: f64, grad: &[f64], stalled_before: bool) -> bool {
    if stalled_before {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    notify(observer, iteration, pos, energy, grad)
}

/// Packed upper triangle of the symmetric `dim × dim` inverse Hessian: row
/// `i` holds `H[i][i..dim]`, rows stored one after another (`dim·(dim+1)/2`
/// values, half the memory of the full matrix).
struct PackedSym {
    dim: usize,
    a: Vec<f64>,
}

impl PackedSym {
    fn identity(dim: usize) -> Self {
        let mut h = PackedSym { dim, a: vec![0.0; dim * (dim + 1) / 2] };
        for i in 0..dim {
            let o = h.row_start(i);
            h.a[o] = 1.0;
        }
        h
    }
    /// Offset of `H[i][i]`.
    #[inline(always)]
    fn row_start(&self, i: usize) -> usize {
        i * self.dim - i * i.saturating_sub(1) / 2
    }
    fn reset_identity(&mut self) {
        self.a.fill(0.0);
        for i in 0..self.dim {
            let o = self.row_start(i);
            self.a[o] = 1.0;
        }
    }
}

/// Pending rank-2 BFGS update, applied lazily by the next [`fused_pass`]:
/// `H[i][j] += xi[i]·w[j] − fac·hd[i]·xi[j]` with `w = (fac + fae·fac²)·xi
/// − fac·hd`. This is RDKit's `H += fac·xi·xiᵀ − fad·hd·hdᵀ + fae·u·uᵀ`
/// (`hd = H·dgrad`, `u = fac·xi − fad·hd`, `fad = 1/fae`) with the
/// `hd·hdᵀ` terms, which cancel exactly, removed.
struct Update<'a> {
    xi: &'a [f64],
    hd: &'a [f64],
    w: &'a [f64],
    fac: f64,
}

/// Four f64 lanes with multiply-add (fused where the CPU has FMA), for
/// [`fused_pass`].
trait Fma4: Copy {
    fn ld(s: &[f64; 4]) -> Self;
    fn st(self, s: &mut [f64; 4]);
    fn splat(x: f64) -> Self;
    /// `self·b + c`.
    fn fma(self, b: Self, c: Self) -> Self;
    fn hsum(self) -> f64;
}

/// Portable lanes (`a·b + c` without contraction).
impl Fma4 for [f64; 4] {
    #[inline(always)]
    fn ld(s: &[f64; 4]) -> Self {
        *s
    }
    #[inline(always)]
    fn st(self, s: &mut [f64; 4]) {
        *s = self;
    }
    #[inline(always)]
    fn splat(x: f64) -> Self {
        [x; 4]
    }
    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        std::array::from_fn(|l| self[l] * b[l] + c[l])
    }
    #[inline(always)]
    fn hsum(self) -> f64 {
        (self[0] + self[1]) + (self[2] + self[3])
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::*;

    /// `true` if the CPU has AVX (cached by `std`).
    #[inline]
    pub fn has_avx() -> bool {
        std::arch::is_x86_feature_detected!("avx")
    }

    /// `true` if the CPU has AVX and FMA (cached by `std`).
    #[inline]
    pub fn has_fma() -> bool {
        has_avx() && std::arch::is_x86_feature_detected!("fma")
    }

    /// AVX register; `FMA` selects fused multiply-add. Only constructed
    /// inside `#[target_feature]` entry points guarded by [`has_avx`] /
    /// [`has_fma`] (SAFETY of all `unsafe` blocks below).
    #[derive(Clone, Copy)]
    pub struct F4<const FMA: bool>(__m256d);

    impl<const FMA: bool> super::Fma4 for F4<FMA> {
        #[inline(always)]
        fn ld(s: &[f64; 4]) -> Self {
            F4(unsafe { _mm256_loadu_pd(s.as_ptr()) })
        }
        #[inline(always)]
        fn st(self, s: &mut [f64; 4]) {
            unsafe { _mm256_storeu_pd(s.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        fn splat(x: f64) -> Self {
            F4(unsafe { _mm256_set1_pd(x) })
        }
        #[inline(always)]
        fn fma(self, b: Self, c: Self) -> Self {
            F4(unsafe { if FMA { _mm256_fmadd_pd(self.0, b.0, c.0) } else { _mm256_add_pd(_mm256_mul_pd(self.0, b.0), c.0) } })
        }
        #[inline(always)]
        fn hsum(self) -> f64 {
            unsafe {
                let s = _mm_add_pd(_mm256_castpd256_pd128(self.0), _mm256_extractf128_pd::<1>(self.0));
                _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)))
            }
        }
    }
}

/// One sweep over the packed inverse Hessian: applies the pending update
/// `upd` (if any) to every stored element and computes `oa = H·a` and
/// `ob = H·b` for the updated `H`. The matrix is read and written once per
/// BFGS iteration (RDKit traverses it twice: `H·dgrad`, then update and
/// `H·grad`).
///
/// Mathematically identical to RDKit `BFGSOpt.h`; the summation order
/// differs (lane-parallel partial sums, FMA where the CPU has it), so
/// results agree to rounding, not bit for bit.
fn fused_pass(h: &mut PackedSym, upd: Option<&Update<'_>>, a: &[f64], b: &[f64], oa: &mut [f64], ob: &mut [f64]) {
    let dim = h.dim;
    assert!(a.len() == dim && b.len() == dim && oa.len() == dim && ob.len() == dim && h.a.len() == dim * (dim + 1) / 2);
    if let Some(u) = upd {
        assert!(u.xi.len() == dim && u.hd.len() == dim && u.w.len() == dim);
    }
    oa.fill(0.0);
    ob.fill(0.0);
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx,fma")]
        unsafe fn fma(h: &mut PackedSym, upd: Option<&Update<'_>>, a: &[f64], b: &[f64], oa: &mut [f64], ob: &mut [f64]) {
            fused_pass_lanes::<avx::F4<true>>(h, upd, a, b, oa, ob)
        }
        #[target_feature(enable = "avx")]
        unsafe fn no_fma(h: &mut PackedSym, upd: Option<&Update<'_>>, a: &[f64], b: &[f64], oa: &mut [f64], ob: &mut [f64]) {
            fused_pass_lanes::<avx::F4<false>>(h, upd, a, b, oa, ob)
        }
        // SAFETY: the CPU supports the enabled features (checked here).
        if avx::has_fma() {
            return unsafe { fma(h, upd, a, b, oa, ob) };
        }
        if avx::has_avx() {
            return unsafe { no_fma(h, upd, a, b, oa, ob) };
        }
    }
    fused_pass_lanes::<[f64; 4]>(h, upd, a, b, oa, ob)
}

/// [`fused_pass`] two rows `i, i+1` at a time: each row adds its row terms
/// `H[i][j]·a[j]` to `oa[i]` (lane-parallel partial sums) and its column
/// terms `H[i][j]·a[i]` to `oa[j]`, `j > i` (likewise for `b`); pairing
/// the rows halves the loads and stores of `oa` / `ob`.
#[inline(always)]
fn fused_pass_lanes<V: Fma4>(h: &mut PackedSym, upd: Option<&Update<'_>>, a: &[f64], b: &[f64], oa: &mut [f64], ob: &mut [f64]) {
    let dim = h.dim;
    let none = Update { xi: a, hd: a, w: a, fac: 0.0 };
    let mut rest = &mut h.a[..];
    let mut i = 0;
    while i + 1 < dim {
        let (r0, tail) = std::mem::take(&mut rest).split_at_mut(dim - i);
        let (r1, tail) = tail.split_at_mut(dim - i - 1);
        rest = tail;
        match upd {
            Some(u) => fused_rows::<V, true>(r0, r1, u, a, b, oa, ob, i),
            None => fused_rows::<V, false>(r0, r1, &none, a, b, oa, ob, i),
        }
        i += 2;
    }
    if i < dim {
        // last row of an odd dimension: only the diagonal element
        let e = &mut rest[0];
        if let Some(u) = upd {
            *e += u.xi[i] * u.w[i] - (u.fac * u.hd[i]) * u.xi[i];
        }
        oa[i] += *e * a[i];
        ob[i] += *e * b[i];
    }
}

/// Rows `i, i+1` of [`fused_pass_lanes`] (`r0` = stored `H[i][i..dim]`,
/// `r1` = `H[i+1][i+1..dim]`); the update is applied only if `UPD`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn fused_rows<V: Fma4, const UPD: bool>(r0: &mut [f64], r1: &mut [f64], u: &Update<'_>, a: &[f64], b: &[f64], oa: &mut [f64], ob: &mut [f64], i: usize) {
    let k = i + 1;
    // row factors of the update
    let (p0, q0) = (u.xi[i], u.fac * u.hd[i]);
    let (p1, q1) = (u.xi[k], u.fac * u.hd[k]);
    let elem = |e: &mut f64, (p, q): (f64, f64), j: usize| {
        if UPD {
            *e += p * u.w[j] - q * u.xi[j];
        }
    };
    let (ai, bi, ak, bk) = (a[i], b[i], a[k], b[k]);
    // the 2×2 diagonal block
    let ([h00, h01, r0 @ ..], [h11, r1 @ ..]) = (r0, r1) else { unreachable!("rows i, i+1 of a packed matrix") };
    elem(h00, (p0, q0), i);
    elem(h01, (p0, q0), k);
    elem(h11, (p1, q1), k);
    let (mut sa0, mut sb0) = (*h00 * ai + *h01 * ak, *h00 * bi + *h01 * bk);
    let (mut sa1, mut sb1) = (*h01 * ai + *h11 * ak, *h01 * bi + *h11 * bk);
    // columns j >= i+2, shared by both rows
    let j0 = i + 2;
    let (h0c, h0t) = r0.as_chunks_mut::<4>();
    let (h1c, h1t) = r1.as_chunks_mut::<4>();
    let n4 = h0c.len() * 4;
    {
        let (av0, bv0, av1, bv1) = (V::splat(ai), V::splat(bi), V::splat(ak), V::splat(bk));
        let (pv0, nq0) = (V::splat(p0), V::splat(-q0));
        let (pv1, nq1) = (V::splat(p1), V::splat(-q1));
        let zero = V::splat(0.0);
        let (mut va0, mut vb0, mut va1, mut vb1) = (zero, zero, zero, zero);
        let rows = h0c.iter_mut().zip(h1c.iter_mut());
        let outs = oa[j0..].as_chunks_mut::<4>().0.iter_mut().zip(ob[j0..].as_chunks_mut::<4>().0);
        let ins = a[j0..].as_chunks::<4>().0.iter().zip(b[j0..].as_chunks::<4>().0);
        let ucols = u.w[j0..].as_chunks::<4>().0.iter().zip(u.xi[j0..].as_chunks::<4>().0);
        for ((((h0, h1), (oac, obc)), (ac, bc)), (wc, xc)) in rows.zip(outs).zip(ins).zip(ucols) {
            let (mut e0, mut e1) = (V::ld(h0), V::ld(h1));
            if UPD {
                let (w, x) = (V::ld(wc), V::ld(xc));
                e0 = nq0.fma(x, pv0.fma(w, e0));
                e1 = nq1.fma(x, pv1.fma(w, e1));
                e0.st(h0);
                e1.st(h1);
            }
            let (av, bv) = (V::ld(ac), V::ld(bc));
            va0 = e0.fma(av, va0);
            vb0 = e0.fma(bv, vb0);
            va1 = e1.fma(av, va1);
            vb1 = e1.fma(bv, vb1);
            e1.fma(av1, e0.fma(av0, V::ld(oac))).st(oac);
            e1.fma(bv1, e0.fma(bv0, V::ld(obc))).st(obc);
        }
        sa0 += va0.hsum();
        sb0 += vb0.hsum();
        sa1 += va1.hsum();
        sb1 += vb1.hsum();
    }
    for (t, (e0, e1)) in h0t.iter_mut().zip(h1t.iter_mut()).enumerate() {
        let j = j0 + n4 + t;
        elem(e0, (p0, q0), j);
        elem(e1, (p1, q1), j);
        sa0 += *e0 * a[j];
        sb0 += *e0 * b[j];
        sa1 += *e1 * a[j];
        sb1 += *e1 * b[j];
        oa[j] += *e0 * ai + *e1 * ak;
        ob[j] += *e0 * bi + *e1 * bk;
    }
    oa[i] += sa0;
    ob[i] += sb0;
    oa[k] += sa1;
    ob[k] += sb1;
}

/// Scratch vectors of [`minimize_bfgs`] (one allocation).
struct BfgsWork {
    dim: usize,
    buf: Vec<f64>,
}

impl BfgsWork {
    fn new(dim: usize, n: usize) -> Self {
        BfgsWork { dim, buf: vec![0.0; n * dim] }
    }
    fn split<const N: usize>(&mut self) -> [&mut [f64]; N] {
        assert_eq!(self.buf.len(), N * self.dim);
        let mut it = self.buf.chunks_exact_mut(self.dim.max(1));
        std::array::from_fn(|_| it.next().unwrap_or(&mut []))
    }
}

/// RDKit BFGS minimization.
///
/// Same algorithm as RDKit `BFGSOpt.h` (line search, inverse Hessian
/// update, convergence tests), with the symmetric inverse Hessian kept as a
/// packed upper triangle (half the memory, see [`PackedSym`]) and traversed
/// once per iteration: the update of iteration `k` is applied by the sweep
/// of iteration `k + 1` that also computes `H·dgrad` and `H·grad`
/// ([`fused_pass`]); the new direction `-H'·grad` is formed from `H·grad`
/// and the rank-2 update terms. Results agree with RDKit to rounding.
///
/// With `opts.force_tol <= 0` it runs in endless mode (see
/// [`MinimizeOptions::force_tol`]): no convergence test, no iteration
/// limit, and a stalled line search resets the inverse Hessian to identity
/// instead of returning; only the observer can stop it.
pub fn minimize_bfgs(
    ff: &dyn ForceField,
    pos: &mut [f64],
    opts: &MinimizeOptions,
    observer: &mut dyn Observer,
) -> MinimizeResult {
    minimize_bfgs_impl(ff, pos, opts, opts.is_endless(), observer)
}

/// [`minimize_bfgs`] with endless mode chosen explicitly (`false` keeps the
/// RDKit semantics for any `force_tol`, as the distance-geometry embedder needs).
pub(crate) fn minimize_bfgs_impl(
    ff: &dyn ForceField,
    pos: &mut [f64],
    opts: &MinimizeOptions,
    endless: bool,
    observer: &mut dyn Observer,
) -> MinimizeResult {
    let dim = pos.len();
    let scaling = opts.rdkit_grad_scaling;
    let mut work = BfgsWork::new(dim, 9);
    let [grad, mut dgrad, mut hess_dgrad, mut xi, new_pos, hess_grad, mut up_xi, mut up_hd, mut up_w] = work.split::<9>();
    let mut inv_hessian = PackedSym::identity(dim);
    // `fac` of the update pending in `up_*`
    let mut pending: Option<f64> = None;

    let mut fp = ff.energy(pos);
    calc_grad(ff, pos, grad, scaling);
    let mut sum = 0.0;
    for i in 0..dim {
        xi[i] = -grad[i];
        sum += pos[i] * pos[i];
    }
    let max_step = MAXSTEP * sum.sqrt().max(dim as f64);
    let mut func_val = fp;
    let max_iters = if endless { usize::MAX } else { opts.max_iters };
    let mut stalled = false;
    for iter in 1..=max_iters {
        let (status, fv) = linear_search(ff, pos, fp, grad, xi, new_pos, max_step);
        func_val = fv;
        if status < 0 {
            if endless {
                // bad direction: restart from steepest descent
                inv_hessian.reset_identity();
                pending = None;
                for (x, g) in xi.iter_mut().zip(grad.iter()) {
                    *x = -g;
                }
                if endless_restart(observer, iter, pos, fp, grad, stalled) {
                    return MinimizeResult { status: MinimizeStatus::Cancelled, energy: fp, iterations: iter, grad_rms: 0.0 };
                }
                stalled = true;
                continue;
            }
            // bad direction: reset to steepest descent once, otherwise stop
            return MinimizeResult { status: MinimizeStatus::Converged, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        fp = func_val;
        let mut test: f64 = 0.0;
        for i in 0..dim {
            xi[i] = new_pos[i] - pos[i];
            pos[i] = new_pos[i];
            let t = xi[i].abs() / pos[i].abs().max(1.0);
            if t > test {
                test = t;
            }
        }
        dgrad.copy_from_slice(grad);
        ff.on_accepted_step(pos);
        if test < TOLX {
            if endless {
                // no measurable step: restart from steepest descent
                calc_grad(ff, pos, grad, scaling);
                inv_hessian.reset_identity();
                pending = None;
                for (x, g) in xi.iter_mut().zip(grad.iter()) {
                    *x = -g;
                }
                if endless_restart(observer, iter, pos, fp, grad, stalled) {
                    return MinimizeResult { status: MinimizeStatus::Cancelled, energy: fp, iterations: iter, grad_rms: 0.0 };
                }
                stalled = true;
                continue;
            }
            notify(observer, iter, pos, fp, grad);
            return MinimizeResult { status: MinimizeStatus::Converged, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        stalled = false;
        let grad_scale = calc_grad(ff, pos, grad, scaling);
        let converged = !endless && grad_converged(grad, pos, func_val, grad_scale, opts.force_tol);
        for (d, g) in dgrad.iter_mut().zip(grad.iter()) {
            *d = g - *d;
        }
        if notify(observer, iter, pos, fp, grad) {
            return MinimizeResult { status: MinimizeStatus::Cancelled, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        if converged {
            return MinimizeResult { status: MinimizeStatus::Converged, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        // apply the previous update; `H·dgrad` and `H·grad` in the same sweep
        let upd = pending.map(|fac| Update { xi: up_xi, hd: up_hd, w: up_w, fac });
        fused_pass(&mut inv_hessian, upd.as_ref(), dgrad, grad, hess_dgrad, hess_grad);
        let (mut fac, mut fae, mut sum_dgrad, mut sum_xi) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..dim {
            fac += dgrad[i] * xi[i];
            fae += dgrad[i] * hess_dgrad[i];
            sum_dgrad += dgrad[i] * dgrad[i];
            sum_xi += xi[i] * xi[i];
        }
        // New direction `-H'·grad` into `new_pos` (free until the next line
        // search) for the updated `H' = H + xi·wᵀ − fac·hd·xiᵀ` (see
        // [`Update`]): `H'·grad = H·grad + (w·grad)·xi − fac·(xi·grad)·hd`.
        if fac > (EPS * sum_dgrad * sum_xi).sqrt() {
            fac = 1.0 / fac;
            let fw = fac + fae * fac * fac;
            let (mut wg, mut xg) = (0.0, 0.0);
            for i in 0..dim {
                dgrad[i] = fw * xi[i] - fac * hess_dgrad[i];
                wg += dgrad[i] * grad[i];
                xg += xi[i] * grad[i];
            }
            let fx = fac * xg;
            for i in 0..dim {
                new_pos[i] = -(hess_grad[i] + wg * xi[i] - fx * hess_dgrad[i]);
            }
            std::mem::swap(&mut xi, &mut up_xi);
            std::mem::swap(&mut hess_dgrad, &mut up_hd);
            std::mem::swap(&mut dgrad, &mut up_w);
            pending = Some(fac);
        } else {
            for (d, hg) in new_pos.iter_mut().zip(hess_grad.iter()) {
                *d = -hg;
            }
            pending = None;
        }
        xi.copy_from_slice(new_pos);
    }
    MinimizeResult { status: MinimizeStatus::MaxIterations, energy: func_val, iterations: opts.max_iters, grad_rms: 0.0 }
}

/// Ring buffer of the last `m` L-BFGS correction pairs, preallocated.
struct LbfgsHistory {
    dim: usize,
    m: usize,
    s: Vec<f64>,
    y: Vec<f64>,
    rho: Vec<f64>,
    /// Slot of the oldest pair.
    head: usize,
    len: usize,
}

impl LbfgsHistory {
    fn new(dim: usize, m: usize) -> Self {
        LbfgsHistory { dim, m, s: vec![0.0; m * dim], y: vec![0.0; m * dim], rho: vec![0.0; m], head: 0, len: 0 }
    }
    fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }
    /// Slot of the `k`-th oldest pair.
    fn slot(&self, k: usize) -> usize {
        (self.head + k) % self.m
    }
    /// Slot to write the next pair into (dropping the oldest when full).
    fn push_slot(&mut self) -> usize {
        if self.len == self.m {
            let s = self.head;
            self.head = (self.head + 1) % self.m;
            s
        } else {
            self.len += 1;
            self.slot(self.len - 1)
        }
    }
    fn s(&self, slot: usize) -> &[f64] {
        &self.s[slot * self.dim..(slot + 1) * self.dim]
    }
    fn y(&self, slot: usize) -> &[f64] {
        &self.y[slot * self.dim..(slot + 1) * self.dim]
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// L-BFGS (two-loop recursion) with the same line search and convergence
/// criteria as [`minimize_bfgs`]. With `opts.force_tol <= 0` it runs in
/// endless mode like [`minimize_bfgs`] (a stalled line search clears the
/// correction history instead of returning).
pub fn minimize_lbfgs(
    ff: &dyn ForceField,
    pos: &mut [f64],
    opts: &MinimizeOptions,
    m: usize,
    observer: &mut dyn Observer,
) -> MinimizeResult {
    minimize_lbfgs_impl(ff, pos, opts, m, opts.is_endless(), observer)
}

/// [`minimize_lbfgs`] with endless mode chosen explicitly.
pub(crate) fn minimize_lbfgs_impl(
    ff: &dyn ForceField,
    pos: &mut [f64],
    opts: &MinimizeOptions,
    m: usize,
    endless: bool,
    observer: &mut dyn Observer,
) -> MinimizeResult {
    let dim = pos.len();
    let m = m.max(1);
    let scaling = opts.rdkit_grad_scaling;
    let mut work = BfgsWork::new(dim, 6);
    let [grad, new_pos, xi, s, y, q] = work.split::<6>();
    let mut hist = LbfgsHistory::new(dim, m);

    let mut fp = ff.energy(pos);
    calc_grad(ff, pos, grad, scaling);
    for (x, g) in xi.iter_mut().zip(grad.iter()) {
        *x = -g;
    }
    let sum: f64 = pos.iter().map(|p| p * p).sum();
    let max_step = MAXSTEP * sum.sqrt().max(dim as f64);
    let mut alpha = vec![0.0; m];
    let max_iters = if endless { usize::MAX } else { opts.max_iters };
    let mut stalled = false;
    for iter in 1..=max_iters {
        let (status, fv) = linear_search(ff, pos, fp, grad, xi, new_pos, max_step);
        if status < 0 {
            if endless {
                // bad direction: restart from steepest descent
                hist.clear();
                for (x, g) in xi.iter_mut().zip(grad.iter()) {
                    *x = -g;
                }
                if endless_restart(observer, iter, pos, fp, grad, stalled) {
                    return MinimizeResult { status: MinimizeStatus::Cancelled, energy: fp, iterations: iter, grad_rms: 0.0 };
                }
                stalled = true;
                continue;
            }
            if hist.len == 0 {
                return MinimizeResult { status: MinimizeStatus::Converged, energy: fp, iterations: iter, grad_rms: 0.0 };
            }
            // reset memory, retry steepest descent next iteration
            hist.clear();
            for (x, g) in xi.iter_mut().zip(grad.iter()) {
                *x = -g;
            }
            continue;
        }
        fp = fv;
        let mut test: f64 = 0.0;
        for i in 0..dim {
            s[i] = new_pos[i] - pos[i];
            pos[i] = new_pos[i];
            test = test.max(s[i].abs() / pos[i].abs().max(1.0));
        }
        ff.on_accepted_step(pos);
        if test < TOLX {
            if endless {
                // no measurable step: restart from steepest descent
                calc_grad(ff, pos, grad, scaling);
                hist.clear();
                for (x, g) in xi.iter_mut().zip(grad.iter()) {
                    *x = -g;
                }
                if endless_restart(observer, iter, pos, fp, grad, stalled) {
                    return MinimizeResult { status: MinimizeStatus::Cancelled, energy: fp, iterations: iter, grad_rms: 0.0 };
                }
                stalled = true;
                continue;
            }
            notify(observer, iter, pos, fp, grad);
            return MinimizeResult { status: MinimizeStatus::Converged, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        stalled = false;
        y.copy_from_slice(grad); // old gradient
        let grad_scale = calc_grad(ff, pos, grad, scaling);
        let converged = !endless && grad_converged(grad, pos, fp, grad_scale, opts.force_tol);
        if notify(observer, iter, pos, fp, grad) {
            return MinimizeResult { status: MinimizeStatus::Cancelled, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        if converged {
            return MinimizeResult { status: MinimizeStatus::Converged, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        for (yi, g) in y.iter_mut().zip(grad.iter()) {
            *yi = g - *yi;
        }
        let sy = dot(s, y);
        if sy > 1e-12 {
            let slot = hist.push_slot();
            hist.s[slot * dim..(slot + 1) * dim].copy_from_slice(s);
            hist.y[slot * dim..(slot + 1) * dim].copy_from_slice(y);
            hist.rho[slot] = 1.0 / sy;
        }
        // two-loop recursion
        q.copy_from_slice(grad);
        let k = hist.len;
        for idx in (0..k).rev() {
            let sl = hist.slot(idx);
            let a = hist.rho[sl] * dot(hist.s(sl), q);
            alpha[idx] = a;
            for (qi, yi) in q.iter_mut().zip(hist.y(sl)) {
                *qi -= a * yi;
            }
        }
        if k > 0 {
            let sl = hist.slot(k - 1);
            let yy: f64 = hist.y(sl).iter().map(|v| v * v).sum();
            let gamma = 1.0 / (hist.rho[sl] * yy);
            q.iter_mut().for_each(|v| *v *= gamma);
        }
        for idx in 0..k {
            let sl = hist.slot(idx);
            let b = hist.rho[sl] * dot(hist.y(sl), q);
            for (qi, si) in q.iter_mut().zip(hist.s(sl)) {
                *qi += (alpha[idx] - b) * si;
            }
        }
        for (x, v) in xi.iter_mut().zip(q.iter()) {
            *x = -v;
        }
    }
    MinimizeResult { status: MinimizeStatus::MaxIterations, energy: fp, iterations: opts.max_iters, grad_rms: 0.0 }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::ff::{Contrib, ContribForceField};

    struct Rosen;
    impl Contrib for Rosen {
        fn energy(&self, p: &[f64]) -> f64 {
            (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2) + p[2] * p[2]
        }
        fn grad(&self, p: &[f64], g: &mut [f64]) {
            g[0] += -2.0 * (1.0 - p[0]) - 400.0 * p[0] * (p[1] - p[0] * p[0]);
            g[1] += 200.0 * (p[1] - p[0] * p[0]);
            g[2] += 2.0 * p[2];
        }
    }

    #[test]
    fn rosenbrock() {
        for alg in [Algorithm::Bfgs, Algorithm::Lbfgs(5)] {
            let mut ff = ContribForceField::new(1, 3);
            ff.add_contrib(Box::new(Rosen));
            let mut p = vec![-1.2, 1.0, 0.5];
            let opts = MinimizeOptions { max_iters: 2000, force_tol: 1e-8, algorithm: alg, ..Default::default() };
            let r = minimize(&ff, &mut p, &opts, &mut NoObserver);
            assert!((p[0] - 1.0).abs() < 1e-3 && (p[1] - 1.0).abs() < 1e-3, "{alg:?} {p:?} {r:?}");
        }
    }
}

