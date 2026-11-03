//! BFGS / L-BFGS with the RDKit (Numerical Recipes) backtracking line search.
//!
//! Port of RDKit `Code/Numerics/Optimizer/BFGSOpt.h` and the gradient
//! functor in `Code/ForceField/ForceField.cpp` (BSD-3-Clause,
//! Copyright (C) 2004-2008 Greg Landrum and Rational Discovery LLC).

use super::{IterationInfo, MinimizeOptions, MinimizeResult, MinimizeStatus, Observer};
use crate::fast::bonded::{Lane, V4};
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
}

/// Vectors of the BFGS update (see [`packed_pass`]).
struct UpdateVecs<'a> {
    xi: &'a [f64],
    hdgrad: &'a [f64],
    dgrad: &'a [f64],
    pxi: &'a [f64],
    hdg: &'a [f64],
    dgs: &'a [f64],
}

/// RDKit update of one stored element `H[i][j]` (`j >= i`).
#[inline(always)]
fn update_elem(h: &mut f64, u: &UpdateVecs<'_>, i: usize, j: usize) {
    *h += u.pxi[i] * u.xi[j] - u.hdg[i] * u.hdgrad[j] + u.dgs[i] * u.dgrad[j];
}

/// `a + b` (`SUB = false`) or `a - b` (`SUB = true`).
#[inline(always)]
fn acc<const SUB: bool>(a: f64, b: f64) -> f64 {
    if SUB { a - b } else { a + b }
}

/// Row `i` of [`packed_pass`]: updates the stored row, finishes `out[i]`
/// with the row terms `j >= i` (all column terms `j < i` are already in it)
/// and adds this row's column term to every `out[t]`, `t > i`.
#[inline(always)]
fn packed_row<const SUB: bool>(h: &mut PackedSym, upd: Option<&UpdateVecs<'_>>, v: &[f64], out: &mut [f64], i: usize) {
    let dim = h.dim;
    let o = h.row_start(i);
    let row = &mut h.a[o..o + dim - i];
    if let Some(u) = upd {
        for (k, e) in row.iter_mut().enumerate() {
            update_elem(e, u, i, i + k);
        }
    }
    let mut s = out[i];
    for (e, &vj) in row.iter().zip(&v[i..]) {
        s = acc::<SUB>(s, e * vj);
    }
    out[i] = s;
    let vi = v[i];
    for (o, e) in out[i + 1..].iter_mut().zip(&row[1..]) {
        *o = acc::<SUB>(*o, e * vi);
    }
}

/// One sweep over the packed matrix: applies the BFGS update `upd` (if any)
/// to every stored element, and computes `out = H·v` (`SUB = false`) or
/// `out = -H·v` (`SUB = true`) for the updated `H`.
///
/// Arithmetic is identical to RDKit `BFGSOpt.h` operation by operation:
/// every `out[t]` starts from 0 and takes the products `H[t][j]·v[j]` for
/// `j = 0, 1, …` in order. The terms `j < t` come from the rows above (the
/// stored `H[j][t]`, added when row `j` is visited), the terms `j >= t`
/// from row `t` itself. The update writes RDKit's upper-triangle value.
fn packed_pass<const SUB: bool>(h: &mut PackedSym, upd: Option<&UpdateVecs<'_>>, v: &[f64], out: &mut [f64]) {
    let dim = h.dim;
    assert!(v.len() == dim && out.len() == dim && h.a.len() == dim * (dim + 1) / 2);
    if let Some(u) = upd {
        assert!([u.xi, u.hdgrad, u.dgrad, u.pxi, u.hdg, u.dgs].iter().all(|x| x.len() == dim));
    }
    out.fill(0.0);
    #[cfg(target_arch = "x86_64")]
    if crate::fast::avx::available() {
        #[target_feature(enable = "avx")]
        unsafe fn avx<const SUB: bool>(h: &mut PackedSym, upd: Option<&UpdateVecs<'_>>, v: &[f64], out: &mut [f64]) {
            packed_pass_lanes::<crate::fast::avx::A4, SUB>(h, upd, v, out)
        }
        // SAFETY: the CPU supports AVX (checked above).
        return unsafe { avx::<SUB>(h, upd, v, out) };
    }
    packed_pass_lanes::<V4, SUB>(h, upd, v, out)
}

/// [`packed_pass`] four rows `r..r+4` at a time, with 4-wide lanes. Each
/// output still receives its terms one by one in the order of
/// [`packed_row`]: the in-block triangle is done row by row, then for every
/// group of four columns `c ≥ r+4` the column terms of the four rows are
/// added to `out[c..c+4]` in row order, and the row terms of the four rows
/// (one running sum per lane, after a 4×4 transpose of the products) in
/// column order.
#[inline(always)]
fn packed_pass_lanes<V: Lane, const SUB: bool>(h: &mut PackedSym, upd: Option<&UpdateVecs<'_>>, v: &[f64], out: &mut [f64]) {
    // macros, not closures: closures would not inherit the caller's target features
    macro_rules! ld {
        ($s:expr) => {{
            let s: &[f64] = $s;
            V::from_fn(|l| s[l])
        }};
    }
    macro_rules! vacc {
        ($a:expr, $b:expr) => {
            if SUB { $a - $b } else { $a + $b }
        };
    }
    let dim = h.dim;
    let mut r = 0;
    while r + 4 <= dim {
        let (o0, o4) = (h.row_start(r), h.row_start(r + 4));
        let (r0, rest) = h.a[o0..o4].split_at_mut(dim - r);
        let (r1, rest) = rest.split_at_mut(dim - r - 1);
        let (r2, r3) = rest.split_at_mut(dim - r - 2);
        let mut rows = [r0, r1, r2, r3];
        // in-block triangle (columns r..r+4); row k holds column j at j - r - k
        let mut s = [0.0; 4];
        for (k, row) in rows.iter_mut().enumerate() {
            let i = r + k;
            if let Some(u) = upd {
                for j in i..r + 4 {
                    update_elem(&mut row[j - i], u, i, j);
                }
            }
            let mut sk = out[i];
            for j in i..r + 4 {
                sk = acc::<SUB>(sk, row[j - i] * v[j]);
            }
            s[k] = sk;
            for t in i + 1..r + 4 {
                out[t] = acc::<SUB>(out[t], row[t - i] * v[i]);
            }
        }
        // columns r+4.. (index c - r - 4 in every tail)
        let [t0, t1, t2, t3] = rows;
        let tails = [&mut t0[4..], &mut t1[3..], &mut t2[2..], &mut t3[1..]];
        let vr: [f64; 4] = std::array::from_fn(|k| v[r + k]);
        let c4 = r + 4;
        let mut sacc = ld!(&s);
        {
            macro_rules! body {
                ($hc:expr, $oc:expr, $vc:expr, $upd:expr) => {{
                    let ([h0, h1, h2, h3], oc, vc): ([&mut [f64; 4]; 4], &mut [f64; 4], &[f64; 4]) = ($hc, $oc, $vc);
                    let mut a = [ld!(h0), ld!(h1), ld!(h2), ld!(h3)];
                    if let Some((xc, hdc, dgc, p, q, rr)) = $upd {
                        let (x, hd, dg): (V, V, V) = (ld!(xc), ld!(hdc), ld!(dgc));
                        let (p, q, rr): ([f64; 4], [f64; 4], [f64; 4]) = (p, q, rr);
                        for k in 0..4 {
                            a[k] = a[k] + ((x * p[k] - hd * q[k]) + dg * rr[k]);
                        }
                        a[0].write(h0);
                        a[1].write(h1);
                        a[2].write(h2);
                        a[3].write(h3);
                    }
                    let mut o = ld!(oc);
                    for k in 0..4 {
                        o = vacc!(o, a[k] * vr[k]);
                    }
                    o.write(oc);
                    let vv = ld!(vc);
                    let [b0, b1, b2, b3] = a;
                    for q in V::transpose4([b0 * vv, b1 * vv, b2 * vv, b3 * vv]) {
                        sacc = vacc!(sacc, q);
                    }
                }};
            }
            let [h0, h1, h2, h3] = tails;
            let cols = h0.as_chunks_mut::<4>().0.iter_mut().zip(h1.as_chunks_mut::<4>().0.iter_mut()).zip(h2.as_chunks_mut::<4>().0.iter_mut()).zip(h3.as_chunks_mut::<4>().0.iter_mut()).zip(out[c4..].as_chunks_mut::<4>().0.iter_mut()).zip(v[c4..].as_chunks::<4>().0);
            match upd {
                Some(u) => {
                    let f = |w: &[f64]| -> [f64; 4] { [w[r], w[r + 1], w[r + 2], w[r + 3]] };
                    let (p, q, rr) = (f(u.pxi), f(u.hdg), f(u.dgs));
                    let ucols = u.xi[c4..].as_chunks::<4>().0.iter().zip(u.hdgrad[c4..].as_chunks::<4>().0).zip(u.dgrad[c4..].as_chunks::<4>().0);
                    for ((((((h0, h1), h2), h3), oc), vc), ((xc, hdc), dgc)) in cols.zip(ucols) {
                        body!([h0, h1, h2, h3], oc, vc, Some((xc, hdc, dgc, p, q, rr)));
                    }
                }
                None => {
                    for (((((h0, h1), h2), h3), oc), vc) in cols {
                        body!([h0, h1, h2, h3], oc, vc, None::<(&[f64; 4], &[f64; 4], &[f64; 4], [f64; 4], [f64; 4], [f64; 4])>);
                    }
                }
            }
        }
        sacc.write(&mut s);
        let n4 = c4 + (dim - c4) / 4 * 4;
        let base: [usize; 4] = std::array::from_fn(|k| h.row_start(r + k) - (r + k));
        for c in n4..dim {
            for (k, sk) in s.iter_mut().enumerate() {
                let e = &mut h.a[base[k] + c];
                if let Some(u) = upd {
                    update_elem(e, u, r + k, c);
                }
                out[c] = acc::<SUB>(out[c], *e * v[r + k]);
                *sk = acc::<SUB>(*sk, *e * v[c]);
            }
        }
        out[r..r + 4].copy_from_slice(&s);
        r += 4;
    }
    for i in r..dim {
        packed_row::<SUB>(h, upd, v, out, i);
    }
}

/// Scratch vectors of [`minimize_bfgs`] (one allocation).
struct BfgsWork {
    buf: Vec<f64>,
}

impl BfgsWork {
    const N: usize = 8;
    fn new(dim: usize) -> Self {
        BfgsWork { buf: vec![0.0; Self::N * dim] }
    }
    fn split(&mut self) -> [&mut [f64]; Self::N] {
        let dim = self.buf.len() / Self::N;
        let mut it = self.buf.chunks_exact_mut(dim.max(1));
        std::array::from_fn(|_| it.next().unwrap_or(&mut []))
    }
}

/// RDKit BFGS minimization.
///
/// Arithmetic is identical to RDKit `BFGSOpt.h` operation by operation
/// (same products, same summation order), only the storage and loops
/// differ: the symmetric inverse Hessian is kept as a packed upper triangle
/// (half the memory, see [`PackedSym`]), and its update is fused with the
/// product for the next direction ([`packed_pass`]).
pub fn minimize_bfgs(
    ff: &dyn ForceField,
    pos: &mut [f64],
    opts: &MinimizeOptions,
    observer: &mut dyn Observer,
) -> MinimizeResult {
    let dim = pos.len();
    let scaling = opts.rdkit_grad_scaling;
    let mut work = BfgsWork::new(dim);
    let [grad, dgrad, hess_dgrad, xi, new_pos, pxi, hdg, dgs] = work.split();
    let mut inv_hessian = PackedSym::identity(dim);

    let mut fp = ff.energy(pos);
    calc_grad(ff, pos, grad, scaling);
    let mut sum = 0.0;
    for i in 0..dim {
        xi[i] = -grad[i];
        sum += pos[i] * pos[i];
    }
    let max_step = MAXSTEP * sum.sqrt().max(dim as f64);
    let mut func_val = fp;
    for iter in 1..=opts.max_iters {
        let (status, fv) = linear_search(ff, pos, fp, grad, xi, new_pos, max_step);
        func_val = fv;
        if status < 0 {
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
            notify(observer, iter, pos, fp, grad);
            return MinimizeResult { status: MinimizeStatus::Converged, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        let grad_scale = calc_grad(ff, pos, grad, scaling);
        let converged = grad_converged(grad, pos, func_val, grad_scale, opts.force_tol);
        for (d, g) in dgrad.iter_mut().zip(grad.iter()) {
            *d = g - *d;
        }
        if notify(observer, iter, pos, fp, grad) {
            return MinimizeResult { status: MinimizeStatus::Cancelled, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        if converged {
            return MinimizeResult { status: MinimizeStatus::Converged, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        // BFGS inverse Hessian update
        packed_pass::<false>(&mut inv_hessian, None, dgrad, hess_dgrad);
        let (mut fac, mut fae, mut sum_dgrad, mut sum_xi) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..dim {
            fac += dgrad[i] * xi[i];
            fae += dgrad[i] * hess_dgrad[i];
            sum_dgrad += dgrad[i] * dgrad[i];
            sum_xi += xi[i] * xi[i];
        }
        // New direction `-H·grad` (into `new_pos`, free until the next line
        // search).
        let upd = if fac > (EPS * sum_dgrad * sum_xi).sqrt() {
            fac = 1.0 / fac;
            let fad = 1.0 / fae;
            for i in 0..dim {
                dgrad[i] = fac * xi[i] - fad * hess_dgrad[i];
            }
            for i in 0..dim {
                pxi[i] = fac * xi[i];
                hdg[i] = fad * hess_dgrad[i];
                dgs[i] = fae * dgrad[i];
            }
            Some(UpdateVecs { xi, hdgrad: hess_dgrad, dgrad, pxi, hdg, dgs })
        } else {
            None
        };
        packed_pass::<true>(&mut inv_hessian, upd.as_ref(), grad, new_pos);
        let dir = &*new_pos;
        xi.copy_from_slice(dir);
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
/// criteria as [`minimize_bfgs`].
pub fn minimize_lbfgs(
    ff: &dyn ForceField,
    pos: &mut [f64],
    opts: &MinimizeOptions,
    m: usize,
    observer: &mut dyn Observer,
) -> MinimizeResult {
    let dim = pos.len();
    let m = m.max(1);
    let scaling = opts.rdkit_grad_scaling;
    let mut work = BfgsWork::new(dim);
    let [grad, new_pos, xi, s, y, q, _, _] = work.split();
    let mut hist = LbfgsHistory::new(dim, m);

    let mut fp = ff.energy(pos);
    calc_grad(ff, pos, grad, scaling);
    for (x, g) in xi.iter_mut().zip(grad.iter()) {
        *x = -g;
    }
    let sum: f64 = pos.iter().map(|p| p * p).sum();
    let max_step = MAXSTEP * sum.sqrt().max(dim as f64);
    let mut alpha = vec![0.0; m];
    for iter in 1..=opts.max_iters {
        let (status, fv) = linear_search(ff, pos, fp, grad, xi, new_pos, max_step);
        if status < 0 {
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
            notify(observer, iter, pos, fp, grad);
            return MinimizeResult { status: MinimizeStatus::Converged, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        y.copy_from_slice(grad); // old gradient
        let grad_scale = calc_grad(ff, pos, grad, scaling);
        let converged = grad_converged(grad, pos, fp, grad_scale, opts.force_tol);
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
