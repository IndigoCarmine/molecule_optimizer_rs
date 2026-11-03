//! BFGS / L-BFGS with the RDKit (Numerical Recipes) backtracking line search.
//!
//! Port of RDKit `Code/Numerics/Optimizer/BFGSOpt.h` and the gradient
//! functor in `Code/ForceField/ForceField.cpp` (BSD-3-Clause,
//! Copyright (C) 2004-2008 Greg Landrum and Rational Discovery LLC).

use super::{IterationInfo, MinimizeOptions, MinimizeResult, MinimizeStatus, Observer};
use crate::ff::ForceField;
use std::collections::VecDeque;
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

/// RDKit BFGS minimization.
pub fn minimize_bfgs(
    ff: &dyn ForceField,
    pos: &mut [f64],
    opts: &MinimizeOptions,
    observer: &mut dyn Observer,
) -> MinimizeResult {
    let dim = pos.len();
    let scaling = opts.rdkit_grad_scaling;
    let mut grad = vec![0.0; dim];
    let mut dgrad = vec![0.0; dim];
    let mut hess_dgrad = vec![0.0; dim];
    let mut xi = vec![0.0; dim];
    let mut inv_hessian = vec![0.0; dim * dim];
    let mut new_pos = vec![0.0; dim];

    let mut fp = ff.energy(pos);
    calc_grad(ff, pos, &mut grad, scaling);
    let mut sum = 0.0;
    for i in 0..dim {
        inv_hessian[i * dim + i] = 1.0;
        xi[i] = -grad[i];
        sum += pos[i] * pos[i];
    }
    let max_step = MAXSTEP * sum.sqrt().max(dim as f64);
    let mut func_val = fp;
    for iter in 1..=opts.max_iters {
        let (status, fv) = linear_search(ff, pos, fp, &grad, &mut xi, &mut new_pos, max_step);
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
            dgrad[i] = grad[i];
        }
        if test < TOLX {
            notify(observer, iter, pos, fp, &grad);
            return MinimizeResult { status: MinimizeStatus::Converged, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        let grad_scale = calc_grad(ff, pos, &mut grad, scaling);
        let converged = grad_converged(&grad, pos, func_val, grad_scale, opts.force_tol);
        for i in 0..dim {
            dgrad[i] = grad[i] - dgrad[i];
        }
        if notify(observer, iter, pos, fp, &grad) {
            return MinimizeResult { status: MinimizeStatus::Cancelled, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        if converged {
            return MinimizeResult { status: MinimizeStatus::Converged, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        // BFGS inverse Hessian update
        let (mut fac, mut fae, mut sum_dgrad, mut sum_xi) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..dim {
            let row = &inv_hessian[i * dim..(i + 1) * dim];
            let mut h = 0.0;
            for j in 0..dim {
                h += row[j] * dgrad[j];
            }
            hess_dgrad[i] = h;
            fac += dgrad[i] * xi[i];
            fae += dgrad[i] * h;
            sum_dgrad += dgrad[i] * dgrad[i];
            sum_xi += xi[i] * xi[i];
        }
        if fac > (EPS * sum_dgrad * sum_xi).sqrt() {
            fac = 1.0 / fac;
            let fad = 1.0 / fae;
            for i in 0..dim {
                dgrad[i] = fac * xi[i] - fad * hess_dgrad[i];
            }
            for i in 0..dim {
                let pxi = fac * xi[i];
                let hdgi = fad * hess_dgrad[i];
                let dgi = fae * dgrad[i];
                for j in i..dim {
                    let v = inv_hessian[i * dim + j] + pxi * xi[j] - hdgi * hess_dgrad[j] + dgi * dgrad[j];
                    inv_hessian[i * dim + j] = v;
                    inv_hessian[j * dim + i] = v;
                }
            }
        }
        for i in 0..dim {
            let row = &inv_hessian[i * dim..(i + 1) * dim];
            let mut s = 0.0;
            for j in 0..dim {
                s -= row[j] * grad[j];
            }
            xi[i] = s;
        }
    }
    MinimizeResult { status: MinimizeStatus::MaxIterations, energy: func_val, iterations: opts.max_iters, grad_rms: 0.0 }
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
    let scaling = opts.rdkit_grad_scaling;
    let mut grad = vec![0.0; dim];
    let mut new_pos = vec![0.0; dim];
    let mut s_hist: VecDeque<Vec<f64>> = VecDeque::new();
    let mut y_hist: VecDeque<Vec<f64>> = VecDeque::new();
    let mut rho_hist: VecDeque<f64> = VecDeque::new();

    let mut fp = ff.energy(pos);
    calc_grad(ff, pos, &mut grad, scaling);
    let mut xi: Vec<f64> = grad.iter().map(|g| -g).collect();
    let sum: f64 = pos.iter().map(|p| p * p).sum();
    let max_step = MAXSTEP * sum.sqrt().max(dim as f64);
    let mut alpha = vec![0.0; m];
    for iter in 1..=opts.max_iters {
        let (status, fv) = linear_search(ff, pos, fp, &grad, &mut xi, &mut new_pos, max_step);
        if status < 0 {
            if s_hist.is_empty() {
                return MinimizeResult { status: MinimizeStatus::Converged, energy: fp, iterations: iter, grad_rms: 0.0 };
            }
            // reset memory, retry steepest descent next iteration
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            xi = grad.iter().map(|g| -g).collect();
            continue;
        }
        fp = fv;
        let mut test: f64 = 0.0;
        let mut s = vec![0.0; dim];
        for i in 0..dim {
            s[i] = new_pos[i] - pos[i];
            pos[i] = new_pos[i];
            test = test.max(s[i].abs() / pos[i].abs().max(1.0));
        }
        if test < TOLX {
            notify(observer, iter, pos, fp, &grad);
            return MinimizeResult { status: MinimizeStatus::Converged, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        let old_grad = grad.clone();
        let grad_scale = calc_grad(ff, pos, &mut grad, scaling);
        let converged = grad_converged(&grad, pos, fp, grad_scale, opts.force_tol);
        if notify(observer, iter, pos, fp, &grad) {
            return MinimizeResult { status: MinimizeStatus::Cancelled, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        if converged {
            return MinimizeResult { status: MinimizeStatus::Converged, energy: fp, iterations: iter, grad_rms: 0.0 };
        }
        let y: Vec<f64> = grad.iter().zip(&old_grad).map(|(a, b)| a - b).collect();
        let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();
        if sy > 1e-12 {
            if s_hist.len() == m {
                s_hist.pop_front();
                y_hist.pop_front();
                rho_hist.pop_front();
            }
            s_hist.push_back(s);
            y_hist.push_back(y);
            rho_hist.push_back(1.0 / sy);
        }
        // two-loop recursion
        let mut q = grad.clone();
        let k = s_hist.len();
        for idx in (0..k).rev() {
            let a = rho_hist[idx] * s_hist[idx].iter().zip(&q).map(|(x, y)| x * y).sum::<f64>();
            alpha[idx] = a;
            for (qi, yi) in q.iter_mut().zip(&y_hist[idx]) {
                *qi -= a * yi;
            }
        }
        if k > 0 {
            let yy: f64 = y_hist[k - 1].iter().map(|v| v * v).sum();
            let gamma = 1.0 / (rho_hist[k - 1] * yy);
            q.iter_mut().for_each(|v| *v *= gamma);
        }
        for idx in 0..k {
            let b = rho_hist[idx] * y_hist[idx].iter().zip(&q).map(|(x, y)| x * y).sum::<f64>();
            for (qi, si) in q.iter_mut().zip(&s_hist[idx]) {
                *qi += (alpha[idx] - b) * si;
            }
        }
        xi = q.iter().map(|v| -v).collect();
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
