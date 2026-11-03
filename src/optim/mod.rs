//! Minimizers.
//!
//! [`bfgs`] is a port of RDKit `Code/Numerics/Optimizer/BFGSOpt.h`
//! (BSD-3-Clause, Copyright (C) 2004-2008 Greg Landrum and Rational
//! Discovery LLC), including the gradient scaling performed by RDKit's
//! `ForceField::minimize`. An L-BFGS variant using the same line search is
//! provided for large systems.

pub mod bfgs;

use crate::ff::ForceField;
use std::ops::ControlFlow;

/// Minimization algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algorithm {
    /// BFGS for ≤ `AUTO_LBFGS_DIM` coordinates, L-BFGS above.
    #[default]
    Auto,
    Bfgs,
    /// Limited-memory BFGS with the given history size.
    Lbfgs(usize),
}

/// Coordinates (dimension) above which `Auto` switches to L-BFGS.
pub const AUTO_LBFGS_DIM: usize = 3000;

/// Minimizer options. Defaults follow RDKit `ForceField::minimize`.
#[derive(Debug, Clone)]
pub struct MinimizeOptions {
    /// Iteration limit (ignored in endless mode, see [`force_tol`](Self::force_tol)).
    pub max_iters: usize,
    /// Gradient convergence tolerance (RDKit `forceTol`, default 1e-4).
    ///
    /// **Endless mode:** `force_tol <= 0` means "never converge". The
    /// minimizer then skips the gradient and step-size (TOLX) convergence
    /// tests, ignores `max_iters`, and when the line search finds no descent
    /// or cannot move it restarts from steepest descent (BFGS: inverse
    /// Hessian reset to identity; L-BFGS: history cleared) instead of
    /// returning. It runs until the [`Observer`] returns
    /// `ControlFlow::Break` (status [`MinimizeStatus::Cancelled`]); for
    /// [`crate::tasks::Engine`] jobs that is [`crate::tasks::JobHandle::cancel`].
    /// At a point that is stationary to machine precision it keeps calling
    /// the observer (with a 1 ms pause per iteration) so it stays responsive.
    /// Applies to single-structure optimizations only: [`minimize`],
    /// [`crate::calc::optimize::optimize`], [`crate::tasks::Engine::optimize`] and
    /// [`crate::tasks::Engine::optimize_batch`]. Scans, conformer and
    /// assembly searches and the distance-geometry embedder never run
    /// endless; there `force_tol <= 0` only disables the gradient test
    /// (see [`minimize_with`]).
    pub force_tol: f64,
    /// Energy tolerance (accepted for API compatibility; RDKit ignores it).
    pub energy_tol: f64,
    pub algorithm: Algorithm,
    /// Apply RDKit's gradient scaling (×0.1, capped). Default `true`.
    pub rdkit_grad_scaling: bool,
}

impl Default for MinimizeOptions {
    fn default() -> Self {
        MinimizeOptions {
            max_iters: 200,
            force_tol: 1e-4,
            energy_tol: 1e-6,
            algorithm: Algorithm::Auto,
            rdkit_grad_scaling: true,
        }
    }
}

impl MinimizeOptions {
    /// `true` in endless mode (`force_tol <= 0`, see [`force_tol`](Self::force_tol)).
    pub fn is_endless(&self) -> bool {
        self.force_tol <= 0.0
    }
}

/// Outcome of a minimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimizeStatus {
    Converged,
    MaxIterations,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct MinimizeResult {
    pub status: MinimizeStatus,
    pub energy: f64,
    pub iterations: usize,
    /// RMS of the (unscaled) gradient at the final point.
    pub grad_rms: f64,
}

/// Per-iteration information passed to an [`Observer`].
pub struct IterationInfo<'a> {
    pub iteration: usize,
    pub pos: &'a [f64],
    pub energy: f64,
    /// Gradient as used by the minimizer (possibly scaled).
    pub grad: &'a [f64],
}

/// Called after every accepted step; return `ControlFlow::Break(())` to stop.
///
/// This is the synchronous way to get every intermediate structure:
/// [`IterationInfo::pos`] holds the current coordinates (copy them if they
/// are needed later), e.g. with a closure
/// `|info: &IterationInfo<'_>| { frames.push(info.pos.to_vec()); ControlFlow::Continue(()) }`.
/// For [`crate::tasks::Engine`] jobs use [`crate::tasks::JobHandle::stream`]
/// instead.
pub trait Observer {
    fn on_iteration(&mut self, info: &IterationInfo<'_>) -> ControlFlow<()>;
}

/// Observer that never stops.
pub struct NoObserver;
impl Observer for NoObserver {
    fn on_iteration(&mut self, _: &IterationInfo<'_>) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

impl<F: FnMut(&IterationInfo<'_>) -> ControlFlow<()>> Observer for F {
    fn on_iteration(&mut self, info: &IterationInfo<'_>) -> ControlFlow<()> {
        self(info)
    }
}

/// Minimizes `ff` starting from `pos` (updated in place). Runs in endless
/// mode when `opts.force_tol <= 0` (see [`MinimizeOptions::force_tol`]);
/// [`minimize_with`] chooses the mode explicitly.
pub fn minimize(
    ff: &dyn ForceField,
    pos: &mut [f64],
    opts: &MinimizeOptions,
    observer: &mut dyn Observer,
) -> MinimizeResult {
    minimize_with(ff, pos, opts, opts.is_endless(), observer)
}

/// [`minimize`] with endless mode chosen explicitly instead of from
/// `opts.force_tol`. With `endless == false` a `force_tol <= 0` only
/// disables the gradient convergence test: the step-size (TOLX) test, a
/// failed line search and `max_iters` still end the minimization (the
/// behavior before endless mode existed). Calculations that minimize many
/// structures in turn (scans, conformer and assembly searches, the
/// distance-geometry embedder) always use `endless == false`.
pub fn minimize_with(
    ff: &dyn ForceField,
    pos: &mut [f64],
    opts: &MinimizeOptions,
    endless: bool,
    observer: &mut dyn Observer,
) -> MinimizeResult {
    let dim = pos.len();
    let alg = match opts.algorithm {
        Algorithm::Auto => {
            if dim > AUTO_LBFGS_DIM {
                Algorithm::Lbfgs(10)
            } else {
                Algorithm::Bfgs
            }
        }
        a => a,
    };
    let mut res = match alg {
        Algorithm::Lbfgs(m) => bfgs::minimize_lbfgs_impl(ff, pos, opts, m.max(1), endless, observer),
        _ => bfgs::minimize_bfgs_impl(ff, pos, opts, endless, observer),
    };
    let mut g = vec![0.0; dim];
    res.energy = ff.energy_grad(pos, &mut g);
    res.grad_rms = (g.iter().map(|x| x * x).sum::<f64>() / dim.max(1) as f64).sqrt();
    res
}
