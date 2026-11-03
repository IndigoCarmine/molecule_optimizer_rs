//! Geometry optimization.

use super::{Frame, SharedBuilder, to_flat, to_points};
use crate::core::Molecule;
use crate::error::Result;
use crate::ff::{Constraint, ForceField, ForceFieldBuilder};
use crate::optim::{self, MinimizeOptions, MinimizeStatus, Observer};
use crate::tasks::{CtxObserver, Engine, JobContext, JobHandle};
use rayon::prelude::*;
use std::sync::Arc;

/// Options for [`optimize`].
#[derive(Debug, Clone, Default)]
pub struct OptimizeOptions {
    pub minimize: MinimizeOptions,
    pub constraints: Vec<Constraint>,
}

/// Result of an optimization.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    pub coords: Vec<[f64; 3]>,
    /// Energy of the minimized function (force field + constraint terms).
    pub energy: f64,
    /// Force-field energy without constraint terms.
    pub ff_energy: f64,
    pub status: MinimizeStatus,
    pub iterations: usize,
    pub grad_rms: f64,
}

/// State of a running optimization.
#[derive(Debug, Clone, Default)]
pub struct OptimizeSnapshot {
    pub frame: Frame,
    pub finished: bool,
}

fn energy_without_constraints(ff_with: &dyn ForceField, ff_energy_total: f64, pos: &[f64]) -> f64 {
    let bd = ff_with.energy_breakdown(pos);
    let c: f64 = bd.iter().filter(|(k, _)| k == "Constraint").map(|(_, v)| v).sum();
    ff_energy_total - c
}

/// Optimizes `coords` of `mol` (synchronous). `observer` sees every iteration.
pub fn optimize_coords(
    mol: &Molecule,
    coords: &[[f64; 3]],
    builder: &dyn ForceFieldBuilder,
    opts: &OptimizeOptions,
    observer: &mut dyn Observer,
) -> Result<OptimizeResult> {
    let ff = builder.build_dyn(mol, &opts.constraints)?;
    let mut pos = to_flat(coords);
    let r = optim::minimize(&ff, &mut pos, &opts.minimize, observer);
    let ff_energy = if opts.constraints.is_empty() { r.energy } else { energy_without_constraints(&ff, r.energy, &pos) };
    Ok(OptimizeResult {
        coords: to_points(&pos),
        energy: r.energy,
        ff_energy,
        status: r.status,
        iterations: r.iterations,
        grad_rms: r.grad_rms,
    })
}

/// Optimizes `mol` at its current coordinates (synchronous).
pub fn optimize(mol: &Molecule, builder: &dyn ForceFieldBuilder, opts: &OptimizeOptions) -> Result<OptimizeResult> {
    optimize_coords(mol, &mol.coords, builder, opts, &mut optim::NoObserver)
}

impl JobHandle<OptimizeResult, OptimizeSnapshot> {
    /// Intermediate structures of an [`Engine::optimize`] job as a `Stream`
    /// of [`Frame`]s (coordinates, energy, iteration), ending with the final
    /// structure. Same semantics as [`JobHandle::stream`] (latest value,
    /// never slows the optimization; frames arrive every
    /// [`crate::tasks::EngineConfig::publish_interval`]), with consecutive
    /// duplicates removed.
    ///
    /// ```no_run
    /// # use molecule_optimizer_rs::{calc::OptimizeOptions, ff::uff::UffBuilder, tasks::Engine, Molecule};
    /// # use futures::StreamExt;
    /// # use std::sync::Arc;
    /// # async fn f(mol: Arc<Molecule>) {
    /// let mut opts = OptimizeOptions::default();
    /// opts.minimize.force_tol = 0.0; // endless: run until cancelled
    /// let job = Engine::global().optimize(mol, Arc::new(UffBuilder::default()), opts);
    /// let mut frames = job.frames();
    /// while let Some(f) = frames.next().await {
    ///     println!("iter {} E = {}", f.iteration, f.energy);
    ///     if f.iteration > 1000 {
    ///         job.cancel(); // the stream still yields the final frame, then ends
    ///     }
    /// }
    /// # }
    /// ```
    pub fn frames(&self) -> futures::stream::BoxStream<'static, Frame> {
        use futures::StreamExt;
        let mut last: Option<Frame> = None;
        self.snapshot_stream()
            .filter_map(move |s| {
                let f = &s.data.frame;
                let dup = last.as_ref().is_some_and(|l| l.iteration == f.iteration && l.energy.to_bits() == f.energy.to_bits() && l.coords == f.coords);
                if !dup {
                    last = Some(f.clone());
                }
                futures::future::ready((!dup).then(|| f.clone()))
            })
            .boxed()
    }
}

/// Snapshot of a batch optimization.
#[derive(Debug, Clone, Default)]
pub struct BatchOptimizeSnapshot {
    /// One entry per input structure (`None` until it starts).
    pub items: Vec<Option<OptimizeSnapshot>>,
}

impl Engine {
    /// Optimizes one structure asynchronously. The snapshot always contains
    /// the latest coordinates; `cancel()` returns the current geometry.
    /// Intermediate structures are available as a stream with
    /// [`JobHandle::frames`]. With `opts.minimize.force_tol <= 0` (endless
    /// mode, see [`MinimizeOptions::force_tol`]) the job runs until cancelled
    /// and reports progress with `total == 0`.
    pub fn optimize(
        &self,
        mol: Arc<Molecule>,
        builder: SharedBuilder,
        opts: OptimizeOptions,
    ) -> JobHandle<OptimizeResult, OptimizeSnapshot> {
        let init = OptimizeSnapshot { frame: Frame { coords: mol.coords.clone(), energy: f64::NAN, iteration: 0 }, finished: false };
        // progress total 0 = unknown (endless mode has no iteration limit)
        let total = if opts.minimize.is_endless() { 0 } else { opts.minimize.max_iters };
        self.spawn(init, total, move |ctx: JobContext<OptimizeSnapshot>| {
            let mut obs = CtxObserver::new(&ctx, |ctx, info| {
                let coords = to_points(info.pos);
                let (e, it) = (info.energy, info.iteration);
                ctx.update(|s| s.frame = Frame { coords, energy: e, iteration: it });
                ctx.set_progress(it, total);
            });
            let r = optimize_coords(&mol, &mol.coords, builder.as_ref(), &opts, &mut obs)?;
            let fr = Frame { coords: r.coords.clone(), energy: r.energy, iteration: r.iterations };
            ctx.update(|s| {
                s.frame = fr;
                s.finished = true;
            });
            Ok(r)
        })
    }

    /// Optimizes many structures in parallel. Like the progress of the
    /// running structures, finished structures appear in the snapshot when
    /// one is requested or periodically (and in the final one).
    pub fn optimize_batch(
        &self,
        items: Vec<super::StructureItem>,
        builder: SharedBuilder,
        opts: OptimizeOptions,
    ) -> JobHandle<Vec<Result<OptimizeResult>>, BatchOptimizeSnapshot> {
        let n = items.len();
        self.spawn(BatchOptimizeSnapshot { items: vec![None; n] }, n, move |ctx| {
            let out = items
                .par_iter()
                .enumerate()
                .map(|(i, (mol, coords))| {
                    let start = coords.as_deref().unwrap_or(&mol.coords);
                    ctx.update_silent(|s| {
                        s.items[i] = Some(OptimizeSnapshot { frame: Frame { coords: start.to_vec(), energy: f64::NAN, iteration: 0 }, finished: false })
                    });
                    let mut obs = CtxObserver::new(&ctx, |ctx, info| {
                        let coords = to_points(info.pos);
                        let (e, it) = (info.energy, info.iteration);
                        ctx.update(|s| s.items[i] = Some(OptimizeSnapshot { frame: Frame { coords, energy: e, iteration: it }, finished: false }));
                    });
                    let r = optimize_coords(mol, start, builder.as_ref(), &opts, &mut obs);
                    if let Ok(r) = &r {
                        let fr = Frame { coords: r.coords.clone(), energy: r.energy, iteration: r.iterations };
                        // published on request / periodically only: publishing
                        // clones the snapshot of every item
                        ctx.update_throttled(&mut obs.served_epoch, |s| s.items[i] = Some(OptimizeSnapshot { frame: fr, finished: true }));
                    }
                    ctx.inc_progress();
                    r
                })
                .collect();
            Ok(out)
        })
    }
}
