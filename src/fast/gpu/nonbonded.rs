//! [`WgpuNonbonded`]: GPU implementation of [`NonbondedBackend`] for one
//! (large) system.

use super::GpuContext;
use super::engine::{Engine, SysRef};
use crate::core::Molecule;
use crate::error::Result;
use crate::fast::ApproxOptions;
use crate::fast::approx::{ApproxNonbonded, PairListStats, build_approx_nonbonded};
use crate::fast::backend::{NonbondedBackend, NonbondedEnergy};
use std::sync::Mutex;

/// GPU nonbonded backend (approximate mode, `f32` kernels).
///
/// Same model, options and pair list as [`ApproxNonbonded`] (it wraps one:
/// parameters, exclusions, the cluster-pair list and its lazy rebuild are
/// shared with the CPU code, see [`WgpuNonbonded::cpu`]); the kernels run on
/// the GPU in `f32` with cluster-relative coordinates, Kahan-compensated
/// per-cluster sums and `f64` final energy reductions. Evaluations are
/// bitwise reproducible run-to-run.
pub struct WgpuNonbonded {
    cpu: ApproxNonbonded,
    engine: Mutex<Engine>,
    ctx: GpuContext,
}

impl std::fmt::Debug for WgpuNonbonded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuNonbonded").field("adapter", &self.ctx.adapter_info().name).field("cpu", &self.cpu).finish()
    }
}

/// Builds the GPU nonbonded backend for `mol` (same arguments as
/// [`build_approx_nonbonded`]; PME spline orders up to 12).
pub fn build_gpu_nonbonded(mol: &Molecule, inner_builder: &dyn std::any::Any, opts: &ApproxOptions, ctx: &GpuContext) -> Result<WgpuNonbonded> {
    let cpu = build_approx_nonbonded(mol, inner_builder, opts, true)?;
    let engine = Engine::new(ctx, &[&cpu], true)?;
    Ok(WgpuNonbonded { cpu, engine: Mutex::new(engine), ctx: ctx.clone() })
}

impl WgpuNonbonded {
    /// The wrapped CPU backend (same parameters and pair list); its own
    /// `energy_grad` evaluates the same model in `f64` on the CPU.
    pub fn cpu(&self) -> &ApproxNonbonded {
        &self.cpu
    }
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }
    pub fn options(&self) -> &ApproxOptions {
        self.cpu.options()
    }
    pub fn num_atoms(&self) -> usize {
        self.cpu.num_atoms()
    }
    /// Number of pair-list builds so far (including the initial one).
    pub fn num_builds(&self) -> usize {
        self.cpu.num_builds()
    }
    /// Number of pair-list uploads to the GPU so far.
    pub fn num_uploads(&self) -> usize {
        self.lock().num_uploads()
    }
    pub fn pair_list_stats(&self) -> PairListStats {
        self.cpu.pair_list_stats()
    }
    /// Invocations per i-cluster used by the cluster kernel for the current
    /// list (chosen from the average list length; override with the
    /// environment variable `MOLOPT_GPU_TPC` = 1, 2, 4, ..., 64).
    pub fn threads_per_cluster(&self) -> u32 {
        self.lock().threads_per_cluster()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Engine> {
        self.engine.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn evaluate(&self, pos: &[f64], grad: Option<&mut [f64]>) -> NonbondedEnergy {
        let n = self.cpu.num_atoms();
        assert!(pos.len() >= 3 * n, "position vector too short");
        let list = self.cpu.current_list(pos);
        let mut eng = self.lock();
        eng.evaluate(&[SysRef { nb: &self.cpu, tmpl: 0, list, pos }], grad.is_some());
        if let Some(g) = grad {
            for (o, &v) in g[..3 * n].iter_mut().zip(&eng.r_grad[..3 * n]) {
                *o += v as f64;
            }
        }
        eng.energies[0]
    }
}

impl NonbondedBackend for WgpuNonbonded {
    fn name(&self) -> &str {
        "wgpu-cluster"
    }
    fn energy(&self, pos: &[f64]) -> NonbondedEnergy {
        self.evaluate(pos, None)
    }
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> NonbondedEnergy {
        self.evaluate(pos, Some(grad))
    }
    fn on_accepted_step(&self, pos: &[f64]) {
        self.cpu.on_accepted_step(pos);
    }
}
