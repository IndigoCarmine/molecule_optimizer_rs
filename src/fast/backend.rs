//! Nonbonded evaluation backends.
//!
//! A backend evaluates the nonbonded energy and adds its gradient. CPU
//! implementations live in [`crate::fast::exact`] (fixed pair list) and
//! [`crate::fast::approx`] (cluster-pair Verlet list, PME). Accelerator
//! backends can implement the same trait later.

/// Energy split of a nonbonded evaluation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NonbondedEnergy {
    pub vdw: f64,
    pub electrostatic: f64,
}

impl NonbondedEnergy {
    pub fn total(&self) -> f64 {
        self.vdw + self.electrostatic
    }
}

/// Nonbonded term of a [`super::FastForceField`].
pub trait NonbondedBackend: Send + Sync {
    /// Short name for diagnostics (e.g. `"cpu-simd-cluster"`).
    fn name(&self) -> &str;
    /// Energy only.
    fn energy(&self, pos: &[f64]) -> NonbondedEnergy;
    /// Energy and gradient (gradient is *added* to `grad`).
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> NonbondedEnergy;
    /// Called after each accepted minimizer step (pair-list maintenance).
    fn on_accepted_step(&self, _pos: &[f64]) {}
}
