//! High-performance evaluation path.
//!
//! [`FastBuilder`] wraps any [`ForceFieldBuilder`] (UFF, MMFF94, ...) and
//! produces a [`FastForceField`] that evaluates the same terms with
//! structure-of-arrays tables, SIMD kernels and deterministic parallel
//! reductions. Two modes:
//!
//! * [`FastMode::Exact`]: identical term set and energy expressions as the
//!   RDKit-compatible path (nonbonded pairs selected at build time exactly
//!   like RDKit); results differ only by floating-point summation order.
//! * [`FastMode::Approximate`]: cutoff nonbonded interactions with a
//!   buffered (Verlet) cluster-pair list that is rebuilt only when atoms
//!   moved more than half the buffer, optional periodic boundary
//!   conditions and particle-mesh Ewald electrostatics.
//!
//! The nonbonded part is evaluated through the [`NonbondedBackend`] trait
//! so that accelerators (e.g. GPUs) can be added later.

pub mod approx;
pub mod backend;
pub mod bonded;
pub mod compile;
pub mod exact;
pub mod workspace;

pub use backend::NonbondedBackend;

/// Van der Waals truncation for approximate mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VdwModifier {
    /// Shift the potential to zero at the cutoff.
    PotentialShift,
    /// Smoothly switch the potential to zero between `r_switch` and the cutoff.
    Switch { r_switch: f64 },
}

/// Electrostatics treatment for approximate mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Electrostatics {
    /// No cutoff (all pairs, SIMD N²); isolated systems only.
    Full,
    /// Reaction-field cutoff (`eps_rf = f64::INFINITY` = conducting boundary).
    ReactionField { eps_rf: f64 },
    /// Smooth particle-mesh Ewald (periodic systems only).
    Pme { ewald_rtol: f64, grid_spacing: f64, order: usize },
}

/// Options of the approximate mode.
#[derive(Debug, Clone, PartialEq)]
pub struct ApproxOptions {
    /// Interaction cutoff (Å).
    pub cutoff: f64,
    /// Verlet buffer (Å): the pair list uses `cutoff + buffer` and is rebuilt
    /// when any atom moved more than `buffer / 2` since the last build.
    pub buffer: f64,
    pub vdw_modifier: VdwModifier,
    pub electrostatics: Electrostatics,
}

impl Default for ApproxOptions {
    fn default() -> Self {
        ApproxOptions {
            cutoff: 10.0,
            buffer: 1.0,
            vdw_modifier: VdwModifier::PotentialShift,
            electrostatics: Electrostatics::ReactionField { eps_rf: f64::INFINITY },
        }
    }
}

/// Evaluation mode.
#[derive(Debug, Clone, PartialEq)]
pub enum FastMode {
    Exact,
    Approximate(ApproxOptions),
}

/// Options of [`FastBuilder`].
#[derive(Debug, Clone, PartialEq)]
pub struct FastOptions {
    pub mode: FastMode,
    /// Use SIMD kernels (falls back to scalar reference kernels if `false`).
    pub simd: bool,
    /// Minimum number of terms per parallel task (smaller systems run on
    /// one thread to avoid overhead). Results do not depend on the thread count.
    pub parallel_grain: usize,
}

impl FastOptions {
    pub fn exact() -> Self {
        FastOptions { mode: FastMode::Exact, simd: true, parallel_grain: 2048 }
    }
    pub fn approximate() -> Self {
        FastOptions { mode: FastMode::Approximate(ApproxOptions::default()), simd: true, parallel_grain: 2048 }
    }
    pub fn cutoff(mut self, rc: f64) -> Self {
        if let FastMode::Approximate(a) = &mut self.mode {
            a.cutoff = rc;
        }
        self
    }
    pub fn buffer(mut self, b: f64) -> Self {
        if let FastMode::Approximate(a) = &mut self.mode {
            a.buffer = b;
        }
        self
    }
    pub fn electrostatics(mut self, e: Electrostatics) -> Self {
        if let FastMode::Approximate(a) = &mut self.mode {
            a.electrostatics = e;
        }
        self
    }
    pub fn vdw_modifier(mut self, m: VdwModifier) -> Self {
        if let FastMode::Approximate(a) = &mut self.mode {
            a.vdw_modifier = m;
        }
        self
    }
}

impl Default for FastOptions {
    fn default() -> Self {
        Self::exact()
    }
}
