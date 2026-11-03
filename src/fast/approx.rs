//! Approximate nonbonded evaluation: cutoffs, buffered cluster-pair lists,
//! periodic boundary conditions and particle-mesh Ewald.
//!
//! (Interface stub; implementation pending.)

use super::ApproxOptions;
use super::backend::{NonbondedBackend, NonbondedEnergy};
use crate::core::Molecule;
use crate::error::{Error, Result};

/// Approximate nonbonded backend.
pub struct ApproxNonbonded {
    _private: (),
}

impl NonbondedBackend for ApproxNonbonded {
    fn name(&self) -> &str {
        "approx-stub"
    }
    fn energy(&self, _pos: &[f64]) -> NonbondedEnergy {
        NonbondedEnergy::default()
    }
    fn energy_grad(&self, _pos: &[f64], _grad: &mut [f64]) -> NonbondedEnergy {
        NonbondedEnergy::default()
    }
}

/// Builds the approximate nonbonded backend for `mol`.
///
/// `inner_builder` is the wrapped force-field builder (`&MmffBuilder` or
/// `&UffBuilder` as `&dyn Any`); its settings (dielectric, interfragment
/// handling, ...) and the molecule's typing define the per-pair parameters
/// and exclusions. The periodic cell, if any, is taken from the molecule.
pub fn build_approx_nonbonded(
    mol: &Molecule,
    inner_builder: &dyn std::any::Any,
    opts: &ApproxOptions,
    simd: bool,
) -> Result<ApproxNonbonded> {
    let _ = (mol, inner_builder, opts, simd);
    Err(Error::ForceField("approximate nonbonded mode not implemented yet".into()))
}
