//! MMFF94 / MMFF94s – port of RDKit `Code/ForceField/MMFF` and
//! `Code/GraphMol/ForceFieldHelpers/MMFF`.
//!
//! * [`MmffMolProperties`]: MMFF atom typing (including RDKit's MMFF
//!   aromaticity model) and partial charges.
//! * [`MmffBuilder`]: builds a [`ContribForceField`] with all MMFF terms
//!   (bond stretch, angle bend, stretch-bend, out-of-plane, torsion,
//!   buffered 14-7 van der Waals, electrostatics).
//!
//! Energies and gradients follow the RDKit implementation operation by
//! operation; minimizing with [`crate::optim::minimize`] and the default
//! [`crate::optim::MinimizeOptions`] corresponds to RDKit's
//! `MMFFOptimizeMolecule`.
//!
//! Ported from the RDKit (<https://github.com/rdkit/rdkit>):
//! Copyright (C) 2013-2025 Paolo Tosco, Greg Landrum and other RDKit
//! contributors. BSD 3-Clause license (see `LICENSE-RDKIT.txt`).

mod builder;
pub mod params;
pub mod terms;
mod typer;

pub use typer::MmffMolProperties;

use crate::core::Molecule;
use crate::error::{Error, Result};
use crate::ff::{Constraint, ContribForceField, ForceFieldBuilder};

/// RDKit `MDYNE_A_TO_KCAL_MOL`.
pub const MDYNE_A_TO_KCAL_MOL: f64 = 143.9325;
/// RDKit `DEG2RAD` (`M_PI / 180.0`).
pub const DEG2RAD: f64 = std::f64::consts::PI / 180.0;
/// RDKit `RAD2DEG` (`180.0 / M_PI`).
pub const RAD2DEG: f64 = 180.0 / std::f64::consts::PI;

/// MMFF variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum MmffVariant {
    /// MMFF94 (dynamics variant).
    #[default]
    Mmff94,
    /// MMFF94s (static variant: modified out-of-plane and torsion
    /// parameters for planar delocalized nitrogens).
    Mmff94s,
}

impl MmffVariant {
    pub fn name(self) -> &'static str {
        match self {
            MmffVariant::Mmff94 => "MMFF94",
            MmffVariant::Mmff94s => "MMFF94s",
        }
    }
}

/// Dielectric model for the electrostatic term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum DielectricModel {
    /// `E = 332.0716 qi qj / (D (r + 0.05))`
    #[default]
    Constant,
    /// `E = 332.0716 qi qj / (D (r + 0.05)^2)`
    Distance,
}

impl DielectricModel {
    /// RDKit's numeric value of the model (`CONSTANT = 1`, `DISTANCE = 2`),
    /// which also enters the gradient as the exponent of the distance.
    pub fn as_f64(self) -> f64 {
        match self {
            DielectricModel::Constant => 1.0,
            DielectricModel::Distance => 2.0,
        }
    }
}

/// Builder for MMFF94 / MMFF94s force fields. Defaults match RDKit's
/// `MMFFGetMoleculeForceField` / `MMFFOptimizeMolecule`.
#[derive(Debug, Clone, PartialEq)]
pub struct MmffBuilder {
    pub variant: MmffVariant,
    pub dielectric_model: DielectricModel,
    /// Dielectric constant `D` (default 1.0).
    pub dielectric_constant: f64,
    /// Non-bonded pairs farther apart than this (Å, in the input
    /// coordinates) are not included (default 100.0).
    pub nonbonded_thresh: f64,
    /// Skip non-bonded terms between different fragments (default `true`).
    pub ignore_interfrag_interactions: bool,
}

impl Default for MmffBuilder {
    fn default() -> Self {
        MmffBuilder {
            variant: MmffVariant::Mmff94,
            dielectric_model: DielectricModel::Constant,
            dielectric_constant: 1.0,
            nonbonded_thresh: 100.0,
            ignore_interfrag_interactions: true,
        }
    }
}

impl MmffBuilder {
    /// Default settings for the given variant.
    pub fn new(variant: MmffVariant) -> Self {
        MmffBuilder { variant, ..Default::default() }
    }

    /// Builds the force field from precomputed molecule properties (which
    /// must have been computed for `mol` with `self.variant`).
    pub fn build_with_properties(&self, mol: &Molecule, props: &MmffMolProperties, constraints: &[Constraint]) -> Result<ContribForceField> {
        if !props.is_valid() {
            return Err(Error::MissingParameters(format!("{}: missing MMFF atom types", self.variant.name())));
        }
        if props.atom_types().len() != mol.num_atoms() {
            return Err(Error::InvalidArgument("MMFF properties do not belong to this molecule".into()));
        }
        if props.variant() != self.variant {
            return Err(Error::InvalidArgument("MMFF properties were computed for another variant".into()));
        }
        let mut ff = ContribForceField::new(mol.num_atoms(), 3);
        builder::add_mmff_terms(&mut ff, props, &mol.coords, self);
        ff.add_constraints(&mol.coords, constraints);
        Ok(ff)
    }
}

impl ForceFieldBuilder for MmffBuilder {
    fn name(&self) -> &str {
        self.variant.name()
    }
    fn build(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<ContribForceField> {
        let props = MmffMolProperties::new(mol, self.variant)?;
        self.build_with_properties(mol, &props, constraints)
    }
}

/// `true` if every atom of `mol` can be given an MMFF atom type (RDKit
/// `MMFFHasAllMoleculeParams`).
pub fn mmff_has_all_params(mol: &Molecule) -> bool {
    MmffMolProperties::new(mol, MmffVariant::Mmff94).map(|p| p.is_valid()).unwrap_or(false)
}

#[cfg(test)]
mod tests;
