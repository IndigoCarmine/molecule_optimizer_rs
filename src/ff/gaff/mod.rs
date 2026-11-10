//! GAFF, the general Amber force field: GAFF 1.81, GAFF 2.11 and GAFF
//! 2.2.30.
//!
//! Atom typing, the functional form and the rules for parameters that are
//! not tabulated are implemented from J. Wang, R. M. Wolf, J. W. Caldwell,
//! P. A. Kollman, D. A. Case, "Development and testing of a general Amber
//! force field", J. Comput. Chem. 25, 1157-1174 (2004) and its erratum,
//! J. Comput. Chem. 26, 114 (2005). The conventions GAFF inherits from the
//! Amber protein force fields (combining rules, 1-4 scaling, the columns of
//! the parameter tables) are those of Cornell et al., J. Am. Chem. Soc.
//! 117, 5179 (1995) and Weiner et al., J. Am. Chem. Soc. 106, 765 (1984).
//! The parameter tables are the public-domain files in `src/data/gaff/`; the atom types
//! added after the paper are assigned from the descriptions in those files
//! (see [`typer`]).
//!
//! The energy is eq. (1) of the paper:
//!
//! ```text
//! E = Σ K_r (r - r_eq)² + Σ K_θ (θ - θ_eq)² + Σ (V_n / 2) (1 + cos(n φ - γ))
//!   + Σ_{i<j} [ ε_ij ((R*_ij / r)¹² - 2 (R*_ij / r)⁶) + q_i q_j / (ε r) ]
//! ```
//!
//! with `R*_ij = R*_i + R*_j` and `ε_ij = sqrt(ε_i ε_j)`, 1-4 van der Waals
//! interactions scaled by 1/2 and 1-4 electrostatics by 1/1.2, and no
//! cutoff. The input must be a complete structure with
//! all hydrogens.
//!
//! Partial charges are not computed here (the paper uses RESP or AM1-BCC
//! charges, which need a quantum-chemical calculation): pass them with
//! [`GaffBuilder::with_charges`]. Without charges the force field has no
//! electrostatic term, which [`GaffBuilder::assign`] reports as a warning.
//!
//! ```no_run
//! use molecule_optimizer_rs::ff::{ForceField, ForceFieldBuilder, gaff::{GaffBuilder, GaffVariant}};
//! # fn main() -> molecule_optimizer_rs::Result<()> {
//! let mol = molecule_optimizer_rs::io::read_file("ligand.mol2")?.remove(0);
//! let builder = GaffBuilder::new(GaffVariant::Gaff2);
//! let report = builder.assign(&mol)?;
//! println!("{:?} {:?}", report.types, report.warnings);
//! let ff = builder.build(&mol, &[])?;
//! println!("E = {}", ff.energy(&mol.flat_coords()));
//! # Ok(()) }
//! ```

pub mod builder;
pub mod estimate;
pub mod params;
pub mod terms;
pub mod typer;

#[cfg(test)]
mod tests;

use crate::core::Molecule;
use crate::error::{Error, Result};
use crate::ff::{Constraint, ContribForceField, ForceFieldBuilder};

pub use builder::GaffAssignment;
pub use params::{AngleParam, AtomTypeInfo, BondParam, GaffParams, GaffVariant, TorsionTerm, VdwParam};
pub use terms::{CoulombContrib, FourierTorsionContrib, HarmonicAngleContrib};

/// `e² / (4 π ε0)` in kcal·Å/mol (from the CODATA constants; none of the
/// papers gives a value). Amber programs use 18.2223² = 332.0522; set
/// [`GaffBuilder::coulomb_constant`] to compare with them.
pub const COULOMB_CONSTANT: f64 = 332.063_71;

/// Builder for GAFF force fields.
#[derive(Debug, Clone)]
pub struct GaffBuilder {
    /// Parameter set.
    pub variant: GaffVariant,
    /// Partial charges in e, one per atom. `None`: all charges are zero and
    /// no electrostatic term is created.
    pub charges: Option<Vec<f64>>,
    /// Dielectric constant ε of the electrostatic term (default 1).
    pub dielectric_constant: f64,
    /// Scale of the 1-4 van der Waals interactions (default 1/2).
    pub scale14_vdw: f64,
    /// Scale of the 1-4 electrostatic interactions (default 1/1.2).
    pub scale14_ele: f64,
    /// Coulomb constant in kcal·Å/(mol·e²) (default [`COULOMB_CONSTANT`]).
    pub coulomb_constant: f64,
    /// Skip non-bonded terms between different fragments (default `false`).
    pub ignore_interfrag_interactions: bool,
    /// Fail if an atom has no type or a bond, angle or torsion has no
    /// parameters, instead of leaving the term out (default `false`).
    pub strict: bool,
}

impl Default for GaffBuilder {
    fn default() -> Self {
        GaffBuilder::new(GaffVariant::default())
    }
}

impl GaffBuilder {
    pub fn new(variant: GaffVariant) -> Self {
        GaffBuilder {
            variant,
            charges: None,
            dielectric_constant: 1.0,
            scale14_vdw: 0.5,
            scale14_ele: 1.0 / 1.2,
            coulomb_constant: COULOMB_CONSTANT,
            ignore_interfrag_interactions: false,
            strict: false,
        }
    }

    /// Sets the partial charges (in e, one per atom).
    pub fn with_charges(mut self, charges: Vec<f64>) -> Self {
        self.charges = Some(charges);
        self
    }

    fn check_charges(&self, mol: &Molecule) -> Result<()> {
        match &self.charges {
            Some(q) if q.len() != mol.num_atoms() => {
                Err(Error::InvalidArgument(format!("GAFF: {} charges for {} atoms", q.len(), mol.num_atoms())))
            }
            _ => Ok(()),
        }
    }

    /// Atom types and parameter coverage of `mol` for this builder,
    /// including the warning about missing charges.
    pub fn assign(&self, mol: &Molecule) -> Result<GaffAssignment> {
        self.check_charges(mol)?;
        let (mut a, _) = builder::parameterize(mol, self.variant)?;
        if self.charges.is_none() {
            a.warnings.push("no partial charges were given: all charges are zero and there is no electrostatic term".into());
        }
        Ok(a)
    }
}

impl ForceFieldBuilder for GaffBuilder {
    fn name(&self) -> &str {
        self.variant.name()
    }

    fn build(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<ContribForceField> {
        if mol.coords.len() != mol.num_atoms() {
            return Err(Error::InvalidArgument("molecule has no coordinates".into()));
        }
        self.check_charges(mol)?;
        let (a, terms) = builder::parameterize(mol, self.variant)?;
        if self.strict && !a.is_complete() {
            return Err(Error::MissingParameters(format!(
                "{}: {} untyped atoms, {} bonds, {} angles and {} torsions without parameters",
                self.name(),
                a.untyped.len(),
                a.missing_bonds.len(),
                a.missing_angles.len(),
                a.missing_torsions.len()
            )));
        }
        let mut ff = ContribForceField::new(mol.num_atoms(), 3);
        ff.add_contribs(builder::build_contribs(mol, &terms, self)?);
        ff.add_constraints(&mol.coords, constraints);
        Ok(ff)
    }
}

/// Atom types and parameter coverage of `mol` (see [`GaffAssignment`]).
pub fn gaff_assign(mol: &Molecule, variant: GaffVariant) -> Result<GaffAssignment> {
    Ok(builder::parameterize(mol, variant)?.0)
}

/// GAFF atom types of `mol` (empty string = no type).
pub fn gaff_atom_types(mol: &Molecule, variant: GaffVariant) -> Result<Vec<String>> {
    Ok(gaff_assign(mol, variant)?.types)
}

/// `true` if every atom is typed and every bond, angle and torsion of
/// `mol` has parameters.
pub fn gaff_has_all_params(mol: &Molecule, variant: GaffVariant) -> bool {
    gaff_assign(mol, variant).is_ok_and(|a| a.is_complete())
}
