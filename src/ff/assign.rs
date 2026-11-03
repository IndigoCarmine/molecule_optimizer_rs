//! Force-field atom typing and charge assignment independent of the input
//! file format.
//!
//! [`assign`] re-derives everything it needs from the molecular graph
//! (elements, connectivity, bond orders, formal charges, explicit H): it
//! sanitizes a copy (kekulization, ring perception, RDKit aromaticity,
//! hybridization), so the result does not depend on whether the molecule
//! came from SDF (Kekulé or aromatic bonds), MOL2, PDB or XYZ, nor on
//! aromatic flags set by a reader.

use super::mmff::{MmffMolProperties, MmffVariant};
use super::uff::{atom_typer::atom_types_and_params, uff_atom_types};
use crate::core::Molecule;
use crate::error::Result;
use crate::perception;

/// Force field to assign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForceFieldKind {
    Mmff94,
    Mmff94s,
    Uff,
}

/// Atom types and charges for one molecule.
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub kind: ForceFieldKind,
    /// Atom type per atom: MMFF numeric type as a string (`"37"`), or UFF
    /// label (`"C_R"`). Empty string = untyped.
    pub types: Vec<String>,
    /// MMFF numeric types (`None` for UFF; 0 = untyped).
    pub mmff_types: Option<Vec<u8>>,
    /// Partial charges in e (MMFF bond-charge-increment charges; `None`
    /// for UFF, which has no electrostatics).
    pub partial_charges: Option<Vec<f64>>,
    /// Formal charges used (integer formal charges from the graph; MMFF
    /// reports fractional formal charges for delocalized groups).
    pub formal_charges: Vec<f64>,
    /// Atoms without parameters.
    pub missing: Vec<usize>,
}

impl Assignment {
    /// `true` if every atom is typed and parameterized.
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
    }
    /// Net charge (sum of partial charges, or of formal charges for UFF).
    pub fn net_charge(&self) -> f64 {
        match &self.partial_charges {
            Some(q) => q.iter().sum(),
            None => self.formal_charges.iter().sum(),
        }
    }
}

/// Returns the molecule in the canonical perceived form used for typing.
pub fn normalized(mol: &Molecule) -> Result<Molecule> {
    let mut m = mol.clone();
    perception::sanitize(&mut m)?;
    Ok(m)
}

/// Assigns atom types and charges for `kind` (see module docs).
pub fn assign(mol: &Molecule, kind: ForceFieldKind) -> Result<Assignment> {
    let m = normalized(mol)?;
    match kind {
        ForceFieldKind::Mmff94 | ForceFieldKind::Mmff94s => {
            let variant = if kind == ForceFieldKind::Mmff94 { MmffVariant::Mmff94 } else { MmffVariant::Mmff94s };
            let p = MmffMolProperties::new(&m, variant)?;
            let t = p.atom_types().to_vec();
            Ok(Assignment {
                kind,
                types: t.iter().map(|&x| if x == 0 { String::new() } else { x.to_string() }).collect(),
                missing: (0..t.len()).filter(|&i| t[i] == 0).collect(),
                mmff_types: Some(t),
                partial_charges: Some(p.partial_charges().to_vec()),
                formal_charges: p.formal_charges().to_vec(),
            })
        }
        ForceFieldKind::Uff => {
            let labels = uff_atom_types(&m)?;
            let (_, params, _) = atom_types_and_params(&m);
            Ok(Assignment {
                kind,
                missing: (0..params.len()).filter(|&i| params[i].is_none()).collect(),
                types: labels,
                mmff_types: None,
                partial_charges: None,
                formal_charges: m.atoms().iter().map(|a| a.formal_charge as f64).collect(),
            })
        }
    }
}
