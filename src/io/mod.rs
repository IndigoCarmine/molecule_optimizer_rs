//! File formats: XYZ, MDL Molfile / SDF, PDB.

pub mod molfile;
pub mod pdb;
pub mod xyz;

use crate::core::Molecule;
use crate::error::{Error, Result};
use std::path::Path;

pub use molfile::{read_molfile, read_sdf, write_molblock, write_sdf_record};
pub use pdb::{read_pdb, write_pdb};
pub use xyz::{read_xyz, write_xyz};

/// Supported formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Xyz,
    Mol,
    Sdf,
    Pdb,
}

impl Format {
    pub fn from_path(path: &Path) -> Option<Format> {
        match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "xyz" => Some(Format::Xyz),
            "mol" | "mdl" => Some(Format::Mol),
            "sdf" | "sd" => Some(Format::Sdf),
            "pdb" | "ent" => Some(Format::Pdb),
            _ => None,
        }
    }
}

/// Reads all molecules from a string in the given format.
pub fn read_str(text: &str, format: Format) -> Result<Vec<Molecule>> {
    match format {
        Format::Xyz => xyz::read_xyz_str(text),
        Format::Mol => Ok(vec![molfile::read_molfile(text)?]),
        Format::Sdf => molfile::read_sdf(text),
        Format::Pdb => pdb::read_pdb_models(text),
    }
}

/// Reads all molecules from a file; the format is taken from the extension.
pub fn read_file(path: impl AsRef<Path>) -> Result<Vec<Molecule>> {
    let path = path.as_ref();
    let fmt = Format::from_path(path)
        .ok_or_else(|| Error::InvalidArgument(format!("unknown file extension: {}", path.display())))?;
    let text = std::fs::read_to_string(path)?;
    read_str(&text, fmt)
}

/// Serializes a molecule (with optional replacement coordinates).
pub fn write_str(mol: &Molecule, coords: Option<&[[f64; 3]]>, format: Format) -> String {
    let c = coords.unwrap_or(&mol.coords);
    match format {
        Format::Xyz => xyz::write_xyz_coords(mol, c, None),
        Format::Mol => molfile::write_molblock_coords(mol, c),
        Format::Sdf => molfile::write_sdf_record(mol, Some(c), &[]),
        Format::Pdb => pdb::write_pdb_coords(mol, c),
    }
}

/// Writes molecules to a file; the format is taken from the extension.
pub fn write_file(path: impl AsRef<Path>, mols: &[&Molecule]) -> Result<()> {
    let path = path.as_ref();
    let fmt = Format::from_path(path)
        .ok_or_else(|| Error::InvalidArgument(format!("unknown file extension: {}", path.display())))?;
    let mut s = String::new();
    for m in mols {
        s.push_str(&write_str(m, None, fmt));
    }
    std::fs::write(path, s)?;
    Ok(())
}
