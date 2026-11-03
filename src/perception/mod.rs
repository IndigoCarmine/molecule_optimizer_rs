//! Chemical perception: rings, valence, aromaticity, conjugation,
//! hybridization, kekulization, connectivity from coordinates and
//! stereochemistry from 3D.

pub mod aromaticity;
pub mod bond_perception;
mod canon;
pub mod hybridization;
pub mod hydrogens;
pub mod kekulize;
pub mod legacy_stereo;
pub mod matching;
mod rdkit_kekulize;
mod ring_decomposer;
pub mod rings;
pub mod stereo;
pub mod valence;

use crate::core::Molecule;
use crate::error::Result;

/// Options for [`sanitize_with`].
#[derive(Debug, Clone)]
pub struct SanitizeOptions {
    pub aromaticity: bool,
    /// Assign tetrahedral / double-bond stereo from the 3D coordinates.
    pub stereo_from_3d: bool,
}

impl Default for SanitizeOptions {
    fn default() -> Self {
        SanitizeOptions { aromaticity: true, stereo_from_3d: true }
    }
}

/// RDKit-like sanitization: implicit hydrogens, kekulization, ring
/// perception, aromaticity, conjugation, hybridization, and stereo from 3D.
pub fn sanitize(mol: &mut Molecule) -> Result<()> {
    sanitize_with(mol, &SanitizeOptions::default())
}

pub fn sanitize_with(mol: &mut Molecule, opts: &SanitizeOptions) -> Result<()> {
    valence::compute_implicit_hs(mol);
    kekulize::kekulize(mol)?;
    aromaticity::clear_aromatic_flags(mol);
    mol.ring_info = None;
    mol.rings();
    if opts.aromaticity {
        aromaticity::set_aromaticity(mol);
    }
    hybridization::set_conjugation(mol);
    hybridization::set_hybridization(mol);
    if opts.stereo_from_3d && has_3d_coords(mol) {
        stereo::assign_chiral_tags_from_3d(mol);
        stereo::assign_double_bond_stereo_from_3d(mol);
        // chirality may change hybridization of 4-coordinate atoms
        hybridization::set_hybridization(mol);
        // RDKit: non-tetrahedral chiral tags from 3D set SP2D/SP3D/SP3D2
        stereo::apply_nontetrahedral_hybridization_from_3d(mol);
    }
    Ok(())
}

/// Builds connectivity and bond orders from coordinates (all hydrogens must
/// be present as atoms), then sanitizes.
pub fn perceive_from_coordinates(mol: &mut Molecule) -> Result<()> {
    bond_perception::perceive_connectivity(mol, bond_perception::BOND_TOLERANCE);
    kekulize::assign_bond_orders(mol);
    sanitize(mol)
}

/// `true` if the coordinates are non-planar in z (i.e. real 3D).
pub fn has_3d_coords(mol: &Molecule) -> bool {
    mol.coords.iter().any(|p| p[2].abs() > 1e-4)
}

/// Returns a kekulized copy (aromatic flags cleared).
pub fn kekulized_copy(mol: &Molecule) -> Result<Molecule> {
    let mut m = mol.clone();
    kekulize::kekulize(&mut m)?;
    aromaticity::clear_aromatic_flags(&mut m);
    Ok(m)
}

/// RDKit canonical atom ranks of the whole molecule
/// (`Canon::rankFragmentAtoms` with all atoms and bonds, ties broken; equal
/// to `Chem.CanonicalRankAtomsInFragment(mol, range(n), range(nb))`).  For
/// 3D molecules the chirality used is RDKit's legacy stereo perception from
/// the coordinates; otherwise the molecule's own chiral tags.
pub fn canonical_atom_ranks(mol: &Molecule) -> Vec<usize> {
    let mut stereo = mol.clone();
    stereo.rings();
    if has_3d_coords(&stereo) {
        // changes only stereo tags: the ring info stays valid
        legacy_stereo::assign_stereo_legacy_from_3d(&mut stereo);
    }
    canon::rank_fragment_atoms(&stereo, stereo.ring_info().expect("ring info"))
}

/// Kekulized copy exactly as RDKit's canonical `MolOps::Kekulize` produces
/// it (what RDKit's MMFF typer works on).  The canonical ranks use the
/// chirality RDKit would have assigned (legacy stereo perception from 3D
/// coordinates when the molecule is 3D).  Falls back to
/// [`kekulized_copy`] if the RDKit algorithm cannot kekulize the molecule.
pub fn kekulized_copy_canonical(mol: &Molecule) -> Result<Molecule> {
    let mut m = mol.clone();
    m.rings();
    let ranks = canonical_atom_ranks(&m);
    match rdkit_kekulize::kekulize_canonical(&mut m, Some(&ranks)) {
        Ok(()) => Ok(m),
        Err(_) => kekulized_copy(mol),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, BondOrder, Hybridization};

    fn benzene() -> Molecule {
        let mut m = Molecule::new();
        for k in 0..6 {
            let a = std::f64::consts::PI / 3.0 * k as f64;
            m.add_atom(Atom::new(6), [1.39 * a.cos(), 1.39 * a.sin(), 0.0]);
        }
        for k in 0..6 {
            let a = std::f64::consts::PI / 3.0 * k as f64;
            let h = m.add_atom(Atom::new(1), [2.47 * a.cos(), 2.47 * a.sin(), 0.0]);
            m.add_bond(k, h, BondOrder::Single);
        }
        for k in 0..6 {
            m.add_bond(k, (k + 1) % 6, if k % 2 == 0 { BondOrder::Double } else { BondOrder::Single });
        }
        m
    }

    #[test]
    fn benzene_aromatic() {
        let mut m = benzene();
        sanitize(&mut m).unwrap();
        for i in 0..6 {
            assert!(m.atom(i).is_aromatic);
            assert_eq!(m.atom(i).hybridization, Hybridization::SP2);
            assert_eq!(m.atom(i).implicit_hs, 0);
        }
    }

    #[test]
    fn benzene_from_xyz_coords() {
        let mut m = benzene();
        m.clear_bonds();
        perceive_from_coordinates(&mut m).unwrap();
        assert_eq!(m.num_bonds(), 12);
        assert!((0..6).all(|i| m.atom(i).is_aromatic));
    }

    #[test]
    fn pyrrole_and_furan_aromatic() {
        // pyrrole: N1 C2=C3 C4=C5 with N-H, ring 0-1-2-3-4
        for (het, h_on_het) in [(7u8, true), (8u8, false)] {
            let mut m = Molecule::new();
            let mut ring = Vec::new();
            for k in 0..5 {
                let a = 2.0 * std::f64::consts::PI / 5.0 * k as f64;
                ring.push(m.add_atom(Atom::new(if k == 0 { het } else { 6 }), [1.2 * a.cos(), 1.2 * a.sin(), 0.0]));
            }
            m.add_bond(0, 1, BondOrder::Single);
            m.add_bond(1, 2, BondOrder::Double);
            m.add_bond(2, 3, BondOrder::Single);
            m.add_bond(3, 4, BondOrder::Double);
            m.add_bond(4, 0, BondOrder::Single);
            if h_on_het {
                let h = m.add_atom(Atom::new(1), [2.2, 0.0, 0.0]);
                m.add_bond(0, h, BondOrder::Single);
            }
            sanitize(&mut m).unwrap();
            assert!((0..5).all(|i| m.atom(i).is_aromatic), "het {het}");
        }
    }

    #[test]
    fn cyclohexadiene_not_aromatic() {
        let mut m = Molecule::new();
        for k in 0..6 {
            let a = std::f64::consts::PI / 3.0 * k as f64;
            m.add_atom(Atom::new(6), [1.4 * a.cos(), 1.4 * a.sin(), 0.0]);
        }
        let orders = [BondOrder::Double, BondOrder::Single, BondOrder::Double, BondOrder::Single, BondOrder::Single, BondOrder::Single];
        for k in 0..6 {
            m.add_bond(k, (k + 1) % 6, orders[k]);
        }
        sanitize(&mut m).unwrap();
        assert!((0..6).all(|i| !m.atom(i).is_aromatic));
        assert_eq!(m.atom(4).hybridization, Hybridization::SP3);
    }
}
