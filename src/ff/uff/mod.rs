//! UFF (Universal Force Field) – port of RDKit `Code/ForceField/UFF` and
//! `Code/GraphMol/ForceFieldHelpers/UFF`.
//!
//! The implementation follows RDKit operation-by-operation (including the
//! order in which contribs are created and summed) so that energies,
//! gradients and minimization trajectories reproduce RDKit's to floating
//! point precision for the same input structure.
//!
//! Copyright (C) 2004-2026 Greg Landrum, Rational Discovery LLC, Paolo Tosco
//! and other RDKit contributors. Distributed under the BSD 3-Clause license
//! (see `LICENSE-RDKIT.txt`).
//!
//! ```no_run
//! use molecule_optimizer_rs::ff::{ForceField, ForceFieldBuilder, uff::UffBuilder};
//! use molecule_optimizer_rs::optim::{self, MinimizeOptions, NoObserver};
//! # fn main() -> molecule_optimizer_rs::Result<()> {
//! let mol = molecule_optimizer_rs::io::read_file("ethanol.sdf")?.remove(0);
//! let ff = UffBuilder::default().build(&mol, &[])?;
//! let mut pos = mol.flat_coords();
//! println!("E = {}", ff.energy(&pos));
//! optim::minimize(&ff, &mut pos, &MinimizeOptions::default(), &mut NoObserver);
//! # Ok(()) }
//! ```

pub mod angle_bend;
pub mod atom_typer;
pub mod bond_stretch;
pub mod builder;
pub mod inversion;
pub mod nonbonded;
pub mod params;
pub mod torsion_angle;

use crate::core::Molecule;
use crate::error::{Error, Result};
use crate::ff::{Constraint, ContribForceField, ForceFieldBuilder};

pub use angle_bend::AngleBendContrib;
pub use atom_typer::atom_label;
pub use bond_stretch::BondStretchContrib;
pub use inversion::InversionContrib;
pub use nonbonded::VdwContrib;
pub use params::AtomicParams;
pub use torsion_angle::TorsionAngleContrib;

/// RDKit `isDoubleZero`.
#[inline]
pub(crate) fn is_double_zero(x: f64) -> bool {
    x < 1.0e-10 && x > -1.0e-10
}

/// RDKit `int_pow<N>` (same multiplication sequence).
#[inline]
pub(crate) fn int_pow<const N: u32>(x: f64) -> f64 {
    int_pow_n(x, N)
}

#[inline]
fn int_pow_n(x: f64, n: u32) -> f64 {
    match n {
        0 => 1.0,
        1 => x,
        _ => {
            let half = int_pow_n(x, n / 2);
            if n.is_multiple_of(2) { half * half } else { half * half * x }
        }
    }
}

/// RDKit `ForceField::distance` (indices ordered, squares summed x, y, z).
#[inline]
pub(crate) fn rd_distance(pos: &[f64], i: usize, j: usize) -> f64 {
    let (i, j) = if j < i { (j, i) } else { (i, j) };
    let mut res = 0.0;
    for k in 0..3 {
        let t = pos[3 * i + k] - pos[3 * j + k];
        res += t * t;
    }
    res.sqrt()
}

/// UFF atom type labels (e.g. `"C_3"`, `"C_R"`, `"N_2"`), RDKit
/// `UFF::Tools::getAtomLabel`. Labels of atoms that UFF cannot type are
/// still returned (they have no parameters, see [`uff_has_all_params`]).
pub fn uff_atom_types(mol: &Molecule) -> Result<Vec<String>> {
    Ok((0..mol.num_atoms()).map(|i| atom_label(mol, i)).collect())
}

/// `true` if every atom has UFF parameters (RDKit `UFFHasAllMoleculeParams`).
pub fn uff_has_all_params(mol: &Molecule) -> bool {
    atom_typer::atom_types_and_params(mol).2
}

/// Builder for UFF force fields (RDKit `UFF::constructForceField`).
///
/// Like RDKit, atoms without parameters are silently left out of the
/// terms that need them, and hydrogens that are not explicit atoms are not
/// represented.
#[derive(Debug, Clone)]
pub struct UffBuilder {
    /// Non-bonded pairs farther apart than `vdw_thresh · x_ij` at build time
    /// are skipped (RDKit `vdwThresh`, default 10.0).
    pub vdw_thresh: f64,
    /// Skip non-bonded terms between different fragments (default `true`).
    pub ignore_interfrag_interactions: bool,
}

impl Default for UffBuilder {
    fn default() -> Self {
        UffBuilder { vdw_thresh: 10.0, ignore_interfrag_interactions: true }
    }
}

impl ForceFieldBuilder for UffBuilder {
    fn name(&self) -> &str {
        "UFF"
    }

    fn build(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<ContribForceField> {
        if mol.coords.len() != mol.num_atoms() {
            return Err(Error::InvalidArgument("molecule has no coordinates".into()));
        }
        let (_, params, _) = atom_typer::atom_types_and_params(mol);
        let mut ff =
            builder::construct_force_field(mol, &params, self.vdw_thresh, self.ignore_interfrag_interactions)?;
        ff.add_constraints(&mol.coords, constraints);
        Ok(ff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, BondOrder};
    use crate::ff::{ForceField, gradient_error};
    use crate::perception::sanitize;

    #[test]
    fn int_pow_matches_rdkit_sequence() {
        let x = 1.2345678901234567_f64;
        assert_eq!(int_pow::<4>(x), (x * x) * (x * x));
        assert_eq!(int_pow::<5>(x), (x * x) * (x * x) * x);
        assert_eq!(int_pow::<6>(x), (x * x * x) * (x * x * x));
        assert_eq!(int_pow::<7>(x), (x * x * x) * (x * x * x) * x);
        let r6 = (x * x * x) * (x * x * x);
        assert_eq!(int_pow::<13>(x), r6 * r6 * x);
    }

    /// Small molecule exercising every contrib type: N-methylformamide-like
    /// sp2 carbon (inversion), sp3-sp3 / sp2-sp3 torsions and vdW pairs.
    fn test_mol() -> Molecule {
        let mut m = Molecule::new();
        let c1 = m.add_atom(Atom::new(6), [0.0, 0.0, 0.0]);
        let o = m.add_atom(Atom::new(8), [1.2, 0.1, 0.05]);
        let n = m.add_atom(Atom::new(7), [-0.7, 1.15, 0.1]);
        let c2 = m.add_atom(Atom::new(6), [-2.15, 1.2, -0.05]);
        m.add_bond(c1, o, BondOrder::Double);
        m.add_bond(c1, n, BondOrder::Single);
        m.add_bond(n, c2, BondOrder::Single);
        let hs = [
            (c1, [-0.55, -0.95, -0.1]),
            (n, [-0.2, 2.02, 0.2]),
            (c2, [-2.5, 2.2, 0.3]),
            (c2, [-2.6, 0.45, 0.6]),
            (c2, [-2.5, 1.0, -1.1]),
        ];
        for (a, p) in hs {
            let h = m.add_atom(Atom::new(1), p);
            m.add_bond(a, h, BondOrder::Single);
        }
        sanitize(&mut m).unwrap();
        m
    }

    #[test]
    fn builder_adds_constraints() {
        let m = test_mol();
        let b = UffBuilder::default();
        assert_eq!(b.name(), "UFF");
        let plain = b.build(&m, &[]).unwrap();
        let cs = [Constraint::Fixed { i: 0 }, Constraint::distance(0, 3, 2.0, 100.0)];
        let ff = b.build(&m, &cs).unwrap();
        assert_eq!(ff.num_contribs(), plain.num_contribs() + 1);
        assert_eq!(ff.fixed_points(), &[0]);
        let pos = m.flat_coords();
        let mut g = vec![0.0; pos.len()];
        ff.energy_grad(&pos, &mut g);
        assert_eq!(&g[..3], &[0.0, 0.0, 0.0]);
        let labels: Vec<_> = ff.energy_breakdown(&pos).into_iter().map(|(k, _)| k).collect();
        assert_eq!(labels, ["BondStretch", "AngleBend", "VdW", "Torsion", "Inversion", "Constraint"]);
    }

    #[test]
    fn types_of_formamide() {
        let m = test_mol();
        let t = uff_atom_types(&m).unwrap();
        assert_eq!(&t[..4], &["C_R", "O_R", "N_R", "C_3"]);
        assert!(uff_has_all_params(&m));
    }

    #[test]
    fn gradients_match_numeric_per_contrib_type() {
        let m = test_mol();
        let ff = UffBuilder::default().build(&m, &[]).unwrap();
        let pos = m.flat_coords();
        let mut seen = std::collections::BTreeSet::new();
        for c in ff.contribs() {
            seen.insert(c.label());
            let r = c.as_ref();
            let mut ga = vec![0.0; pos.len()];
            r.grad(&pos, &mut ga);
            let mut p = pos.clone();
            let h = 1e-6;
            for i in 0..pos.len() {
                let x = p[i];
                p[i] = x + h;
                let ep = r.energy(&p);
                p[i] = x - h;
                let em = r.energy(&p);
                p[i] = x;
                let gn = (ep - em) / (2.0 * h);
                assert!((gn - ga[i]).abs() < 1e-5 * (1.0 + gn.abs()), "{} coord {i}: {gn} vs {}", c.label(), ga[i]);
            }
        }
        for l in ["BondStretch", "AngleBend", "Torsion", "Inversion", "VdW"] {
            assert!(seen.contains(l), "missing {l}");
        }
        assert!(gradient_error(&ff, &pos) < 1e-5);
        assert!(ff.energy(&pos).is_finite());
    }
}
