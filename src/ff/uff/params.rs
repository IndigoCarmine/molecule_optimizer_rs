//! UFF atomic parameters.
//!
//! Port of RDKit `Code/ForceField/UFF/Params.{h,cpp}` (the parameter table
//! is `src/data/uff_params.tsv`, extracted verbatim from `Params.cpp`).
//!
//! Copyright (C) 2004-2021 Greg Landrum and other RDKit contributors.
//! Distributed under the BSD 3-Clause license (see `LICENSE-RDKIT.txt`).

use std::collections::HashMap;
use std::sync::OnceLock;

/// Scaling factor for the bond-order correction of the rest length.
pub const LAMBDA: f64 = 0.1332;
/// Bond force constant prefactor.
pub const G: f64 = 332.06;
/// Special-case bond order for amide C-N bonds (defined by RDKit, unused by
/// its builder).
pub const AMIDE_BOND_ORDER: f64 = 1.41;

/// Parameters of one UFF atom type (RDKit `AtomicParams`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicParams {
    /// Valence bond radius.
    pub r1: f64,
    /// Valence angle (radians).
    pub theta0: f64,
    /// vdW characteristic length.
    pub x1: f64,
    /// vdW atomic energy.
    pub d1: f64,
    /// vdW scaling term.
    pub zeta: f64,
    /// Effective charge.
    pub z1: f64,
    /// sp3 torsional barrier parameter.
    pub v1: f64,
    /// Torsional contribution for sp2-sp3 bonds.
    pub u1: f64,
    /// GMP electronegativity.
    pub gmp_xi: f64,
    /// GMP hardness.
    pub gmp_hardness: f64,
    /// GMP radius.
    pub gmp_radius: f64,
}

static PARAM_TEXT: &str = include_str!("../../data/uff_params.tsv");

fn table() -> &'static HashMap<&'static str, AtomicParams> {
    static TABLE: OnceLock<HashMap<&'static str, AtomicParams>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut out = HashMap::new();
        for line in PARAM_TEXT.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let f: Vec<&'static str> = line.split('\t').collect();
            let v = |k: usize| -> f64 { f[k].trim().parse().expect("bad UFF parameter") };
            out.insert(
                f[0],
                AtomicParams {
                    r1: v(1),
                    // RDKit: theta0 * M_PI / 180.
                    theta0: v(2) * std::f64::consts::PI / 180.0,
                    x1: v(3),
                    d1: v(4),
                    zeta: v(5),
                    z1: v(6),
                    v1: v(7),
                    u1: v(8),
                    gmp_xi: v(9),
                    gmp_hardness: v(10),
                    gmp_radius: v(11),
                },
            );
        }
        out
    })
}

/// Looks up the parameters of a UFF atom type label (e.g. `"C_3"`).
pub fn params(label: &str) -> Option<&'static AtomicParams> {
    table().get(label)
}

/// All known atom type labels.
pub fn labels() -> impl Iterator<Item = &'static str> {
    table().keys().copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn table_loaded() {
        assert_eq!(labels().count(), 127);
        let c = params("C_3").unwrap();
        assert_eq!(c.r1, 0.757);
        assert!((c.theta0 - 109.47f64.to_radians()).abs() < 1e-12);
        assert_eq!(params("Lw6+3").unwrap().x1, 3.236);
        assert!(params("C_5").is_none());
    }
}
