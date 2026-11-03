//! UFF atom typing and bond rest lengths, as used by the distance-geometry
//! bounds-matrix builder for 1-2 distances.
//!
//! Ported from RDKit 2026.09.1:
//! * `Code/ForceField/UFF/Params.{h,cpp}` (`AtomicParams`, `defaultParamData`,
//!   the table lives in `data/uff_params.txt`)
//! * `Code/GraphMol/ForceFieldHelpers/UFF/AtomTyper.cpp`
//!   (`Tools::addAtomChargeFlags`, `Tools::getAtomLabel`, `getAtomTypes`)
//! * `Code/ForceField/UFF/BondStretch.cpp` (`Utils::calcBondRestLength`)
//!
//! Copyright (C) 2004-2026 Greg Landrum and other RDKit contributors,
//! BSD-3-Clause.

use crate::core::{Hybridization, Molecule, element};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Scaling factor for the Pauling bond-order correction (RDKit `Params::lambda`).
pub const LAMBDA: f64 = 0.1332;

/// UFF atomic parameters (RDKit `ForceFields::UFF::AtomicParams`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UffAtomParams {
    /// Valence bond radius (Å).
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

static PARAM_TEXT: &str = include_str!("data/uff_params.txt");

fn param_table() -> &'static HashMap<String, UffAtomParams> {
    static TABLE: OnceLock<HashMap<String, UffAtomParams>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut out = HashMap::new();
        for line in PARAM_TEXT.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            assert!(f.len() >= 12, "bad UFF parameter line: {line}");
            let p = |i: usize| -> f64 {
                f[i].trim().parse().unwrap_or_else(|_| panic!("bad UFF parameter value {:?}", f[i]))
            };
            out.insert(
                f[0].to_string(),
                UffAtomParams {
                    r1: p(1),
                    theta0: p(2) * std::f64::consts::PI / 180.,
                    x1: p(3),
                    d1: p(4),
                    zeta: p(5),
                    z1: p(6),
                    v1: p(7),
                    u1: p(8),
                    gmp_xi: p(9),
                    gmp_hardness: p(10),
                    gmp_radius: p(11),
                },
            );
        }
        out
    })
}

/// Lookup by UFF label, e.g. `"C_R"`.
pub fn uff_params_by_label(label: &str) -> Option<UffAtomParams> {
    param_table().get(label).copied()
}

/// RDKit `Tools::addAtomChargeFlags` with `tolerateChargeMismatch = true`
/// (the default used by `getAtomLabel`).  With tolerance on, the valence /
/// charge checks of the original only decide which (identical) suffix is
/// appended, so they are folded away here except where they change the
/// result.
fn add_atom_charge_flags(mol: &Molecule, i: usize, key: &mut String) {
    let atom = mol.atom(i);
    let total_valence = mol.total_valence(i);
    let anum = atom.atomic_num;
    match anum {
        // only +1 in default UFF params: Cu, Ag
        29 | 47 => key.push_str("+1"),
        // only +2: Be, Ca, Mn, Fe, Ni, Pd, Pt
        4 | 20 | 25 | 26 | 28 | 46 | 78 => key.push_str("+2"),
        // only +3: Sc, Cr, Co, Au, Ac, Cm..Lr
        21 | 24 | 27 | 79 | 89 | 96..=103 => key.push_str("+3"),
        // only +4: He, Ar, Ti, Kr, Xe, Th, Pa, U, Np, Pu, Am
        2 | 18 | 22 | 36 | 54 | 90..=95 => key.push_str("+4"),
        // only +5: V, Nb, Tc, Ta
        23 | 41 | 43 | 73 => key.push_str("+5"),
        // only +6: Mo
        42 => key.push_str("+6"),
        // Mg, Zn, Se, Cd, Te, Hg, Po: +2
        12 | 30 | 34 | 48 | 52 | 80 | 84 => key.push_str("+2"),
        // Al, Si: no suffix
        13 | 14 => {}
        // P
        15 => key.push_str(if total_valence == 3 { "+3" } else { "+5" }),
        // S
        16 => {
            if atom.hybridization != Hybridization::SP2 {
                key.push_str(match total_valence {
                    2 => "+2",
                    4 => "+4",
                    _ => "+6",
                });
            }
        }
        // Ga, As, In, Sb, Tl, Pb, Bi: +3
        31 | 33 | 49 | 51 | 81 | 82 | 83 => key.push_str("+3"),
        // Re
        75 => {
            if key == "Re6" {
                *key = "Re6+5".to_string();
            } else if key == "Re3" {
                *key = "Re3+7".to_string();
            }
        }
        _ => {}
    }
    // lanthanides
    if (57..=71).contains(&anum) {
        key.push_str("+3");
    }
}

fn atom_has_conjugated_bond(mol: &Molecule, i: usize) -> bool {
    mol.atom_bonds(i).iter().any(|&(_, b)| mol.bond(b).is_conjugated)
}

/// RDKit `UFF::Tools::getAtomLabel` for atom `i`.
pub fn uff_atom_label(mol: &Molecule, i: usize) -> String {
    let atom = mol.atom(i);
    let anum = atom.atomic_num;
    let mut key = atom.symbol().to_string();
    if key.len() == 1 {
        key.push('_');
    }
    if anum != 0 {
        let el = element(anum);
        // do not do hybridization on alkali metals or halogens:
        if el.default_valence() == -1 || (el.n_outer != 1 && el.n_outer != 7) {
            match anum {
                12 | 13 | 14 | 15 | 50 | 51 | 52 | 81 | 82 | 83 | 84 => key.push('3'),
                80 => key.push('1'),
                _ => match atom.hybridization {
                    Hybridization::S => {}
                    Hybridization::SP => key.push('1'),
                    Hybridization::SP2 => {
                        if (atom.is_aromatic || atom_has_conjugated_bond(mol, i))
                            && matches!(anum, 6 | 7 | 8 | 16)
                        {
                            key.push('R');
                        } else {
                            key.push('2');
                        }
                    }
                    Hybridization::SP3 => key.push('3'),
                    Hybridization::SP2D => key.push('4'),
                    Hybridization::SP3D => key.push('5'),
                    Hybridization::SP3D2 => key.push('6'),
                    // RDKit logs "Unrecognized hybridization" and leaves the key as is
                    Hybridization::Unspecified | Hybridization::Other => {}
                },
            }
        }
    }
    add_atom_charge_flags(mol, i, &mut key);
    key
}

/// RDKit `UFF::Tools::getAtomLabel` for every atom.
pub fn uff_atom_labels(mol: &Molecule) -> Vec<String> {
    (0..mol.num_atoms()).map(|i| uff_atom_label(mol, i)).collect()
}

/// RDKit `UFF::getAtomTypes`: per-atom parameters (`None` if the label has
/// no parameters) and whether all atoms were parametrized.
pub fn uff_atom_params(mol: &Molecule) -> (Vec<Option<UffAtomParams>>, bool) {
    let params: Vec<Option<UffAtomParams>> =
        uff_atom_labels(mol).iter().map(|l| uff_params_by_label(l)).collect();
    let found_all = params.iter().all(Option::is_some);
    (params, found_all)
}

/// RDKit `ForceFields::UFF::Utils::calcBondRestLength`.
///
/// `bond_order` must be positive (aromatic = 1.5).
pub fn calc_bond_rest_length(bond_order: f64, a: &UffAtomParams, b: &UffAtomParams) -> f64 {
    debug_assert!(bond_order > 0.0, "bad bond order");
    let ri = a.r1;
    let rj = b.r1;
    // Pauling correction:
    let r_bo = -LAMBDA * (ri + rj) * bond_order.ln();
    // O'Keefe and Breese electronegativity correction:
    let xi = a.gmp_xi;
    let xj = b.gmp_xi;
    let r_en = ri * rj * (xi.sqrt() - xj.sqrt()) * (xi.sqrt() - xj.sqrt()) / (xi * ri + xj * rj);
    ri + rj + r_bo - r_en
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_complete() {
        assert_eq!(param_table().len(), 127);
        let c = uff_params_by_label("C_R").unwrap();
        assert_eq!(c.r1, 0.729);
        assert!((c.theta0 - 120f64.to_radians()).abs() < 1e-12);
        assert_eq!(c.gmp_xi, 5.343);
        let lw = uff_params_by_label("Lw6+3").unwrap();
        assert_eq!(lw.gmp_radius, 1.9);
        assert!(uff_params_by_label("C_9").is_none());
    }

    #[test]
    fn rest_lengths_hand_checked() {
        // C_3 - C_3 single: no corrections at all
        let c3 = uff_params_by_label("C_3").unwrap();
        assert!((calc_bond_rest_length(1.0, &c3, &c3) - 1.514).abs() < 1e-12);
        // C_R - C_R aromatic: 1.458 - 0.1332 * 1.458 * ln(1.5)
        let cr = uff_params_by_label("C_R").unwrap();
        let expect = 1.458 - 0.1332 * 1.458 * 1.5f64.ln();
        assert!((calc_bond_rest_length(1.5, &cr, &cr) - expect).abs() < 1e-12);
        assert!((calc_bond_rest_length(1.5, &cr, &cr) - 1.379_256_405_400_789).abs() < 1e-12);
        // C_3 - H_: electronegativity correction only
        let h = uff_params_by_label("H_").unwrap();
        let (ri, rj, xi, xj) = (0.757f64, 0.354f64, 5.343f64, 4.528f64);
        let ren = ri * rj * (xi.sqrt() - xj.sqrt()).powi(2) / (xi * ri + xj * rj);
        let r = calc_bond_rest_length(1.0, &c3, &h);
        assert!((r - (1.111 - ren)).abs() < 1e-12);
        assert!((r - 1.1094).abs() < 1e-3);
        // symmetric
        assert_eq!(r, calc_bond_rest_length(1.0, &h, &c3));
    }
}
