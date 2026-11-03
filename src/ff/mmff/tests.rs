//! Unit tests for the MMFF port (analytic vs. numeric gradients).

use super::*;
use crate::ff::{Contrib, ForceField};

const SDF: &str = include_str!("../../../tests/data/mmff/MMFF94_dative.sdf");

fn molecules(n: usize) -> Vec<Molecule> {
    crate::io::molfile::read_sdf_lenient(SDF).into_iter().filter_map(|r| r.ok()).take(n).collect()
}

/// Central-difference check of one contrib at slightly perturbed
/// coordinates (so that no term sits exactly at its minimum).
fn check_contrib(c: &dyn Contrib, pos: &[f64]) -> f64 {
    let mut ga = vec![0.0; pos.len()];
    c.grad(pos, &mut ga);
    let mut p = pos.to_vec();
    let h = 1e-6;
    let mut max_err: f64 = 0.0;
    for i in 0..pos.len() {
        let x = p[i];
        p[i] = x + h;
        let ep = c.energy(&p);
        p[i] = x - h;
        let em = c.energy(&p);
        p[i] = x;
        let gn = (ep - em) / (2.0 * h);
        max_err = max_err.max((gn - ga[i]).abs() / ga[i].abs().max(1.0));
    }
    max_err
}

fn perturbed(mol: &Molecule) -> Vec<f64> {
    let mut pos = mol.flat_coords();
    for (k, x) in pos.iter_mut().enumerate() {
        // deterministic pseudo-random displacement
        *x += 0.05 * (((k * 7919) % 101) as f64 / 50.0 - 1.0);
    }
    pos
}

#[test]
fn analytic_gradients_match_numeric_for_every_term() {
    let mut seen = std::collections::BTreeMap::<&'static str, usize>::new();
    for variant in [MmffVariant::Mmff94, MmffVariant::Mmff94s] {
        for diel in [DielectricModel::Constant, DielectricModel::Distance] {
            let b = MmffBuilder { variant, dielectric_model: diel, dielectric_constant: 2.0, ..Default::default() };
            for mol in molecules(40) {
                let ff = b.build(&mol, &[]).unwrap();
                let pos = perturbed(&mol);
                for c in ff.contribs() {
                    let err = check_contrib(c.as_ref(), &pos);
                    assert!(err < 1e-5, "{} {}: gradient error {err}", mol.name, c.label());
                    *seen.entry(c.label()).or_default() += 1;
                }
                // whole force field (including fixed-point handling)
                assert!(crate::ff::gradient_error(&ff, &pos) < 1e-4);
            }
        }
    }
    for l in ["BondStretch", "AngleBend", "StretchBend", "OopBend", "Torsion", "Nonbonded"] {
        assert!(seen.get(l).copied().unwrap_or(0) > 0, "no {l} terms tested");
    }
}

#[test]
fn breakdown_reports_all_labels_and_sums_to_total() {
    let mols = molecules(100);
    let (mol, ff) = mols
        .iter()
        .map(|m| (m, MmffBuilder::default().build(m, &[]).unwrap()))
        .find(|(_, ff)| ff.num_contribs() == 6)
        .expect("a molecule with all term types");
    let pos = mol.flat_coords();
    let bd = ff.energy_breakdown(&pos);
    let labels: Vec<&str> = bd.iter().map(|(k, _)| k.as_str()).collect();
    for l in ["BondStretch", "AngleBend", "StretchBend", "OopBend", "Torsion", "VdW", "Electrostatic"] {
        assert!(labels.contains(&l), "{l} missing in {labels:?}");
    }
    let sum: f64 = bd.iter().map(|(_, e)| e).sum();
    let e = ff.energy(&pos);
    assert!((sum - e).abs() < 1e-9 * e.abs().max(1.0));
}

#[test]
fn builder_names_and_validity() {
    assert_eq!(MmffBuilder::default().name(), "MMFF94");
    assert_eq!(MmffBuilder::new(MmffVariant::Mmff94s).name(), "MMFF94s");
    let mol = &molecules(1)[0];
    assert!(mmff_has_all_params(mol));
    let p = MmffMolProperties::new(mol, MmffVariant::Mmff94).unwrap();
    assert!(p.is_valid());
    assert_eq!(p.atom_types().len(), mol.num_atoms());
    let q: f64 = p.partial_charges().iter().sum();
    assert!(q.abs() < 1e-9);
}

#[test]
fn constraints_are_added() {
    let mol = &molecules(1)[0];
    let plain = MmffBuilder::default().build(mol, &[]).unwrap();
    let c = MmffBuilder::default().build(mol, &[Constraint::distance(0, 1, 2.0, 100.0)]).unwrap();
    assert!(c.num_contribs() > plain.num_contribs());
}
