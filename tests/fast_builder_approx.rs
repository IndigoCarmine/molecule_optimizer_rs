//! `FastBuilder` in approximate mode, end to end: the bonded part comes
//! from the fast tables, the nonbonded part from the approximate backend.

#[allow(dead_code)]
mod common;

use common::water::water_box;
use molecule_optimizer_rs::calc::{self, OptimizeOptions};
use molecule_optimizer_rs::fast::{Electrostatics, FastBuilder, FastOptions, VdwModifier};
use molecule_optimizer_rs::ff::mmff::{MmffBuilder, MmffVariant};
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::ff::{ForceField, ForceFieldBuilder};
use molecule_optimizer_rs::io::molfile::read_sdf_lenient;
use molecule_optimizer_rs::optim::MinimizeOptions;
use molecule_optimizer_rs::Molecule;
use std::path::PathBuf;

fn paracetamol() -> Molecule {
    molecule_optimizer_rs::io::read_file(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data/paracetamol.mol")).unwrap().remove(0)
}

/// Large cutoff, no electrostatic cutoff, plain van der Waals truncation:
/// identical interactions to the exact path for an isolated molecule.
fn no_cutoff() -> FastOptions {
    FastOptions::approximate()
        .cutoff(1000.0)
        .buffer(0.0)
        .electrostatics(Electrostatics::Full)
        .vdw_modifier(VdwModifier::Switch { r_switch: 1000.0 })
}

fn compare(name: &str, exact: &dyn ForceField, approx: &dyn ForceField, pos: &[f64]) -> (f64, f64) {
    let mut ge = vec![0.0; pos.len()];
    let mut ga = vec![0.0; pos.len()];
    let ee = exact.energy_grad(pos, &mut ge);
    let ea = approx.energy_grad(pos, &mut ga);
    let er = (ee - ea).abs() / ee.abs().max(1.0);
    assert!((exact.energy(pos) - approx.energy(pos)).abs() / ee.abs().max(1.0) <= 1e-10, "{name}: energy()");
    let gmax = ge.iter().fold(1.0f64, |a, b| a.max(b.abs()));
    let gr = ge.iter().zip(&ga).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / gmax;
    assert!(er <= 1e-10, "{name}: energy {ea} vs exact {ee} (rel {er:.2e})");
    assert!(gr <= 1e-9, "{name}: gradient rel {gr:.2e}");
    // breakdown: same labels, same values (the exact path omits a nonbonded
    // label when no pair has that part)
    let be = exact.energy_breakdown(pos);
    for (k, v) in approx.energy_breakdown(pos) {
        let r = be.iter().find(|(l, _)| *l == k).map(|(_, x)| *x).unwrap_or(0.0);
        assert!((r - v).abs() <= 1e-10 * ee.abs().max(1.0), "{name}: breakdown {k}: {v} vs {r}");
    }
    (er, gr)
}

#[test]
fn approximate_without_cutoff_equals_exact() {
    let mol = paracetamol();
    let pos = mol.flat_coords();
    let mut worst: (f64, f64) = (0.0, 0.0);
    let mut acc = |r: (f64, f64)| worst = (worst.0.max(r.0), worst.1.max(r.1));
    // MMFF (both variants) and UFF on paracetamol
    for b in [MmffBuilder::default(), MmffBuilder::new(MmffVariant::Mmff94s)] {
        let exact = b.build(&mol, &[]).unwrap();
        let approx = FastBuilder::new(b.clone(), no_cutoff()).build_dyn(&mol, &[]).unwrap();
        acc(compare("paracetamol MMFF", &exact, approx.as_ref(), &pos));
    }
    let exact = UffBuilder::default().build(&mol, &[]).unwrap();
    let approx = FastBuilder::new(UffBuilder::default(), no_cutoff()).build_dyn(&mol, &[]).unwrap();
    acc(compare("paracetamol UFF", &exact, approx.as_ref(), &pos));
    // MMFF fixtures (isolated molecules, all pairs within the exact path's 100 Å threshold)
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/mmff/MMFF94_dative.sdf");
    let b = MmffBuilder::default();
    let fast = FastBuilder::new(b.clone(), no_cutoff());
    let mut n = 0;
    for m in read_sdf_lenient(&std::fs::read_to_string(p).unwrap()).into_iter().step_by(7).flatten() {
        let Ok(exact) = b.build(&m, &[]) else { continue };
        let approx = fast.build_dyn(&m, &[]).unwrap();
        acc(compare(&m.name, &exact, approx.as_ref(), &m.flat_coords()));
        n += 1;
    }
    println!("approximate (no cutoff) vs exact: {n} MMFF fixtures + paracetamol, max rel dE {:.2e}, max rel dgrad {:.2e}", worst.0, worst.1);
}

#[test]
fn optimize_periodic_water_box_with_pme() {
    let mol = water_box([3, 3, 3], 3.1, 11);
    assert!(mol.cell.is_some());
    let b = MmffBuilder { ignore_interfrag_interactions: false, ..Default::default() };
    let opts = FastOptions::approximate()
        .cutoff(4.0)
        .buffer(0.5)
        .electrostatics(Electrostatics::Pme { ewald_rtol: 1e-5, grid_spacing: 0.8, order: 4 });
    let fast = FastBuilder::new(b, opts);
    let ff = fast.build_dyn(&mol, &[]).unwrap();
    let pos = mol.flat_coords();
    let e0 = ff.energy(&pos);
    let labels: Vec<String> = ff.energy_breakdown(&pos).into_iter().map(|(k, _)| k).collect();
    assert!(labels.iter().any(|k| k == "VdW") && labels.iter().any(|k| k == "Electrostatic"), "{labels:?}");
    let oo = OptimizeOptions { minimize: MinimizeOptions { max_iters: 300, ..Default::default() }, ..Default::default() };
    let r = calc::optimize::optimize(&mol, &fast, &oo).unwrap();
    println!("water box (27 H2O, PME): E {e0:.4} -> {:.4} ({:?}, {} iterations, grad rms {:.2e})", r.energy, r.status, r.iterations, r.grad_rms);
    assert!(r.energy.is_finite() && r.energy < e0 - 1.0, "{e0} -> {}", r.energy);
    // the reported energy is the force field's energy at the returned coordinates
    let flat: Vec<f64> = r.coords.iter().flatten().copied().collect();
    assert!((ff.energy(&flat) - r.energy).abs() < 1e-8 * r.energy.abs().max(1.0));
}
