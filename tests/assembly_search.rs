//! Assembly search (soft-core λ-continuation): known MMFF94 minima of the
//! water and acetic acid dimers, parity of the final energies with RDKit's
//! own MMFF minimization, λ-continuation vs direct minimization,
//! determinism, cancellation and snapshots.
//!
//! The RDKit reference (`tests/data/assembly/*_rdkit.ref`) is produced by
//! `tools/gen_assembly_reference.py reference` (RDKit 2026.09.1) from the
//! `*_found.sdf` files, which the ignored test `write_assembly_fixtures`
//! writes:
//!
//! ```text
//! cargo test --release --test assembly_search -- --ignored write_assembly_fixtures
//! py tools/gen_assembly_reference.py reference
//! ```

use molecule_optimizer_rs::Molecule;
use molecule_optimizer_rs::calc::assembly::{
    AssemblyInput, AssemblyMode, AssemblySearchOptions, AssemblySearchResult, MonomerConformers, TrialStage,
    assembly_search, combine_molecules,
};
use molecule_optimizer_rs::core::geometry::{V3, angle, dist};
use molecule_optimizer_rs::distgeom::EmbedParams;
use molecule_optimizer_rs::ff::mmff::MmffBuilder;
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::ff::{ForceField, ForceFieldBuilder};
use molecule_optimizer_rs::io;
use molecule_optimizer_rs::io::molfile::write_sdf_record;
use molecule_optimizer_rs::optim::{MinimizeOptions, NoObserver, minimize};
use molecule_optimizer_rs::tasks::{Engine, EngineConfig};
use std::path::PathBuf;
use std::time::Duration;

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/assembly").join(name)
}

fn monomer(name: &str) -> Molecule {
    io::read_file(data(&format!("{name}.sdf"))).unwrap().remove(0)
}

fn mmff_interfrag() -> MmffBuilder {
    MmffBuilder { ignore_interfrag_interactions: false, ..Default::default() }
}

fn water_dimer_search() -> AssemblySearchResult {
    let opts = AssemblySearchOptions { num_trials: 24, ..Default::default() };
    assembly_search(&AssemblyInput::Molecules(vec![(monomer("water"), 2)]), &MmffBuilder::default(), &opts).unwrap()
}

fn acetic_acid_dimer_search(mode: AssemblyMode, trials: usize) -> AssemblySearchResult {
    let opts = AssemblySearchOptions { num_trials: trials, mode, ..Default::default() };
    assembly_search(&AssemblyInput::Molecules(vec![(monomer("acetic_acid"), 2)]), &MmffBuilder::default(), &opts)
        .unwrap()
}

/// Water dimer atoms: O0 H1 H2 | O3 H4 H5. Returns (d(O-O), best O-H...O angle in degrees, d(H...O)).
fn water_dimer_geometry(c: &[V3]) -> (f64, f64, f64) {
    let doo = dist(c[0], c[3]);
    let mut best = (0.0, f64::INFINITY);
    for (o, hs, acc) in [(0, [1, 2], 3), (3, [4, 5], 0)] {
        for h in hs {
            let a = angle(c[o], c[h], c[acc]).to_degrees();
            let d = dist(c[h], c[acc]);
            if d < best.1 {
                best = (a, d);
            }
        }
    }
    (doo, best.0, best.1)
}

/// Acetic acid atoms (per molecule): C0 C1 O2(=O) O3(H) H4 H5 H6 H7(O).
/// Returns the two O-H...O=C distances of the cyclic dimer.
fn acid_hbonds(c: &[V3]) -> (f64, f64) {
    (dist(c[7], c[8 + 2]), dist(c[8 + 7], c[2]))
}

#[test]
fn water_dimer_is_the_linear_hydrogen_bonded_structure() {
    let r = water_dimer_search();
    let best = r.global_minimum().unwrap();
    let (doo, ang, dho) = water_dimer_geometry(&best.coords);
    println!(
        "water dimer: E = {:.6}, ΔE_bind = {:.4}, d(OO) = {doo:.4}, ∠OH…O = {ang:.1}, d(H…O) = {dho:.3}",
        best.energy, best.binding_energy
    );
    // MMFF94 gives d(O-O) = 2.750 Å (RDKit: same structure, see below)
    assert!((2.70..=3.0).contains(&doo), "O-O {doo}");
    assert!(ang > 160.0, "O-H...O angle {ang}");
    assert!(best.connected);
    // binding energy against separately minimized waters
    let w = monomer("water");
    let ff = MmffBuilder::default().build(&w, &[]).unwrap();
    let mut pos = w.flat_coords();
    let e_w =
        minimize(&ff, &mut pos, &MinimizeOptions { max_iters: 5000, ..Default::default() }, &mut NoObserver).energy;
    assert!((r.monomer_energies[0] - e_w).abs() < 1e-8);
    assert!((best.binding_energy - (best.energy - 2.0 * e_w)).abs() < 1e-8);
    assert!(best.binding_energy < -4.0 && best.binding_energy > -8.0, "{}", best.binding_energy);
    // the final energy is the plain MMFF energy with interfragment terms
    let mut m = r.complex.clone();
    m.coords = best.coords.clone();
    let e = mmff_interfrag().build(&m, &[]).unwrap().energy(&m.flat_coords());
    assert!((e - best.energy).abs() < 1e-9);
    assert_eq!(r.monomers.len(), 1);
    assert_eq!(r.copies.len(), 2);
}

#[test]
fn acetic_acid_dimer_is_cyclic() {
    let r = acetic_acid_dimer_search(AssemblyMode::LambdaContinuation, 32);
    let best = r.global_minimum().unwrap();
    let (h1, h2) = acid_hbonds(&best.coords);
    println!(
        "acetic acid dimer: E = {:.6}, ΔE_bind = {:.4}, H…O = {h1:.3} / {h2:.3}",
        best.energy, best.binding_energy
    );
    assert!(h1 < 2.0 && h2 < 2.0, "not the cyclic dimer: {h1} {h2}");
    assert!(best.binding_energy < -10.0);
}

#[test]
#[ignore = "writes the inputs of tools/gen_assembly_reference.py"]
fn write_assembly_fixtures() {
    for (name, r) in [
        ("water_dimer", water_dimer_search()),
        ("acetic_acid_dimer", acetic_acid_dimer_search(AssemblyMode::LambdaContinuation, 32)),
    ] {
        let best = r.global_minimum().unwrap();
        let mut m = r.complex.clone();
        m.name = name.into();
        let rec = write_sdf_record(&m, Some(&best.coords), &[("energy", format!("{:?}", best.energy))]);
        std::fs::write(data(&format!("{name}_found.sdf")), rec).unwrap();
    }
}

struct RdkitRef {
    e0: f64,
    e1: f64,
    x: Vec<V3>,
}

fn read_ref(name: &str) -> RdkitRef {
    let text = std::fs::read_to_string(data(&format!("{name}_rdkit.ref"))).unwrap();
    let (mut e0, mut e1, mut x) = (f64::NAN, f64::NAN, vec![]);
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first() {
            Some(&"E0") => e0 = f[1].parse().unwrap(),
            Some(&"E1") => {
                e1 = f[1].parse().unwrap();
                assert_eq!(f[2], "0", "RDKit minimization did not converge");
            }
            Some(&"X") => {
                let v: Vec<f64> = f[1..].iter().map(|s| s.parse().unwrap()).collect();
                x = v.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
            }
            _ => {}
        }
    }
    RdkitRef { e0, e1, x }
}

/// Our lowest structures are minima of RDKit's MMFF94 too: RDKit's energy
/// at our coordinates equals ours, and re-minimizing them with RDKit
/// changes neither the energy nor the structure.
#[test]
fn lowest_structures_agree_with_rdkit_reminimization() {
    for (name, r) in [
        ("water_dimer", water_dimer_search()),
        ("acetic_acid_dimer", acetic_acid_dimer_search(AssemblyMode::LambdaContinuation, 32)),
    ] {
        let rf = read_ref(name);
        let found = io::read_file(data(&format!("{name}_found.sdf"))).unwrap().remove(0);
        // same energy function as RDKit at the stored (4-decimal) coordinates
        let e_ours = mmff_interfrag().build(&found, &[]).unwrap().energy(&found.flat_coords());
        assert!((e_ours - rf.e0).abs() <= 1e-10 * rf.e0.abs().max(1.0), "{name}: {e_ours} vs RDKit {}", rf.e0);
        // the search (re-run now) reaches RDKit's re-minimized energy and structure
        let best = r.global_minimum().unwrap();
        println!(
            "{name}: search E = {:.8}, RDKit E(at our coords) = {:.8}, RDKit re-minimized E = {:.8}",
            best.energy, rf.e0, rf.e1
        );
        assert!((best.energy - rf.e1).abs() < 1e-4, "{name}: {} vs RDKit {}", best.energy, rf.e1);
        assert!((rf.e0 - rf.e1).abs() < 1e-3, "{name}: RDKit moved the structure: {} -> {}", rf.e0, rf.e1);
        let rmsd = r.rmsd(&best.coords, &rf.x, true);
        assert!(rmsd < 0.01, "{name}: RMSD to RDKit-reminimized structure {rmsd}");
    }
}

fn hits(r: &AssemblySearchResult, g: &[V3], e: f64) -> usize {
    r.all.iter().filter(|s| (s.energy - e).abs() < 0.01 && r.rmsd(&s.coords, g, true) < 0.25).count()
}

/// The user's hypothesis: λ-continuation from weak, soft intermolecular
/// interactions finds the global minimum more often than direct
/// full-strength minimization from the same random placements. The numbers
/// are printed (run with `--nocapture`); only weak properties are asserted.
#[test]
fn lambda_continuation_vs_direct_minimization() {
    let systems: Vec<(&str, AssemblyInput, usize)> = vec![
        ("(H2O)2", AssemblyInput::Molecules(vec![(monomer("water"), 2)]), 24),
        ("(AcOH)2", AssemblyInput::Molecules(vec![(monomer("acetic_acid"), 2)]), 64),
        ("(C6H6)2", AssemblyInput::Molecules(vec![(monomer("benzene"), 2)]), 32),
        ("(H2O)6", AssemblyInput::Molecules(vec![(monomer("water"), 6)]), 48),
    ];
    println!("system   trials  E_min(MMFF94)  λ hits  direct hits  λ unique  direct unique  λ iters  direct iters");
    for (name, input, n) in systems {
        let opts = AssemblySearchOptions { num_trials: n, ..Default::default() };
        let lam = assembly_search(&input, &MmffBuilder::default(), &opts).unwrap();
        let dir = assembly_search(
            &input,
            &MmffBuilder::default(),
            &AssemblySearchOptions { mode: AssemblyMode::Direct, ..opts },
        )
        .unwrap();
        let (a, b) = (lam.global_minimum().unwrap(), dir.global_minimum().unwrap());
        let g = if a.energy <= b.energy { a } else { b };
        let (hl, hd) = (hits(&lam, &g.coords, g.energy), hits(&dir, &g.coords, g.energy));
        let it = |r: &AssemblySearchResult| r.all.iter().map(|s| s.iterations).sum::<usize>();
        println!(
            "{name:<8} {n:<7} {:<14.4} {hl:<7} {hd:<12} {:<9} {:<14} {:<8} {}",
            g.energy,
            lam.structures.len(),
            dir.structures.len(),
            it(&lam),
            it(&dir)
        );
        assert_eq!(lam.all.len(), n);
        assert_eq!(dir.all.len(), n);
        assert!(hl + hd >= 1);
        for s in lam.all.iter().chain(&dir.all) {
            assert!(s.energy.is_finite());
        }
        if name == "(AcOH)2" {
            let (h1, h2) = acid_hbonds(&g.coords);
            assert!(h1 < 2.0 && h2 < 2.0);
        }
    }
}

#[test]
fn deterministic_independent_of_thread_count() {
    let input = AssemblyInput::Molecules(vec![(monomer("water"), 3)]);
    let opts = AssemblySearchOptions { num_trials: 12, ..Default::default() };
    let run = |threads: usize| {
        let eng = Engine::new(EngineConfig { num_threads: Some(threads), ..Default::default() }).unwrap();
        eng.install(|| assembly_search(&input, &MmffBuilder::default(), &opts).unwrap())
    };
    let (a, b) = (run(1), run(4));
    assert_eq!(a.all.len(), b.all.len());
    for (x, y) in a.all.iter().zip(&b.all) {
        assert_eq!(x.trial, y.trial);
        assert_eq!(x.energy.to_bits(), y.energy.to_bits());
        assert_eq!(x.coords, y.coords);
        assert_eq!(x.lambda_energies, y.lambda_energies);
    }
    assert_eq!(a.structures, b.structures);
    // the asynchronous job gives the same result
    let eng = Engine::new(EngineConfig { num_threads: Some(3), ..Default::default() }).unwrap();
    let c = eng.assembly_search(input.clone(), MmffBuilder::default(), opts.clone()).wait_blocking().unwrap();
    assert_eq!(a.structures, c.structures);
    assert!(!c.cancelled);
}

#[test]
fn cancellation_returns_completed_trials() {
    let eng = Engine::new(EngineConfig { num_threads: Some(2), ..Default::default() }).unwrap();
    let n = 400;
    let opts = AssemblySearchOptions { num_trials: n, ..Default::default() };
    let job = eng.assembly_search(AssemblyInput::Molecules(vec![(monomer("water"), 6)]), MmffBuilder::default(), opts);
    let t0 = std::time::Instant::now();
    while job.snapshot().data.completed < 3 && t0.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(5));
    }
    job.cancel();
    let r = job.wait_blocking().unwrap();
    assert!(r.cancelled);
    assert!(r.all.len() >= 3 && r.all.len() < n, "{}", r.all.len());
    assert!(!r.structures.is_empty());
    assert!(r.all.windows(2).all(|w| w[0].trial < w[1].trial));
}

#[test]
fn snapshot_shows_trials_lambda_and_found_structures() {
    let eng = Engine::new(EngineConfig { num_threads: Some(2), publish_interval: Duration::ZERO }).unwrap();
    let opts = AssemblySearchOptions { num_trials: 16, ..Default::default() };
    let schedule = opts.lambda_schedule.clone();
    let job = eng.assembly_search(AssemblyInput::Molecules(vec![(monomer("water"), 6)]), MmffBuilder::default(), opts);
    let (mut saw_lambda, mut saw_final, mut saw_found) = (false, false, false);
    let t0 = std::time::Instant::now();
    while !job.is_finished() && t0.elapsed() < Duration::from_secs(60) {
        let s = job.request_snapshot_blocking(Duration::from_millis(200));
        let d = &s.data;
        assert_eq!(d.total, 16);
        for (t, p) in &d.in_flight {
            assert!(*t < 16);
            match p.stage {
                TrialStage::Lambda { step } => {
                    assert_eq!((p.lambda_vdw, p.lambda_ele), schedule[step]);
                    assert_eq!(p.frame.coords.len(), 18);
                    saw_lambda = true;
                }
                TrialStage::Final => {
                    assert_eq!((p.lambda_vdw, p.lambda_ele), (1.0, 1.0));
                    saw_final = true;
                }
                TrialStage::Placing => {}
            }
        }
        if !d.found.is_empty() {
            saw_found = true;
            assert!(d.found.windows(2).all(|w| w[0].energy <= w[1].energy));
        }
    }
    let last = job.snapshot();
    let r = job.wait_blocking().unwrap();
    assert!(saw_lambda && saw_final && saw_found, "{saw_lambda} {saw_final} {saw_found}");
    assert_eq!(last.data.completed, 16);
    assert!(last.data.in_flight.is_empty());
    assert_eq!(last.data.best().unwrap().energy, r.global_minimum().unwrap().energy);
}

#[test]
fn complex_input_and_identical_fragments() {
    // a single two-fragment molecule (the stored water dimer)
    let mol = io::read_file(data("water_dimer_found.sdf")).unwrap().remove(0);
    let opts = AssemblySearchOptions { num_trials: 8, ..Default::default() };
    let r = assembly_search(&AssemblyInput::Complex(mol.clone()), &MmffBuilder::default(), &opts).unwrap();
    assert_eq!(r.monomers.len(), 1);
    assert_eq!(r.copies.iter().map(|c| c.atoms.clone()).collect::<Vec<_>>(), vec![vec![0, 1, 2], vec![3, 4, 5]]);
    let best = r.global_minimum().unwrap();
    let rf = read_ref("water_dimer");
    assert!((best.energy - rf.e1).abs() < 1e-4, "{} vs {}", best.energy, rf.e1);
    // swapping the two molecules gives RMSD 0 with permutations
    let swapped: Vec<V3> = [3, 4, 5, 0, 1, 2].iter().map(|&i| best.coords[i]).collect();
    assert!(r.rmsd(&best.coords, &swapped, true) < 1e-6);
    // and so does swapping the hydrogens of one water
    let hswap: Vec<V3> = [0, 2, 1, 3, 4, 5].iter().map(|&i| best.coords[i]).collect();
    assert!(r.rmsd(&best.coords, &hswap, true) < 1e-6);
    assert!(r.rmsd(&best.coords, &mol.coords, true) < 0.05);
}

#[test]
fn hetero_assembly_and_etkdg_monomers() {
    let mut params = EmbedParams::etkdg_v3();
    params.random_seed = 42;
    let opts = AssemblySearchOptions {
        num_trials: 12,
        monomer_conformers: MonomerConformers::Etkdg { params, num_conformers: 4 },
        ..Default::default()
    };
    let input = AssemblyInput::Molecules(vec![(monomer("acetic_acid"), 1), (monomer("water"), 1)]);
    let r = assembly_search(&input, &MmffBuilder::default(), &opts).unwrap();
    assert_eq!(r.monomers.len(), 2);
    assert_eq!(r.complex.num_atoms(), 11);
    let best = r.global_minimum().unwrap();
    assert!(best.binding_energy < -5.0, "{}", best.binding_energy);
    assert!(best.connected);
    // monomer reference energies are minimized ETKDG conformers
    assert!(r.monomer_energies.iter().all(|e| e.is_finite()));
    assert!((r.reference_energy - r.monomer_energies.iter().sum::<f64>()).abs() < 1e-12);
}

#[test]
fn uff_assembly_search_runs() {
    let input = AssemblyInput::Molecules(vec![(monomer("benzene"), 2)]);
    let opts = AssemblySearchOptions { num_trials: 6, ..Default::default() };
    let r = assembly_search(&input, &UffBuilder::default(), &opts).unwrap();
    let best = r.global_minimum().unwrap();
    assert!(best.binding_energy < -1.0, "{}", best.binding_energy);
    let mut m = combine_molecules(&[(monomer("benzene"), 2)]);
    m.coords = best.coords.clone();
    let e = UffBuilder { ignore_interfrag_interactions: false, ..Default::default() }
        .build(&m, &[])
        .unwrap()
        .energy(&m.flat_coords());
    assert!((e - best.energy).abs() < 1e-9);
}
