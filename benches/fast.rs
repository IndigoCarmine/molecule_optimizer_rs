//! Fast path vs `ContribForceField`.
//!
//! `cargo bench --bench fast` (molecules: `examples/data/paracetamol.mol`,
//! `benches/data/peptide_{200,2000}.mol`, see `benches/data/make_bench_mols.py`).
//!
//! * `eg/<mol>/<ff>/<impl>`: one energy + gradient evaluation.
//!   `contrib` = RDKit-compatible `ContribForceField`, `fast` = SIMD fast
//!   path, `fast-scalar` = fast path with scalar kernels; `-1t` variants run
//!   in a one-thread pool.
//! * `batch/paracetamol-1000/<ff>/<impl>`: `Engine::optimize_batch` of 1000
//!   perturbed copies of paracetamol.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use molecule_optimizer_rs::Molecule;
use molecule_optimizer_rs::calc::OptimizeOptions;
use molecule_optimizer_rs::fast::{FastBuilder, FastOptions};
use molecule_optimizer_rs::ff::mmff::MmffBuilder;
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::ff::{ForceField, ForceFieldBuilder};
use molecule_optimizer_rs::tasks::{Engine, EngineConfig};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

fn load(path: &str) -> Option<Molecule> {
    let p = format!("{}/{path}", env!("CARGO_MANIFEST_DIR"));
    molecule_optimizer_rs::io::read_file(&p).ok().map(|mut v| v.remove(0))
}

fn molecules() -> Vec<(&'static str, Molecule)> {
    let mut out = Vec::new();
    for (name, path) in [
        ("paracetamol", "examples/data/paracetamol.mol"),
        ("peptide200", "benches/data/peptide_200.mol"),
        ("peptide2000", "benches/data/peptide_2000.mol"),
    ] {
        match load(path) {
            Some(m) => out.push((name, m)),
            None => eprintln!("skipping {path} (not found)"),
        }
    }
    out
}

fn bench_eg(c: &mut Criterion) {
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let mut g = c.benchmark_group("eg");
    g.warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(3));
    for (name, mol) in molecules() {
        if mol.num_atoms() > 1000 {
            g.sample_size(10);
        } else {
            g.sample_size(50);
        }
        let pos = mol.flat_coords();
        let builders: [(&str, Box<dyn ForceFieldBuilder>, Box<dyn ForceFieldBuilder>, Box<dyn ForceFieldBuilder>); 2] = [
            (
                "MMFF",
                Box::new(MmffBuilder::default()),
                Box::new(FastBuilder::new(MmffBuilder::default(), FastOptions::exact())),
                Box::new(FastBuilder::new(MmffBuilder::default(), FastOptions::exact().simd(false))),
            ),
            (
                "UFF",
                Box::new(UffBuilder::default()),
                Box::new(FastBuilder::new(UffBuilder::default(), FastOptions::exact())),
                Box::new(FastBuilder::new(UffBuilder::default(), FastOptions::exact().simd(false))),
            ),
        ];
        for (ffname, contrib, fast, scalar) in &builders {
            let ffs: Vec<(&str, Box<dyn ForceField>)> = vec![
                ("contrib", contrib.build_dyn(&mol, &[]).unwrap()),
                ("fast", fast.build_dyn(&mol, &[]).unwrap()),
                ("fast-scalar", scalar.build_dyn(&mol, &[]).unwrap()),
            ];
            let mut grad = vec![0.0; pos.len()];
            for (imp, ff) in &ffs {
                let id = format!("{name}/{ffname}");
                g.bench_with_input(BenchmarkId::new(id.clone(), imp), &(), |b, _| {
                    b.iter(|| black_box(ff.energy_grad(black_box(&pos), &mut grad)))
                });
                if *imp != "fast-scalar" {
                    g.bench_with_input(BenchmarkId::new(id, format!("{imp}-1t")), &(), |b, _| {
                        one.install(|| b.iter(|| black_box(ff.energy_grad(black_box(&pos), &mut grad))))
                    });
                }
            }
        }
    }
    g.finish();
}

fn perturbed(mol: &Molecule, k: usize) -> Vec<[f64; 3]> {
    mol.coords
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut c = *c;
            for (d, x) in c.iter_mut().enumerate() {
                let h = (i * 2654435761 + d * 40503 + k * 7919) % 1000;
                *x += 0.3 * (h as f64 / 1000.0 - 0.5);
            }
            c
        })
        .collect()
}

fn bench_batch(c: &mut Criterion) {
    let mol = Arc::new(load("examples/data/paracetamol.mol").unwrap());
    let items: Vec<_> = (0..1000).map(|k| (mol.clone(), Some(perturbed(&mol, k)))).collect();
    let engine = Engine::new(EngineConfig { num_threads: None, publish_interval: Duration::ZERO }).unwrap();
    let mut g = c.benchmark_group("batch");
    g.sample_size(10).warm_up_time(Duration::from_secs(1)).measurement_time(Duration::from_secs(8));
    let cases: Vec<(&str, &str, Arc<dyn ForceFieldBuilder>)> = vec![
        ("MMFF", "contrib", Arc::new(MmffBuilder::default())),
        ("MMFF", "fast", Arc::new(FastBuilder::new(MmffBuilder::default(), FastOptions::exact()))),
        ("UFF", "contrib", Arc::new(UffBuilder::default())),
        ("UFF", "fast", Arc::new(FastBuilder::new(UffBuilder::default(), FastOptions::exact()))),
    ];
    for (ff, imp, b) in cases {
        g.bench_function(BenchmarkId::new(format!("paracetamol-1000/{ff}"), imp), |bch| {
            bch.iter(|| {
                let r = engine.optimize_batch(items.clone(), b.clone(), OptimizeOptions::default()).wait_blocking().unwrap();
                black_box(r.len())
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench_eg, bench_batch);
criterion_main!(benches);
