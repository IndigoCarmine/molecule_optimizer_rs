//! Approximate nonbonded mode on MMFF water boxes (~5k and ~50k atoms):
//! energy + gradient with reaction field and PME, pair-list build time and
//! thread scaling.
//!
//! `cargo bench --bench approx` (set `APPROX_BENCH_QUICK=1` to skip the
//! 50k-atom system).

#[path = "../tests/common/water.rs"]
mod water;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use molecule_optimizer_rs::Molecule;
use molecule_optimizer_rs::fast::approx::{ApproxNonbonded, build_approx_nonbonded};
use molecule_optimizer_rs::fast::backend::NonbondedBackend;
use molecule_optimizer_rs::fast::{ApproxOptions, Electrostatics, VdwModifier};
use molecule_optimizer_rs::ff::mmff::MmffBuilder;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn systems() -> Vec<(String, Molecule)> {
    let mut sides = vec![12usize]; // 1728 waters, 5184 atoms
    if std::env::var_os("APPROX_BENCH_QUICK").is_none() {
        sides.push(26); // 17576 waters, 52728 atoms
    }
    sides
        .into_iter()
        .map(|k| {
            let t = Instant::now();
            let m = water::water_box([k, k, k], 3.1, 1);
            eprintln!("built water box {k}^3 ({} atoms) in {:.2?}", m.num_atoms(), t.elapsed());
            (format!("{}atoms", m.num_atoms()), m)
        })
        .collect()
}

fn electrostatics() -> Vec<(&'static str, Electrostatics)> {
    vec![
        ("rf", Electrostatics::ReactionField { eps_rf: f64::INFINITY }),
        ("pme", Electrostatics::Pme { ewald_rtol: 1e-5, grid_spacing: 1.0, order: 4 }),
    ]
}

fn backend(m: &Molecule, el: Electrostatics, simd: bool) -> ApproxNonbonded {
    let inner = MmffBuilder { ignore_interfrag_interactions: false, ..MmffBuilder::default() };
    let o = ApproxOptions { cutoff: 10.0, buffer: 1.0, vdw_modifier: VdwModifier::PotentialShift, electrostatics: el };
    let t = Instant::now();
    let b = build_approx_nonbonded(m, &inner, &o, simd).unwrap();
    let s = b.pair_list_stats();
    eprintln!(
        "setup {} atoms {:?}: {:.2?}; {} clusters, {} cluster pairs, {} atom pairs in list, PME grid {:?}",
        m.num_atoms(),
        el,
        t.elapsed(),
        s.clusters,
        s.cluster_pairs,
        s.atom_pairs,
        b.pme_grid()
    );
    b
}

fn bench(c: &mut Criterion) {
    let systems = systems();
    let mut g = c.benchmark_group("energy_grad");
    g.sample_size(20).measurement_time(Duration::from_secs(4)).warm_up_time(Duration::from_secs(1));
    for (name, m) in &systems {
        let pos = m.flat_coords();
        let mut grad = vec![0.0; pos.len()];
        for (en, el) in electrostatics() {
            for simd in [true, false] {
                let b = backend(m, el, simd);
                let id = format!("{name}/{en}/{}", if simd { "simd" } else { "scalar" });
                g.bench_function(BenchmarkId::from_parameter(id), |bb| {
                    bb.iter(|| black_box(b.energy_grad(black_box(&pos), &mut grad)))
                });
            }
        }
    }
    g.finish();

    let mut g = c.benchmark_group("pair_list_build");
    g.sample_size(10).measurement_time(Duration::from_secs(3)).warm_up_time(Duration::from_secs(1));
    for (name, m) in &systems {
        let pos = m.flat_coords();
        let b = backend(m, Electrostatics::ReactionField { eps_rf: f64::INFINITY }, true);
        g.bench_function(BenchmarkId::from_parameter(name), |bb| bb.iter(|| b.rebuild(black_box(&pos))));
    }
    g.finish();

    let mut g = c.benchmark_group("thread_scaling");
    g.sample_size(10).measurement_time(Duration::from_secs(3)).warm_up_time(Duration::from_secs(1));
    let max_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut counts = vec![1usize, 2, 4, 8, 16];
    counts.retain(|&t| t < max_threads);
    counts.push(max_threads);
    let (name, m) = systems.last().unwrap();
    let pos = m.flat_coords();
    for (en, el) in electrostatics() {
        let b = backend(m, el, true);
        for &t in &counts {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
            let mut grad = vec![0.0; pos.len()];
            g.bench_function(BenchmarkId::new(format!("{name}/{en}"), t), |bb| {
                bb.iter(|| pool.install(|| black_box(b.energy_grad(black_box(&pos), &mut grad))))
            });
        }
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
