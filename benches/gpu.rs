//! GPU (wgpu) vs CPU SIMD approximate nonbonded backend.
//!
//! `cargo bench --features gpu --bench gpu` (set `GPU_BENCH_QUICK=1` to skip
//! the ~50k-atom system). Prints a markdown table:
//!
//! * E + gradient of MMFF water boxes (~5k and ~50k atoms), reaction field
//!   and PME, single system;
//! * batch throughput: 1000 paracetamol conformers (MMFF approximate, `Full`
//!   and reaction field) per call, GPU batch vs CPU (rayon over conformers).

#[path = "../tests/common/water.rs"]
mod water;

use molecule_optimizer_rs::Molecule;
use molecule_optimizer_rs::fast::approx::build_approx_nonbonded;
use molecule_optimizer_rs::fast::backend::{NonbondedBackend, NonbondedEnergy};
use molecule_optimizer_rs::fast::gpu::{GpuBatch, GpuContext, build_gpu_nonbonded};
use molecule_optimizer_rs::fast::{ApproxOptions, Electrostatics, VdwModifier};
use molecule_optimizer_rs::ff::mmff::MmffBuilder;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Median time per call of `f` (after warm-up), at least `min_iters` calls
/// and about `budget` of total time.
fn time<F: FnMut()>(mut f: F, budget: Duration, min_iters: usize) -> Duration {
    for _ in 0..2 {
        f();
    }
    let mut samples = Vec::new();
    let start = Instant::now();
    while samples.len() < min_iters || (start.elapsed() < budget && samples.len() < 200) {
        let t = Instant::now();
        f();
        samples.push(t.elapsed());
    }
    samples.sort();
    samples[samples.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let ctx = match GpuContext::new_blocking() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return;
        }
    };
    let threads = rayon::current_num_threads();
    println!("adapter: {}", ctx.description());
    println!("CPU: {threads} rayon threads\n");
    let budget = Duration::from_secs(3);

    // ------------------------------------------------------------------ single system
    let mut sides = vec![12usize];
    if std::env::var_os("GPU_BENCH_QUICK").is_none() {
        sides.push(26);
    }
    let inner = MmffBuilder { ignore_interfrag_interactions: false, ..MmffBuilder::default() };
    println!("| system | electrostatics | CPU SIMD E+grad (ms) | GPU E+grad (ms) | GPU energy only (ms) | speedup | GPU TPC |");
    println!("|---|---|---|---|---|---|---|");
    for k in sides {
        let m: Molecule = water::water_box([k, k, k], 3.1, 1);
        let pos = m.flat_coords();
        let mut grad = vec![0.0; pos.len()];
        for (en, el) in [
            ("RF", Electrostatics::ReactionField { eps_rf: f64::INFINITY }),
            ("PME", Electrostatics::Pme { ewald_rtol: 1e-5, grid_spacing: 1.0, order: 4 }),
        ] {
            let o = ApproxOptions { cutoff: 10.0, buffer: 1.0, vdw_modifier: VdwModifier::PotentialShift, electrostatics: el };
            let cpu = build_approx_nonbonded(&m, &inner, &o, true).unwrap();
            let gpu = build_gpu_nonbonded(&m, &inner, &o, &ctx).unwrap();
            let tc = time(|| _ = black_box(cpu.energy_grad(black_box(&pos), &mut grad)), budget, 5);
            let tg = time(|| _ = black_box(gpu.energy_grad(black_box(&pos), &mut grad)), budget, 5);
            let te = time(|| _ = black_box(gpu.energy(black_box(&pos))), budget, 5);
            println!(
                "| water {}^3 ({} atoms) | {en}{} | {:.2} | {:.2} | {:.2} | {:.2}x | {} |",
                k,
                m.num_atoms(),
                cpu.pme_grid().map(|g| format!(" grid {g:?}")).unwrap_or_default(),
                ms(tc),
                ms(tg),
                ms(te),
                tc.as_secs_f64() / tg.as_secs_f64(),
                gpu.threads_per_cluster()
            );
        }
    }

    // ------------------------------------------------------------------ batch
    let para = molecule_optimizer_rs::io::read_file(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data/paracetamol.mol")).unwrap().remove(0);
    let nconf = 1000;
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let base = para.flat_coords();
    let confs: Vec<Vec<f64>> = (0..nconf).map(|_| base.iter().map(|x| x + rng.gen_range(-0.2..0.2)).collect()).collect();
    let refs: Vec<&[f64]> = confs.iter().map(|c| c.as_slice()).collect();
    println!("\n| batch | electrostatics | CPU (rayon) E+grad (ms) | GPU batch E+grad (ms) | GPU conformers/s | speedup |");
    println!("|---|---|---|---|---|---|");
    let mmff = MmffBuilder::default();
    for (en, o) in [
        ("Full", ApproxOptions { cutoff: 10.0, buffer: 1.0, vdw_modifier: VdwModifier::PotentialShift, electrostatics: Electrostatics::Full }),
        ("RF 8 A", ApproxOptions { cutoff: 8.0, buffer: 1.0, vdw_modifier: VdwModifier::PotentialShift, electrostatics: Electrostatics::ReactionField { eps_rf: f64::INFINITY } }),
    ] {
        let cpu = build_approx_nonbonded(&para, &mmff, &o, true).unwrap();
        let batch = GpuBatch::new(&ctx, &[(&para, nconf)], &mmff, &o).unwrap();
        let mut grads: Vec<Vec<f64>> = vec![vec![0.0; base.len()]; nconf];
        let tc = time(
            || {
                grads.par_iter_mut().zip(refs.par_iter()).for_each(|(g, p)| {
                    black_box(cpu.energy_grad(p, g));
                })
            },
            budget,
            5,
        );
        let mut e = vec![NonbondedEnergy::default(); nconf];
        let tg = time(
            || {
                let mut gr: Vec<&mut [f64]> = grads.iter_mut().map(|g| g.as_mut_slice()).collect();
                batch.evaluate(&refs, &mut e, Some(&mut gr));
            },
            budget,
            5,
        );
        println!(
            "| {nconf} paracetamol conformers (20 atoms) | {en} | {:.2} | {:.2} | {:.0} | {:.2}x |",
            ms(tc),
            ms(tg),
            nconf as f64 / tg.as_secs_f64(),
            tc.as_secs_f64() / tg.as_secs_f64()
        );
    }
}
