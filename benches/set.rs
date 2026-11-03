//! Benchmark over the diverse molecule set (`benches/data/set/`): energy +
//! gradient and full minimization from the ETKDG initial structures, for the
//! RDKit-compatible path and the fast exact path (MMFF94 and UFF).
//!
//! `cargo bench --bench set`

use criterion::{Criterion, criterion_group, criterion_main};
use molecule_optimizer_rs::Molecule;
use molecule_optimizer_rs::fast::{FastBuilder, FastOptions};
use molecule_optimizer_rs::ff::mmff::{MmffBuilder, mmff_has_all_params};
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::ff::{ForceField, ForceFieldBuilder};
use molecule_optimizer_rs::io;
use molecule_optimizer_rs::optim::{MinimizeOptions, NoObserver, minimize};
use std::hint::black_box;

fn load() -> Vec<Molecule> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/data/set/initial.sdf");
    io::read_sdf(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn bench(c: &mut Criterion) {
    let all = load();
    let mmff_mols: Vec<&Molecule> = all.iter().filter(|m| mmff_has_all_params(m)).collect();
    let uff_mols: Vec<&Molecule> = all.iter().collect();
    let cases: Vec<(&str, Box<dyn ForceFieldBuilder>, &Vec<&Molecule>)> = vec![
        ("MMFF94", Box::new(MmffBuilder::default()), &mmff_mols),
        ("MMFF94/fast", Box::new(FastBuilder::new(MmffBuilder::default(), FastOptions::exact())), &mmff_mols),
        ("UFF", Box::new(UffBuilder::default()), &uff_mols),
        ("UFF/fast", Box::new(FastBuilder::new(UffBuilder::default(), FastOptions::exact())), &uff_mols),
    ];
    let mut g = c.benchmark_group("set");
    g.sample_size(10);
    for (name, b, mols) in &cases {
        let ffs: Vec<(Box<dyn ForceField>, Vec<f64>)> =
            mols.iter().map(|m| (b.build_dyn(m, &[]).unwrap(), m.flat_coords())).collect();
        g.bench_function(format!("{name}/energy_grad_all"), |bch| {
            bch.iter(|| {
                let mut e = 0.0;
                for (ff, pos) in &ffs {
                    let mut grad = vec![0.0; pos.len()];
                    e += ff.energy_grad(pos, &mut grad);
                }
                black_box(e)
            })
        });
        g.bench_function(format!("{name}/build_all"), |bch| {
            bch.iter(|| {
                for m in mols.iter() {
                    black_box(b.build_dyn(m, &[]).unwrap());
                }
            })
        });
        let opts = MinimizeOptions { max_iters: 2000, ..Default::default() };
        g.bench_function(format!("{name}/minimize_all"), |bch| {
            bch.iter(|| {
                let mut e = 0.0;
                for (ff, pos0) in &ffs {
                    let mut pos = pos0.clone();
                    e += minimize(ff.as_ref(), &mut pos, &opts, &mut NoObserver).energy;
                }
                black_box(e)
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
