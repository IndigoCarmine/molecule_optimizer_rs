//! Quick timing of the approximate nonbonded backend on a water box.
//!
//! `cargo run --release --example approx_profile -- [waters_per_side]`

#[path = "../tests/common/water.rs"]
mod water;

use molecule_optimizer_rs::fast::approx::build_approx_nonbonded;
use molecule_optimizer_rs::fast::backend::NonbondedBackend;
use molecule_optimizer_rs::fast::{ApproxOptions, Electrostatics, VdwModifier};
use molecule_optimizer_rs::ff::mmff::MmffBuilder;
use std::time::Instant;

fn main() {
    let k: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(26);
    let m = water::water_box([k, k, k], 3.1, 1);
    let pos = m.flat_coords();
    let inner = MmffBuilder { ignore_interfrag_interactions: false, ..MmffBuilder::default() };
    for el in [
        Electrostatics::ReactionField { eps_rf: f64::INFINITY },
        Electrostatics::Pme { ewald_rtol: 1e-5, grid_spacing: 1.0, order: 4 },
    ] {
        let o = ApproxOptions { cutoff: 10.0, buffer: 1.0, vdw_modifier: VdwModifier::PotentialShift, electrostatics: el };
        let b = build_approx_nonbonded(&m, &inner, &o, true).unwrap();
        let mut g = vec![0.0; pos.len()];
        let reps = 20;
        let time = |f: &mut dyn FnMut()| {
            f();
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            t.elapsed() / reps
        };
        let t_eg = time(&mut || {
            b.energy_grad(&pos, &mut g);
        });
        let t_e = time(&mut || {
            b.energy(&pos);
        });
        let t_rec = time(&mut || {
            b.pme_reciprocal_energy(&pos);
        });
        let t_build = time(&mut || b.rebuild(&pos));
        println!("{} atoms {el:?}: E+grad {t_eg:.2?}, E {t_e:.2?}, PME recip (E only) {t_rec:.2?}, list build {t_build:.2?}", m.num_atoms());
    }
}
