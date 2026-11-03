//! Shows reading the current geometry of a running job at any time:
//! starts an optimization and writes the in-progress coordinates to
//! `trajectory.xyz` every few milliseconds until it finishes.
//! `cargo run --release --example live_snapshot`
use molecule_optimizer_rs::calc::OptimizeOptions;
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::tasks::Engine;
use molecule_optimizer_rs::{Result, io};
use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "examples/data/paracetamol.mol".into());
    let mol = Arc::new(io::read_file(&path)?.remove(0));
    let mut opts = OptimizeOptions::default();
    opts.minimize.max_iters = 2000;
    let job = Engine::global().optimize(mol.clone(), Arc::new(UffBuilder::default()), opts);
    let mut traj = String::new();
    let mut last = u64::MAX;
    while !job.is_finished() {
        let snap = job.request_snapshot().await; // fresh coordinates from the worker
        if snap.version != last {
            last = snap.version;
            let f = &snap.data.frame;
            println!("v{:<4} iter {:<5} E = {:.6}", snap.version, f.iteration, f.energy);
            traj.push_str(&io::xyz::write_xyz_coords(&mol, &f.coords, Some(&format!("iter {} E {:.6}", f.iteration, f.energy))));
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let r = job.await?;
    println!("final E = {:.6} kcal/mol after {} iterations", r.energy, r.iterations);
    std::fs::write("trajectory.xyz", traj)?;
    Ok(())
}
