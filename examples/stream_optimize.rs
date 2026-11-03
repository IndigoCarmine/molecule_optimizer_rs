//! Streams the intermediate structures of a running optimization and writes
//! them to `stream_trajectory.xyz`.
//!
//! * `cargo run --release --example stream_optimize [file]` optimizes to
//!   convergence; the stream ends with the final structure.
//! * `cargo run --release --example stream_optimize [file] endless` sets
//!   `force_tol = 0` (endless mode: never converges) and cancels the job
//!   after 2 s; the stream still delivers the final frame, then ends.
use futures::StreamExt;
use molecule_optimizer_rs::calc::OptimizeOptions;
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::tasks::{Engine, EngineConfig};
use molecule_optimizer_rs::{Result, io};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let endless = args.iter().any(|a| a == "endless");
    let path = args.iter().find(|a| *a != "endless").cloned().unwrap_or_else(|| "examples/data/paracetamol.mol".into());
    let mol = Arc::new(io::read_file(&path)?.remove(0));
    // publish a frame every 1 ms (the stream never makes the worker publish more often)
    let engine = Engine::new(EngineConfig { publish_interval: Duration::from_millis(1), ..Default::default() })?;
    let mut opts = OptimizeOptions::default();
    opts.minimize.max_iters = 2000;
    if endless {
        opts.minimize.force_tol = 0.0;
    }
    let job = engine.optimize(mol.clone(), Arc::new(UffBuilder::default()), opts);
    let t0 = Instant::now();
    let mut frames = job.frames();
    let mut traj = String::new();
    let mut n = 0;
    while let Some(f) = frames.next().await {
        n += 1;
        println!("frame {n:<4} iter {:<6} E = {:.6}", f.iteration, f.energy);
        traj.push_str(&io::xyz::write_xyz_coords(&mol, &f.coords, Some(&format!("iter {} E {:.6}", f.iteration, f.energy))));
        if endless && t0.elapsed() > Duration::from_secs(2) {
            job.cancel(); // the stream yields the final frame and ends
        }
    }
    let r = job.await?;
    println!("{:?}: E = {:.6} kcal/mol after {} iterations, {n} frames", r.status, r.energy, r.iterations);
    std::fs::write("stream_trajectory.xyz", traj)?;
    Ok(())
}
