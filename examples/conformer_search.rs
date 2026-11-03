//! ETKDGv3 + MMFF94 conformer search with live progress.
//! `cargo run --release --example conformer_search -- examples/data/paracetamol.mol 100`
use molecule_optimizer_rs::calc::ConformerSearchOptions;
use molecule_optimizer_rs::distgeom::EmbedParams;
use molecule_optimizer_rs::ff::mmff::MmffBuilder;
use molecule_optimizer_rs::tasks::Engine;
use molecule_optimizer_rs::{Result, io};
use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "examples/data/paracetamol.mol".into());
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);
    let mol = Arc::new(io::read_file(&path)?.remove(0));
    let mut params = EmbedParams::etkdg_v3();
    params.random_seed = 0xf00d;
    let opts = ConformerSearchOptions::etkdg(params, n);
    let job = Engine::global().conformer_search(mol.clone(), Arc::new(MmffBuilder::default()), opts);
    while !job.is_finished() {
        let s = job.snapshot();
        let best = s.data.best().map(|c| c.energy).unwrap_or(f64::NAN);
        println!("{}/{} done, {} in flight, best E = {:.4}", s.data.completed, s.data.total, s.data.in_flight.len(), best);
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let r = job.await?;
    let best = r.global_minimum().unwrap();
    println!("{} unique conformers; global minimum E(MMFF94) = {:.6} kcal/mol (conformer {})", r.conformers.len(), best.energy, best.id);
    std::fs::write("global_min.xyz", io::xyz::write_xyz_coords(&mol, &best.coords, Some("global minimum")))?;
    Ok(())
}
