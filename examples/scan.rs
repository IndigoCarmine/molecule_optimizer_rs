//! Relaxed dihedral scan of butane C-C-C-C, run in parallel.
//! `cargo run --release --example scan`
use molecule_optimizer_rs::calc::{ScanCoordinate, ScanOptions};
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::tasks::Engine;
use molecule_optimizer_rs::{Result, io};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let mol = Arc::new(io::read_file("examples/data/butane.mol")?.remove(0));
    let opts = ScanOptions {
        coordinates: vec![ScanCoordinate::Dihedral { i: 0, j: 1, k: 2, l: 3, values: ScanCoordinate::steps(-180.0, 15.0, 24) }],
        ..Default::default()
    };
    let res = Engine::global().scan(mol, Arc::new(UffBuilder::default()), opts).await?;
    for (p, rel) in res.points.iter().zip(res.relative_energies()) {
        println!("{:>8.1} deg  {:>10.4} kcal/mol", p.values[0], rel);
    }
    Ok(())
}
