//! Asynchronous geometry optimization of all molecules in a file, in parallel.
//! `cargo run --release --example optimize -- input.sdf out.sdf`
use molecule_optimizer_rs::calc::OptimizeOptions;
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::tasks::Engine;
use molecule_optimizer_rs::{Result, io};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().unwrap_or_else(|| "examples/data/paracetamol.mol".into());
    let output = args.next().unwrap_or_else(|| "optimized.sdf".into());
    let mols: Vec<Arc<_>> = io::read_file(&input)?.into_iter().map(Arc::new).collect();
    let items = mols.iter().map(|m| (m.clone(), None)).collect();
    let job = Engine::global().optimize_batch(items, Arc::new(UffBuilder::default()), OptimizeOptions::default());
    let results = job.await?;
    let mut out = String::new();
    for (m, r) in mols.iter().zip(results) {
        let r = r?;
        println!("{}: E = {:.6} kcal/mol ({:?}, {} iterations)", m.name, r.energy, r.status, r.iterations);
        out.push_str(&io::write_sdf_record(m, Some(&r.coords), &[("UFF_ENERGY", format!("{:.6}", r.energy))]));
    }
    std::fs::write(&output, out)?;
    println!("wrote {output}");
    Ok(())
}
