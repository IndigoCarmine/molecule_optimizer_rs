//! Single-point energy with breakdown.
//! `cargo run --release --example energy -- examples/data/paracetamol.mol`
use molecule_optimizer_rs::{calc, ff::uff::UffBuilder, io};

fn main() -> molecule_optimizer_rs::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "examples/data/paracetamol.mol".into());
    for mol in io::read_file(&path)? {
        let e = calc::energy::energy(&mol, &UffBuilder::default())?;
        println!("{} ({}): E(UFF) = {:.6} kcal/mol, grad RMS = {:.4}", mol.name, mol.formula(), e.total, e.grad_rms);
        for (k, v) in &e.breakdown {
            println!("  {k:<12} {v:>12.6}");
        }
    }
    Ok(())
}
