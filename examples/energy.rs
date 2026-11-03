//! Single-point energy with breakdown.
//! `cargo run --release --example energy -- examples/data/paracetamol.mol`
use molecule_optimizer_rs::ff::{ForceFieldBuilder, mmff::MmffBuilder, uff::UffBuilder};
use molecule_optimizer_rs::{calc, io};

fn main() -> molecule_optimizer_rs::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "examples/data/paracetamol.mol".into());
    for mol in io::read_file(&path)? {
        let builders: [Box<dyn ForceFieldBuilder>; 2] = [Box::new(UffBuilder::default()), Box::new(MmffBuilder::default())];
        for b in &builders {
            let e = calc::energy::energy(&mol, b.as_ref())?;
            println!("{} ({}): E({}) = {:.6} kcal/mol, grad RMS = {:.4}", mol.name, mol.formula(), b.name(), e.total, e.grad_rms);
            for (k, v) in &e.breakdown {
                println!("  {k:<14} {v:>12.6}");
            }
        }
    }
    Ok(())
}
