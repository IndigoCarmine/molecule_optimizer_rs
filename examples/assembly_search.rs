//! Assembly (cluster / dimer) search with soft-core λ-continuation (MMFF94).
//!
//! ```text
//! cargo run --release --example assembly_search -- [monomer.sdf] [copies] [trials] [out.sdf]
//! ```
//!
//! Defaults: acetic acid dimer (`tests/data/assembly/acetic_acid.sdf`, 2
//! copies), 32 trials, output `assembly.sdf`. Runs the λ-continuation search
//! and, from the same random placements, plain full-strength minimization,
//! then prints the unique structures, binding energies and how often each
//! strategy reached the lowest structure.
use molecule_optimizer_rs::calc::assembly::{
    AssemblyInput, AssemblyMode, AssemblySearchOptions, AssemblySearchResult, assembly_search,
};
use molecule_optimizer_rs::ff::mmff::MmffBuilder;
use molecule_optimizer_rs::io::molfile::write_sdf_record;
use molecule_optimizer_rs::tasks::Engine;
use molecule_optimizer_rs::{Result, io};
use std::time::{Duration, Instant};

fn hits(r: &AssemblySearchResult, gmin: &[[f64; 3]], emin: f64) -> usize {
    r.all.iter().filter(|s| (s.energy - emin).abs() < 0.01 && r.rmsd(&s.coords, gmin, true) < 0.25).count()
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "tests/data/assembly/acetic_acid.sdf".into());
    let copies: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    let trials: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let out = args.next().unwrap_or_else(|| "assembly.sdf".into());
    let mono = io::read_file(&path)?.remove(0);
    let input = AssemblyInput::Molecules(vec![(mono, copies)]);
    let opts = AssemblySearchOptions { num_trials: trials, ..Default::default() };

    // λ-continuation, asynchronously with a live view of the trials
    let t0 = Instant::now();
    let job = Engine::global().assembly_search(input.clone(), MmffBuilder::default(), opts.clone());
    while !job.is_finished() {
        let s = job.snapshot();
        let lam: Vec<String> = s.data.in_flight.values().map(|t| format!("{:.3}", t.lambda_vdw)).collect();
        let best = s.data.best().map(|b| b.energy).unwrap_or(f64::NAN);
        println!(
            "{}/{} done, λ of in-flight trials [{}], best E = {best:.4}",
            s.data.completed,
            s.data.total,
            lam.join(" ")
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let lam = job.await?;
    let t_lam = t0.elapsed();

    // baseline: direct minimization from the same placements
    let t0 = Instant::now();
    let direct = assembly_search(
        &input,
        &MmffBuilder::default(),
        &AssemblySearchOptions { mode: AssemblyMode::Direct, ..opts },
    )?;
    let t_dir = t0.elapsed();

    println!("\nmonomer energies {:?}, reference Σ = {:.4} kcal/mol", lam.monomer_energies, lam.reference_energy);
    println!("unique structures (λ-continuation):");
    println!("  #   trial   E(MMFF94)    ΔE_bind   connected");
    for (k, s) in lam.structures.iter().enumerate() {
        println!("  {k:<3} {:<7} {:<12.4} {:<9.4} {}", s.trial, s.energy, s.binding_energy, s.connected);
    }
    let (gl, gd) = (lam.global_minimum().unwrap(), direct.global_minimum().unwrap());
    let g = if gl.energy <= gd.energy { gl } else { gd };
    println!("\nlowest: E = {:.4} kcal/mol, binding energy {:.4} kcal/mol", g.energy, g.energy - lam.reference_energy);
    println!(
        "λ-continuation: {}/{} trials reached it, {} unique, {:.2?}",
        hits(&lam, &g.coords, g.energy),
        lam.all.len(),
        lam.structures.len(),
        t_lam
    );
    println!(
        "direct:         {}/{} trials reached it, {} unique, {:.2?}",
        hits(&direct, &g.coords, g.energy),
        direct.all.len(),
        direct.structures.len(),
        t_dir
    );

    let mut sdf = String::new();
    for (k, s) in lam.structures.iter().enumerate() {
        let mut m = lam.complex.clone();
        m.name = format!("assembly_{k}");
        sdf.push_str(&write_sdf_record(
            &m,
            Some(&s.coords),
            &[("energy", format!("{:.6}", s.energy)), ("binding_energy", format!("{:.6}", s.binding_energy))],
        ));
    }
    std::fs::write(&out, sdf)?;
    println!("wrote {} structures to {out}", lam.structures.len());
    Ok(())
}
