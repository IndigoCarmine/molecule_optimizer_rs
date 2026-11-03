//! User-defined energy terms on top of MMFF94: a flat-bottom spherical wall
//! that keeps every atom inside a sphere, plus a custom Lennard-Jones
//! attraction scaled by a global `lambda` acting only between two copies of
//! a molecule. The dimer is optimized for several `lambda` values without
//! rebuilding the force field.
//!
//! `cargo run --release --example custom_terms -- [input.mol]`
use molecule_optimizer_rs::calc::{self, OptimizeOptions};
use molecule_optimizer_rs::core::geometry::{centroid, norm};
use molecule_optimizer_rs::ff::custom::{
    AtomParams, CustomExternal, CustomNonbonded, CustomTermsExt, InteractionGroups, WithCustomTerms,
};
use molecule_optimizer_rs::ff::mmff::MmffBuilder;
use molecule_optimizer_rs::ff::{ForceField, ForceFieldBuilder};
use molecule_optimizer_rs::{Molecule, Result, io, perception};

/// Two copies of `mol`, the second shifted by `shift`, centered at the origin.
fn dimer(mol: &Molecule, shift: [f64; 3]) -> Result<Molecule> {
    let mut d = mol.clone();
    let n = mol.num_atoms();
    for a in 0..n {
        let p = mol.coords[a];
        d.add_atom(mol.atom(a).clone(), [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]]);
    }
    for b in mol.bonds() {
        d.add_bond(b.begin + n, b.end + n, b.order);
    }
    perception::sanitize(&mut d)?;
    let c = centroid(&d.coords);
    for p in &mut d.coords {
        *p = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
    }
    d.name = format!("{} dimer", mol.name);
    Ok(d)
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "examples/data/paracetamol.mol".into());
    let mono = io::read_file(&path)?.remove(0);
    let n = mono.num_atoms();
    let mol = dimer(&mono, [7.0, 2.0, 1.0])?;
    let first: Vec<usize> = (0..n).collect();
    let second: Vec<usize> = (n..2 * n).collect();

    // Flat-bottom spherical wall: zero inside radius R, harmonic outside.
    let wall =
        CustomExternal::new("wall", "k*max(0, d - R)^2; d = sqrt(x^2 + y^2 + z^2)", &[]).atoms(0..mol.num_atoms(), &[]);
    // Lennard-Jones between the two copies only, with per-atom parameters by
    // element and Lorentz-Berthelot combining rules, smoothly switched off
    // between 8 and 10 Angstrom.
    let lj = CustomNonbonded::new("lj", "lambda*4*eps*(s6^2 - s6); s6 = (sig/r)^6")
        .per_atom(
            &["sig", "eps"],
            AtomParams::Function(std::sync::Arc::new(|m: &Molecule, i: usize| match m.atom(i).atomic_num {
                1 => vec![2.5, 0.03],
                6 => vec![3.4, 0.09],
                7 => vec![3.3, 0.17],
                _ => vec![3.1, 0.21],
            })),
        )
        .combining("sig = 0.5*(sig1 + sig2); eps = sqrt(eps1*eps2)")
        .groups(InteractionGroups::Between(first, second))
        .cutoff(10.0, Some(8.0));

    let builder = WithCustomTerms::new(MmffBuilder::default())
        .with_global("k", 20.0)
        .with_global("R", 5.5)
        .with_global("lambda", 1.0)
        .with(wall)?
        .with(lj)?;

    let ff = builder.build(&mol, &[])?;
    let mut opts = OptimizeOptions::default();
    opts.minimize.max_iters = 5000;
    let mut coords = mol.coords.clone();
    println!("{} ({} atoms), builder {}", mol.name, mol.num_atoms(), builder.name());
    for lambda in [0.0, 0.5, 1.0, 2.0] {
        // Changing a global updates `ff` (and any force field built by
        // `builder`) without rebuilding.
        builder.set_global("lambda", lambda)?;
        let r = calc::optimize::optimize_coords(
            &mol,
            &coords,
            &builder,
            &opts,
            &mut molecule_optimizer_rs::optim::NoObserver,
        )?;
        coords = r.coords.clone();
        let pos: Vec<f64> = coords.iter().flatten().copied().collect();
        let a = centroid(&coords[..n]);
        let b = centroid(&coords[n..]);
        let sep = norm([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
        let rmax = coords.iter().map(|p| norm(*p)).fold(0.0, f64::max);
        println!(
            "lambda = {lambda:3.1}: E = {:10.4} kcal/mol ({:?}, {} iterations), dE/dlambda = {:8.4}, \
             centroid distance {sep:5.2} A, max |r| {rmax:5.2} A",
            r.energy,
            r.status,
            r.iterations,
            ff.energy_param_derivative("lambda", &pos),
        );
        for (k, v) in ff.energy_breakdown(&pos).iter().filter(|(k, _)| k.starts_with("Custom:")) {
            println!("    {k:<12} {v:10.4}");
        }
    }

    // The same LJ kernel as WGSL source for a GPU pipeline.
    let kernel = &builder.terms()[1].kernel;
    println!("\nWGSL for the custom LJ term:\n{}", kernel.to_wgsl_function("lj_energy_and_deriv").unwrap());
    Ok(())
}
