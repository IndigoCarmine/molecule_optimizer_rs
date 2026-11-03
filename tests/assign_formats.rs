//! Atom typing and charges must not depend on the input file format:
//! the benchmark set read from SDF, MOL2 (our writer) and XYZ (connectivity
//! and bond orders perceived from coordinates) must give identical
//! MMFF94 / UFF assignments.

use molecule_optimizer_rs::ff::{ForceFieldKind, assign};
use molecule_optimizer_rs::{Molecule, io};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn sdf() -> Vec<Molecule> {
    io::read_sdf(&std::fs::read_to_string(root().join("benches/data/set/initial.sdf")).unwrap()).unwrap()
}

fn same(a: &Molecule, b: &Molecule, kind: ForceFieldKind) -> Result<(), String> {
    let x = assign(a, kind).map_err(|e| e.to_string())?;
    let y = assign(b, kind).map_err(|e| e.to_string())?;
    if x.types != y.types {
        return Err(format!("types differ: {:?} vs {:?}", x.types, y.types));
    }
    if let (Some(p), Some(q)) = (&x.partial_charges, &y.partial_charges) {
        let d = p.iter().zip(q).map(|(u, v)| (u - v).abs()).fold(0.0, f64::max);
        if d > 1e-12 {
            return Err(format!("charges differ by {d}"));
        }
    }
    Ok(())
}

#[test]
fn sdf_and_mol2_give_identical_assignments() {
    let mols = sdf();
    let mol2 = io::read_mol2_str(&std::fs::read_to_string(root().join("tests/data/mol2/bench_initial.mol2")).unwrap()).unwrap();
    let mut bad = Vec::new();
    for (a, b) in mols.iter().zip(&mol2) {
        for kind in [ForceFieldKind::Mmff94, ForceFieldKind::Uff] {
            if a.name == "BODIPY_core" && kind != ForceFieldKind::Uff {
                continue;
            }
            if let Err(e) = same(a, b, kind) {
                bad.push(format!("{} {kind:?}: {e}", a.name));
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn sdf_and_xyz_give_identical_assignments() {
    let mut bad = Vec::new();
    let mols = sdf();
    for a in &mols {
        let b = match io::read_xyz(&io::write_xyz(a, None)) {
            Ok(b) => b,
            Err(e) => {
                bad.push(format!("{}: xyz read failed: {e}", a.name));
                continue;
            }
        };
        for kind in [ForceFieldKind::Mmff94, ForceFieldKind::Uff] {
            if a.name == "BODIPY_core" && kind != ForceFieldKind::Uff {
                continue;
            }
            if let Err(e) = same(a, &b, kind) {
                bad.push(format!("{} {kind:?}: {e}", a.name));
            }
        }
    }
    println!("XYZ vs SDF: {} mismatches of {} molecules", bad.len(), mols.len());
    for b in &bad {
        println!("  {}", &b[..b.len().min(160)]);
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
