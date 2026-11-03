//! MOL2 interoperability with RDKit: the benchmark set is written as MOL2
//! by our writer (`write_fixture`, ignored), RDKit reads that file
//! (`tools/gen_mol2_reference.py`) and computes MMFF94/UFF energies; our
//! reader must give the same molecules and energies.

use molecule_optimizer_rs::ff::mmff::MmffBuilder;
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::ff::{ForceField, ForceFieldBuilder};
use molecule_optimizer_rs::io;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
#[ignore = "regenerates tests/data/mol2/bench_initial.mol2"]
fn write_fixture() {
    let text = std::fs::read_to_string(root().join("benches/data/set/initial.sdf")).unwrap();
    let mut out = String::new();
    for m in io::read_sdf(&text).unwrap() {
        out.push_str(&io::write_mol2(&m));
        out.push('\n');
    }
    std::fs::write(root().join("tests/data/mol2/bench_initial.mol2"), out).unwrap();
}

#[test]
fn mol2_energies_match_rdkit() {
    let dir = root().join("tests/data/mol2");
    let mols = io::read_mol2_str(&std::fs::read_to_string(dir.join("bench_initial.mol2")).unwrap()).unwrap();
    let refs = std::fs::read_to_string(dir.join("rdkit_energies.tsv")).unwrap();
    let mut checked = 0;
    let mut bad = Vec::new();
    for line in refs.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let (idx, name, kind, e_ref): (usize, &str, &str, &str) = (f[0].parse().unwrap(), f[1], f[2], f[3]);
        let m = &mols[idx];
        assert_eq!(m.name, name);
        let Ok(e_ref) = e_ref.parse::<f64>() else { continue };
        let b: Box<dyn ForceFieldBuilder> = if kind == "MMFF" { Box::new(MmffBuilder::default()) } else { Box::new(UffBuilder::default()) };
        let e = match b.build_dyn(m, &[]) {
            Ok(ff) => ff.energy(&m.flat_coords()),
            Err(err) => {
                bad.push(format!("{name} {kind}: build failed: {err}"));
                continue;
            }
        };
        if (e - e_ref).abs() > 1e-6 * e_ref.abs().max(1.0) {
            bad.push(format!("{name} {kind}: ours {e:.6} rdkit {e_ref:.6}"));
        }
        checked += 1;
    }
    println!("checked {checked} energies, {} mismatches", bad.len());
    assert!(checked > 200, "fixture too small");
    assert!(bad.is_empty(), "mismatches:\n{}", bad.join("\n"));
}
