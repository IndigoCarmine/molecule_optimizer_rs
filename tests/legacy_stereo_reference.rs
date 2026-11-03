//! Compares `perception::legacy_stereo::assign_stereo_legacy_from_3d` with the
//! stereo RDKit (legacy perception) assigns when reading the same molblocks
//! with `Chem.SDMolSupplier(path, removeHs=False)`.

use molecule_optimizer_rs::core::{BondStereo, ChiralTag, Molecule};
use molecule_optimizer_rs::io;
use molecule_optimizer_rs::perception::legacy_stereo::assign_stereo_legacy_from_3d;
use serde_json::Value;
use std::path::Path;

fn tag_str(t: ChiralTag) -> &'static str {
    match t {
        ChiralTag::Cw => "CW",
        ChiralTag::Ccw => "CCW",
        ChiralTag::Unspecified => "",
    }
}

/// Returns a list of mismatch descriptions (empty = match).
fn compare(mol: &Molecule, exp: &Value) -> Vec<String> {
    let mut errs = Vec::new();
    let tags = exp["chiral_tags"].as_array().unwrap();
    assert_eq!(tags.len(), mol.num_atoms(), "{}: atom count", mol.name);
    for (i, t) in tags.iter().enumerate() {
        let got = tag_str(mol.atom(i).chiral_tag);
        if got != t.as_str().unwrap() {
            errs.push(format!("atom {i}: got '{got}' expected '{}'", t.as_str().unwrap()));
        }
    }
    let mut expected_bonds = std::collections::BTreeMap::new();
    for b in exp["bond_stereo"].as_array().unwrap() {
        let idx = b["bond"].as_u64().unwrap() as usize;
        let sa: Vec<usize> = b["stereo_atoms"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let st = match b["stereo"].as_str().unwrap() {
            "Z" | "CIS" => BondStereo::Cis,
            "E" | "TRANS" => BondStereo::Trans,
            s => panic!("unexpected stereo {s}"),
        };
        expected_bonds.insert(idx, (st, (sa[0], sa[1])));
    }
    for (b, bond) in mol.bonds().iter().enumerate() {
        let got = if bond.stereo == BondStereo::None { None } else { Some((bond.stereo, bond.stereo_atoms.unwrap())) };
        let want = expected_bonds.get(&b).copied();
        if got != want {
            errs.push(format!("bond {b} ({}-{}): got {got:?} expected {want:?}", bond.begin, bond.end));
        }
    }
    errs
}

fn run_fixture(sdf: &Path, json: &Path) -> (usize, usize, Vec<String>) {
    let text = std::fs::read_to_string(sdf).unwrap();
    let exp: Value = serde_json::from_str(&std::fs::read_to_string(json).unwrap()).unwrap();
    let mols = io::read_sdf(&text).unwrap();
    let mut ok = 0;
    let mut report = Vec::new();
    for mut m in mols.clone() {
        let e = exp.get(&m.name).unwrap_or_else(|| panic!("no reference for {}", m.name));
        assign_stereo_legacy_from_3d(&mut m);
        let errs = compare(&m, e);
        if errs.is_empty() {
            ok += 1;
        } else {
            report.push(format!("{}: {}", m.name, errs.join("; ")));
        }
    }
    (ok, mols.len(), report)
}

fn data(p: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(p)
}

#[test]
fn etkdg_molecules_match_rdkit_legacy_stereo() {
    let (ok, total, report) = run_fixture(&data("etkdg/molecules.sdf"), &data("etkdg/stereo.json"));
    for r in &report {
        eprintln!("{r}");
    }
    eprintln!("etkdg fixture: {ok}/{total} molecules match");
    assert_eq!(total, 79);
    assert!(report.is_empty(), "{} mismatching molecules", report.len());
}

#[test]
fn hard_cases_match_rdkit_legacy_stereo() {
    let (ok, total, report) = run_fixture(&data("stereo/cases.sdf"), &data("stereo/cases.json"));
    for r in &report {
        eprintln!("{r}");
    }
    eprintln!("hard-case fixture: {ok}/{total} molecules match");
    assert!(total > 0);
    assert!(report.is_empty(), "{} mismatching molecules", report.len());
}

#[test]
fn cip_ranks_match_rdkit() {
    // ranks as written by RDKit (_CIPRank after assignStereochemistry may be
    // reranked, so compare against the fixture of plain assignAtomCIPRanks)
    let p = data("stereo/cip_ranks.json");
    let exp: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    let text = std::fs::read_to_string(data("stereo/cases.sdf")).unwrap();
    let mut bad = Vec::new();
    let mut n = 0;
    for m in io::read_sdf(&text).unwrap() {
        let Some(arr) = exp.get(&m.name).and_then(|v| v.as_array()) else { continue };
        let want: Vec<u32> = arr.iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let got = molecule_optimizer_rs::perception::legacy_stereo::cip_ranks(&m);
        n += 1;
        if got != want {
            bad.push(format!("{}: got {got:?} expected {want:?}", m.name));
        }
    }
    assert!(n > 0);
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
