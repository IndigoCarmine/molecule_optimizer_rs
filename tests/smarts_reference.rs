//! SMARTS matcher vs. RDKit 2026.09.1 reference data
//! (`tools/gen_etkdg_reference.py smarts`): every torsion-preference pattern
//! of the v1, v2, small-ring and macrocycle tables against every test
//! molecule, comparing the ordered `GetSubstructMatches` result
//! (uniquify=False and uniquify=True).

use molecule_optimizer_rs::io;
use molecule_optimizer_rs::smarts::Smarts;
use serde_json::Value;
use std::collections::BTreeMap;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/etkdg");

/// Molecules whose RDKit perception (aromaticity, hybridization, rings, H
/// counts) differs from ours, so that SMARTS mismatches on them are not
/// matcher bugs.  `(name, reason)`; checked against `stereo.json`.
const KNOWN_PERCEPTION_MISMATCHES: &[(&str, &str)] = &[];

fn to_matches(v: &Value) -> Vec<Vec<usize>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|m| {
            m.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect()
        })
        .collect()
}

#[test]
fn torsion_patterns_match_rdkit() {
    let sdf = std::fs::read_to_string(format!("{DATA}/molecules.sdf")).unwrap();
    let mols = io::read_sdf(&sdf).unwrap();
    let json: Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{DATA}/torsion_matches.json")).unwrap(),
    )
    .unwrap();

    // parse every pattern
    let mut tables: BTreeMap<String, Vec<Smarts>> = BTreeMap::new();
    let mut parse_errors = Vec::new();
    let mut n_patterns = 0;
    for (tname, pats) in json["tables"].as_object().unwrap() {
        let mut v = Vec::new();
        for p in pats.as_array().unwrap() {
            let s = p.as_str().unwrap();
            n_patterns += 1;
            match Smarts::parse(s) {
                Ok(q) => v.push(q),
                Err(e) => {
                    parse_errors.push(format!("{tname}: {s}: {e}"));
                    v.push(Smarts::parse("[#0]").unwrap());
                }
            }
        }
        tables.insert(tname.clone(), v);
    }
    assert!(
        parse_errors.is_empty(),
        "parse errors:\n{}",
        parse_errors.join("\n")
    );

    let matches = json["matches"].as_object().unwrap();
    let mut n_lists = 0;
    let mut n_nonempty = 0;
    let (mut ok_all, mut ok_unique) = (0, 0);
    let mut failures = Vec::new();
    let mut skipped = Vec::new();
    let start = std::time::Instant::now();
    for mol in &mols {
        let skip = KNOWN_PERCEPTION_MISMATCHES
            .iter()
            .find(|(n, _)| *n == mol.name);
        let reference = matches
            .get(&mol.name)
            .unwrap_or_else(|| panic!("no reference for {}", mol.name));
        for (tname, qs) in &tables {
            for (k, q) in qs.iter().enumerate() {
                let key = format!("{tname}:{k}");
                let (exp_all, exp_unique) = match reference.get(&key) {
                    Some(r) => (to_matches(&r["all"]), to_matches(&r["unique"])),
                    None => (Vec::new(), Vec::new()),
                };
                n_lists += 1;
                if !exp_all.is_empty() {
                    n_nonempty += 1;
                }
                let got_all = q.find_matches(mol, false);
                let got_unique = q.find_matches(mol, true);
                let good_all = got_all == exp_all;
                let good_unique = got_unique == exp_unique;
                ok_all += good_all as usize;
                ok_unique += good_unique as usize;
                if !(good_all && good_unique) {
                    let msg = format!(
                        "{} {key} {}\n   all: expected {exp_all:?}\n        got      {got_all:?}\n   unique: expected {exp_unique:?}\n           got      {got_unique:?}",
                        mol.name,
                        q.as_str()
                    );
                    if skip.is_some() {
                        skipped.push(msg);
                    } else {
                        failures.push(msg);
                    }
                }
            }
        }
    }
    let elapsed = start.elapsed();
    println!(
        "{} molecules x {n_patterns} patterns = {n_lists} match lists ({n_nonempty} non-empty); \
         identical incl. order: all {ok_all}/{n_lists}, unique {ok_unique}/{n_lists}; {:.2?}",
        mols.len(),
        elapsed
    );
    // timing: all patterns over the molecule closest to 40 atoms
    if let Some(m) = mols.iter().min_by_key(|m| m.num_atoms().abs_diff(40)) {
        let t0 = std::time::Instant::now();
        let mut total = 0;
        for qs in tables.values() {
            for q in qs {
                total += q.find_matches(m, false).len();
            }
        }
        println!(
            "all {n_patterns} patterns on {} ({} atoms): {:.2?} ({total} matches)",
            m.name,
            m.num_atoms(),
            t0.elapsed()
        );
    }
    for (name, reason) in KNOWN_PERCEPTION_MISMATCHES {
        println!("skipped (known perception mismatch) {name}: {reason}");
    }
    for s in &skipped {
        println!("  [perception] {s}");
    }
    assert!(
        failures.is_empty(),
        "{} mismatching match lists:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Extra general patterns (`tools/gen_smarts_reference.py`): query atom count,
/// RDKit query bond order, and ordered matches.
#[test]
fn extra_patterns_match_rdkit() {
    let sdf = std::fs::read_to_string(format!("{DATA}/molecules.sdf")).unwrap();
    let mols = io::read_sdf(&sdf).unwrap();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/smarts/extra_matches.json"
    );
    let json: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut qs = Vec::new();
    for p in json["patterns"].as_array().unwrap() {
        let s = p["smarts"].as_str().unwrap();
        let q = Smarts::parse(s).unwrap_or_else(|e| panic!("{s}: {e}"));
        assert_eq!(
            q.num_atoms(),
            p["num_atoms"].as_u64().unwrap() as usize,
            "{s}"
        );
        let bonds: Vec<(usize, usize)> = p["bonds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                (
                    b[0].as_u64().unwrap() as usize,
                    b[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        assert_eq!(q.bond_atoms(), bonds, "query bond order of {s}");
        qs.push(q);
    }
    let matches = json["matches"].as_object().unwrap();
    let mut failures = Vec::new();
    let mut n = 0;
    for mol in &mols {
        let reference = &matches[&mol.name];
        for (k, q) in qs.iter().enumerate() {
            let (exp_all, exp_unique) = match reference.get(k.to_string()) {
                Some(r) => (to_matches(&r["all"]), to_matches(&r["unique"])),
                None => (Vec::new(), Vec::new()),
            };
            n += 1;
            let got_all = q.find_matches(mol, false);
            let got_unique = q.find_matches(mol, true);
            if got_all != exp_all || got_unique != exp_unique {
                failures.push(format!(
                    "{} {}\n   all: expected {exp_all:?}\n        got      {got_all:?}\n   unique: expected {exp_unique:?}\n           got      {got_unique:?}",
                    mol.name,
                    q.as_str()
                ));
            }
            assert_eq!(
                q.has_match(mol),
                !exp_all.is_empty(),
                "has_match {} {}",
                mol.name,
                q.as_str()
            );
        }
    }
    println!(
        "extra patterns: {}/{n} match lists identical",
        n - failures.len()
    );
    assert!(
        failures.is_empty(),
        "{} mismatching match lists:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
