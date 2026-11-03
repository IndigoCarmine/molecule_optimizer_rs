//! ETKDG port vs RDKit 2026.09.1 reference data (`tools/gen_etkdg_reference.py`).

use molecule_optimizer_rs::core::geometry::kabsch_rmsd;
use molecule_optimizer_rs::distgeom::{self, EmbedParams, Embedding, StopAfter};
use molecule_optimizer_rs::optim::NoObserver;
use molecule_optimizer_rs::{Molecule, io};
use serde_json::Value;
use std::collections::BTreeMap;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/etkdg");

fn molecules() -> BTreeMap<String, Molecule> {
    let text = std::fs::read_to_string(format!("{DIR}/molecules.sdf")).unwrap();
    io::molfile::read_sdf(&text).unwrap().into_iter().map(|m| (m.name.clone(), m)).collect()
}

fn json(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(format!("{DIR}/{name}")).unwrap()).unwrap()
}

fn coords(v: &Value) -> Option<Vec<[f64; 3]>> {
    v.as_array().map(|a| a.iter().map(|p| [p[0].as_f64().unwrap(), p[1].as_f64().unwrap(), p[2].as_f64().unwrap()]).collect())
}

fn max_dev(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter().zip(b).flat_map(|(p, q)| (0..3).map(move |k| (p[k] - q[k]).abs())).fold(0.0, f64::max)
}

fn params(name: &str) -> EmbedParams {
    let mut p = match name {
        "etkdgv3" => EmbedParams::etkdg_v3(),
        "etkdg" => EmbedParams::etkdg(),
        "kdg" => EmbedParams::kdg(),
        "etdg" => EmbedParams::etdg(),
        "sretkdgv3" => EmbedParams::sr_etkdg_v3(),
        _ => unreachable!(),
    };
    p.random_seed = 42;
    p
}

#[test]
#[allow(clippy::needless_range_loop)]
fn bounds_match_rdkit() {
    let mols = molecules();
    let data = json("bounds.json");
    let mut bad = Vec::new();
    for (name, rec) in data.as_object().unwrap() {
        let mol = &mols[name];
        let p = EmbedParams::etkdg_v3();
        let raw = distgeom::raw_bounds_matrix(mol, &p);
        let smooth = distgeom::bounds_matrix(mol, &p).unwrap();
        let mut plain_p = EmbedParams::dg();
        plain_p.use_macrocycle14config = false;
        let plain = distgeom::raw_bounds_matrix(mol, &plain_p);
        for (key, ours) in [("raw", &raw), ("smoothed", &smooth), ("plain_raw", &plain)] {
            let r = rec[key].as_array().unwrap();
            let n = mol.num_atoms();
            let mut worst = (0.0, 0, 0);
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let e = r[i][j].as_f64().unwrap();
                    let o = ours.val(i, j);
                    if (e - o).abs() > worst.0 {
                        worst = ((e - o).abs(), i, j);
                    }
                }
            }
            if worst.0 > 1e-6 {
                bad.push(format!("{name}/{key}: max dev {:.3e} at ({}, {})", worst.0, worst.1, worst.2));
            }
        }
    }
    println!("bounds mismatches: {}/{}", bad.len(), data.as_object().unwrap().len() * 3);
    for b in &bad {
        println!("  {b}");
    }
    assert!(bad.is_empty());
}

#[test]
fn experimental_torsions_match_rdkit() {
    let mols = molecules();
    let data = json("exp_torsions.json");
    let mut bad = Vec::new();
    let mut total = 0;
    for (name, rec) in data.as_object().unwrap() {
        for (pname, list) in rec.as_object().unwrap() {
            total += 1;
            let e = Embedding::new(&mols[name], &params(pname)).unwrap();
            let ours: Vec<[usize; 4]> = e.fragments.iter().flat_map(|f| f.details.exp_torsions.iter().filter(|t| !t.smarts.is_empty()).map(|t| t.atoms.map(|a| f.atoms[a]))).collect();
            let theirs: Vec<[usize; 4]> = list
                .as_array()
                .unwrap()
                .iter()
                .map(|t| {
                    let a = t["atoms"].as_array().unwrap();
                    [0, 1, 2, 3].map(|k| a[k].as_u64().unwrap() as usize)
                })
                .collect();
            if ours != theirs {
                bad.push(format!("{name}/{pname}: ours {ours:?}\n      rdkit {theirs:?}"));
            }
        }
    }
    println!("torsion list mismatches: {}/{}", bad.len(), total);
    for b in &bad {
        println!("  {b}");
    }
    assert!(bad.is_empty());
}

/// Seed parity: our coordinates vs RDKit `EmbedMolecule(randomSeed=42)`.
#[test]
fn seed_parity_report() {
    let mols = molecules();
    let data = json("embed_seed42.json");
    let mut counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut lines = Vec::new();
    let t0 = std::time::Instant::now();
    let mut n_embed = 0;
    let mut exact: BTreeMap<String, usize> = BTreeMap::new();
    for (name, rec) in data.as_object().unwrap() {
        let mol = &mols[name];
        for key in ["initial", "etkdgv3", "etkdg", "kdg", "etdg", "random_coords"] {
            let Some(expected) = coords(&rec[key]) else { continue };
            let (pname, stop, random) = match key {
                "initial" => ("etkdgv3", StopAfter { initial_only: true }, false),
                "random_coords" => ("etkdgv3", StopAfter::default(), true),
                k => (k, StopAfter::default(), false),
            };
            let mut p = params(pname);
            p.use_random_coords = random;
            let e = Embedding::new(mol, &p).unwrap();
            let ours = e.embed_with_seed(42, stop, &mut NoObserver).unwrap();
            n_embed += 1;
            let entry = counts.entry(key.to_string()).or_default();
            entry.1 += 1;
            match ours {
                Some(c) => {
                    let dev = max_dev(&c, &expected);
                    let rmsd = kabsch_rmsd(&expected, &c);
                    if dev < 1e-8 {
                        *exact.entry(key.to_string()).or_insert(0) += 1;
                    } else if rmsd < 1e-3 {
                        lines.push(format!("{name}/{key}: not bit-exact, max dev {dev:.2e}"));
                    }
                    if rmsd < 1e-3 {
                        entry.0 += 1;
                    } else {
                        lines.push(format!("{name}/{key}: max dev {dev:.2e}, rmsd {rmsd:.3}"));
                    }
                }
                None => lines.push(format!("{name}/{key}: embedding failed")),
            }
        }
        // multiple conformers (per-conformer seeds)
        if let Some(list) = rec["multi3"].as_array() {
            let e = Embedding::new(mol, &params("etkdgv3")).unwrap();
            for (ci, c) in list.iter().enumerate() {
                let expected = coords(c).unwrap();
                let entry = counts.entry("multi3".into()).or_default();
                entry.1 += 1;
                if let Some(ours) = e.embed_conformer(ci, &mut NoObserver).unwrap() {
                    let rmsd = kabsch_rmsd(&expected, &ours);
                    if max_dev(&expected, &ours) < 1e-8 {
                        *exact.entry("multi3".to_string()).or_insert(0) += 1;
                    }
                    if rmsd < 1e-3 {
                        entry.0 += 1;
                    } else {
                        lines.push(format!("{name}/multi3[{ci}]: rmsd {rmsd:.3}"));
                    }
                }
                n_embed += 1;
            }
        }
    }
    println!("seed parity (RMSD < 1e-3 A vs RDKit):");
    for (k, (ok, n)) in &counts {
        println!("  {k:14} {ok:3}/{n}   (max |dx| < 1e-8: {})", exact.get(k).copied().unwrap_or(0));
    }
    println!("average time per embedding: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3 / n_embed as f64);
    for l in &lines {
        println!("  {l}");
    }
    // the initial (metric-matrix) coordinates must match everywhere
    let (ok, n) = counts["initial"];
    assert_eq!(ok, n, "initial coordinates differ from RDKit");
}
