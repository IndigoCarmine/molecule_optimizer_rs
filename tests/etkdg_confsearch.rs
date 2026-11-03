//! Conformer-search acceptance test vs RDKit 2026.09.1:
//! `EmbedMultipleConfs(numConfs=N, ETKDGv3, randomSeed=0xf00d)` followed by
//! `MMFFOptimizeMoleculeConfs` / `UFFOptimizeMoleculeConfs(maxIters=2000)`;
//! the global minimum must be the same structure with the same energy.
//!
//! The quick test always runs: it re-embeds only RDKit's global-minimum
//! conformer (same per-conformer seed) and optimizes it. The full search
//! (all N conformers through `calc::conformer::conformer_search`) runs when
//! the environment variable `ETKDG_CONFSEARCH=1` is set.

use molecule_optimizer_rs::calc::conformer::{ConformerSearchOptions, conformer_search};
use molecule_optimizer_rs::core::geometry::kabsch_rmsd;
use molecule_optimizer_rs::distgeom::{EmbedParams, Embedding};
use molecule_optimizer_rs::ff::mmff::MmffBuilder;
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::ff::{ForceField, ForceFieldBuilder};
use molecule_optimizer_rs::optim::{self, MinimizeOptions, NoObserver};
use molecule_optimizer_rs::{Molecule, io};
use serde_json::Value;
use std::collections::BTreeMap;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/etkdg");

fn molecules() -> BTreeMap<String, Molecule> {
    let text = std::fs::read_to_string(format!("{DIR}/molecules.sdf")).unwrap();
    io::molfile::read_sdf(&text).unwrap().into_iter().map(|m| (m.name.clone(), m)).collect()
}

fn coords(v: &Value) -> Vec<[f64; 3]> {
    v.as_array().unwrap().iter().map(|p| [p[0].as_f64().unwrap(), p[1].as_f64().unwrap(), p[2].as_f64().unwrap()]).collect()
}

fn heavy(mol: &Molecule, c: &[[f64; 3]]) -> Vec<[f64; 3]> {
    mol.heavy_atoms().iter().map(|&i| c[i]).collect()
}

fn params(seed: i32) -> EmbedParams {
    let mut p = EmbedParams::etkdg_v3();
    p.random_seed = seed;
    p
}

fn builders() -> Vec<(&'static str, Box<dyn ForceFieldBuilder>)> {
    vec![("mmff", Box::new(MmffBuilder::default())), ("uff", Box::new(UffBuilder::default()))]
}

#[test]
fn global_minimum_conformer_matches_rdkit_quick() {
    let mols = molecules();
    let data: Value = serde_json::from_str(&std::fs::read_to_string(format!("{DIR}/confsearch.json")).unwrap()).unwrap();
    let mut ok = 0;
    let mut total = 0;
    let mut report = Vec::new();
    for (name, rec) in data.as_object().unwrap() {
        let mol = &mols[name];
        if rec["num_embedded"] != rec["num_confs"] {
            continue; // conformer indices would be shifted
        }
        let e = Embedding::new(mol, &params(rec["seed"].as_i64().unwrap() as i32)).unwrap();
        for (ffname, builder) in builders() {
            total += 1;
            let r = &rec[ffname];
            let ci = r["min_conf_index"].as_u64().unwrap() as usize;
            let start = e.embed_conformer(ci, &mut NoObserver).unwrap().expect("embedding");
            let emb_rmsd = kabsch_rmsd(&coords(&r["min_embedded_coords"]), &start);
            let mut m = mol.clone();
            m.coords = start.clone();
            let ff = builder.build(&m, &[]).unwrap();
            let mut pos: Vec<f64> = start.iter().flatten().copied().collect();
            let opts = MinimizeOptions { max_iters: 2000, ..Default::default() };
            optim::minimize(&ff, &mut pos, &opts, &mut NoObserver);
            let energy = ff.energy(&pos);
            let opt: Vec<[f64; 3]> = pos.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
            let rmsd = kabsch_rmsd(&heavy(mol, &coords(&r["min_coords"])), &heavy(mol, &opt));
            let de = (energy - r["min_energy"].as_f64().unwrap()).abs();
            if de < 1e-3 && rmsd < 0.05 {
                ok += 1;
            }
            report.push(format!(
                "{name:22} {ffname}: conf {ci:3} embed rmsd {emb_rmsd:.1e}, dE {de:.1e} kcal/mol, heavy rmsd {rmsd:.1e}"
            ));
        }
    }
    for l in &report {
        println!("{l}");
    }
    println!("global minimum reproduced: {ok}/{total}");
    assert_eq!(ok, total);
}

#[test]
fn full_conformer_search_matches_rdkit() {
    if std::env::var("ETKDG_CONFSEARCH").ok().as_deref() != Some("1") {
        println!("skipped (set ETKDG_CONFSEARCH=1 to run the full conformer search)");
        return;
    }
    let mols = molecules();
    let data: Value = serde_json::from_str(&std::fs::read_to_string(format!("{DIR}/confsearch.json")).unwrap()).unwrap();
    let mut bad = Vec::new();
    for (name, rec) in data.as_object().unwrap() {
        let mol = &mols[name];
        let n = rec["num_confs"].as_u64().unwrap() as usize;
        let opts = ConformerSearchOptions::etkdg(params(rec["seed"].as_i64().unwrap() as i32), n);
        for (ffname, builder) in builders() {
            let r = &rec[ffname];
            let t0 = std::time::Instant::now();
            let res = conformer_search(mol, builder.as_ref(), &opts).unwrap();
            let best = res.global_minimum().unwrap();
            let de = (best.energy - r["min_energy"].as_f64().unwrap()).abs();
            let rmsd = kabsch_rmsd(&heavy(mol, &coords(&r["min_coords"])), &heavy(mol, &best.coords));
            println!(
                "{name:22} {ffname}: {n} confs in {:.2}s, E {:.6} (RDKit {:.6}), heavy rmsd {rmsd:.1e}",
                t0.elapsed().as_secs_f64(),
                best.energy,
                r["min_energy"].as_f64().unwrap()
            );
            if de >= 1e-3 || rmsd >= 0.05 {
                bad.push(format!("{name}/{ffname}"));
            }
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}
