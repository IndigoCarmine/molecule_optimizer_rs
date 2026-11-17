//! Compares `depict::compute_2d_coords` with RDKit's
//! `rdDepictor.Compute2DCoords` (tools/gen_depict_reference.py) on the same
//! molblocks read with `SDMolSupplier(removeHs=False)`.

use molecule_optimizer_rs::depict::{Compute2DCoordsParams, compute_2d_coords};
use molecule_optimizer_rs::io;
use serde_json::Value;
use std::collections::BTreeMap;

/// Rounding-level agreement (Å).
const TOL: f64 = 1e-6;

/// Smallest fraction of depictions that must agree with RDKit where exact
/// agreement cannot be required (see [`check`]). The lowest fraction seen in
/// CI is 92 % (explicit-hydrogen set on Linux); over all three sets it is
/// 95 % on Linux and 94 % on macOS.
const MIN_MATCH_FRACTION: f64 = 0.85;

fn variant_params(name: &str, exp: &Value) -> Compute2DCoordsParams {
    let mut p = Compute2DCoordsParams::default();
    match name {
        "default" => {}
        "noorient" => p.canon_orient = false,
        "sampled" => {
            p.n_flips_per_sample = 3;
            p.n_samples = 20;
            p.sample_seed = 7;
        }
        "nopae" => p.use_path_angle_expansion = false,
        "bondlen" => p.bond_length = 1.0,
        "coordmap" => {
            let mut m = BTreeMap::new();
            for (k, v) in exp["map"].as_object().unwrap() {
                m.insert(
                    k.parse().unwrap(),
                    [v[0].as_f64().unwrap(), v[1].as_f64().unwrap()],
                );
            }
            p.coord_map = m;
        }
        _ => panic!("unknown variant {name}"),
    }
    p
}

fn max_dev(got: &[[f64; 3]], want: &Value) -> f64 {
    let want = want.as_array().unwrap();
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| {
            let (x, y) = (w[0].as_f64().unwrap(), w[1].as_f64().unwrap());
            ((g[0] - x).powi(2) + (g[1] - y).powi(2)).sqrt()
        })
        .fold(0.0, f64::max)
}

fn run(key: &str, sdf: &str) -> (usize, Vec<String>) {
    let reference: Value =
        serde_json::from_str(&std::fs::read_to_string("tests/data/depict/reference.json").unwrap())
            .unwrap();
    let text = std::fs::read_to_string(sdf).unwrap();
    let mols = io::read_sdf(&text).unwrap();
    let mut checked = 0;
    let mut fails = Vec::new();
    for mol in &mols {
        let exp = &reference[key][&mol.name];
        assert!(exp.is_object(), "no reference for {key}/{}", mol.name);
        for (variant, v) in exp.as_object().unwrap() {
            let params = variant_params(variant, v);
            let want = if variant == "coordmap" {
                &v["coords"]
            } else {
                v
            };
            let got = compute_2d_coords(mol, &params).unwrap();
            assert_eq!(got.len(), mol.num_atoms(), "{key}/{}/{variant}", mol.name);
            assert!(
                got.iter().flatten().all(|x| x.is_finite()),
                "{key}/{}/{variant}: non-finite coordinate",
                mol.name
            );
            let d = max_dev(&got, want);
            checked += 1;
            if d.is_nan() || d > TOL {
                fails.push(format!(
                    "{key}/{}/{variant}: max deviation {d:.3e}",
                    mol.name
                ));
            }
        }
    }
    (checked, fails)
}

/// Where the layout has a geometric tie (for example equal gaps around an
/// atom to put the next neighbor in), the algorithm, like RDKit's, takes
/// whichever candidate is larger in the last bit, and that bit comes from the
/// platform math library (`sin`, `cos`, `acos`, `atan2`). The reference was
/// generated on Windows, so every depiction must match there. On other
/// systems a few percent of the molecules get a different, equally valid
/// layout: there every depiction must be complete and finite, and at least
/// [`MIN_MATCH_FRACTION`] of them must match.
fn check(key: &str, sdf: &str) {
    let (checked, fails) = run(key, sdf);
    for f in &fails {
        eprintln!("{f}");
    }
    eprintln!(
        "{key}: {} of {checked} depictions differ from RDKit",
        fails.len()
    );
    if cfg!(windows) {
        assert!(
            fails.is_empty(),
            "{} of {checked} depictions differ from RDKit",
            fails.len()
        );
    } else {
        let matching = (checked - fails.len()) as f64 / checked as f64;
        assert!(
            matching >= MIN_MATCH_FRACTION,
            "only {:.1} % of {checked} depictions match RDKit",
            100.0 * matching
        );
    }
}

#[test]
#[cfg_attr(
    not(target_os = "windows"),
    ignore = "RDKit depiction coordinates are not portable across math libraries"
)]
fn depict_matches_rdkit_explicit_h() {
    check("h", "tests/data/depict/molecules_h.sdf");
}

#[test]
#[cfg_attr(
    not(target_os = "windows"),
    ignore = "RDKit depiction coordinates are not portable across math libraries"
)]
fn depict_matches_rdkit_implicit_h() {
    check("noh", "tests/data/depict/molecules_noh.sdf");
}

#[test]
#[cfg_attr(
    not(target_os = "windows"),
    ignore = "RDKit depiction coordinates are not portable across math libraries"
)]
fn depict_matches_rdkit_bench_set() {
    check("set", "benches/data/set/initial.sdf");
}
