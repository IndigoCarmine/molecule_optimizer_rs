//! Properties of embedded conformers (independent of RDKit reference data).

use molecule_optimizer_rs::core::geometry::{cross, dihedral, dot, sub};
use molecule_optimizer_rs::core::{BondStereo, ChiralTag};
use molecule_optimizer_rs::distgeom::{self, EmbedParams, Embedding, force_fields};
use molecule_optimizer_rs::error::Error;
use molecule_optimizer_rs::ff::gradient_error;
use molecule_optimizer_rs::optim::{IterationInfo, NoObserver};
use molecule_optimizer_rs::{Molecule, io};
use rayon::prelude::*;
use std::ops::ControlFlow;

fn molecules() -> Vec<Molecule> {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/etkdg/molecules.sdf")).unwrap();
    io::molfile::read_sdf(&text).unwrap()
}

fn seeded(seed: i32) -> EmbedParams {
    let mut p = EmbedParams::etkdg_v3();
    p.random_seed = seed;
    p
}

fn signed_volume(c: &[[f64; 3]], center: usize, n: &[usize]) -> f64 {
    let v = |k: usize| sub(c[n[k]], c[center]);
    dot(v(0), cross(v(1), v(2)))
}

#[test]
fn conformers_satisfy_bounds_and_keep_stereo() {
    let mut worst_bound: (f64, String) = (0.0, String::new());
    for mol in molecules() {
        let e = Embedding::new(&mol, &seeded(7)).unwrap();
        let Some(c) = e.embed_conformer(0, &mut NoObserver).unwrap() else {
            // RDKit 2026.09.1 cannot embed bicyclo[1.1.1]pentane with ETKDGv3 either
            assert_eq!(mol.name, "bicyclo111", "{} failed", mol.name);
            continue;
        };
        // bounds (smoothed), relative violation
        for f in &e.fragments {
            let n = f.atoms.len();
            for i in 0..n {
                for j in 0..i {
                    let (a, b) = (f.atoms[i], f.atoms[j]);
                    let d = molecule_optimizer_rs::core::geometry::dist(c[a], c[b]);
                    let (lb, ub) = (f.bounds.lower(i, j), f.bounds.upper(i, j));
                    let viol = if d < lb { lb - d } else if d > ub { d - ub } else { 0.0 };
                    if viol > worst_bound.0 {
                        worst_bound = (viol, format!("{} {a}-{b}: d {d:.3} bounds [{lb:.3}, {ub:.3}]", mol.name));
                    }
                }
            }
        }
        // chirality of the enforced stereocenters is that of the input
        let m = &e.mol;
        for i in 0..m.num_atoms() {
            if m.atom(i).chiral_tag == ChiralTag::Unspecified {
                continue;
            }
            let nb: Vec<usize> = m.neighbors(i).collect();
            let v_in = signed_volume(&mol.coords, i, &nb);
            let v_out = signed_volume(&c, i, &nb);
            assert!(v_in * v_out > 0.0, "{}: chirality of atom {i} inverted", mol.name);
        }
        // double-bond stereo
        for b in m.bonds() {
            if b.stereo == BondStereo::None {
                continue;
            }
            let (s0, s1) = b.stereo_atoms.unwrap();
            let d = dihedral(c[s0], c[b.begin], c[b.end], c[s1]).abs();
            let cis = d < std::f64::consts::FRAC_PI_2;
            assert_eq!(cis, b.stereo == BondStereo::Cis, "{}: E/Z of bond {}-{} lost", mol.name, b.begin, b.end);
        }
    }
    println!("largest bounds violation: {:.3} A ({})", worst_bound.0, worst_bound.1);
    // pairs covered by torsion terms are not bounds-restrained in the ETK stage
    assert!(worst_bound.0 < 0.75, "{}", worst_bound.1);
}

#[test]
fn deterministic_and_parallel_equals_sequential() {
    for mol in molecules().into_iter().filter(|m| ["ibuprofen", "l_menthol", "macrolactone_14", "glucose"].contains(&m.name.as_str())) {
        let p = seeded(0xf00d);
        let seq = distgeom::embed_multiple(&mol, 8, &p).unwrap();
        let again = distgeom::embed_multiple(&mol, 8, &p).unwrap();
        assert_eq!(seq, again, "{}: not deterministic", mol.name);
        let e = Embedding::new(&mol, &p).unwrap();
        let par: Vec<Vec<[f64; 3]>> = (0..8usize)
            .into_par_iter()
            .map(|ci| e.embed_conformer(ci, &mut NoObserver).unwrap())
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(seq, par, "{}: parallel != sequential", mol.name);
        // single-conformer API equals conformer 0
        let one = distgeom::embed_molecule(&mol, &p, &mut NoObserver).unwrap();
        assert_eq!(one, seq[0]);
    }
}

#[test]
fn observer_sees_4d_positions_and_can_cancel() {
    let mol = molecules().into_iter().find(|m| m.name == "l_menthol").unwrap();
    let n = mol.num_atoms();
    let mut dims = std::collections::BTreeSet::new();
    let mut obs = |info: &IterationInfo<'_>| {
        dims.insert(info.pos.len() / n);
        ControlFlow::Continue(())
    };
    distgeom::embed_molecule(&mol, &seeded(1), &mut obs).unwrap();
    assert!(dims.contains(&4) && dims.contains(&3), "{dims:?}");
    let mut stop = |_: &IterationInfo<'_>| ControlFlow::Break(());
    let r = distgeom::embed_molecule(&mol, &seeded(1), &mut stop);
    assert!(matches!(r, Err(Error::Cancelled)), "{r:?}");
}

#[test]
fn pruning_removes_duplicates() {
    let mol = molecules().into_iter().find(|m| m.name == "cyclohexane").unwrap();
    let mut p = seeded(3);
    let all = distgeom::embed_multiple(&mol, 10, &p).unwrap();
    p.prune_rms_thresh = 0.5;
    let pruned = distgeom::embed_multiple(&mol, 10, &p).unwrap();
    assert!(pruned.len() < all.len() && !pruned.is_empty(), "{} vs {}", pruned.len(), all.len());
    assert_eq!(pruned[0], all[0]);
}

#[test]
fn etk_force_field_gradient_matches_finite_differences() {
    for mol in molecules().into_iter().filter(|m| ["aspirin", "propyne", "e_stilbene"].contains(&m.name.as_str())) {
        let e = Embedding::new(&mol, &seeded(5)).unwrap();
        let f = &e.fragments[0];
        let c = e.embed_conformer(0, &mut NoObserver).unwrap().unwrap();
        let mut pos: Vec<f64> = c.iter().flatten().copied().collect();
        for (k, x) in pos.iter_mut().enumerate() {
            *x += 0.05 * ((k as f64 * 0.37).sin()); // move off the minimum
        }
        let ff = force_fields::construct_3d_force_field(&f.bounds, &pos, &f.details);
        let err = gradient_error(&ff, &pos);
        assert!(err < 1e-3, "{}: {err}", mol.name);
        let dg = force_fields::construct_force_field(&f.bounds, 3, &f.chiral_centers, 1.0, 0.0, 0.0, 5.0);
        assert!(gradient_error(&dg, &pos) < 1e-3);
    }
}

#[test]
fn bounds_matrix_is_smoothed() {
    for mol in molecules() {
        let b = distgeom::bounds_matrix(&mol, &EmbedParams::etkdg_v3()).unwrap();
        assert!(b.check_valid(), "{}", mol.name);
        let n = b.num_rows();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if i != j && j != k && i != k {
                        assert!(b.upper(i, j) <= b.upper(i, k) + b.upper(k, j) + 1e-9, "{}", mol.name);
                    }
                }
            }
        }
    }
}
