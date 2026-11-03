//! Soft-core intermolecular force field: analytic gradients and dE/dλ
//! against finite differences, finiteness at r → 0, and λ = 1 exactness
//! against the plain MMFF / UFF builders with interfragment interactions.

use molecule_optimizer_rs::Molecule;
use molecule_optimizer_rs::calc::assembly::combine_molecules;
use molecule_optimizer_rs::ff::mmff::{DielectricModel, MmffBuilder, MmffVariant};
use molecule_optimizer_rs::ff::softcore::{Confinement, SoftCoreBuilder, SoftCoreInner, SoftCoreParams};
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::ff::{ForceField, ForceFieldBuilder, numerical_gradient};
use molecule_optimizer_rs::io;

const LAMBDAS: [f64; 5] = [0.0, 0.1, 0.5, 0.9, 1.0];

fn monomer(name: &str) -> Molecule {
    let p = format!("{}/tests/data/assembly/{name}.sdf", env!("CARGO_MANIFEST_DIR"));
    io::read_file(p).unwrap().remove(0)
}

/// `m` and a copy rotated about z by `angle` (rad) and shifted by `shift`.
fn dimer(m: &Molecule, angle: f64, shift: [f64; 3]) -> Molecule {
    let mut b = m.clone();
    let (s, c) = angle.sin_cos();
    for p in &mut b.coords {
        *p = [c * p[0] - s * p[1] + shift[0], s * p[0] + c * p[1] + shift[1], p[2] + shift[2]];
    }
    combine_molecules(&[(m.clone(), 1), (b, 1)])
}

fn systems() -> Vec<(&'static str, Molecule)> {
    vec![
        ("water dimer", dimer(&monomer("water"), 1.9, [2.1, 1.3, 0.8])),
        ("acetic acid dimer", dimer(&monomer("acetic_acid"), 3.0, [0.5, 0.3, 3.0])),
        ("benzene dimer", dimer(&monomer("benzene"), 0.4, [0.6, 0.3, 3.6])),
        (
            "benzene/water",
            combine_molecules(&[(monomer("benzene"), 1), (shifted(&monomer("water"), [0.2, -0.1, 3.1]), 1)]),
        ),
    ]
}

fn shifted(m: &Molecule, d: [f64; 3]) -> Molecule {
    let mut b = m.clone();
    for p in &mut b.coords {
        *p = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
    }
    b
}

fn rel_close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

fn check_exact<B: SoftCoreInner>(inner: B, reference: B, mol: &Molecule, what: &str) {
    let refff = reference.build(mol, &[]).unwrap();
    let sc = SoftCoreBuilder::new(inner);
    let ff = sc.build(mol, &[]).unwrap();
    let pos = mol.flat_coords();
    let (mut g1, mut g2) = (vec![0.0; pos.len()], vec![0.0; pos.len()]);
    let e1 = refff.energy_grad(&pos, &mut g1);
    let e2 = ff.energy_grad(&pos, &mut g2);
    assert!(rel_close(e1, e2, 1e-12), "{what}: E {e1} vs {e2}");
    assert!(rel_close(e1, ff.energy(&pos), 1e-12), "{what}");
    let gmax = g1.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for (k, (a, b)) in g1.iter().zip(&g2).enumerate() {
        assert!((a - b).abs() <= 1e-12 * gmax.max(1.0), "{what}: grad {k}: {a} vs {b}");
    }
    // the breakdown labels merge with the inner force field's
    let mut b1 = refff.energy_breakdown(&pos);
    let mut b2 = ff.energy_breakdown(&pos);
    b1.sort_by(|x, y| x.0.cmp(&y.0));
    b2.sort_by(|x, y| x.0.cmp(&y.0));
    assert_eq!(b1.len(), b2.len(), "{what}: {b1:?} vs {b2:?}");
    for (x, y) in b1.iter().zip(&b2) {
        assert_eq!(x.0, y.0);
        assert!(rel_close(x.1, y.1, 1e-11), "{what}: {} {} vs {}", x.0, x.1, y.1);
    }
    // and it is a genuine intermolecular contribution
    let intra = inner_only(&sc, mol);
    assert!((intra - e1).abs() > 1e-6, "{what}: no intermolecular energy");
}

fn inner_only<B: SoftCoreInner>(sc: &SoftCoreBuilder<B>, mol: &Molecule) -> f64 {
    sc.inner.with_interfrag(false).build(mol, &[]).unwrap().energy(&mol.flat_coords())
}

#[test]
fn lambda_one_equals_mmff_with_interfragment_terms() {
    for (name, mol) in systems() {
        for (variant, model, d) in [
            (MmffVariant::Mmff94, DielectricModel::Constant, 1.0),
            (MmffVariant::Mmff94s, DielectricModel::Constant, 1.0),
            (MmffVariant::Mmff94, DielectricModel::Distance, 4.0),
        ] {
            let b = MmffBuilder { variant, dielectric_model: model, dielectric_constant: d, ..Default::default() };
            let r = MmffBuilder { ignore_interfrag_interactions: false, ..b.clone() };
            check_exact(b, r, &mol, &format!("MMFF {name} {variant:?} {model:?}"));
        }
    }
}

#[test]
fn lambda_one_equals_uff_with_interfragment_terms() {
    for (name, mol) in systems() {
        let b = UffBuilder::default();
        let r = UffBuilder { ignore_interfrag_interactions: false, ..b.clone() };
        check_exact(b, r, &mol, &format!("UFF {name}"));
    }
}

/// The build-time distance filter of the inner builders is reproduced.
#[test]
fn lambda_one_respects_build_time_cutoffs() {
    let mol = dimer(&monomer("acetic_acid"), 0.7, [5.0, 0.5, 0.0]);
    let m = MmffBuilder { nonbonded_thresh: 5.0, ..Default::default() };
    check_exact(m.clone(), MmffBuilder { ignore_interfrag_interactions: false, ..m }, &mol, "MMFF thresh 5");
    let u = UffBuilder { vdw_thresh: 1.6, ..Default::default() };
    check_exact(u.clone(), UffBuilder { ignore_interfrag_interactions: false, ..u }, &mol, "UFF thresh 1.6");
}

fn params_set() -> Vec<SoftCoreParams> {
    vec![
        SoftCoreParams::default(),
        SoftCoreParams { alpha_vdw: 0.3, alpha_ele: 2.5, p: 2.0, n: 2.0, include_all_pairs: true },
        SoftCoreParams { alpha_vdw: 1.0, alpha_ele: 0.5, p: 1.0, n: 3.0, include_all_pairs: false },
    ]
}

fn check_derivatives<B: SoftCoreInner>(inner: B, mol: &Molecule, what: &str, lambdas: &[f64]) {
    for params in params_set() {
        let sc = SoftCoreBuilder::new(inner.clone())
            .with_params(params)
            .with_confinement(Some(Confinement { radius: 0.4, force_constant: 5.0 }));
        let (ff, h) = sc.build_with_handle(mol, &[]).unwrap();
        let pos = mol.flat_coords();
        for &lv in lambdas {
            for &le in lambdas {
                h.set_lambda(lv, le);
                // coordinates
                let mut ga = vec![0.0; pos.len()];
                ff.energy_grad(&pos, &mut ga);
                let gn = numerical_gradient(&ff, &pos, 1e-6);
                for k in 0..pos.len() {
                    assert!(
                        (ga[k] - gn[k]).abs() <= 1e-5 * (1.0 + gn[k].abs()),
                        "{what} {params:?} λ=({lv},{le}) coord {k}: {} vs {}",
                        ga[k],
                        gn[k]
                    );
                }
                // λ: second-order differences, one-sided at the ends
                let (dv, de) = h.de_dlambda(&pos);
                let e_at = |v: f64, e: f64| {
                    h.set_lambda(v, e);
                    ff.energy(&pos)
                };
                let fd = |f: &dyn Fn(f64) -> f64, x: f64| {
                    let s = 1e-5;
                    if x < s {
                        (-3.0 * f(x) + 4.0 * f(x + s) - f(x + 2.0 * s)) / (2.0 * s)
                    } else if x > 1.0 - s {
                        (3.0 * f(x) - 4.0 * f(x - s) + f(x - 2.0 * s)) / (2.0 * s)
                    } else {
                        (f(x + s) - f(x - s)) / (2.0 * s)
                    }
                };
                let nv = fd(&|x| e_at(x, le), lv);
                let ne = fd(&|x| e_at(lv, x), le);
                h.set_lambda(lv, le);
                assert!(
                    (dv - nv).abs() <= 1e-5 * (1.0 + nv.abs()),
                    "{what} {params:?} λ=({lv},{le}) dE/dλ_vdw {dv} vs {nv}"
                );
                assert!(
                    (de - ne).abs() <= 1e-5 * (1.0 + ne.abs()),
                    "{what} {params:?} λ=({lv},{le}) dE/dλ_ele {de} vs {ne}"
                );
                // energy breakdown consistency
                let (ev, ee) = h.inter_energy(&pos);
                if lv == 0.0 && le == 0.0 {
                    assert_eq!((ev, ee), (0.0, 0.0));
                }
            }
        }
    }
}

#[test]
fn gradients_and_lambda_derivatives_match_finite_differences() {
    for (name, mol) in systems() {
        check_derivatives(MmffBuilder::default(), &mol, &format!("MMFF {name}"), &LAMBDAS);
        check_derivatives(UffBuilder::default(), &mol, &format!("UFF {name}"), &LAMBDAS);
    }
    // overlapping fragments (soft-core regime; at λ_vdw = 1 the hard core
    // makes ∂E/∂λ ~ 1e7 and too stiff for finite differences)
    let close = dimer(&monomer("water"), 2.5, [0.9, 0.4, 0.3]);
    check_derivatives(MmffBuilder::default(), &close, "MMFF overlapping waters", &LAMBDAS[..4]);
    check_derivatives(UffBuilder::default(), &close, "UFF overlapping waters", &LAMBDAS[..4]);
}

#[test]
fn finite_at_zero_distance_below_lambda_one() {
    // the O atoms of the two waters coincide: r = 0 for an O-O pair
    let w = monomer("water");
    let o = w.coords[0];
    let mut b = w.clone();
    let (s, c) = (1.3f64).sin_cos();
    for p in &mut b.coords {
        let d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
        *p = [o[0] + c * d[0] - s * d[2], o[1] + d[1], o[2] + s * d[0] + c * d[2]];
    }
    let mol = combine_molecules(&[(w, 1), (b, 1)]);
    let pos = mol.flat_coords();
    for params in params_set() {
        for inner in [0, 1] {
            let (ff, h) = if inner == 0 {
                SoftCoreBuilder::new(MmffBuilder::default()).with_params(params).build_with_handle(&mol, &[]).unwrap()
            } else {
                SoftCoreBuilder::new(UffBuilder::default()).with_params(params).build_with_handle(&mol, &[]).unwrap()
            };
            for &l in &[0.0, 0.1, 0.5, 0.9, 0.99] {
                h.set_lambda(l, l);
                let mut g = vec![0.0; pos.len()];
                let e = ff.energy_grad(&pos, &mut g);
                let (dv, de) = h.de_dlambda(&pos);
                assert!(e.is_finite() && g.iter().all(|x| x.is_finite()), "inner {inner} λ={l}: E={e}");
                assert!(dv.is_finite() && de.is_finite());
                let (ev, ee) = h.inter_energy(&pos);
                assert!(ev.is_finite() && ee.is_finite());
                // soft: no r^-12 / r^-7 singularity at the coinciding pair
                if l < 0.95 {
                    assert!(e < 1e6, "inner {inner} λ={l}: E={e} not soft");
                }
            }
        }
    }
}

#[test]
fn lambda_changes_without_rebuilding_and_shared_state() {
    let mol = dimer(&monomer("water"), 1.9, [2.1, 1.3, 0.8]);
    let pos = mol.flat_coords();
    let sc = SoftCoreBuilder::new(MmffBuilder::default());
    let ff = sc.build(&mol, &[]).unwrap();
    let full = ff.energy(&pos);
    sc.set_lambda(0.0, 0.0);
    let none = ff.energy(&pos);
    let intra = inner_only(&sc, &mol);
    assert!(rel_close(none, intra, 1e-12), "{none} vs {intra}");
    assert!((full - none).abs() > 1e-3);
    // a handle-built force field has its own state
    let (ff2, h2) = sc.build_with_handle(&mol, &[]).unwrap();
    assert_eq!(h2.lambda(), (0.0, 0.0));
    h2.set_lambda(1.0, 1.0);
    assert!(rel_close(ff2.energy(&pos), full, 1e-12));
    assert!(rel_close(ff.energy(&pos), none, 1e-12));
    assert_eq!(h2.fragments().len(), 2);
    assert_eq!(h2.pairs().len(), 9);
    assert_eq!(sc.name(), "MMFF94+SoftCore");
}
