//! Approximate nonbonded mode: cluster-pair kernels, pair-list maintenance,
//! periodic boundary conditions and particle-mesh Ewald.

mod common;

use common::water::{BondedWater, WaterFf, water_box};
use molecule_optimizer_rs::core::{Molecule, UnitCell};
use molecule_optimizer_rs::fast::approx::{ApproxNonbonded, build_approx_nonbonded};
use molecule_optimizer_rs::fast::backend::NonbondedBackend;
use molecule_optimizer_rs::fast::{ApproxOptions, Electrostatics, VdwModifier};
use molecule_optimizer_rs::ff::mmff::{DielectricModel, MmffBuilder, MmffMolProperties, MmffVariant};
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::ff::{ForceField, ForceFieldBuilder};
use molecule_optimizer_rs::io::molfile::read_sdf_lenient;
use molecule_optimizer_rs::optim::{Algorithm, MinimizeOptions, NoObserver, minimize};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::path::PathBuf;

const RF_INF: Electrostatics = Electrostatics::ReactionField { eps_rf: f64::INFINITY };

fn pme(rtol: f64, spacing: f64, order: usize) -> Electrostatics {
    Electrostatics::Pme { ewald_rtol: rtol, grid_spacing: spacing, order }
}

fn opts(cutoff: f64, buffer: f64, vdw: VdwModifier, el: Electrostatics) -> ApproxOptions {
    ApproxOptions { cutoff, buffer, vdw_modifier: vdw, electrostatics: el }
}

fn mmff_periodic() -> MmffBuilder {
    MmffBuilder { ignore_interfrag_interactions: false, ..MmffBuilder::default() }
}

fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-300)
}

/// max |a - b| / max |b|
fn grad_diff(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs())) / scale
}

/// A drug-sized MMFF fixture molecule (single fragment, ≥ 30 atoms).
fn fixture_molecule() -> Molecule {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/mmff/MMFF94_dative.sdf");
    let text = std::fs::read_to_string(path).unwrap();
    for m in read_sdf_lenient(&text).into_iter().flatten() {
        if m.num_atoms() >= 30
            && m.fragments().len() == 1
            && MmffMolProperties::new(&m, MmffVariant::Mmff94).map(|p| p.is_valid()).unwrap_or(false)
        {
            return m;
        }
    }
    panic!("no fixture molecule");
}

fn eval(b: &ApproxNonbonded, pos: &[f64]) -> (f64, f64, Vec<f64>) {
    let mut g = vec![0.0; pos.len()];
    let e = b.energy_grad(pos, &mut g);
    (e.vdw, e.electrostatic, g)
}

// ---------------------------------------------------------------------------
// kernels vs scalar reference
// ---------------------------------------------------------------------------

fn check_against_reference(mol: &Molecule, inner: &dyn std::any::Any, o: &ApproxOptions, tol: f64) {
    let pos = mol.flat_coords();
    for simd in [true, false] {
        let b = build_approx_nonbonded(mol, inner, o, simd).unwrap();
        let (v, e, g) = eval(&b, &pos);
        let mut gr = vec![0.0; pos.len()];
        let r = b.reference_energy_grad(&pos, Some(&mut gr));
        assert!(rel_diff(v, r.vdw) < tol, "{o:?} simd={simd}: vdw {v} vs {}", r.vdw);
        assert!(rel_diff(e, r.electrostatic) < tol || (e - r.electrostatic).abs() < 1e-12, "{o:?} simd={simd}: ele {e} vs {}", r.electrostatic);
        let gd = grad_diff(&g, &gr);
        println!(
            "{} {:?} simd={simd}: |dE_vdw|/E {:.1e}, |dE_ele|/E {:.1e}, max|dg|/max|g| {gd:.1e}",
            if mol.cell.is_some() { "periodic" } else { "isolated" },
            o.electrostatics,
            rel_diff(v, r.vdw),
            rel_diff(e, r.electrostatic),
        );
        assert!(gd < tol, "{o:?} simd={simd}: gradient rel diff {gd}");
        let en = b.energy(&pos);
        assert!(rel_diff(en.vdw, v) < 1e-14 && rel_diff(en.electrostatic, e) < 1e-14);
    }
}

#[test]
fn cluster_kernels_match_per_pair_reference_periodic() {
    let mol = water_box([5, 5, 5], 3.1, 1);
    let inner = mmff_periodic();
    let mods = [VdwModifier::PotentialShift, VdwModifier::Switch { r_switch: 5.0 }, VdwModifier::Switch { r_switch: 7.0 }];
    let els = [RF_INF, Electrostatics::ReactionField { eps_rf: 78.5 }, pme(1e-5, 0.9, 4)];
    for m in mods {
        for el in els {
            check_against_reference(&mol, &inner, &opts(7.0, 0.6, m, el), 1e-12);
        }
    }
    // triclinic cell
    let mut tri = mol.clone();
    tri.cell = Some(UnitCell::from_parameters(15.5, 15.5, 15.5, 80.0, 75.0, 100.0).unwrap());
    check_against_reference(&tri, &inner, &opts(6.5, 0.6, VdwModifier::PotentialShift, pme(1e-5, 0.9, 5)), 1e-12);
    check_against_reference(&tri, &inner, &opts(6.5, 0.6, VdwModifier::Switch { r_switch: 5.5 }, RF_INF), 1e-12);
}

#[test]
fn cluster_kernels_match_per_pair_reference_isolated() {
    let mol = fixture_molecule();
    let mmff = MmffBuilder::default();
    let mmff_dd = MmffBuilder { dielectric_model: DielectricModel::Distance, dielectric_constant: 4.0, ..MmffBuilder::default() };
    for inner in [&mmff, &mmff_dd] {
        for el in [Electrostatics::Full, RF_INF] {
            for m in [VdwModifier::PotentialShift, VdwModifier::Switch { r_switch: 3.5 }] {
                check_against_reference(&mol, inner, &opts(5.0, 0.5, m, el), 1e-12);
            }
        }
    }
    let uff = UffBuilder::default();
    for m in [VdwModifier::PotentialShift, VdwModifier::Switch { r_switch: 3.5 }] {
        check_against_reference(&mol, &uff, &opts(5.0, 0.5, m, RF_INF), 1e-12);
    }
}

// ---------------------------------------------------------------------------
// parity with the RDKit-exact contributions
// ---------------------------------------------------------------------------

fn exact_nonbonded(ff: &molecule_optimizer_rs::ff::ContribForceField, labels: &[&str], pos: &[f64]) -> (f64, Vec<f64>) {
    let mut e = 0.0;
    let mut g = vec![0.0; pos.len()];
    for c in ff.contribs() {
        if labels.contains(&c.label()) {
            e += c.energy(pos);
            c.grad(pos, &mut g);
        }
    }
    (e, g)
}

#[test]
fn isolated_full_matches_rdkit_exact_mmff() {
    let mol = fixture_molecule();
    let pos = mol.flat_coords();
    for inner in [
        MmffBuilder::default(),
        MmffBuilder { dielectric_model: DielectricModel::Distance, dielectric_constant: 4.0, ..MmffBuilder::default() },
    ] {
        let exact = inner.build(&mol, &[]).unwrap();
        let (ee, ge) = exact_nonbonded(&exact, &["Nonbonded"], &pos);
        // cutoff beyond the molecule size, no vdW modification
        let o = opts(60.0, 0.0, VdwModifier::Switch { r_switch: 60.0 }, Electrostatics::Full);
        for simd in [true, false] {
            let b = build_approx_nonbonded(&mol, &inner, &o, simd).unwrap();
            let (v, e, g) = eval(&b, &pos);
            println!("MMFF {:?}: exact {ee:.12} approx {:.12} (vdw {v:.6}, ele {e:.6})", inner.dielectric_model, v + e);
            assert!(rel_diff(v + e, ee) < 1e-12, "energy {} vs {ee}", v + e);
            assert!(grad_diff(&g, &ge) < 1e-12, "gradient {}", grad_diff(&g, &ge));
        }
    }
}

#[test]
fn isolated_matches_rdkit_exact_uff() {
    let mol = fixture_molecule();
    let pos = mol.flat_coords();
    let inner = UffBuilder::default();
    let exact = inner.build(&mol, &[]).unwrap();
    let (ee, ge) = exact_nonbonded(&exact, &["VdW"], &pos);
    let o = opts(60.0, 0.0, VdwModifier::Switch { r_switch: 60.0 }, Electrostatics::Full);
    for simd in [true, false] {
        let b = build_approx_nonbonded(&mol, &inner, &o, simd).unwrap();
        let (v, e, g) = eval(&b, &pos);
        println!("UFF: exact {ee:.12} approx {v:.12}");
        assert_eq!(e, 0.0);
        assert!(rel_diff(v, ee) < 1e-12, "energy {v} vs {ee}");
        assert!(grad_diff(&g, &ge) < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// pair-list maintenance
// ---------------------------------------------------------------------------

fn displace(pos: &[f64], max: f64, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let mut p = pos.to_vec();
    for a in p.chunks_mut(3) {
        // random direction, length < max
        loop {
            let d = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
            let l2: f64 = d.iter().map(|x| x * x).sum();
            if l2 <= 1.0 {
                for k in 0..3 {
                    a[k] += max * d[k];
                }
                break;
            }
        }
    }
    p
}

#[test]
fn pair_list_reuse_and_rebuild() {
    let mol = water_box([6, 6, 6], 3.1, 2);
    let inner = mmff_periodic();
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    for el in [RF_INF, pme(1e-5, 1.0, 4)] {
        let o = opts(8.0, 1.0, VdwModifier::PotentialShift, el);
        let b = build_approx_nonbonded(&mol, &inner, &o, true).unwrap();
        assert_eq!(b.num_builds(), 1);
        let pos = mol.flat_coords();
        for _ in 0..3 {
            let p = displace(&pos, 0.499, &mut rng);
            assert!(!b.needs_rebuild(&p));
            let (v, e, g) = eval(&b, &p);
            assert_eq!(b.num_builds(), 1, "no rebuild expected");
            let mut fresh_mol = mol.clone();
            fresh_mol.set_flat_coords(&p);
            let f = build_approx_nonbonded(&fresh_mol, &inner, &o, true).unwrap();
            let (v2, e2, g2) = eval(&f, &p);
            assert!(rel_diff(v, v2) < 1e-12 && rel_diff(e, e2) < 1e-12, "{v} {v2} {e} {e2}");
            assert!(grad_diff(&g, &g2) < 1e-12);
        }
        // larger moves trigger a rebuild, and the result is still exact
        let p = displace(&pos, 0.8, &mut rng);
        if b.max_displacement(&p) > 0.5 {
            let (v, e, _) = eval(&b, &p);
            assert_eq!(b.num_builds(), 2);
            let r = b.reference_energy_grad(&p, None);
            assert!(rel_diff(v, r.vdw) < 1e-12 && rel_diff(e, r.electrostatic) < 1e-12);
        }
        // on_accepted_step rebuilds proactively
        let p2 = displace(&p, 1.5, &mut rng);
        b.on_accepted_step(&p2);
        assert_eq!(b.num_builds(), 3);
        assert!(!b.needs_rebuild(&p2));
    }
}

#[test]
fn atoms_leaving_the_box_are_handled() {
    let mol = water_box([5, 5, 5], 3.1, 4);
    let inner = mmff_periodic();
    let o = opts(7.0, 0.8, VdwModifier::PotentialShift, pme(1e-5, 1.0, 4));
    let b = build_approx_nonbonded(&mol, &inner, &o, true).unwrap();
    let pos = mol.flat_coords();
    let (v0, e0, g0) = eval(&b, &pos);
    // translate whole molecules by lattice vectors: same energy
    let mut p = pos.clone();
    let l = 15.5;
    for (w, x) in p.chunks_mut(9).enumerate() {
        let s = [(w % 3) as f64 - 1.0, ((w / 3) % 3) as f64 - 1.0, 2.0 * ((w % 2) as f64)];
        for a in 0..3 {
            for d in 0..3 {
                x[3 * a + d] += s[d] * l;
            }
        }
    }
    let (v1, e1, g1) = eval(&b, &p);
    assert!(rel_diff(v0, v1) < 1e-12 && rel_diff(e0, e1) < 1e-9, "{v0} {v1} {e0} {e1}");
    assert!(grad_diff(&g1, &g0) < 1e-8);
}

// ---------------------------------------------------------------------------
// gradients vs finite differences
// ---------------------------------------------------------------------------

fn fd_check(b: &ApproxNonbonded, pos: &[f64], ncheck: usize, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut g = vec![0.0; pos.len()];
    b.energy_grad(pos, &mut g);
    let gmax = g.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let h = 1e-5;
    let mut p = pos.to_vec();
    for _ in 0..ncheck {
        let i = rng.gen_range(0..pos.len());
        let x = p[i];
        p[i] = x + h;
        let ep = b.energy(&p).total();
        p[i] = x - h;
        let em = b.energy(&p).total();
        p[i] = x;
        let fd = (ep - em) / (2.0 * h);
        assert!((fd - g[i]).abs() < 1e-5 * gmax.max(1.0), "{:?}: coord {i}: fd {fd} vs analytic {}", b.options(), g[i]);
    }
}

#[test]
fn gradients_match_finite_differences() {
    let water = water_box([4, 4, 4], 3.1, 5);
    let mut tri = water.clone();
    tri.cell = Some(UnitCell::from_parameters(12.4, 12.4, 12.4, 85.0, 80.0, 95.0).unwrap());
    let inner = mmff_periodic();
    let mods = [VdwModifier::PotentialShift, VdwModifier::Switch { r_switch: 4.0 }, VdwModifier::Switch { r_switch: 9.0 }];
    let periodic_els = [RF_INF, Electrostatics::ReactionField { eps_rf: 78.5 }, pme(1e-5, 0.8, 4), pme(1e-6, 1.0, 6)];
    for mol in [&water, &tri] {
        for m in mods {
            for el in periodic_els {
                let b = build_approx_nonbonded(mol, &inner, &opts(5.5, 0.5, m, el), true).unwrap();
                fd_check(&b, &mol.flat_coords(), 40, 7);
            }
        }
    }
    // isolated systems: Full and reaction field, both dielectric models, UFF
    let mol = fixture_molecule();
    let dd = MmffBuilder { dielectric_model: DielectricModel::Distance, ..MmffBuilder::default() };
    for inner in [MmffBuilder::default(), dd] {
        for m in mods {
            for el in [Electrostatics::Full, RF_INF, Electrostatics::ReactionField { eps_rf: 4.0 }] {
                let b = build_approx_nonbonded(&mol, &inner, &opts(5.5, 0.5, m, el), true).unwrap();
                fd_check(&b, &mol.flat_coords(), 40, 8);
            }
        }
    }
    for m in mods {
        let b = build_approx_nonbonded(&mol, &UffBuilder::default(), &opts(5.5, 0.5, m, RF_INF), true).unwrap();
        fd_check(&b, &mol.flat_coords(), 40, 9);
    }
}

#[test]
fn reaction_field_vanishes_smoothly_at_cutoff() {
    // two waters (isolated); the second one is pulled away rigidly along x.
    // With eps_rf = inf and a vdW switch, energy and force go to zero
    // continuously at the cutoff.
    let mut mol = water_box([2, 1, 1], 3.1, 6);
    mol.cell = None;
    let inner = mmff_periodic();
    let rc = 6.0;
    let b = build_approx_nonbonded(&mol, &inner, &opts(rc, 0.3, VdwModifier::Switch { r_switch: 4.5 }, RF_INF), true).unwrap();
    let base = mol.flat_coords();
    let energy_at = |dx: f64| {
        let mut p = base.clone();
        for a in 3..6 {
            p[3 * a] += dx;
        }
        let mut g = vec![0.0; p.len()];
        let e = b.energy_grad(&p, &mut g).total();
        (e, g.iter().fold(0.0f64, |m, x| m.max(x.abs())))
    };
    // start beyond the first-neighbor distance (all atom pairs > 3.5 Å)
    let mut x = 1.5;
    let mut last = energy_at(x).0;
    let step = 0.002;
    let mut max_jump: f64 = 0.0;
    while x < 8.0 {
        x += step;
        let (e, _) = energy_at(x);
        max_jump = max_jump.max((e - last).abs());
        last = e;
    }
    let (e_far, g_far) = energy_at(8.0);
    assert_eq!(e_far, 0.0);
    assert_eq!(g_far, 0.0);
    // the largest step change is consistent with a continuous energy
    assert!(max_jump < 5e-3, "max energy change per {step} Å step: {max_jump}");
}

// ---------------------------------------------------------------------------
// PME
// ---------------------------------------------------------------------------

/// High-precision direct Ewald sum of the Coulomb energy (kcal/mol) of point
/// charges in an orthorhombic box, excluding the listed pairs (n = 0 image).
/// Pairs in `scaled14` have their (n = 0) Coulomb energy scaled by 0.75.
fn direct_ewald(pos: &[[f64; 3]], q: &[f64], l: [f64; 3], excluded: &[(usize, usize)], scaled14: &[(usize, usize)]) -> f64 {
    let c = 332.0716;
    let n = q.len();
    let beta = 0.45;
    let rcut: f64 = 12.0;
    let erfc = molecule_optimizer_rs::fast::approx::erfc::erfc;
    let mut real = 0.0;
    let nimg = [0, 1, 2].map(|d| (rcut / l[d]).ceil() as i64 + 1);
    let is_excl = |i: usize, j: usize| excluded.iter().any(|&(a, b)| (a == i && b == j) || (a == j && b == i));
    for i in 0..n {
        for j in 0..n {
            for a in -nimg[0]..=nimg[0] {
                for b in -nimg[1]..=nimg[1] {
                    for k in -nimg[2]..=nimg[2] {
                        let zero = a == 0 && b == 0 && k == 0;
                        if zero && (i == j || is_excl(i, j)) {
                            continue;
                        }
                        let d = [
                            pos[i][0] - pos[j][0] + a as f64 * l[0],
                            pos[i][1] - pos[j][1] + b as f64 * l[1],
                            pos[i][2] - pos[j][2] + k as f64 * l[2],
                        ];
                        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        if r < rcut {
                            real += 0.5 * q[i] * q[j] * erfc(beta * r) / r;
                        }
                    }
                }
            }
        }
    }
    let v = l[0] * l[1] * l[2];
    let pi = std::f64::consts::PI;
    let mmax = [0, 1, 2].map(|d| (6.5 * beta * l[d] / pi).ceil() as i64);
    let mut recip = 0.0;
    for m1 in -mmax[0]..=mmax[0] {
        for m2 in -mmax[1]..=mmax[1] {
            for m3 in -mmax[2]..=mmax[2] {
                if m1 == 0 && m2 == 0 && m3 == 0 {
                    continue;
                }
                let m = [m1 as f64 / l[0], m2 as f64 / l[1], m3 as f64 / l[2]];
                let m2v = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
                let (mut sr, mut si) = (0.0, 0.0);
                for i in 0..n {
                    let arg = 2.0 * pi * (m[0] * pos[i][0] + m[1] * pos[i][1] + m[2] * pos[i][2]);
                    sr += q[i] * arg.cos();
                    si += q[i] * arg.sin();
                }
                recip += (-pi * pi * m2v / (beta * beta)).exp() / m2v * (sr * sr + si * si);
            }
        }
    }
    recip /= 2.0 * pi * v;
    let selfe = -beta / pi.sqrt() * q.iter().map(|x| x * x).sum::<f64>();
    let mut excl = 0.0;
    for &(i, j) in excluded {
        let d = [0, 1, 2].map(|k| pos[i][k] - pos[j][k]);
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        excl -= q[i] * q[j] * (1.0 - erfc(beta * r)) / r;
    }
    for &(i, j) in scaled14 {
        let d = [0, 1, 2].map(|k| pos[i][k] - pos[j][k]);
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        excl -= 0.25 * q[i] * q[j] / r;
    }
    let qt: f64 = q.iter().sum();
    let net = -pi * qt * qt / (2.0 * v * beta * beta);
    c * (real + recip + selfe + excl + net)
}

fn water_charges(mol: &Molecule) -> Vec<f64> {
    MmffMolProperties::new(mol, MmffVariant::Mmff94).unwrap().partial_charges().to_vec()
}

fn water_exclusions(nwat: usize) -> Vec<(usize, usize)> {
    (0..nwat).flat_map(|w| [(3 * w, 3 * w + 1), (3 * w, 3 * w + 2), (3 * w + 1, 3 * w + 2)]).collect()
}

/// Magnitude of the Ewald self energy `C β/√π Σ q²`: the natural scale of
/// the reciprocal-space energy, used to express PME errors.
fn ewald_scale(b: &ApproxNonbonded, q: &[f64]) -> f64 {
    332.0716 * b.ewald_beta().unwrap() / std::f64::consts::PI.sqrt() * q.iter().map(|x| x * x).sum::<f64>()
}

#[test]
fn pme_matches_direct_ewald_sum() {
    let mol = water_box([4, 4, 4], 3.1, 11);
    let q = water_charges(&mol);
    let l = [12.4; 3];
    let reference = direct_ewald(&mol.coords, &q, l, &water_exclusions(64), &[]);
    let inner = mmff_periodic();
    let pos = mol.flat_coords();
    println!("direct Ewald reference: {reference:.10} kcal/mol");
    for (rtol, spacing, order, tol) in [(1e-5, 0.8, 4, 2e-5), (1e-6, 0.35, 8, 2e-6), (1e-8, 0.3, 10, 2e-8), (1e-10, 0.25, 12, 1e-9)] {
        let b = build_approx_nonbonded(&mol, &inner, &opts(6.0, 0.5, VdwModifier::PotentialShift, pme(rtol, spacing, order)), true).unwrap();
        let e = b.energy(&pos).electrostatic;
        let scale = ewald_scale(&b, &q);
        let err = (e - reference).abs() / scale;
        println!(
            "PME rtol={rtol:e} spacing={spacing} order={order} grid={:?} beta={:.5}: E={e:.10} |dE|={:.3e} |dE|/E={:.3e} |dE|/E_self={err:.3e}",
            b.pme_grid().unwrap(),
            b.ewald_beta().unwrap(),
            (e - reference).abs(),
            rel_diff(e, reference),
        );
        assert!(err < tol, "relative error {err} (tol {tol})");
    }
}

/// PME with MMFF: plain Coulomb `332.0716 qi qj / r` (no 0.05 Å buffer),
/// 1-2/1-3 pairs excluded, 1-4 pairs scaled by 0.75. Checked against a
/// direct Ewald sum for a drug-like molecule in a periodic box.
#[test]
fn pme_mmff_molecule_plain_coulomb_with_scaled_14() {
    let mut mol = fixture_molecule();
    let q = water_charges(&mol);
    let topo = mol.topological_distances();
    let n = mol.num_atoms();
    let (mut excl, mut p14) = (Vec::new(), Vec::new());
    for (i, row) in topo.iter().enumerate() {
        for (j, &d) in row.iter().enumerate().skip(i + 1) {
            match d {
                1 | 2 => excl.push((i, j)),
                3 => p14.push((i, j)),
                _ => {}
            }
        }
    }
    assert!(!p14.is_empty());
    // center the molecule in a 24 Å box
    let mut c = [0.0; 3];
    for p in &mol.coords {
        for d in 0..3 {
            c[d] += p[d] / n as f64;
        }
    }
    for p in mol.coords.iter_mut() {
        for d in 0..3 {
            p[d] += 12.0 - c[d];
        }
    }
    let l = 24.0;
    mol.cell = Some(UnitCell::cubic(l).unwrap());
    let reference = direct_ewald(&mol.coords, &q, [l; 3], &excl, &p14);
    let b = build_approx_nonbonded(&mol, &MmffBuilder::default(), &opts(11.0, 0.5, VdwModifier::PotentialShift, pme(1e-10, 0.3, 10)), true).unwrap();
    let e = b.energy(&mol.flat_coords()).electrostatic;
    let err = (e - reference).abs() / ewald_scale(&b, &q);
    println!("molecule in box: PME {e:.10} vs direct Ewald {reference:.10} (|dE|/E_self {err:.2e})");
    assert!(err < 1e-9, "{e} vs {reference}");
    // the buffered isolated energy differs (as documented)
    let mut iso = mol.clone();
    iso.cell = None;
    let full = build_approx_nonbonded(&iso, &MmffBuilder::default(), &opts(60.0, 0.0, VdwModifier::PotentialShift, Electrostatics::Full), true).unwrap();
    println!("isolated buffered (r + 0.05) electrostatics: {:.6}", full.energy(&iso.flat_coords()).electrostatic);
}

#[test]
fn pme_converges_with_grid_refinement() {
    let mol = water_box([4, 4, 4], 3.1, 12);
    let q = water_charges(&mol);
    let inner = mmff_periodic();
    let pos = mol.flat_coords();
    let fine = build_approx_nonbonded(&mol, &inner, &opts(6.0, 0.5, VdwModifier::PotentialShift, pme(1e-6, 0.25, 10)), true).unwrap();
    let r = fine.pme_reciprocal_energy(&pos);
    let scale = ewald_scale(&fine, &q);
    for order in [4, 6] {
        let mut last = f64::INFINITY;
        for spacing in [1.5, 1.0, 0.7, 0.5] {
            let b = build_approx_nonbonded(&mol, &inner, &opts(6.0, 0.5, VdwModifier::PotentialShift, pme(1e-6, spacing, order)), true).unwrap();
            let err = (b.pme_reciprocal_energy(&pos) - r).abs() / scale;
            println!("order {order} spacing {spacing}: grid {:?} |E_rec - E_rec(fine)|/E_self = {err:.3e}", b.pme_grid().unwrap());
            assert!(err < last, "no convergence at spacing {spacing}");
            last = err;
        }
        assert!(last < 1e-4);
    }
}

#[test]
fn pme_translation_invariance() {
    let mol = water_box([4, 4, 4], 3.1, 13);
    let q = water_charges(&mol);
    let inner = mmff_periodic();
    let b = build_approx_nonbonded(&mol, &inner, &opts(6.0, 0.5, VdwModifier::PotentialShift, pme(1e-6, 0.6, 6)), true).unwrap();
    let scale = ewald_scale(&b, &q);
    let pos = mol.flat_coords();
    let (v0, e0, g0) = eval(&b, &pos);
    for t in [[0.37, -1.1, 2.3], [0.05, 0.05, 0.05], [6.2, 0.0, -3.1]] {
        let p: Vec<f64> = pos.iter().enumerate().map(|(i, x)| x + t[i % 3]).collect();
        let (v, e, g) = eval(&b, &p);
        println!("translation {t:?}: |dE|/E_self = {:.3e}, gradient rel. change {:.3e}", (e - e0).abs() / scale, grad_diff(&g, &g0));
        assert!(rel_diff(v, v0) < 1e-11);
        assert!((e - e0).abs() < 1e-6 * scale, "translation {t:?}: {e} vs {e0}");
        assert!(grad_diff(&g, &g0) < 1e-3);
    }
}

#[test]
fn results_do_not_depend_on_thread_count() {
    let mol = water_box([6, 6, 6], 3.1, 14);
    let inner = mmff_periodic();
    let pos = mol.flat_coords();
    for el in [RF_INF, pme(1e-5, 0.9, 4)] {
        for simd in [true, false] {
            let o = opts(8.0, 1.0, VdwModifier::Switch { r_switch: 6.5 }, el);
            let run = |threads: usize| {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                pool.install(|| {
                    let b = build_approx_nonbonded(&mol, &inner, &o, simd).unwrap();
                    eval(&b, &pos)
                })
            };
            let r1 = run(1);
            for t in [2, 3, 8] {
                let r = run(t);
                assert_eq!(r1.0.to_bits(), r.0.to_bits(), "vdw differs with {t} threads");
                assert_eq!(r1.1.to_bits(), r.1.to_bits(), "electrostatics differ with {t} threads");
                assert!(r1.2.iter().zip(&r.2).all(|(a, b)| a.to_bits() == b.to_bits()), "gradient differs with {t} threads");
            }
        }
    }
}

#[test]
fn invalid_settings_are_rejected() {
    let mol = water_box([4, 4, 4], 3.1, 15);
    let inner = mmff_periodic();
    // cutoff > half box
    assert!(build_approx_nonbonded(&mol, &inner, &opts(6.3, 0.5, VdwModifier::PotentialShift, RF_INF), true).is_err());
    // full electrostatics with PBC
    assert!(build_approx_nonbonded(&mol, &inner, &opts(6.0, 0.5, VdwModifier::PotentialShift, Electrostatics::Full), true).is_err());
    // PME needs a cell
    let mut iso = mol.clone();
    iso.cell = None;
    assert!(build_approx_nonbonded(&iso, &inner, &opts(6.0, 0.5, VdwModifier::PotentialShift, pme(1e-5, 1.0, 4)), true).is_err());
    // PME with distance-dependent dielectric
    let dd = MmffBuilder { dielectric_model: DielectricModel::Distance, ..mmff_periodic() };
    assert!(build_approx_nonbonded(&mol, &dd, &opts(6.0, 0.5, VdwModifier::PotentialShift, pme(1e-5, 1.0, 4)), true).is_err());
    // PME cannot ignore interfragment interactions
    assert!(build_approx_nonbonded(&mol, &MmffBuilder::default(), &opts(6.0, 0.5, VdwModifier::PotentialShift, pme(1e-5, 1.0, 4)), true).is_err());
    // but reaction field can (and then waters do not interact)
    let b = build_approx_nonbonded(&mol, &MmffBuilder::default(), &opts(6.0, 0.5, VdwModifier::PotentialShift, RF_INF), true).unwrap();
    assert_eq!(b.energy(&mol.flat_coords()).total(), 0.0);
    // unsupported builder
    assert!(build_approx_nonbonded(&mol, &3usize, &ApproxOptions::default(), true).is_err());
}

// ---------------------------------------------------------------------------
// minimization of water boxes
// ---------------------------------------------------------------------------

fn minimize_water(n: usize, el: Electrostatics, cutoff: f64, iters: usize) {
    let mol = water_box([n, n, n], 3.1, 20 + n as u64);
    let inner = mmff_periodic();
    let nb = build_approx_nonbonded(&mol, &inner, &opts(cutoff, 1.0, VdwModifier::PotentialShift, el), true).unwrap();
    let ff = WaterFf { bonded: BondedWater::new(n * n * n), nb };
    let mut pos = mol.flat_coords();
    let mut g = vec![0.0; pos.len()];
    let e0 = ff.energy_grad(&pos, &mut g);
    let rms0 = (g.iter().map(|x| x * x).sum::<f64>() / g.len() as f64).sqrt();
    let o = MinimizeOptions { max_iters: iters, algorithm: Algorithm::Lbfgs(10), ..MinimizeOptions::default() };
    let t = std::time::Instant::now();
    let res = minimize(&ff, &mut pos, &o, &mut NoObserver);
    println!(
        "{} waters, {:?}: E {e0:.2} -> {:.2} kcal/mol, grad rms {rms0:.3} -> {:.4}, {} iterations, {} list builds, {:.2?}",
        n * n * n,
        el,
        res.energy,
        res.grad_rms,
        res.iterations,
        ff.nb.num_builds(),
        t.elapsed()
    );
    assert!(res.energy.is_finite() && res.energy < e0);
    assert!(res.grad_rms < 0.5 * rms0);
    // energy per water in a sensible range (no collapse)
    let per = res.energy / (n * n * n) as f64;
    assert!(per > -30.0 && per < 0.0, "energy per water {per}");
    // the final energy agrees with a freshly built backend and with the reference sum
    let mut m2 = mol.clone();
    m2.set_flat_coords(&pos);
    let fresh = build_approx_nonbonded(&m2, &inner, ff.nb.options(), true).unwrap();
    let a = ff.nb.energy(&pos).total();
    let b = fresh.energy(&pos).total();
    assert!(rel_diff(a, b) < 1e-11, "{a} vs {b}");
    // bonded geometry stays reasonable
    for w in 0..n * n * n {
        for h in 1..3 {
            let d: f64 = (0..3).map(|k| (pos[9 * w + 3 * h + k] - pos[9 * w + k]).powi(2)).sum::<f64>().sqrt();
            assert!((0.85..1.1).contains(&d), "O-H distance {d}");
        }
    }
}

#[test]
fn water_box_216_minimizes_with_pme() {
    minimize_water(6, pme(1e-5, 1.0, 4), 8.0, 300);
}

#[test]
fn water_box_216_minimizes_with_reaction_field() {
    minimize_water(6, RF_INF, 8.0, 300);
}

#[test]
fn water_box_1728_minimizes() {
    minimize_water(12, pme(1e-5, 1.0, 4), 9.0, 200);
    minimize_water(12, RF_INF, 9.0, 200);
}
