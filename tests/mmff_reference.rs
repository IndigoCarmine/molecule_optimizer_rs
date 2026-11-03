//! MMFF94 / MMFF94s parity with RDKit on the MMFF validation suite.
//!
//! Reference data (`tests/data/mmff/*.ref`) is produced by
//! `tools/gen_mmff_reference.py` with RDKit 2026.09.1 from the SDF files in
//! the same directory (see that script for the format).

use molecule_optimizer_rs::Molecule;
use molecule_optimizer_rs::ff::mmff::{DielectricModel, MmffBuilder, MmffMolProperties, MmffVariant};
use molecule_optimizer_rs::ff::{ForceField, ForceFieldBuilder};
use molecule_optimizer_rs::io::molfile::read_sdf_lenient;
use molecule_optimizer_rs::optim::{MinimizeOptions, NoObserver, minimize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Default, Debug)]
struct Ref {
    name: String,
    n: usize,
    status: String,
    types: Vec<u8>,
    charges: Vec<f64>,
    energy: Vec<f64>,
    extra: Vec<f64>,
    grad: Vec<f64>,
    o10: (i32, f64, Vec<f64>),
    o200: (i32, f64, Vec<f64>),
    pert: Vec<f64>,
    pert_e: f64,
    po200: (i32, f64, Vec<f64>),
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/mmff")
}

fn floats(it: &[&str]) -> Vec<f64> {
    it.iter().map(|s| s.parse::<f64>().unwrap()).collect()
}

fn parse_ref(text: &str) -> Vec<Ref> {
    let mut out: Vec<Ref> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.is_empty() {
            continue;
        }
        if f[0] == "MOL" {
            out.push(Ref { name: f[1].into(), n: f[2].parse().unwrap(), status: f[3].into(), ..Default::default() });
            continue;
        }
        let r = out.last_mut().unwrap();
        let opt = |f: &[&str]| (f[1].parse::<i32>().unwrap(), f[2].parse::<f64>().unwrap(), floats(&f[3..]));
        match f[0] {
            "T" => r.types = f[1..].iter().map(|s| s.parse().unwrap()).collect(),
            "Q" => r.charges = floats(&f[1..]),
            "E" => r.energy = floats(&f[1..]),
            "X" => r.extra = floats(&f[1..]),
            "G" => r.grad = floats(&f[1..]),
            "O10" => r.o10 = opt(&f),
            "O200" => r.o200 = opt(&f),
            "P" => r.pert = floats(&f[1..]),
            "PE" => r.pert_e = f[1].parse().unwrap(),
            "PO200" => r.po200 = opt(&f),
            _ => panic!("unknown tag {}", f[0]),
        }
    }
    out
}

fn heavy_rmsd(mol: &Molecule, a: &[f64], b: &[f64]) -> f64 {
    let heavy = mol.heavy_atoms();
    let mut s = 0.0;
    for &i in &heavy {
        for d in 0..3 {
            let x = a[3 * i + d] - b[3 * i + d];
            s += x * x;
        }
    }
    (s / heavy.len().max(1) as f64).sqrt()
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(1.0)
}

#[derive(Default, Debug)]
struct Stats {
    n: usize,
    type_mismatch: Vec<String>,
    charge_mismatch: Vec<String>,
    max_charge_err: f64,
    energy_mismatch: Vec<String>,
    max_energy_rel: f64,
    max_term_rel: f64,
    max_extra_rel: f64,
    max_grad_err: f64,
    o10_max_rmsd: f64,
    o200_bad: Vec<String>,
    o200_max_rmsd: f64,
    o200_max_de: f64,
    po200_bad: Vec<String>,
    po200_max_rmsd: f64,
    po200_max_de: f64,
    po200_exact: usize,
}

const TERM_LABELS: [&str; 7] = ["BondStretch", "AngleBend", "StretchBend", "OopBend", "Torsion", "VdW", "Electrostatic"];

fn run_set(sdf: &str, reff: &str, variant: MmffVariant) -> Stats {
    let sdf_text = std::fs::read_to_string(data_dir().join(sdf)).unwrap();
    let ref_text = std::fs::read_to_string(data_dir().join(reff)).unwrap();
    let mols = read_sdf_lenient(&sdf_text);
    let refs = parse_ref(&ref_text);
    assert_eq!(mols.len(), refs.len(), "{sdf}: record count");
    let mut st = Stats::default();
    let builder = MmffBuilder::new(variant);
    let opts = MinimizeOptions::default();
    for (m, r) in mols.into_iter().zip(&refs) {
        if r.status != "ok" {
            continue;
        }
        let mol = m.unwrap_or_else(|e| panic!("{sdf} {}: read error {e}", r.name));
        assert_eq!(mol.num_atoms(), r.n);
        st.n += 1;
        let props = MmffMolProperties::new(&mol, variant).unwrap();
        if props.atom_types() != r.types.as_slice() {
            let diff: Vec<String> = (0..r.n)
                .filter(|&i| props.atom_types()[i] != r.types[i])
                .map(|i| format!("{}{}:{}!={}", mol.atom(i).symbol(), i + 1, props.atom_types()[i], r.types[i]))
                .collect();
            st.type_mismatch.push(format!("{} [{}]", r.name, diff.join(" ")));
            continue;
        }
        let ce = props.partial_charges().iter().zip(&r.charges).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        st.max_charge_err = st.max_charge_err.max(ce);
        if ce > 1e-4 {
            st.charge_mismatch.push(format!("{} ({ce:.2e})", r.name));
        }
        let ff = builder.build_with_properties(&mol, &props, &[]).unwrap();
        let pos = mol.flat_coords();
        let e = ff.energy(&pos);
        let er = rel(r.energy[0], e);
        st.max_energy_rel = st.max_energy_rel.max(er);
        if (e - r.energy[0]).abs() > 1e-6 * r.energy[0].abs().max(1.0) {
            st.energy_mismatch.push(format!("{} ({} vs {})", r.name, e, r.energy[0]));
        }
        let bd: HashMap<String, f64> = ff.energy_breakdown(&pos).into_iter().collect();
        for (k, l) in TERM_LABELS.iter().enumerate() {
            let ours = bd.get(*l).copied().unwrap_or(0.0);
            st.max_term_rel = st.max_term_rel.max((ours - r.energy[k + 1]).abs() / r.energy[0].abs().max(1.0));
        }
        // extra settings
        let variants = [
            MmffBuilder { dielectric_model: DielectricModel::Distance, dielectric_constant: 4.0, ..builder.clone() },
            MmffBuilder { ignore_interfrag_interactions: false, ..builder.clone() },
            MmffBuilder { nonbonded_thresh: 9.0, ..builder.clone() },
        ];
        for (k, b) in variants.iter().enumerate() {
            let e2 = b.build_with_properties(&mol, &props, &[]).unwrap().energy(&pos);
            st.max_extra_rel = st.max_extra_rel.max(rel(r.extra[k], e2));
        }
        let mut g = vec![0.0; pos.len()];
        ff.energy_grad(&pos, &mut g);
        let gmax = r.grad.iter().fold(1.0f64, |a, b| a.max(b.abs()));
        let ge = g.iter().zip(&r.grad).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / gmax;
        st.max_grad_err = st.max_grad_err.max(ge);
        // short trajectory
        let mut p10 = pos.clone();
        minimize(&ff, &mut p10, &MinimizeOptions { max_iters: 10, ..opts.clone() }, &mut NoObserver);
        st.o10_max_rmsd = st.o10_max_rmsd.max(heavy_rmsd(&mol, &p10, &r.o10.2));
        // full optimisation
        let mut p200 = pos.clone();
        let res = minimize(&ff, &mut p200, &opts, &mut NoObserver);
        let rmsd = heavy_rmsd(&mol, &p200, &r.o200.2);
        let de = (res.energy - r.o200.1).abs();
        st.o200_max_rmsd = st.o200_max_rmsd.max(rmsd);
        st.o200_max_de = st.o200_max_de.max(de);
        if rmsd > 1e-3 || de > 1e-4 {
            st.o200_bad.push(format!("{} (rmsd {rmsd:.2e}, dE {de:.2e})", r.name));
        }
        // from the perturbed start
        let mut mp = mol.clone();
        mp.set_flat_coords(&r.pert);
        let ffp = builder.build_with_properties(&mp, &props, &[]).unwrap();
        let mut pp = r.pert.clone();
        assert!(rel(r.pert_e, ffp.energy(&pp)) < 1e-6, "{}: perturbed energy", r.name);
        let res = minimize(&ffp, &mut pp, &opts, &mut NoObserver);
        let rmsd = heavy_rmsd(&mol, &pp, &r.po200.2);
        let de = (res.energy - r.po200.1).abs();
        st.po200_max_rmsd = st.po200_max_rmsd.max(rmsd);
        st.po200_max_de = st.po200_max_de.max(de);
        // RDKit status != 0: RDKit itself stopped at 200 iterations, so the
        // reference is an arbitrary point on an unconverged trajectory, whose
        // phase is sensitive to rounding (the BFGS kernel agrees with RDKit
        // to rounding, not bit for bit). Such points are compared with the
        // "same structure" tolerance of the conformer tests (observed: up to
        // RMSD 2.3e-2 A / dE 3.2e-4 kcal/mol, while run to convergence both
        // trajectories reach the same minimum within 2e-8 kcal/mol).
        let (tol_rmsd, tol_de) = if r.po200.0 != 0 { (5e-2, 1e-3) } else { (1e-3, 1e-4) };
        if rmsd > tol_rmsd || de > tol_de {
            st.po200_bad.push(format!("{} (rmsd {rmsd:.2e}, dE {de:.2e}, status {:?} vs {})", r.name, res.status, r.po200.0));
        } else if rmsd < 1e-6 {
            st.po200_exact += 1;
        }
    }
    println!("{sdf} ({}):", variant.name());
    println!("  molecules: {}", st.n);
    println!("  atom-type mismatches: {} {:?}", st.type_mismatch.len(), st.type_mismatch);
    println!("  charge mismatches: {} (max |dq| {:.2e}) {:?}", st.charge_mismatch.len(), st.max_charge_err, st.charge_mismatch);
    println!(
        "  energy mismatches: {} (max rel {:.2e}; terms {:.2e}; diel/interfrag/thresh {:.2e}) {:?}",
        st.energy_mismatch.len(),
        st.max_energy_rel,
        st.max_term_rel,
        st.max_extra_rel,
        st.energy_mismatch
    );
    println!("  gradient max rel err: {:.2e}", st.max_grad_err);
    println!("  10-iteration heavy-atom RMSD max: {:.2e}", st.o10_max_rmsd);
    println!("  MMFFOptimizeMolecule: max RMSD {:.2e}, max dE {:.2e}, outliers {} {:?}", st.o200_max_rmsd, st.o200_max_de, st.o200_bad.len(), st.o200_bad);
    println!(
        "  perturbed start: max RMSD {:.2e}, max dE {:.2e}, RMSD<1e-6: {}, outliers {} {:?}",
        st.po200_max_rmsd,
        st.po200_max_de,
        st.po200_exact,
        st.po200_bad.len(),
        st.po200_bad
    );
    st
}

fn check(st: &Stats) {
    assert!(st.type_mismatch.is_empty(), "atom type mismatches: {:?}", st.type_mismatch);
    assert!(st.charge_mismatch.is_empty(), "charge mismatches: {:?}", st.charge_mismatch);
    assert!(st.energy_mismatch.is_empty(), "energy mismatches: {:?}", st.energy_mismatch);
    assert!(st.max_term_rel < 1e-6 && st.max_extra_rel < 1e-6);
    assert!(st.max_grad_err < 1e-6);
    assert!(st.o10_max_rmsd < 1e-4);
    assert!(st.o200_bad.is_empty(), "optimization outliers: {:?}", st.o200_bad);
    // minimizations from perturbed starts (all agree to the last printed
    // digit at the time of writing)
    assert!(st.po200_bad.is_empty(), "perturbed-start outliers: {:?}", st.po200_bad);
}

fn hyper_ref(name: &str) -> String {
    let own = format!("{name}.ref");
    if data_dir().join(&own).exists() { own } else { format!("{}.ref", name.replace("hypervalent", "dative")) }
}

#[test]
fn mmff94_dative() {
    check(&run_set("MMFF94_dative.sdf", "MMFF94_dative.ref", MmffVariant::Mmff94));
}

#[test]
fn mmff94_hypervalent() {
    check(&run_set("MMFF94_hypervalent.sdf", &hyper_ref("MMFF94_hypervalent"), MmffVariant::Mmff94));
}

#[test]
fn mmff94s_dative() {
    check(&run_set("MMFF94s_dative.sdf", "MMFF94s_dative.ref", MmffVariant::Mmff94s));
}

#[test]
fn mmff94s_hypervalent() {
    check(&run_set("MMFF94s_hypervalent.sdf", &hyper_ref("MMFF94s_hypervalent"), MmffVariant::Mmff94s));
}

#[test]
fn builder_via_trait() {
    let sdf = std::fs::read_to_string(data_dir().join("MMFF94_dative.sdf")).unwrap();
    let mol = read_sdf_lenient(&sdf).into_iter().next().unwrap().unwrap();
    let b: &dyn ForceFieldBuilder = &MmffBuilder::default();
    assert_eq!(b.name(), "MMFF94");
    let ff = b.build(&mol, &[]).unwrap();
    assert!(ff.energy(&mol.flat_coords()).is_finite());
}
