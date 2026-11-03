//! GPU (wgpu) backend of the approximate nonbonded mode: accuracy against the
//! CPU backend, GPU FFT against rustfft, reproducibility, batch mode and
//! minimization. Tests print a note and pass if no GPU adapter is available.
#![cfg(feature = "gpu")]

mod common;

use common::water::{BondedWater, WaterFf, water_box};
use molecule_optimizer_rs::core::{Molecule, UnitCell};
use molecule_optimizer_rs::fast::approx::build_approx_nonbonded;
use molecule_optimizer_rs::fast::backend::{NonbondedBackend, NonbondedEnergy};
use molecule_optimizer_rs::fast::gpu::{GpuBatch, GpuContext, GpuFft3d, WgpuNonbonded, build_gpu_nonbonded, enumerate_adapters};
use molecule_optimizer_rs::fast::{ApproxOptions, Electrostatics, VdwModifier};
use molecule_optimizer_rs::ff::ForceField;
use molecule_optimizer_rs::ff::mmff::{DielectricModel, MmffBuilder, MmffMolProperties, MmffVariant};
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::io::molfile::read_sdf_lenient;
use molecule_optimizer_rs::optim::{Algorithm, MinimizeOptions, NoObserver, minimize};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Shared context, or `None` (tests print a note and return) if no adapter.
fn ctx() -> Option<&'static GpuContext> {
    static CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| match GpuContext::new_blocking() {
        Ok(c) => {
            eprintln!("GPU adapter: {}", c.description());
            Some(c)
        }
        Err(e) => {
            eprintln!("note: skipping GPU tests: {e}");
            None
        }
    })
    .as_ref()
}

macro_rules! gpu_or_skip {
    () => {
        match ctx() {
            Some(c) => c,
            None => return,
        }
    };
}

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

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

/// ||a - b|| / ||b||
fn rel_rms(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let n: f64 = b.iter().map(|y| y * y).sum();
    (d / n.max(1e-300)).sqrt()
}

/// A drug-sized MMFF fixture molecule (single fragment, >= 30 atoms).
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

fn paracetamol() -> Molecule {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/data/paracetamol.mol");
    let text = std::fs::read_to_string(path).unwrap();
    read_sdf_lenient(&text).into_iter().flatten().next().unwrap()
}

fn eval(b: &dyn NonbondedBackend, pos: &[f64]) -> (NonbondedEnergy, Vec<f64>) {
    let mut g = vec![0.0; pos.len()];
    let e = b.energy_grad(pos, &mut g);
    (e, g)
}

/// Randomly perturbed copy of `pos` (|d| < `max` per atom).
fn displace(pos: &[f64], max: f64, rng: &mut ChaCha8Rng) -> Vec<f64> {
    pos.iter().map(|x| x + rng.gen_range(-max..max) / 3f64.sqrt()).collect()
}

// ---------------------------------------------------------------------------
// adapters, FFT
// ---------------------------------------------------------------------------

#[test]
fn list_adapters() {
    for a in enumerate_adapters() {
        println!("adapter: {} ({:?}, {:?}, driver {} {})", a.name, a.backend, a.device_type, a.driver, a.driver_info);
    }
    if let Some(c) = ctx() {
        println!("selected: {}", c.description());
    }
}

/// 3D FFT with rustfft in f64 (row-major, last index fastest).
fn cpu_fft3(data: &mut [Complex64], dims: [usize; 3], inverse: bool) {
    let mut planner = FftPlanner::<f64>::new();
    let strides = [dims[1] * dims[2], dims[2], 1];
    let total: usize = dims.iter().product();
    for d in 0..3 {
        let n = dims[d];
        let f = if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
        let mut line = vec![Complex64::new(0.0, 0.0); n];
        for start in 0..total {
            if (start / strides[d]) % n != 0 {
                continue;
            }
            for k in 0..n {
                line[k] = data[start + k * strides[d]];
            }
            f.process(&mut line);
            for k in 0..n {
                data[start + k * strides[d]] = line[k];
            }
        }
    }
}

#[test]
fn gpu_fft_matches_rustfft() {
    let ctx = gpu_or_skip!();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for dims in [[4, 4, 4], [8, 6, 5], [12, 10, 14], [7, 9, 16], [20, 25, 28], [48, 45, 50], [80, 81, 84]] {
        let fft = GpuFft3d::new(ctx, dims).unwrap();
        let total: usize = dims.iter().product();
        let x: Vec<[f32; 2]> = (0..total).map(|_| [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)]).collect();
        for inverse in [false, true] {
            let mut g = x.clone();
            fft.transform(&mut g, inverse);
            let mut c: Vec<Complex64> = x.iter().map(|v| Complex64::new(v[0] as f64, v[1] as f64)).collect();
            cpu_fft3(&mut c, dims, inverse);
            let norm = (c.iter().map(|v| v.norm_sqr()).sum::<f64>() / total as f64).sqrt();
            let err = (g.iter().zip(&c).map(|(a, b)| (Complex64::new(a[0] as f64, a[1] as f64) - b).norm_sqr()).sum::<f64>() / total as f64).sqrt();
            println!("FFT {dims:?} inverse={inverse} ({} passes): rms error / rms value = {:.2e}", fft.num_passes(), err / norm);
            assert!(err / norm < 2e-6, "{dims:?}: {}", err / norm);
            let mut g2 = x.clone();
            fft.transform(&mut g2, inverse);
            assert!(g.iter().zip(&g2).all(|(a, b)| a[0].to_bits() == b[0].to_bits() && a[1].to_bits() == b[1].to_bits()));
        }
    }
}

// ---------------------------------------------------------------------------
// GPU vs CPU approximate backend (same options, same pair list)
// ---------------------------------------------------------------------------

struct Acc {
    de: f64,
    dg: f64,
}

/// GPU vs CPU; prints and returns the errors.
fn compare(label: &str, gpu: &WgpuNonbonded, pos: &[f64]) -> Acc {
    let (eg, gg) = eval(gpu, pos);
    let (ec, gc) = eval(gpu.cpu(), pos);
    let de_ele = if ec.electrostatic == 0.0 { eg.electrostatic.abs() } else { rel(eg.electrostatic, ec.electrostatic) };
    let a = Acc { de: rel(eg.total(), ec.total()), dg: rel_rms(&gg, &gc) };
    println!(
        "ACCURACY | {label} | {} atoms | E_cpu {:.6} | E_gpu {:.6} | dE/E {:.2e} (vdw {:.2e}, ele {:.2e}) | force rel RMS {:.2e} | TPC {}",
        gpu.num_atoms(),
        ec.total(),
        eg.total(),
        a.de,
        rel(eg.vdw, ec.vdw),
        de_ele,
        a.dg,
        gpu.threads_per_cluster()
    );
    // the energy-only path gives the same energy
    let e2 = gpu.energy(pos);
    assert!(rel(e2.total(), eg.total()) < 1e-6, "energy-only {} vs {}", e2.total(), eg.total());
    a
}

fn check(label: &str, mol: &Molecule, inner: &dyn std::any::Any, o: &ApproxOptions, etol: f64, gtol: f64) {
    let ctx = gpu_or_skip!();
    let gpu = build_gpu_nonbonded(mol, inner, o, ctx).unwrap();
    let a = compare(label, &gpu, &mol.flat_coords());
    assert!(a.de < etol, "{label}: energy rel error {:e}", a.de);
    assert!(a.dg < gtol, "{label}: force rel RMS error {:e}", a.dg);
}

#[test]
fn water_216_reaction_field_and_pme() {
    let mol = water_box([6, 6, 6], 3.1, 1);
    let inner = mmff_periodic();
    check("water216 RF", &mol, &inner, &opts(8.0, 1.0, VdwModifier::PotentialShift, RF_INF), 1e-5, 1e-4);
    let rf78 = Electrostatics::ReactionField { eps_rf: 78.5 };
    check("water216 RF(78.5) switch", &mol, &inner, &opts(8.0, 1.0, VdwModifier::Switch { r_switch: 6.5 }, rf78), 1e-5, 1e-4);
    check("water216 PME", &mol, &inner, &opts(8.0, 1.0, VdwModifier::PotentialShift, pme(1e-5, 1.0, 4)), 1e-5, 1e-4);
    check("water216 PME order 6", &mol, &inner, &opts(8.0, 1.0, VdwModifier::Switch { r_switch: 7.0 }, pme(1e-6, 0.8, 6)), 1e-5, 1e-4);
}

#[test]
fn water_1728_reaction_field_and_pme() {
    let mol = water_box([12, 12, 12], 3.1, 2);
    let inner = mmff_periodic();
    check("water1728 RF", &mol, &inner, &opts(10.0, 1.0, VdwModifier::PotentialShift, RF_INF), 1e-5, 1e-4);
    check("water1728 PME", &mol, &inner, &opts(10.0, 1.0, VdwModifier::PotentialShift, pme(1e-5, 1.0, 4)), 1e-5, 1e-4);
}

#[test]
fn water_triclinic_pme_and_rf() {
    let mut mol = water_box([5, 5, 5], 3.1, 3);
    mol.cell = Some(UnitCell::from_parameters(15.5, 15.5, 15.5, 80.0, 75.0, 100.0).unwrap());
    check("water125 triclinic PME", &mol, &mmff_periodic(), &opts(6.5, 0.6, VdwModifier::PotentialShift, pme(1e-5, 0.9, 5)), 1e-5, 1e-4);
    check("water125 triclinic RF", &mol, &mmff_periodic(), &opts(6.5, 0.6, VdwModifier::Switch { r_switch: 5.5 }, RF_INF), 1e-5, 1e-4);
}

#[test]
fn drug_like_mmff_full_and_reaction_field() {
    let mol = fixture_molecule();
    let mmff = MmffBuilder::default();
    let dd = MmffBuilder { dielectric_model: DielectricModel::Distance, dielectric_constant: 4.0, ..MmffBuilder::default() };
    for (name, inner) in [("MMFF", &mmff), ("MMFF dist-diel", &dd)] {
        let full = Electrostatics::Full;
        check(&format!("drug {name} Full"), &mol, inner, &opts(60.0, 0.0, VdwModifier::Switch { r_switch: 60.0 }, full), 1e-5, 1e-4);
        check(&format!("drug {name} Full, vdW shift 8 A"), &mol, inner, &opts(8.0, 0.5, VdwModifier::PotentialShift, full), 1e-5, 1e-4);
        check(&format!("drug {name} RF 8 A"), &mol, inner, &opts(8.0, 0.5, VdwModifier::Switch { r_switch: 6.0 }, RF_INF), 1e-5, 1e-4);
    }
}

#[test]
fn drug_like_uff() {
    let mol = fixture_molecule();
    let uff = UffBuilder::default();
    check("drug UFF", &mol, &uff, &opts(60.0, 0.0, VdwModifier::Switch { r_switch: 60.0 }, Electrostatics::Full), 1e-5, 1e-4);
    check("drug UFF shift 6 A", &mol, &uff, &opts(6.0, 0.5, VdwModifier::PotentialShift, RF_INF), 1e-5, 1e-4);
    check("drug UFF switch 6 A", &mol, &uff, &opts(6.0, 0.5, VdwModifier::Switch { r_switch: 4.5 }, RF_INF), 1e-5, 1e-4);
}

#[test]
fn drug_in_box_pme_with_14_pairs() {
    let ctx = gpu_or_skip!();
    let mut mol = fixture_molecule();
    let n = mol.num_atoms() as f64;
    let c: Vec<f64> = (0..3).map(|d| mol.coords.iter().map(|p| p[d]).sum::<f64>() / n).collect();
    for p in mol.coords.iter_mut() {
        for d in 0..3 {
            p[d] += 12.0 - c[d];
        }
    }
    mol.cell = Some(UnitCell::cubic(24.0).unwrap());
    let gpu = build_gpu_nonbonded(&mol, &MmffBuilder::default(), &opts(10.0, 0.5, VdwModifier::PotentialShift, pme(1e-6, 0.8, 6)), ctx).unwrap();
    let a = compare("drug in 24 A box PME", &gpu, &mol.flat_coords());
    assert!(a.de < 1e-5 && a.dg < 1e-4);
}

// ---------------------------------------------------------------------------
// pair-list maintenance, reproducibility
// ---------------------------------------------------------------------------

#[test]
fn pair_list_rebuilds_are_uploaded() {
    let ctx = gpu_or_skip!();
    let mol = water_box([6, 6, 6], 3.1, 4);
    let gpu = build_gpu_nonbonded(&mol, &mmff_periodic(), &opts(8.0, 1.0, VdwModifier::PotentialShift, pme(1e-5, 1.0, 4)), ctx).unwrap();
    let pos = mol.flat_coords();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let _ = eval(&gpu, &pos);
    assert_eq!((gpu.num_builds(), gpu.num_uploads()), (1, 1));
    // small moves: list reused
    let p = displace(&pos, 0.45, &mut rng);
    let a = compare("water216 PME displaced (no rebuild)", &gpu, &p);
    assert!(a.de < 1e-5 && a.dg < 1e-4);
    assert_eq!((gpu.num_builds(), gpu.num_uploads()), (1, 1));
    // large moves: rebuild on the CPU, re-upload
    let p = displace(&pos, 1.5, &mut rng);
    let a = compare("water216 PME displaced (rebuild)", &gpu, &p);
    assert!(a.de < 1e-5 && a.dg < 1e-4);
    assert_eq!((gpu.num_builds(), gpu.num_uploads()), (2, 2));
    // on_accepted_step rebuilds proactively
    let p2 = displace(&p, 1.5, &mut rng);
    gpu.on_accepted_step(&p2);
    assert_eq!(gpu.num_builds(), 3);
    let _ = eval(&gpu, &p2);
    assert_eq!(gpu.num_uploads(), 3);
    // translating whole waters by lattice vectors does not change anything
    let mut t = p2.clone();
    for (w, x) in t.chunks_mut(9).enumerate() {
        let s = [(w % 3) as f64 - 1.0, ((w / 3) % 3) as f64 - 1.0, 2.0 * ((w % 2) as f64)];
        for a in 0..3 {
            for d in 0..3 {
                x[3 * a + d] += s[d] * 18.6;
            }
        }
    }
    let (e0, g0) = eval(&gpu, &p2);
    let (e1, g1) = eval(&gpu, &t);
    println!("lattice translation: dE/E {:.2e}, force rel RMS {:.2e}", rel(e1.total(), e0.total()), rel_rms(&g1, &g0));
    assert!(rel(e1.total(), e0.total()) < 1e-5 && rel_rms(&g1, &g0) < 1e-4);
}

fn bits_equal(a: &(NonbondedEnergy, Vec<f64>), b: &(NonbondedEnergy, Vec<f64>)) -> bool {
    a.0.vdw.to_bits() == b.0.vdw.to_bits()
        && a.0.electrostatic.to_bits() == b.0.electrostatic.to_bits()
        && a.1.iter().zip(&b.1).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn results_are_bitwise_reproducible() {
    let ctx = gpu_or_skip!();
    let mol = water_box([8, 8, 8], 3.1, 6);
    for el in [RF_INF, pme(1e-5, 1.0, 4)] {
        let o = opts(9.0, 1.0, VdwModifier::Switch { r_switch: 7.5 }, el);
        let gpu = build_gpu_nonbonded(&mol, &mmff_periodic(), &o, ctx).unwrap();
        let pos = mol.flat_coords();
        let r0 = eval(&gpu, &pos);
        for _ in 0..5 {
            assert!(bits_equal(&r0, &eval(&gpu, &pos)), "{el:?}: run-to-run difference");
        }
        // a second, independently built backend gives the same bits
        let gpu2 = build_gpu_nonbonded(&mol, &mmff_periodic(), &o, ctx).unwrap();
        assert!(bits_equal(&r0, &eval(&gpu2, &pos)), "{el:?}: differs between instances");
        println!("{el:?}: bitwise reproducible (E = {:.8})", r0.0.total());
    }
}

// ---------------------------------------------------------------------------
// batch mode
// ---------------------------------------------------------------------------

#[test]
fn batch_equals_individual_evaluations() {
    let ctx = gpu_or_skip!();
    let para = paracetamol();
    let drug = fixture_molecule();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    for (o, label) in [
        (opts(60.0, 0.0, VdwModifier::Switch { r_switch: 60.0 }, Electrostatics::Full), "Full"),
        (opts(6.0, 0.5, VdwModifier::PotentialShift, RF_INF), "RF 6 A"),
    ] {
        let inner = MmffBuilder::default();
        let batch = GpuBatch::new(ctx, &[(&para, 5), (&drug, 3)], &inner, &o).unwrap();
        assert_eq!(batch.num_systems(), 8);
        let pos: Vec<Vec<f64>> = (0..8)
            .map(|s| {
                let base = if s < 5 { para.flat_coords() } else { drug.flat_coords() };
                displace(&base, 0.2, &mut rng)
            })
            .collect();
        let refs: Vec<&[f64]> = pos.iter().map(|p| p.as_slice()).collect();
        let res = batch.energy_grad(&refs);
        for (s, (e, g)) in res.iter().enumerate() {
            let mol = if s < 5 { &para } else { &drug };
            let single = build_gpu_nonbonded(mol, &inner, &o, ctx).unwrap();
            let (es, gs) = eval(&single, &pos[s]);
            let (ec, gc) = eval(batch.template(s), &pos[s]);
            println!(
                "batch {label} system {s}: E_batch {:.8} E_single {:.8} E_cpu {:.8} | batch vs single dE {:.1e} dg {:.1e} | vs CPU dE/E {:.1e} dg {:.1e}",
                e.total(),
                es.total(),
                ec.total(),
                (e.total() - es.total()).abs(),
                rel_rms(g, &gs),
                rel(e.total(), ec.total()),
                rel_rms(g, &gc)
            );
            assert!(rel(e.total(), es.total()) < 1e-6, "system {s}");
            assert!(rel_rms(g, &gs) < 1e-5, "system {s}");
            assert!(rel(e.total(), ec.total()) < 1e-5 && rel_rms(g, &gc) < 1e-4, "system {s} vs CPU");
        }
        // energies-only path
        let en = batch.energies(&refs);
        for (a, (b, _)) in en.iter().zip(&res) {
            assert!(rel(a.total(), b.total()) < 1e-6);
        }
        // UFF batch
        let ub = GpuBatch::new(ctx, &[(&para, 2)], &UffBuilder::default(), &o).unwrap();
        let ur = ub.energy_grad(&refs[..2]);
        for (s, (e, g)) in ur.iter().enumerate() {
            let (ec, gc) = eval(ub.template(s), &pos[s]);
            assert!(rel(e.total(), ec.total()) < 1e-5 && rel_rms(g, &gc) < 1e-4, "UFF system {s}");
        }
    }
    // PME is rejected
    assert!(GpuBatch::new(ctx, &[(&para, 2)], &MmffBuilder::default(), &opts(6.0, 0.5, VdwModifier::PotentialShift, pme(1e-5, 1.0, 4))).is_err());
}

// ---------------------------------------------------------------------------
// minimization
// ---------------------------------------------------------------------------

#[test]
fn water_box_minimization_gpu_matches_cpu() {
    let ctx = gpu_or_skip!();
    let n = 6;
    let mol = water_box([n, n, n], 3.1, 26);
    for el in [RF_INF, pme(1e-5, 1.0, 4)] {
        let o = opts(8.0, 1.0, VdwModifier::PotentialShift, el);
        let mo = MinimizeOptions { max_iters: 300, algorithm: Algorithm::Lbfgs(10), ..MinimizeOptions::default() };
        let cpu = WaterFf { bonded: BondedWater::new(n * n * n), nb: build_approx_nonbonded(&mol, &mmff_periodic(), &o, true).unwrap() };
        let gpu = WaterFf { bonded: BondedWater::new(n * n * n), nb: build_gpu_nonbonded(&mol, &mmff_periodic(), &o, ctx).unwrap() };
        let mut pc = mol.flat_coords();
        let mut pg = mol.flat_coords();
        let e0 = cpu.energy(&pc);
        let t = std::time::Instant::now();
        let rc = minimize(&cpu, &mut pc, &mo, &mut NoObserver);
        let tc = t.elapsed();
        let t = std::time::Instant::now();
        let rg = minimize(&gpu, &mut pg, &mo, &mut NoObserver);
        let tg = t.elapsed();
        // GPU final energy re-evaluated on the CPU
        let eg_cpu = cpu.energy(&pg);
        println!(
            "minimize {} waters {el:?}: E0 {e0:.3}; CPU -> {:.4} ({} it, {tc:.2?}); GPU -> {:.4} ({} it, {tg:.2?}, {} list builds), GPU geometry on CPU {:.4}",
            n * n * n,
            rc.energy,
            rc.iterations,
            rg.energy,
            rg.iterations,
            gpu.nb.num_builds(),
            eg_cpu
        );
        assert!(rg.energy < e0 && rg.energy.is_finite());
        // both reach a similar minimum (different paths through a rugged landscape)
        let drop_c = e0 - rc.energy;
        let drop_g = e0 - eg_cpu;
        assert!((drop_g - drop_c).abs() < 0.02 * drop_c, "CPU drop {drop_c}, GPU drop {drop_g}");
        // the GPU energy at the final geometry agrees with the CPU model
        assert!(rel(rg.energy, eg_cpu) < 1e-5, "{} vs {eg_cpu}", rg.energy);
    }
}
