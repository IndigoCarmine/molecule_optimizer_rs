//! Accuracy of the fast exact path against the RDKit-compatible
//! `ContribForceField` on the committed UFF and MMFF fixtures.

// NaN-safe `!(x <= tol)` checks and index loops over xyz are intended.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::needless_range_loop, clippy::type_complexity)]

use molecule_optimizer_rs::calc::conformer::RandomTorsionGenerator;
use molecule_optimizer_rs::calc::{self, ConformerSearchOptions, OptimizeOptions, ScanCoordinate, ScanOptions};
use molecule_optimizer_rs::fast::{FastBuilder, FastForceField, FastOptions};
use molecule_optimizer_rs::ff::mmff::{DielectricModel, MmffBuilder, MmffVariant};
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::ff::{Constraint, ForceField, ForceFieldBuilder};
use molecule_optimizer_rs::io::{molfile::read_sdf_lenient, read_molfile};
use molecule_optimizer_rs::optim::{MinimizeOptions, NoObserver, minimize};
use molecule_optimizer_rs::{Atom, Molecule};
use std::path::PathBuf;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

struct UffCase {
    name: String,
    mol: Molecule,
    builder: UffBuilder,
}

fn parse_xyz(l: &str) -> [f64; 3] {
    let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
    [v[0], v[1], v[2]]
}

fn uff_cases() -> Vec<UffCase> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/uff/uff_reference.txt");
    let text = std::fs::read_to_string(path).expect("fixture");
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let (mut name, mut thresh, mut interfrag, mut mb, mut coords) = (String::new(), 10.0, true, String::new(), Vec::new());
    let mut k = 0;
    while k < lines.len() {
        let l = lines[k];
        k += 1;
        if let Some(v) = l.strip_prefix("$NAME ") {
            name = v.to_string();
        } else if let Some(v) = l.strip_prefix("$OPTS ") {
            let f: Vec<&str> = v.split_whitespace().collect();
            thresh = f[0].parse().unwrap();
            interfrag = f[1] == "1";
        } else if l == "$MOLBLOCK" {
            mb.clear();
            while lines[k] != "M  END" {
                mb.push_str(lines[k]);
                mb.push('\n');
                k += 1;
            }
            mb.push_str("M  END\n");
            k += 1;
        } else if let Some(v) = l.strip_prefix("$COORDS ") {
            let n: usize = v.trim().parse().unwrap();
            coords = (0..n).map(|i| parse_xyz(lines[k + i])).collect();
            k += n;
        } else if l == "$END" {
            let mut mol = read_molfile(&mb).unwrap_or_else(|e| panic!("{name}: {e}"));
            mol.coords = coords.clone();
            out.push(UffCase {
                name: name.clone(),
                mol,
                builder: UffBuilder { vdw_thresh: thresh, ignore_interfrag_interactions: interfrag },
            });
        }
    }
    assert!(out.len() > 300);
    out
}

fn mmff_mols(sdf: &str) -> Vec<(String, Molecule)> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/mmff").join(sdf);
    let text = std::fs::read_to_string(p).unwrap();
    read_sdf_lenient(&text)
        .into_iter()
        .enumerate()
        .filter_map(|(i, m)| m.ok().map(|m| (format!("{sdf}#{i} {}", m.name), m)))
        .collect()
}

const MMFF_SETS: [(&str, MmffVariant); 4] = [
    ("MMFF94_dative.sdf", MmffVariant::Mmff94),
    ("MMFF94_hypervalent.sdf", MmffVariant::Mmff94),
    ("MMFF94s_dative.sdf", MmffVariant::Mmff94s),
    ("MMFF94s_hypervalent.sdf", MmffVariant::Mmff94s),
];

fn paracetamol() -> Molecule {
    molecule_optimizer_rs::io::read_file(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data/paracetamol.mol")).unwrap().remove(0)
}

// ---------------------------------------------------------------------------
// comparison helpers
// ---------------------------------------------------------------------------

#[derive(Default, Debug)]
struct Diff {
    n: usize,
    max_e_rel: f64,
    max_g_rel: f64,
    max_bd_rel: f64,
    /// systems whose energy and gradient are bit-identical to the reference
    bit_identical: usize,
    failures: Vec<String>,
}

impl Diff {
    fn check(&mut self, name: &str, reference: &dyn ForceField, fast: &dyn ForceField, pos: &[f64]) {
        self.n += 1;
        let e_ref = reference.energy(pos);
        let e_fast = fast.energy(pos);
        let mut g_ref = vec![0.0; pos.len()];
        let mut g_fast = vec![0.0; pos.len()];
        let eg_ref = reference.energy_grad(pos, &mut g_ref);
        let eg_fast = fast.energy_grad(pos, &mut g_fast);
        let scale = e_ref.abs().max(1.0);
        let er = ((e_fast - e_ref).abs().max((eg_fast - eg_ref).abs())) / scale;
        let gs = g_ref.iter().fold(1.0f64, |a, b| a.max(b.abs()));
        let gr = g_ref.iter().zip(&g_fast).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / gs;
        if eg_ref == eg_fast && e_ref == e_fast && g_ref == g_fast {
            self.bit_identical += 1;
        }
        self.max_e_rel = self.max_e_rel.max(er);
        self.max_g_rel = self.max_g_rel.max(gr);
        if !(er <= 1e-10) {
            self.failures.push(format!("{name}: energy {e_fast} vs {e_ref} (rel {er:.2e})"));
        }
        if !(gr <= 1e-9) {
            self.failures.push(format!("{name}: gradient rel {gr:.2e}"));
        }
        let br = reference.energy_breakdown(pos);
        let bf = fast.energy_breakdown(pos);
        let lr: Vec<&str> = br.iter().map(|(k, _)| k.as_str()).collect();
        let lf: Vec<&str> = bf.iter().map(|(k, _)| k.as_str()).collect();
        if lr != lf {
            self.failures.push(format!("{name}: breakdown labels {lf:?} vs {lr:?}"));
        } else {
            for ((_, a), (_, b)) in br.iter().zip(&bf) {
                let d = (a - b).abs() / scale;
                self.max_bd_rel = self.max_bd_rel.max(d);
                if !(d <= 1e-10) {
                    self.failures.push(format!("{name}: breakdown value {b} vs {a}"));
                }
            }
        }
        if reference.fixed_points() != fast.fixed_points() {
            self.failures.push(format!("{name}: fixed points differ"));
        }
    }
    fn assert_ok(&self, what: &str) {
        println!(
            "{what}: {} systems ({} bit-identical), max rel |dE| {:.2e}, max rel |dgrad| {:.2e}, max breakdown {:.2e}",
            self.n, self.bit_identical, self.max_e_rel, self.max_g_rel, self.max_bd_rel
        );
        assert!(self.failures.is_empty(), "{what}: {} failures:\n{}", self.failures.len(), self.failures.join("\n"));
    }
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

#[derive(Default)]
struct MinStats {
    n: usize,
    max_de: f64,
    max_rmsd: f64,
    outliers: Vec<String>,
}

impl MinStats {
    fn run(&mut self, name: &str, mol: &Molecule, reference: &dyn ForceField, fast: &dyn ForceField) {
        let opts = MinimizeOptions::default();
        let mut pr = mol.flat_coords();
        let mut pf = pr.clone();
        let rr = minimize(reference, &mut pr, &opts, &mut NoObserver);
        let rf = minimize(fast, &mut pf, &opts, &mut NoObserver);
        let de = (rr.energy - rf.energy).abs();
        let rmsd = heavy_rmsd(mol, &pr, &pf);
        self.n += 1;
        self.max_de = self.max_de.max(de);
        self.max_rmsd = self.max_rmsd.max(rmsd);
        if !(de <= 1e-6 && rmsd < 1e-3) {
            self.outliers.push(format!(
                "{name}: dE {de:.2e}, RMSD {rmsd:.2e} (contrib {:?} {} it E {:.6}; fast {:?} {} it E {:.6})",
                rr.status, rr.iterations, rr.energy, rf.status, rf.iterations, rf.energy
            ));
        }
    }
    fn assert_ok(&self, what: &str) {
        println!(
            "{what}: {} minimizations, max |dE| {:.2e}, max heavy RMSD {:.2e}, outliers {}",
            self.n,
            self.max_de,
            self.max_rmsd,
            self.outliers.len()
        );
        for o in &self.outliers {
            println!("  outlier {o}");
        }
        assert!(self.outliers.len() * 100 <= self.n, "{what}: more than 1% outliers");
    }
}

fn fast_of<B: ForceFieldBuilder + Clone + 'static>(b: &B, opts: FastOptions) -> FastBuilder<B> {
    FastBuilder::new(b.clone(), opts)
}

// ---------------------------------------------------------------------------
// energies / gradients / breakdowns
// ---------------------------------------------------------------------------

#[test]
fn uff_energy_gradient_breakdown() {
    let mut d = Diff::default();
    let mut d_scalar = Diff::default();
    let mut d_nocache = Diff::default();
    for c in uff_cases() {
        let reference = c.builder.build(&c.mol, &[]);
        let fast = fast_of(&c.builder, FastOptions::exact()).build_fast(&c.mol, &[]);
        let (Ok(reference), Ok(fast)) = (reference, fast) else {
            assert!(
                c.builder.build(&c.mol, &[]).is_err() && fast_of(&c.builder, FastOptions::exact()).build_fast(&c.mol, &[]).is_err(),
                "{}: build result differs",
                c.name
            );
            continue;
        };
        let pos = c.mol.flat_coords();
        d.check(&c.name, &reference, &fast, &pos);
        let scalar = fast_of(&c.builder, FastOptions::exact().simd(false)).build_fast(&c.mol, &[]).unwrap();
        d_scalar.check(&c.name, &reference, &scalar, &pos);
        let nc = fast_of(&c.builder, FastOptions::exact().cache(false)).build_fast(&c.mol, &[]).unwrap();
        d_nocache.check(&c.name, &reference, &nc, &pos);
    }
    d.assert_ok("UFF simd");
    d_scalar.assert_ok("UFF scalar");
    d_nocache.assert_ok("UFF uncached");
}

#[test]
fn mmff_energy_gradient_breakdown() {
    for (sdf, variant) in MMFF_SETS {
        let mut d = Diff::default();
        let mut extra = Diff::default();
        let b = MmffBuilder::new(variant);
        let variants = [
            MmffBuilder { dielectric_model: DielectricModel::Distance, dielectric_constant: 4.0, ..b.clone() },
            MmffBuilder { ignore_interfrag_interactions: false, ..b.clone() },
            MmffBuilder { nonbonded_thresh: 9.0, ..b.clone() },
        ];
        let fb = fast_of(&b, FastOptions::exact());
        for (name, mol) in mmff_mols(sdf) {
            let pos = mol.flat_coords();
            match (b.build(&mol, &[]), fb.build_fast(&mol, &[])) {
                (Ok(r), Ok(f)) => d.check(&name, &r, &f, &pos),
                (Err(_), Err(_)) => continue,
                (r, f) => panic!("{name}: build result differs ({:?} vs {:?})", r.err(), f.err()),
            }
            for v in &variants {
                let r = v.build(&mol, &[]).unwrap();
                let f = fast_of(v, FastOptions::exact()).build_fast(&mol, &[]).unwrap();
                extra.check(&name, &r, &f, &pos);
            }
        }
        d.assert_ok(sdf);
        extra.assert_ok(&format!("{sdf} (dielectric/interfrag/thresh variants)"));
    }
}

#[test]
fn constraints_and_fixed_points() {
    let mol = paracetamol();
    let cs = [
        Constraint::distance(0, 8, 6.0, 50.0),
        Constraint::angle(0, 1, 3, 100.0, 0.5),
        Constraint::dihedral(0, 1, 3, 4, 120.0, 0.3),
        Constraint::Position { i: 6, pos: None, max_displ: 0.0, k: 20.0 },
        Constraint::Position { i: 7, pos: Some([2.5, 0.0, 0.5]), max_displ: 0.05, k: 10.0 },
        Constraint::Fixed { i: 2 },
    ];
    let mut d = Diff::default();
    let mut mins = MinStats::default();
    let mut p = mol.flat_coords();
    for (k, x) in p.iter_mut().enumerate() {
        *x += 0.05 * ((k * 7919 % 13) as f64 / 13.0 - 0.5);
    }
    for b in [&MmffBuilder::default() as &dyn std::any::Any, &UffBuilder::default()] {
        let (r, f): (Box<dyn ForceField>, Box<dyn ForceField>) = if let Some(m) = b.downcast_ref::<MmffBuilder>() {
            (Box::new(m.build(&mol, &cs).unwrap()), Box::new(fast_of(m, FastOptions::exact()).build_fast(&mol, &cs).unwrap()))
        } else {
            let u = b.downcast_ref::<UffBuilder>().unwrap();
            (Box::new(u.build(&mol, &cs).unwrap()), Box::new(fast_of(u, FastOptions::exact()).build_fast(&mol, &cs).unwrap()))
        };
        d.check("paracetamol+constraints", r.as_ref(), f.as_ref(), &mol.flat_coords());
        d.check("paracetamol+constraints (moved)", r.as_ref(), f.as_ref(), &p);
        mins.run("paracetamol+constraints", &mol, r.as_ref(), f.as_ref());
        assert!(f.energy_breakdown(&p).iter().any(|(k, _)| k == "Constraint"));
    }
    d.assert_ok("constraints");
    mins.assert_ok("constraints");
}

// ---------------------------------------------------------------------------
// minimization
// ---------------------------------------------------------------------------

#[test]
fn uff_minimization_matches() {
    let mut st = MinStats::default();
    for c in uff_cases() {
        let Ok(r) = c.builder.build(&c.mol, &[]) else { continue };
        let f = fast_of(&c.builder, FastOptions::exact()).build_fast(&c.mol, &[]).unwrap();
        st.run(&c.name, &c.mol, &r, &f);
    }
    st.assert_ok("UFF");
}

#[test]
fn mmff_minimization_matches() {
    for (sdf, variant) in MMFF_SETS {
        let mut st = MinStats::default();
        let b = MmffBuilder::new(variant);
        let fb = fast_of(&b, FastOptions::exact());
        for (name, mol) in mmff_mols(sdf) {
            let Ok(r) = b.build(&mol, &[]) else { continue };
            let f = fb.build_fast(&mol, &[]).unwrap();
            st.run(&name, &mol, &r, &f);
        }
        st.assert_ok(sdf);
    }
}

// ---------------------------------------------------------------------------
// determinism: SIMD vs scalar, thread counts
// ---------------------------------------------------------------------------

/// `n³` translated copies of `mol` on a cubic lattice (one molecule with
/// several fragments).
fn lattice(mol: &Molecule, n: usize, spacing: f64) -> Molecule {
    let mut out = Molecule::new();
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let off = out.num_atoms();
                let shift = [x as f64 * spacing, y as f64 * spacing, z as f64 * spacing];
                for i in 0..mol.num_atoms() {
                    let a: Atom = mol.atom(i).clone();
                    let p = mol.coords[i];
                    out.add_atom(a, [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]]);
                }
                for b in mol.bonds() {
                    let k = out.add_bond(b.begin + off, b.end + off, b.order);
                    out.bond_mut(k).is_aromatic = b.is_aromatic;
                    out.bond_mut(k).is_conjugated = b.is_conjugated;
                }
            }
        }
    }
    molecule_optimizer_rs::perception::sanitize(&mut out).unwrap();
    out
}

fn eval_bits(ff: &dyn ForceField, pos: &[f64]) -> (u64, u64, Vec<u64>) {
    let mut g = vec![0.0; pos.len()];
    let e = ff.energy_grad(pos, &mut g);
    (ff.energy(pos).to_bits(), e.to_bits(), g.iter().map(|x| x.to_bits()).collect())
}

#[test]
fn thread_count_and_simd_independence() {
    let base = paracetamol();
    let big = lattice(&base, 3, 7.0);
    let mut pos = big.flat_coords();
    for (k, x) in pos.iter_mut().enumerate() {
        *x += 0.03 * ((k * 104729 % 17) as f64 / 17.0 - 0.5);
    }
    let cases: Vec<(&str, Box<dyn Fn(FastOptions) -> FastForceField>)> = vec![
        ("MMFF", Box::new(|o| fast_of(&MmffBuilder { ignore_interfrag_interactions: false, ..Default::default() }, o).build_fast(&big, &[]).unwrap())),
        ("UFF", Box::new(|o| fast_of(&UffBuilder { ignore_interfrag_interactions: false, ..Default::default() }, o).build_fast(&big, &[]).unwrap())),
    ];
    for (name, mk) in &cases {
        for grain in [16usize, 256, 1 << 30] {
            let opts = FastOptions::exact().parallel_grain(grain);
            let ff = mk(opts.clone());
            let mut results = Vec::new();
            for threads in [1usize, 4, 16] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                results.push(pool.install(|| eval_bits(&ff, &pos)));
            }
            assert_eq!(results[0], results[1], "{name} grain {grain}: 1 vs 4 threads");
            assert_eq!(results[0], results[2], "{name} grain {grain}: 1 vs 16 threads");
            // SIMD vs scalar
            let scalar = mk(opts.simd(false));
            let mut gs = vec![0.0; pos.len()];
            let mut gv = vec![0.0; pos.len()];
            let es = scalar.energy_grad(&pos, &mut gs);
            let ev = ff.energy_grad(&pos, &mut gv);
            assert!((es - ev).abs() <= 1e-12 * es.abs().max(1.0), "{name}: simd {ev} vs scalar {es}");
            let gmax = gs.iter().fold(1.0f64, |a, b| a.max(b.abs()));
            let gd = gs.iter().zip(&gv).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(gd <= 1e-12 * gmax, "{name}: simd/scalar gradient {gd:.2e}");
            println!("{name} grain {grain}: bonded chunks {}, E {ev}", ff.bonded().chunks.len());
        }
    }
}

// ---------------------------------------------------------------------------
// typing cache and pair re-selection
// ---------------------------------------------------------------------------

fn perturbed(mol: &Molecule, amp: f64, seed: usize) -> Molecule {
    let mut m = mol.clone();
    for (i, c) in m.coords.iter_mut().enumerate() {
        for d in 0..3 {
            let h = (i * 2654435761 + d * 40503 + seed * 7919) % 1000;
            c[d] += amp * (h as f64 / 1000.0 - 0.5);
        }
    }
    m
}

fn assert_same(name: &str, a: &FastForceField, b: &FastForceField, pos: &[f64]) {
    let pa = a.exact_nonbonded().map(|e| e.pairs()).unwrap_or_default();
    let pb = b.exact_nonbonded().map(|e| e.pairs()).unwrap_or_default();
    assert_eq!(pa, pb, "{name}: pair sets differ");
    let (ea, eb) = (eval_bits(a, pos), eval_bits(b, pos));
    assert_eq!(ea.0, eb.0, "{name}: energy differs");
    assert_eq!(ea.1, eb.1, "{name}: energy_grad energy differs");
    let ga: Vec<f64> = ea.2.iter().map(|&x| f64::from_bits(x)).collect();
    let gb: Vec<f64> = eb.2.iter().map(|&x| f64::from_bits(x)).collect();
    assert!(ga == gb, "{name}: gradient differs");
    assert_eq!(a.energy_breakdown(pos), b.energy_breakdown(pos), "{name}: breakdown differs");
}

#[test]
fn cached_build_equals_fresh_build() {
    // scan-like use: the template is built from the first geometry, later
    // geometries only re-select pairs
    let mut n = 0;
    for c in uff_cases().into_iter().step_by(3) {
        let b = UffBuilder { vdw_thresh: 1.3, ..c.builder.clone() };
        let cached = fast_of(&b, FastOptions::exact());
        let fresh = fast_of(&b, FastOptions::exact().cache(false));
        if cached.build_fast(&c.mol, &[]).is_err() {
            continue;
        }
        for seed in 0..3 {
            let m = perturbed(&c.mol, 0.6, seed);
            let a = cached.build_fast(&m, &[]).unwrap();
            let f = fresh.build_fast(&m, &[]).unwrap();
            assert_same(&c.name, &a, &f, &m.flat_coords());
            n += 1;
        }
    }
    for (sdf, variant) in MMFF_SETS {
        let b = MmffBuilder { nonbonded_thresh: 4.5, ..MmffBuilder::new(variant) };
        let cached = fast_of(&b, FastOptions::exact());
        let fresh = fast_of(&b, FastOptions::exact().cache(false));
        for (name, mol) in mmff_mols(sdf).into_iter().step_by(5) {
            if cached.build_fast(&mol, &[]).is_err() {
                continue;
            }
            for seed in 0..2 {
                let m = perturbed(&mol, 0.6, seed);
                let a = cached.build_fast(&m, &[]).unwrap();
                let f = fresh.build_fast(&m, &[]).unwrap();
                assert_same(&name, &a, &f, &m.flat_coords());
                n += 1;
            }
        }
        assert!(cached.cache_len() > 10);
    }
    println!("{n} cached builds identical to fresh builds");
}

#[test]
fn grid_selection_equals_builder_selection() {
    let base = paracetamol();
    let big = lattice(&base, 3, 6.5);
    assert!(big.num_atoms() >= 500);
    for interfrag in [true, false] {
        for thresh in [3.0, 6.0, 9.5, 100.0] {
            let b = MmffBuilder { nonbonded_thresh: thresh, ignore_interfrag_interactions: interfrag, ..Default::default() };
            let cached = fast_of(&b, FastOptions::exact());
            let fresh = fast_of(&b, FastOptions::exact().cache(false));
            cached.build_fast(&big, &[]).unwrap();
            for seed in 0..2 {
                let m = perturbed(&big, 1.0, seed);
                assert_same(&format!("MMFF thresh {thresh} interfrag {interfrag}"), &cached.build_fast(&m, &[]).unwrap(), &fresh.build_fast(&m, &[]).unwrap(), &m.flat_coords());
            }
        }
        for vt in [0.8, 1.5, 3.0, 10.0] {
            let b = UffBuilder { vdw_thresh: vt, ignore_interfrag_interactions: interfrag };
            let cached = fast_of(&b, FastOptions::exact());
            let fresh = fast_of(&b, FastOptions::exact().cache(false));
            cached.build_fast(&big, &[]).unwrap();
            for seed in 0..2 {
                let m = perturbed(&big, 1.0, seed);
                assert_same(&format!("UFF vdw_thresh {vt} interfrag {interfrag}"), &cached.build_fast(&m, &[]).unwrap(), &fresh.build_fast(&m, &[]).unwrap(), &m.flat_coords());
            }
        }
    }
}

#[test]
fn builder_api() {
    let mol = paracetamol();
    let fb = FastBuilder::new(MmffBuilder::default(), FastOptions::exact());
    assert_eq!(fb.name(), "MMFF94/fast");
    // `build` returns the wrapped force field unchanged
    let c = fb.build(&mol, &[]).unwrap();
    let r = MmffBuilder::default().build(&mol, &[]).unwrap();
    let pos = mol.flat_coords();
    assert_eq!(c.energy(&pos).to_bits(), r.energy(&pos).to_bits());
    assert_eq!(c.num_contribs(), r.num_contribs());
    // errors are reported like the wrapped builder
    let mut nocoords = mol.clone();
    nocoords.coords.clear();
    assert!(fb.build_dyn(&nocoords, &[]).is_err());
    assert!(FastBuilder::new(UffBuilder::default(), FastOptions::exact()).build_dyn(&nocoords, &[]).is_err());
    // approximate mode goes through the approximate backend
    let approx = FastBuilder::new(MmffBuilder::default(), FastOptions::approximate());
    let _ = approx.build_dyn(&mol, &[]);
}

// ---------------------------------------------------------------------------
// calc integration
// ---------------------------------------------------------------------------

#[test]
fn calc_with_fast_builder() {
    let mol = paracetamol();
    let fast = FastBuilder::new(MmffBuilder::default(), FastOptions::exact());
    let plain = MmffBuilder::default();
    let a = calc::optimize::optimize(&mol, &fast, &OptimizeOptions::default()).unwrap();
    let b = calc::optimize::optimize(&mol, &plain, &OptimizeOptions::default()).unwrap();
    assert!((a.energy - b.energy).abs() < 1e-6, "{} vs {}", a.energy, b.energy);

    let dih = ScanCoordinate::Dihedral { i: 0, j: 1, k: 3, l: 4, values: ScanCoordinate::steps(-180.0, 60.0, 6) };
    let so = ScanOptions { coordinates: vec![dih], ..Default::default() };
    let sa = calc::scan::scan(&mol, &fast, &so).unwrap();
    let sb = calc::scan::scan(&mol, &plain, &so).unwrap();
    assert_eq!(sa.points.len(), 6);
    for (p, q) in sa.points.iter().zip(&sb.points) {
        assert!((p.energy - q.energy).abs() < 1e-4, "scan point {:?}: {} vs {}", p.values, p.energy, q.energy);
    }
    assert_eq!(fast.cache_len(), 1);

    let mut co = ConformerSearchOptions::new(Arc::new(RandomTorsionGenerator { seed: 7 }));
    co.num_conformers = 6;
    let ca = calc::conformer::conformer_search(&mol, &fast, &co).unwrap();
    let cb = calc::conformer::conformer_search(&mol, &plain, &co).unwrap();
    let (ma, mb) = (ca.global_minimum().unwrap().energy, cb.global_minimum().unwrap().energy);
    assert!((ma - mb).abs() < 1e-4, "{ma} vs {mb}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn engine_batch_with_fast_builder() {
    use molecule_optimizer_rs::tasks::{Engine, EngineConfig};
    let mol = Arc::new(paracetamol());
    let items: Vec<_> = (0..8).map(|k| (mol.clone(), Some(perturbed(&mol, 0.2, k).coords))).collect();
    let engine = Engine::new(EngineConfig { num_threads: Some(4), ..Default::default() }).unwrap();
    let fast = Arc::new(FastBuilder::new(UffBuilder::default(), FastOptions::exact()));
    let ra = engine.optimize_batch(items.clone(), fast, OptimizeOptions::default()).await.unwrap();
    let rb = engine.optimize_batch(items, Arc::new(UffBuilder::default()), OptimizeOptions::default()).await.unwrap();
    for (a, b) in ra.iter().zip(&rb) {
        let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
        assert!((a.energy - b.energy).abs() < 1e-6);
    }
}
