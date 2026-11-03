//! Async/parallel calculation tests with a toy force field (independent of
//! the UFF/MMFF ports).

use molecule_optimizer_rs::calc::conformer::RandomTorsionGenerator;
use molecule_optimizer_rs::calc::{self, ConformerSearchOptions, OptimizeOptions, ScanCoordinate, ScanMode, ScanOptions};
use molecule_optimizer_rs::core::geometry::dihedral;
use molecule_optimizer_rs::ff::constraints::{AngleConstraintContrib, DistanceConstraintContrib};
use molecule_optimizer_rs::ff::{Constraint, Contrib, ContribForceField, ForceFieldBuilder, utils::p3};
use molecule_optimizer_rs::tasks::{Engine, EngineConfig};
use molecule_optimizer_rs::{Molecule, Result, io};
use std::sync::Arc;
use std::time::Duration;

/// 3-fold torsion term with numerical gradient.
struct Tors {
    idx: [usize; 4],
    v: f64,
}
impl Contrib for Tors {
    fn energy(&self, pos: &[f64]) -> f64 {
        let [a, b, c, d] = self.idx;
        let phi = dihedral(p3(pos, a), p3(pos, b), p3(pos, c), p3(pos, d));
        0.5 * self.v * (1.0 + (3.0 * phi).cos())
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let mut p = pos.to_vec();
        for &i in &self.idx {
            for k in 0..3 {
                let x = p[3 * i + k];
                p[3 * i + k] = x + 1e-6;
                let ep = self.energy(&p);
                p[3 * i + k] = x - 1e-6;
                let em = self.energy(&p);
                p[3 * i + k] = x;
                grad[3 * i + k] += (ep - em) / 2e-6;
            }
        }
    }
    fn label(&self) -> &'static str {
        "Torsion"
    }
}

/// Repulsive pair term.
struct Rep {
    i: usize,
    j: usize,
}
impl Contrib for Rep {
    fn energy(&self, pos: &[f64]) -> f64 {
        let d = molecule_optimizer_rs::ff::utils::distance(pos, self.i, self.j);
        0.5 * (2.5 / d).powi(12)
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let a = p3(pos, self.i);
        let b = p3(pos, self.j);
        let d = molecule_optimizer_rs::ff::utils::distance(pos, self.i, self.j);
        let de = -6.0 * (2.5f64).powi(12) / d.powi(13);
        for k in 0..3 {
            let g = de * (a[k] - b[k]) / d;
            grad[3 * self.i + k] += g;
            grad[3 * self.j + k] -= g;
        }
    }
    fn label(&self) -> &'static str {
        "VdW"
    }
}

struct Toy;
#[allow(clippy::needless_range_loop)]
impl ForceFieldBuilder for Toy {
    fn name(&self) -> &str {
        "Toy"
    }
    fn build(&self, mol: &Molecule, cons: &[Constraint]) -> Result<ContribForceField> {
        let n = mol.num_atoms();
        let mut ff = ContribForceField::new(n, 3);
        for b in mol.bonds() {
            let d0 = if mol.atom(b.begin).atomic_num == 1 || mol.atom(b.end).atomic_num == 1 { 1.09 } else { 1.53 };
            ff.add_contrib(Box::new(DistanceConstraintContrib { i: b.begin, j: b.end, min: d0, max: d0, k: 600.0 }));
        }
        let topo = mol.topological_distances();
        for j in 0..n {
            let nb: Vec<usize> = mol.neighbors(j).collect();
            for x in 0..nb.len() {
                for y in x + 1..nb.len() {
                    ff.add_contrib(Box::new(AngleConstraintContrib { i: nb[x], j, k: nb[y], min: 109.5, max: 109.5, force: 0.02 }));
                }
            }
        }
        for b in mol.bonds() {
            let (j, k) = (b.begin, b.end);
            let a = mol.neighbors(j).find(|&x| x != k && mol.atom(x).atomic_num > 1);
            let d = mol.neighbors(k).find(|&x| x != j && mol.atom(x).atomic_num > 1);
            if let (Some(a), Some(d)) = (a, d) {
                ff.add_contrib(Box::new(Tors { idx: [a, j, k, d], v: 3.0 }));
            }
        }
        for i in 0..n {
            for j in i + 1..n {
                if topo[i][j] >= 3 && topo[i][j] != usize::MAX {
                    ff.add_contrib(Box::new(Rep { i, j }));
                }
            }
        }
        ff.add_constraints(&mol.coords, cons);
        Ok(ff)
    }
}

const PENTANE_XYZ: &str = "17
pentane
C   0.000  0.000  0.000
C   1.530  0.000  0.000
C   2.040  1.440  0.000
C   3.570  1.440  0.100
C   4.080  2.880  0.150
H  -0.360 -1.030  0.000
H  -0.360  0.510  0.890
H  -0.360  0.510 -0.890
H   1.890 -0.510  0.890
H   1.890 -0.510 -0.890
H   1.680  1.950 -0.890
H   1.680  1.950  0.890
H   3.930  0.930  0.990
H   3.930  0.930 -0.790
H   5.170  2.880  0.150
H   3.720  3.390  1.040
H   3.720  3.390 -0.740
";

fn pentane() -> Molecule {
    io::read_xyz(PENTANE_XYZ).unwrap()
}

fn engine(n: usize, interval_ms: u64) -> Engine {
    Engine::new(EngineConfig { num_threads: Some(n), publish_interval: Duration::from_millis(interval_ms) }).unwrap()
}

#[test]
fn pentane_perceived() {
    let m = pentane();
    assert_eq!(m.num_bonds(), 16);
    assert_eq!(m.formula(), "C5H12");
}

#[tokio::test(flavor = "multi_thread")]
async fn optimize_with_live_snapshots() {
    let eng = engine(2, 0);
    let mol = Arc::new(pentane());
    let mut opts = OptimizeOptions::default();
    opts.minimize.max_iters = 5000;
    opts.minimize.force_tol = 1e-10;
    let h = eng.optimize(mol.clone(), Arc::new(Toy), opts);
    let mut last = 0;
    let mut energies = Vec::new();
    for _ in 0..10 {
        let s = h.request_snapshot_timeout(Duration::from_secs(5)).await;
        assert!(s.version >= last);
        last = s.version;
        assert_eq!(s.data.frame.coords.len(), mol.num_atoms());
        assert!(s.data.frame.coords.iter().flatten().all(|x| x.is_finite()));
        if s.data.frame.energy.is_finite() {
            energies.push(s.data.frame.energy);
        }
        if s.data.finished {
            break;
        }
    }
    let r = h.await.unwrap();
    for w in energies.windows(2) {
        assert!(w[1] <= w[0] + 1e-9, "energy must not increase: {energies:?}");
    }
    let sync = calc::optimize::optimize(&mol, &Toy, &OptimizeOptions {
        minimize: molecule_optimizer_rs::optim::MinimizeOptions { max_iters: 5000, force_tol: 1e-10, ..Default::default() },
        ..Default::default()
    })
    .unwrap();
    assert!((r.energy - sync.energy).abs() < 1e-9, "async and sync results must agree");
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_returns_current_geometry() {
    let eng = engine(1, 0);
    let mol = Arc::new(pentane());
    let mut opts = OptimizeOptions::default();
    opts.minimize.max_iters = 1_000_000;
    opts.minimize.force_tol = 1e-300;
    let h = eng.optimize(mol, Arc::new(Toy), opts);
    let _ = h.request_snapshot().await;
    h.cancel();
    let r = h.await.unwrap();
    assert!(matches!(r.status, molecule_optimizer_rs::optim::MinimizeStatus::Cancelled | molecule_optimizer_rs::optim::MinimizeStatus::Converged));
}

fn scan_opts(mode: ScanMode) -> ScanOptions {
    ScanOptions {
        coordinates: vec![ScanCoordinate::Dihedral { i: 0, j: 1, k: 2, l: 3, values: ScanCoordinate::steps(-180.0, 30.0, 12) }],
        mode,
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn dihedral_scan_parallel_matches_sequential() {
    let mol = Arc::new(pentane());
    for mode in [ScanMode::Relaxed, ScanMode::Independent] {
        let r4 = engine(4, 10).scan(mol.clone(), Arc::new(Toy), scan_opts(mode)).await.unwrap();
        let r1 = engine(1, 10).scan(mol.clone(), Arc::new(Toy), scan_opts(mode)).await.unwrap();
        assert_eq!(r4.points.len(), 12);
        for (a, b) in r4.points.iter().zip(&r1.points) {
            assert!((a.energy - b.energy).abs() < 1e-10);
            let ach = a.achieved[0];
            let mut diff = (ach - a.values[0]).rem_euclid(360.0);
            if diff > 180.0 {
                diff -= 360.0;
            }
            assert!(diff.abs() < 1.0, "target {} achieved {}", a.values[0], ach);
        }
        // 3-fold torsion: staggered (180) lower than eclipsed (120 / 0)
        let e = |v: f64| r4.points.iter().find(|p| (p.values[0] - v).abs() < 1e-9).unwrap().energy;
        assert!(e(-180.0) < e(0.0));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn two_dimensional_scan() {
    let mol = Arc::new(pentane());
    let opts = ScanOptions {
        coordinates: vec![
            ScanCoordinate::Dihedral { i: 0, j: 1, k: 2, l: 3, values: ScanCoordinate::steps(-180.0, 60.0, 6) },
            ScanCoordinate::Dihedral { i: 1, j: 2, k: 3, l: 4, values: ScanCoordinate::steps(-180.0, 60.0, 6) },
        ],
        ..Default::default()
    };
    let h = engine(4, 5).scan(mol, Arc::new(Toy), opts);
    let snap = h.request_snapshot().await;
    assert_eq!(snap.data.points.len(), 36);
    let r = h.await.unwrap();
    assert_eq!(r.points.len(), 36);
    assert_eq!(r.shape, vec![6, 6]);
}

#[tokio::test(flavor = "multi_thread")]
async fn conformer_search_finds_all_anti() {
    let mol = Arc::new(pentane());
    let mut opts = ConformerSearchOptions::new(Arc::new(RandomTorsionGenerator { seed: 7 }));
    opts.num_conformers = 40;
    let h = engine(4, 5).conformer_search(mol.clone(), Arc::new(Toy), opts.clone());
    let s = h.request_snapshot().await;
    assert_eq!(s.data.total, 40);
    let r = h.await.unwrap();
    assert!(r.conformers.len() >= 2);
    let best = r.global_minimum().unwrap();
    let d1 = dihedral(best.coords[0], best.coords[1], best.coords[2], best.coords[3]).to_degrees();
    let d2 = dihedral(best.coords[1], best.coords[2], best.coords[3], best.coords[4]).to_degrees();
    assert!(d1.abs() > 150.0 && d2.abs() > 150.0, "global minimum should be anti/anti: {d1} {d2}");
    // deterministic regardless of thread count
    let r1 = engine(1, 5).conformer_search(mol, Arc::new(Toy), opts).await.unwrap();
    assert_eq!(r.conformers.len(), r1.conformers.len());
    assert!((r.conformers[0].energy - r1.conformers[0].energy).abs() < 1e-10);
}
