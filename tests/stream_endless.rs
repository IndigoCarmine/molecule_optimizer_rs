//! Streaming intermediate structures (`JobHandle::stream` / `frames`) and
//! endless minimization (`force_tol <= 0`), with a small toy force field.

use futures::StreamExt;
use molecule_optimizer_rs::calc::{Frame, OptimizeOptions};
use molecule_optimizer_rs::ff::constraints::{AngleConstraintContrib, DistanceConstraintContrib};
use molecule_optimizer_rs::ff::{Constraint, Contrib, ContribForceField, ForceFieldBuilder};
use molecule_optimizer_rs::optim::{Algorithm, IterationInfo, MinimizeOptions, MinimizeStatus, NoObserver, minimize};
use molecule_optimizer_rs::tasks::{Engine, EngineConfig};
use molecule_optimizer_rs::{Molecule, Result, io};
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::{Duration, Instant};

const BUTANE_XYZ: &str = "14
butane (distorted)
C   0.000  0.000  0.000
C   1.700  0.100  0.000
C   2.000  1.600  0.300
C   3.300  1.700  0.200
H  -0.400 -1.000  0.100
H  -0.300  0.600  0.900
H  -0.500  0.400 -0.800
H   2.000 -0.500  0.900
H   2.100 -0.400 -0.900
H   1.600  2.100 -0.600
H   1.600  2.000  1.300
H   3.800  0.800  0.700
H   3.700  2.600  0.700
H   3.600  1.800 -0.900
";

fn butane() -> Molecule {
    io::read_xyz(BUTANE_XYZ).unwrap()
}

/// Wraps a term and sleeps on every evaluation (slow force field, so a job
/// lasts long enough to publish several intermediate states).
struct Slow(Box<dyn Contrib>, Duration);
impl Contrib for Slow {
    fn energy(&self, pos: &[f64]) -> f64 {
        std::thread::sleep(self.1);
        self.0.energy(pos)
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        self.0.grad(pos, grad)
    }
    fn label(&self) -> &'static str {
        "Slow"
    }
}

/// Bond springs and angle terms; `delay` slows down every energy evaluation.
struct Springs {
    delay: Duration,
}
impl ForceFieldBuilder for Springs {
    fn name(&self) -> &str {
        "Springs"
    }
    fn build(&self, mol: &Molecule, cons: &[Constraint]) -> Result<ContribForceField> {
        let n = mol.num_atoms();
        let mut ff = ContribForceField::new(n, 3);
        let mut terms: Vec<Box<dyn Contrib>> = Vec::new();
        for b in mol.bonds() {
            let d0 = if mol.atom(b.begin).atomic_num == 1 || mol.atom(b.end).atomic_num == 1 { 1.09 } else { 1.53 };
            terms.push(Box::new(DistanceConstraintContrib { i: b.begin, j: b.end, min: d0, max: d0, k: 300.0 }));
        }
        for j in 0..n {
            let nb: Vec<usize> = mol.neighbors(j).collect();
            for x in 0..nb.len() {
                for y in x + 1..nb.len() {
                    terms.push(Box::new(AngleConstraintContrib { i: nb[x], j, k: nb[y], min: 109.5, max: 109.5, force: 0.02 }));
                }
            }
        }
        for (k, t) in terms.into_iter().enumerate() {
            // one slow term is enough to slow down every evaluation
            if k == 0 && !self.delay.is_zero() {
                ff.add_contrib(Box::new(Slow(t, self.delay)));
            } else {
                ff.add_contrib(t);
            }
        }
        ff.add_constraints(&mol.coords, cons);
        Ok(ff)
    }
}

fn engine(interval: Duration) -> Engine {
    Engine::new(EngineConfig { num_threads: Some(2), publish_interval: interval }).unwrap()
}

#[test]
fn endless_minimize_runs_past_convergence_until_break() {
    let mol = butane();
    let ff = Springs { delay: Duration::ZERO }.build(&mol, &[]).unwrap();
    for alg in [Algorithm::Bfgs, Algorithm::Lbfgs(5)] {
        let mut p0 = mol.flat_coords();
        let normal = minimize(&ff, &mut p0, &MinimizeOptions { max_iters: 5000, algorithm: alg, ..Default::default() }, &mut NoObserver);
        assert_eq!(normal.status, MinimizeStatus::Converged, "{alg:?}");
        let stop_at = normal.iterations + 300;
        let mut p = mol.flat_coords();
        let mut seen = 0usize;
        let mut last_iter = 0usize;
        let mut obs = |info: &IterationInfo<'_>| {
            seen += 1;
            assert!(info.iteration > last_iter, "iterations must increase");
            last_iter = info.iteration;
            if info.iteration >= stop_at { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
        };
        // `max_iters` is ignored in endless mode
        let opts = MinimizeOptions { max_iters: 3, force_tol: 0.0, algorithm: alg, ..Default::default() };
        let r = minimize(&ff, &mut p, &opts, &mut obs);
        assert_eq!(r.status, MinimizeStatus::Cancelled, "{alg:?}");
        assert_eq!(r.iterations, stop_at, "{alg:?}");
        assert!(seen >= normal.iterations, "{alg:?}: observer must see every iteration ({seen})");
        assert!(r.energy <= normal.energy + 1e-6, "{alg:?}: endless {} vs converged {}", r.energy, normal.energy);
        assert!(p.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn negative_force_tol_is_endless() {
    let mol = butane();
    let ff = Springs { delay: Duration::ZERO }.build(&mol, &[]).unwrap();
    let mut p = mol.flat_coords();
    let mut obs = |info: &IterationInfo<'_>| if info.iteration >= 500 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) };
    let r = minimize(&ff, &mut p, &MinimizeOptions { max_iters: 10, force_tol: -1.0, ..Default::default() }, &mut obs);
    assert_eq!((r.status, r.iterations), (MinimizeStatus::Cancelled, 500));
}

#[tokio::test(flavor = "multi_thread")]
async fn endless_job_streams_frames_and_stops_promptly_on_cancel() {
    let mol = Arc::new(butane());
    let normal = molecule_optimizer_rs::calc::optimize::optimize(&mol, &Springs { delay: Duration::ZERO }, &OptimizeOptions::default()).unwrap();
    for alg in [Algorithm::Bfgs, Algorithm::Lbfgs(5)] {
        let eng = engine(Duration::from_millis(1));
        let mut opts = OptimizeOptions::default();
        opts.minimize.force_tol = 0.0;
        opts.minimize.max_iters = 5;
        opts.minimize.algorithm = alg;
        let job = eng.optimize(mol.clone(), Arc::new(Springs { delay: Duration::ZERO }), opts);
        let mut frames = job.frames();
        let target = 3 * normal.iterations + 50;
        let mut got: Vec<Frame> = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(60);
        while let Some(f) = tokio::time::timeout(Duration::from_secs(10), frames.next()).await.expect("frame") {
            let past = f.iteration >= target;
            got.push(f);
            if past {
                break;
            }
            assert!(Instant::now() < deadline, "{alg:?}: endless job too slow");
        }
        assert!(!job.is_finished(), "{alg:?}: endless job must not finish by itself");
        assert_eq!(job.progress().borrow().total, 0, "unknown total in endless mode");
        let t0 = Instant::now();
        job.cancel();
        // the stream ends after the final frame
        let rest: Vec<Frame> = tokio::time::timeout(Duration::from_secs(5), frames.collect()).await.expect("stream must end after cancel");
        assert!(t0.elapsed() < Duration::from_secs(2), "{alg:?}: cancel took {:?}", t0.elapsed());
        let r = job.await.unwrap();
        assert_eq!(r.status, MinimizeStatus::Cancelled);
        assert!(r.iterations >= target);
        let last = rest.last().or(got.last()).unwrap();
        assert_eq!(last.coords, r.coords, "{alg:?}: last streamed frame is the result");
        assert_eq!(last.iteration, r.iterations);
        assert!(got.len() >= 3, "{alg:?}: only {} frames", got.len());
        assert!(r.energy <= normal.energy + 1e-6);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn frames_stream_yields_intermediate_and_final_then_ends() {
    let mol = Arc::new(butane());
    let eng = engine(Duration::from_millis(1));
    let mut opts = OptimizeOptions::default();
    opts.minimize.max_iters = 2000;
    let job = eng.optimize(mol.clone(), Arc::new(Springs { delay: Duration::from_micros(300) }), opts);
    // two concurrent streams
    let (a, b) = (job.frames(), job.frames());
    let (fa, fb): (Vec<Frame>, Vec<Frame>) =
        tokio::time::timeout(Duration::from_secs(60), async { futures::join!(a.collect(), b.collect()) }).await.expect("streams must end");
    assert!(job.is_finished());
    let r = job.await.unwrap();
    assert_eq!(r.status, MinimizeStatus::Converged);
    for frames in [&fa, &fb] {
        assert!(frames.len() >= 3, "expected several intermediate frames, got {}", frames.len());
        let last = frames.last().unwrap();
        assert_eq!((last.coords.clone(), last.iteration), (r.coords.clone(), r.iterations), "final frame delivered");
        assert_eq!(last.energy.to_bits(), r.energy.to_bits());
        for w in frames.windows(2) {
            assert!(w[1].iteration >= w[0].iteration);
            assert!(w[1] != w[0], "no consecutive duplicates");
        }
        assert!(frames.iter().filter(|f| f.iteration < r.iterations).count() >= 2, "intermediate frames");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn generic_stream_ends_with_final_state() {
    let mol = Arc::new(butane());
    let eng = engine(Duration::from_millis(1));
    let items: Vec<_> = (0..4).map(|_| (mol.clone(), None)).collect();
    let job = eng.optimize_batch(items, Arc::new(Springs { delay: Duration::from_micros(100) }), OptimizeOptions::default());
    let states: Vec<_> = tokio::time::timeout(Duration::from_secs(60), job.stream().collect()).await.expect("stream must end");
    assert!(!states.is_empty());
    let last = states.last().unwrap();
    assert!(last.items.iter().all(|i| i.as_ref().is_some_and(|s| s.finished)), "final state has every item finished");
    let r = job.await.unwrap();
    assert_eq!(r.len(), 4);
    // a stream opened after the job finished yields the final state once
    let job = eng.optimize(mol, Arc::new(Springs { delay: Duration::ZERO }), OptimizeOptions::default());
    while !job.is_finished() {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let late: Vec<_> = job.stream().collect().await;
    assert_eq!(late.len(), 1);
    assert!(late[0].finished);
}
