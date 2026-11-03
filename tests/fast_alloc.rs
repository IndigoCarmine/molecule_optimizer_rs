//! Steady-state `energy_grad` / `energy` calls of a `FastForceField` do not
//! allocate (counting global allocator; one test in this binary so nothing
//! else runs concurrently).

use molecule_optimizer_rs::fast::{FastBuilder, FastOptions};
use molecule_optimizer_rs::ff::mmff::MmffBuilder;
use molecule_optimizer_rs::ff::uff::UffBuilder;
use molecule_optimizer_rs::ff::{Constraint, ForceField};
use molecule_optimizer_rs::{Atom, Molecule};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn paracetamol() -> Molecule {
    molecule_optimizer_rs::io::read_file(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data/paracetamol.mol")).unwrap().remove(0)
}

fn lattice(mol: &Molecule, n: usize, spacing: f64) -> Molecule {
    let mut out = Molecule::new();
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let off = out.num_atoms();
                for i in 0..mol.num_atoms() {
                    let a: Atom = mol.atom(i).clone();
                    let p = mol.coords[i];
                    out.add_atom(a, [p[0] + x as f64 * spacing, p[1] + y as f64 * spacing, p[2] + z as f64 * spacing]);
                }
                for b in mol.bonds() {
                    out.add_bond(b.begin + off, b.end + off, b.order);
                }
            }
        }
    }
    molecule_optimizer_rs::perception::sanitize(&mut out).unwrap();
    out
}

fn count(ff: &dyn ForceField, pos: &[f64]) -> usize {
    let mut g = vec![0.0; pos.len()];
    // warm up (first calls may size the workspaces)
    for _ in 0..3 {
        ff.energy_grad(pos, &mut g);
        ff.energy(pos);
    }
    let before = ALLOCS.load(Ordering::SeqCst);
    let mut s = 0.0;
    for _ in 0..50 {
        s += ff.energy_grad(pos, &mut g);
        s += ff.energy(pos);
    }
    let n = ALLOCS.load(Ordering::SeqCst) - before;
    assert!(s.is_finite());
    n
}

#[test]
fn energy_grad_does_not_allocate() {
    let mol = paracetamol();
    let pos = mol.flat_coords();
    let cons = [Constraint::distance(0, 8, 6.0, 10.0), Constraint::Fixed { i: 3 }];
    let serial_m = FastBuilder::new(MmffBuilder::default(), FastOptions::exact()).build_fast(&mol, &cons).unwrap();
    let serial_u = FastBuilder::new(UffBuilder::default(), FastOptions::exact()).build_fast(&mol, &cons).unwrap();
    assert_eq!(count(&serial_m, &pos), 0, "serial MMFF");
    assert_eq!(count(&serial_u, &pos), 0, "serial UFF");

    // parallel plan (fixed chunks, per-chunk buffers), run inside a pool
    let big = lattice(&mol, 3, 7.0);
    let bpos = big.flat_coords();
    let opts = FastOptions::exact().parallel_grain(64);
    let par_m = FastBuilder::new(MmffBuilder { ignore_interfrag_interactions: false, ..Default::default() }, opts.clone())
        .build_fast(&big, &[])
        .unwrap();
    let par_u = FastBuilder::new(UffBuilder { ignore_interfrag_interactions: false, ..Default::default() }, opts).build_fast(&big, &[]).unwrap();
    assert!(!par_m.bonded().chunks.is_empty());
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let (a, b) = pool.install(|| (count(&par_m, &bpos), count(&par_u, &bpos)));
    println!("parallel allocations per 100 calls: MMFF {a}, UFF {b}");
    assert_eq!(a, 0, "parallel MMFF");
    assert_eq!(b, 0, "parallel UFF");
}
