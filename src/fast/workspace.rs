//! Preallocated evaluation buffers.
//!
//! Parallel evaluation gives every fixed chunk its own gradient buffer
//! (`3N` values) and energy slot; the buffers are reduced in chunk order.
//! A [`WorkspacePool`] keeps such buffer sets for reuse: a call takes one
//! (allocating only if every set is in use by a concurrent caller) and puts
//! it back afterwards, so steady-state `energy_grad` calls do not touch the
//! heap.

use rayon::prelude::*;
use std::sync::Mutex;

/// Buffers of one evaluation.
#[derive(Debug, Default)]
pub struct Workspace {
    bufs: Vec<f64>,
    energies: Vec<[f64; 3]>,
}

impl Workspace {
    pub fn new(n_chunks: usize, len: usize) -> Self {
        Workspace { bufs: vec![0.0; n_chunks * len], energies: vec![[0.0; 3]; n_chunks] }
    }
    /// Gradient buffers (chunk-major, `len` values each) and energy slots.
    pub fn split(&mut self) -> (&mut [f64], &mut [[f64; 3]]) {
        (&mut self.bufs, &mut self.energies)
    }
}

/// Pool of [`Workspace`]s of one force field.
#[derive(Debug)]
pub struct WorkspacePool {
    n_chunks: usize,
    len: usize,
    pool: Mutex<Vec<Workspace>>,
}

impl WorkspacePool {
    /// One workspace is allocated up front (none if `n_chunks == 0`).
    pub fn new(n_chunks: usize, len: usize) -> Self {
        let mut v = Vec::with_capacity(4);
        if n_chunks > 0 {
            v.push(Workspace::new(n_chunks, len));
        }
        WorkspacePool { n_chunks, len, pool: Mutex::new(v) }
    }

    /// Runs `f` with a workspace from the pool.
    pub fn with<R>(&self, f: impl FnOnce(&mut Workspace) -> R) -> R {
        let taken = self.pool.lock().unwrap_or_else(|e| e.into_inner()).pop();
        let mut ws = taken.unwrap_or_else(|| Workspace::new(self.n_chunks, self.len));
        let r = f(&mut ws);
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).push(ws);
        r
    }
}

/// Block size of the parallel reduction.
const REDUCE_BLOCK: usize = 4096;

/// Calls `f(k0, k1, &mut out[k0..k1])` for fixed blocks of `out[..len]`
/// (in parallel for large `len`). The block boundaries do not depend on the
/// thread count, and `f` must add the chunk buffers in chunk order, so the
/// result is deterministic.
pub fn reduce_into(out: &mut [f64], len: usize, f: impl Fn(usize, usize, &mut [f64]) + Sync) {
    let out = &mut out[..len];
    if len <= REDUCE_BLOCK {
        f(0, len, out);
    } else {
        out.par_chunks_mut(REDUCE_BLOCK).enumerate().for_each(|(b, o)| {
            let k0 = b * REDUCE_BLOCK;
            f(k0, k0 + o.len(), o);
        });
    }
}
