//! Exact nonbonded backend: the pair set the RDKit-compatible builder
//! selects (frozen at build time), evaluated with the same per-pair energy
//! expressions from a compressed-row (CSR) pair list with SIMD kernels.
//!
//! * Pairs are stored row-wise (`i` = first atom of the RDKit pair); per
//!   pair only the partner index, a parameter-table index, flags and (MMFF)
//!   the charge term are stored. Pair parameters live in a small table
//!   indexed by the (atom type, atom type) combination, so the hot loop
//!   streams ~15 bytes per pair.
//! * Four pairs of a row are evaluated at once; the force on the row atom
//!   is accumulated in registers, forces on the partners are scattered in
//!   pair order.
//! * Parallel evaluation splits the rows into fixed chunks with balanced
//!   pair counts (depending only on the pair list and the grain); each
//!   chunk has its own gradient buffer and the buffers are reduced in chunk
//!   order, so results are bit-identical for any thread count.
//! * [`NbTemplate::select`] re-selects the pair set for a new geometry with
//!   the builder's exact rule (topological exclusions, fragments and the
//!   build-time distance threshold) using a cell grid, i.e. in
//!   O(N + pairs) instead of O(N²) when the threshold is short compared to
//!   the molecule.

use super::backend::{NonbondedBackend, NonbondedEnergy};
use super::bonded::{Lane, P, V4, dot3, sub3};
use super::workspace::WorkspacePool;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

// MMFF buffered 14-7 constants (RDKit `MMFF::Utils`).
const VDW1: f64 = 1.07;
const VDW1M1: f64 = VDW1 - 1.0;
const VDW2: f64 = 1.12;
const VDW2M1: f64 = VDW2 - 1.0;
const VDW2T7: f64 = VDW2 * 7.0;
const DIEL: f64 = 332.0716;

/// Pair flags (MMFF).
pub(crate) const F_VDW: u8 = 1;
pub(crate) const F_ELE: u8 = 2;
pub(crate) const F_14: u8 = 4;
pub(crate) const F_DIST: u8 = 8;

/// Pair-parameter table: one record per (type, type) combination.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ParamTable {
    /// UFF Lennard-Jones: `[x_ij, D_ij, 10·x_ij (runtime cutoff), 12·D_ij/x_ij]`.
    Uff(Vec<[f64; 4]>),
    /// MMFF buffered 14-7: `[R*, eps, eps/R*, 1.07·R*, 0.07·R*, 1.12·R*⁷,
    /// 0.12·R*⁷, 0]`.
    Mmff(Vec<[f64; 8]>),
    #[default]
    Empty,
}

impl ParamTable {
    fn new_uff() -> Self {
        ParamTable::Uff(Vec::new())
    }
    fn new_mmff() -> Self {
        ParamTable::Mmff(Vec::new())
    }
    fn len(&self) -> usize {
        match self {
            ParamTable::Uff(v) => v.len(),
            ParamTable::Mmff(v) => v.len(),
            ParamTable::Empty => 0,
        }
    }
    /// Adds a UFF entry (`xij`, `well_depth`, `thresh` as in `VdwContrib`).
    fn push_uff(&mut self, x: f64, d: f64, th: f64) {
        if let ParamTable::Uff(v) = self {
            v.push([x, d, th, 12. * d / x]);
        }
    }
    /// Adds an MMFF entry (`R*_ij`, `eps_ij`).
    fn push_mmff(&mut self, rs: f64, e: f64) {
        if let ParamTable::Mmff(v) = self {
            let r2 = rs * rs;
            let r7 = r2 * r2 * r2 * rs;
            v.push([rs, e, e / rs, VDW1 * rs, VDW1M1 * rs, VDW2 * r7, VDW2M1 * r7, 0.0]);
        }
    }
}

/// One pair before compression.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pair {
    pub i: u32,
    pub j: u32,
    pub p: u32,
    pub flags: u8,
    pub ct: f64,
}

/// A parallel chunk: rows and the lowest atom index it touches.
#[derive(Debug, Clone, PartialEq)]
struct NbChunk {
    rows: Range<usize>,
    lo: usize,
}

/// Exact nonbonded backend (see the module documentation).
pub struct ExactNonbonded {
    mmff: bool,
    n_atoms: usize,
    row_start: Vec<u32>,
    j: Vec<u32>,
    p: Vec<u32>,
    flags: Vec<u8>,
    ct: Vec<f64>,
    params: Arc<ParamTable>,
    has_vdw: bool,
    has_ele: bool,
    chunks: Vec<NbChunk>,
    simd: bool,
    ws: WorkspacePool,
}

impl std::fmt::Debug for ExactNonbonded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExactNonbonded")
            .field("mmff", &self.mmff)
            .field("n_atoms", &self.n_atoms)
            .field("pairs", &self.j.len())
            .field("chunks", &self.chunks.len())
            .finish()
    }
}

impl ExactNonbonded {
    /// Builds the backend from pairs ordered by row (`i`).
    pub(crate) fn from_sorted_pairs(
        mmff: bool,
        n_atoms: usize,
        pairs: &[Pair],
        params: Arc<ParamTable>,
        grain: usize,
        simd: bool,
    ) -> Self {
        // the kernels index without bounds checks: validate here
        let np = params.len();
        assert!(n_atoms < u32::MAX as usize / 3, "too many atoms");
        let mut row_start = vec![0u32; n_atoms + 1];
        let mut last = 0u32;
        for pr in pairs {
            assert!(pr.i >= last, "pairs must be ordered by their first atom");
            assert!((pr.i as usize) < n_atoms && (pr.j as usize) < n_atoms, "pair atom index out of range");
            assert!((pr.p as usize) < np, "pair parameter index out of range");
            last = pr.i;
            row_start[pr.i as usize + 1] += 1;
        }
        for k in 0..n_atoms {
            row_start[k + 1] += row_start[k];
        }
        let j: Vec<u32> = pairs.iter().map(|p| p.j).collect();
        let p: Vec<u32> = pairs.iter().map(|p| p.p).collect();
        let flags: Vec<u8> = pairs.iter().map(|p| p.flags).collect();
        let ct: Vec<f64> = if mmff { pairs.iter().map(|p| p.ct).collect() } else { Vec::new() };
        let has_vdw = if mmff { flags.iter().any(|f| f & F_VDW != 0) } else { !pairs.is_empty() };
        let has_ele = mmff && flags.iter().any(|f| f & F_ELE != 0);
        let mut nb = ExactNonbonded {
            mmff,
            n_atoms,
            row_start,
            j,
            p,
            flags,
            ct,
            params,
            has_vdw,
            has_ele,
            chunks: Vec::new(),
            simd,
            ws: WorkspacePool::new(0, 0),
        };
        nb.plan(grain);
        nb
    }

    /// Fixed chunk plan (rows with balanced pair counts).
    fn plan(&mut self, grain: usize) {
        let total = self.j.len();
        let n_chunks = (total / grain.max(1)).clamp(1, super::bonded::MAX_CHUNKS);
        self.chunks.clear();
        if n_chunks > 1 {
            let per = total.div_ceil(n_chunks);
            let mut start = 0;
            let mut acc = 0;
            for i in 0..self.n_atoms {
                acc += (self.row_start[i + 1] - self.row_start[i]) as usize;
                if acc >= per || i + 1 == self.n_atoms {
                    if acc > 0 {
                        let rows = start..i + 1;
                        let (s, e) = (self.row_start[rows.start] as usize, self.row_start[rows.end] as usize);
                        let lo = self.j[s..e].iter().map(|&j| j as usize).min().unwrap_or(rows.start).min(rows.start);
                        self.chunks.push(NbChunk { rows, lo });
                    }
                    start = i + 1;
                    acc = 0;
                }
            }
            if self.chunks.len() <= 1 {
                self.chunks.clear();
            }
        }
        self.ws = WorkspacePool::new(self.chunks.len(), 3 * self.n_atoms);
    }

    /// Number of pairs.
    pub fn num_pairs(&self) -> usize {
        self.j.len()
    }

    /// The pairs `(i, j)` in evaluation order.
    pub fn pairs(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(self.j.len());
        for i in 0..self.n_atoms {
            for t in self.row_start[i] as usize..self.row_start[i + 1] as usize {
                out.push((i, self.j[t] as usize));
            }
        }
        out
    }

    /// `true` if any pair has a van der Waals / electrostatic part.
    pub fn has_vdw(&self) -> bool {
        self.has_vdw
    }
    pub fn has_electrostatics(&self) -> bool {
        self.has_ele
    }

    /// Selects SIMD or scalar kernels.
    pub fn set_simd(&mut self, simd: bool) {
        self.simd = simd;
    }

    /// Adds `dg` to row atom `i` and `-dg` to the partners, pair by pair
    /// (the reference contribs' order).
    ///
    /// Safety: `t + L::N <= num_pairs`, `grad.len() >= 3 * n_atoms`, `i < n_atoms`.
    #[inline(always)]
    unsafe fn scatter_pairs<L: Lane>(&self, grad: &mut [f64], i: usize, t: usize, dg: P<L>) {
        let mut v = [[0.0f64; 4]; 3];
        for d in 0..3 {
            dg[d].write(&mut v[d]);
        }
        let a = 3 * i;
        for k in 0..L::N {
            unsafe {
                let b = 3 * *self.j.get_unchecked(t + k) as usize;
                for d in 0..3 {
                    *grad.get_unchecked_mut(a + d) += v[d][k];
                    *grad.get_unchecked_mut(b + d) += -v[d][k];
                }
            }
        }
    }

    /// Partner coordinates of pairs `t .. t + L::N`.
    ///
    /// Safety: `t + L::N <= num_pairs`, `pos.len() >= 3 * n_atoms`.
    #[inline(always)]
    unsafe fn ldj<L: Lane>(&self, pos: &[f64], t: usize) -> P<L> {
        unsafe {
            let j = |k: usize| 3 * *self.j.get_unchecked(t + k) as usize;
            [
                L::from_fn(|k| *pos.get_unchecked(j(k))),
                L::from_fn(|k| *pos.get_unchecked(j(k) + 1)),
                L::from_fn(|k| *pos.get_unchecked(j(k) + 2)),
            ]
        }
    }

    /// Parameter records of pairs `t .. t + L::N`, field `f`.
    ///
    /// Safety: `t + L::N <= num_pairs` (parameter indices are validated).
    #[inline(always)]
    unsafe fn par<L: Lane, const W: usize>(&self, tab: &[[f64; W]], t: usize, f: usize) -> L {
        unsafe { L::from_fn(|k| *tab.get_unchecked(*self.p.get_unchecked(t + k) as usize).get_unchecked(f)) }
    }

    /// Safety: see [`Self::eval_rows`].
    #[inline(always)]
    unsafe fn uff_block<L: Lane, const G: bool>(&self, pos: &[f64], grad: &mut [f64], i: usize, t: usize, xi: P<L>) -> L {
        let ParamTable::Uff(tab) = &*self.params else { unreachable!() };
        let pj = unsafe { self.ldj::<L>(pos, t) };
        let g = |f: usize| unsafe { self.par::<L, 4>(tab, t, f) };
        let (x, d_, th) = (g(0), g(1), g(2));
        let d = sub3(xi, pj);
        let dist = dot3(d, d).sqrt();
        let in_r = !dist.gtv(th);
        let posd = dist.gt(0.0);
        let r = x / dist;
        let r3 = r * r * r;
        let r6 = r3 * r3;
        let r12 = r6 * r6;
        let e = L::select(in_r & posd, d_ * (r12 - r6 * 2.0), L::splat(0.0));
        if G {
            let pre = g(3) * (r6 * r - r12 * r);
            let z = L::splat(0.0);
            let h = L::splat(100.0);
            let dg: P<L> = [
                L::select(in_r, L::select(posd, pre * d[0] / dist, h), z),
                L::select(in_r, L::select(posd, pre * d[1] / dist, h), z),
                L::select(in_r, L::select(posd, pre * d[2] / dist, h), z),
            ];
            unsafe { self.scatter_pairs(grad, i, t, dg) };
        }
        e
    }

    /// Safety: see [`Self::eval_rows`].
    #[inline(always)]
    unsafe fn mmff_block<L: Lane, const G: bool>(&self, pos: &[f64], grad: &mut [f64], i: usize, t: usize, xi: P<L>) -> (L, L) {
        let ParamTable::Mmff(tab) = &*self.params else { unreachable!() };
        let pj = unsafe { self.ldj::<L>(pos, t) };
        let g = |f: usize| unsafe { self.par::<L, 8>(tab, t, f) };
        let fl = |k: usize| unsafe { *self.flags.get_unchecked(t + k) };
        let bit = |b: u8| L::from_fn(|k| if fl(k) & b != 0 { 1.0 } else { 0.0 }).gt(0.5);
        let (vm, em, m14, dm) = (bit(F_VDW), bit(F_ELE), bit(F_14), bit(F_DIST));
        let ct = L::from_fn(|k| unsafe { *self.ct.get_unchecked(t + k) });
        let z = L::splat(0.0);
        let d = sub3(xi, pj);
        let dist = dot3(d, d).sqrt();
        // van der Waals energy (calcVdWEnergy)
        let dist2 = dist * dist;
        let dist7 = dist2 * dist2 * dist2 * dist;
        let a = g(3) / (dist + g(4));
        let a2 = a * a;
        let a7 = a2 * a2 * a2 * a;
        let b = g(5) / (dist7 + g(6)) - 2.0;
        let ev = L::select(vm, g(1) * a7 * b, z);
        // electrostatics (calcEleEnergy)
        let corr = dist + 0.05;
        let sc = L::select(m14, L::splat(0.75), L::splat(1.0));
        let ee = L::select(em, ct * DIEL / L::select(dm, corr * corr, corr) * sc, z);
        if G {
            let rr = g(0);
            let q = dist / rr;
            let q2 = q * q;
            let q6 = q2 * q2 * q2;
            let q7 = q6 * q;
            let q7p = q7 + VDW2M1;
            let tt = L::splat(VDW1) / (q + VDW1 - 1.0);
            let t2 = tt * tt;
            let t7 = t2 * t2 * t2 * tt;
            let de_v = g(2) * t7 * (q6 * -VDW2T7 / (q7p * q7p) + ((L::splat(-VDW2T7) / q7p + 14.0) / (q + VDW1M1)));
            let vg = L::select(vm, de_v / dist, z);
            let cg = corr * L::select(dm, corr * corr, corr);
            let bm = L::select(dm, L::splat(-DIEL * 2.0), L::splat(-DIEL * 1.0));
            let de_e = bm * ct / cg * sc;
            let eg = L::select(em, de_e / dist, z);
            let de = vg + eg;
            // coincident atoms: RDKit's constant push-apart
            let posd = dist.gt(0.0);
            let special = L::select(vm, rr * 0.01, z) + L::select(em, L::splat(0.02), z);
            let dg: P<L> = [
                L::select(posd, de * d[0], special),
                L::select(posd, de * d[1], special),
                L::select(posd, de * d[2], special),
            ];
            unsafe { self.scatter_pairs(grad, i, t, dg) };
        }
        (ev, ee)
    }

    /// Evaluates rows `rows` (SIMD blocks, scalar tails). Pair energies are
    /// added to `acc` pair by pair (vdW then electrostatics, as the
    /// reference contribs do); returns the separate (vdW, electrostatic)
    /// sums.
    fn eval_rows<L: Lane, const G: bool>(&self, pos: &[f64], grad: &mut [f64], rows: Range<usize>, acc: &mut f64) -> (f64, f64) {
        // bounds for the unchecked kernels (indices validated at construction)
        assert!(pos.len() >= 3 * self.n_atoms && (!G || grad.len() >= 3 * self.n_atoms) && rows.end <= self.n_atoms);
        let (mut ev, mut ee) = (0.0, 0.0);
        let mut lv = [0.0f64; 4];
        let mut le = [0.0f64; 4];
        for i in rows {
            let (s, e) = (self.row_start[i] as usize, self.row_start[i + 1] as usize);
            if s == e {
                continue;
            }
            let xs = [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];
            let xi = [L::splat(xs[0]), L::splat(xs[1]), L::splat(xs[2])];
            let mut t = s;
            if self.mmff {
                // Absent parts are 0.0; a running sum that starts at +0.0 can
                // never be -0.0, so adding them is bitwise neutral.
                while t + L::N <= e {
                    let (v, el) = unsafe { self.mmff_block::<L, G>(pos, grad, i, t, xi) };
                    v.write(&mut lv);
                    el.write(&mut le);
                    for k in 0..L::N {
                        *acc += lv[k];
                        *acc += le[k];
                        ev += lv[k];
                        ee += le[k];
                    }
                    t += L::N;
                }
                while t < e {
                    let (v, el) = unsafe { self.mmff_block::<f64, G>(pos, grad, i, t, xs) };
                    *acc += v;
                    *acc += el;
                    ev += v;
                    ee += el;
                    t += 1;
                }
            } else {
                while t + L::N <= e {
                    unsafe { self.uff_block::<L, G>(pos, grad, i, t, xi) }.write(&mut lv);
                    for x in &lv[..L::N] {
                        *acc += *x;
                        ev += *x;
                    }
                    t += L::N;
                }
                while t < e {
                    let x = unsafe { self.uff_block::<f64, G>(pos, grad, i, t, xs) };
                    *acc += x;
                    ev += x;
                    t += 1;
                }
            }
        }
        (ev, ee)
    }

    fn eval_rows_dispatch<const G: bool>(&self, pos: &[f64], grad: &mut [f64], rows: Range<usize>, acc: &mut f64) -> (f64, f64) {
        if self.simd { self.eval_rows::<V4, G>(pos, grad, rows, acc) } else { self.eval_rows::<f64, G>(pos, grad, rows, acc) }
    }

    /// Evaluates all pairs, adds the energy to the running total `acc` in
    /// the reference contribs' order (UFF: pair by pair; MMFF: the pair sum
    /// of the single nonbonded contrib) and the gradient to `grad` (if `G`).
    pub fn eval_into<const G: bool>(&self, pos: &[f64], grad: &mut [f64], acc: &mut f64) -> NonbondedEnergy {
        self.eval_overlapped::<G, ()>(pos, grad, acc, |_| ()).0
    }

    /// `true` if the pairs are evaluated in parallel chunks.
    pub fn is_parallel(&self) -> bool {
        !self.chunks.is_empty()
    }

    /// Like [`Self::eval_into`], but runs `other(grad)` concurrently with
    /// the parallel pair chunks (which write to their own buffers); the
    /// chunk buffers are added to `grad` after both have finished, so the
    /// result is deterministic. Without a parallel plan the pairs are
    /// evaluated first, then `other`.
    pub fn eval_overlapped<const G: bool, R: Send>(
        &self,
        pos: &[f64],
        grad: &mut [f64],
        acc: &mut f64,
        other: impl FnOnce(&mut [f64]) -> R + Send,
    ) -> (NonbondedEnergy, R) {
        if self.chunks.is_empty() {
            let (v, e) = if self.mmff {
                let mut local = 0.0;
                let r = self.eval_rows_dispatch::<G>(pos, grad, 0..self.n_atoms, &mut local);
                *acc += local;
                r
            } else {
                self.eval_rows_dispatch::<G>(pos, grad, 0..self.n_atoms, acc)
            };
            return (NonbondedEnergy { vdw: v, electrostatic: e }, other(grad));
        }
        let len = 3 * self.n_atoms;
        self.ws.with(|ws| {
            let (bufs, energies) = ws.split();
            let r = if G {
                let (r, ()) = rayon::join(
                    || other(&mut *grad),
                    || {
                        bufs.par_chunks_mut(len).zip(energies.par_iter_mut()).zip(self.chunks.par_iter()).for_each(
                            |((buf, en), ch)| {
                                buf[3 * ch.lo..].fill(0.0);
                                let mut c = 0.0;
                                let (v, e) = self.eval_rows_dispatch::<true>(pos, buf, ch.rows.clone(), &mut c);
                                *en = [c, v, e];
                            },
                        )
                    },
                );
                let chunks = &self.chunks;
                let bufs: &[f64] = bufs;
                super::workspace::reduce_into(grad, len, |k0, k1, out| {
                    for (c, ch) in chunks.iter().enumerate() {
                        let lo = (3 * ch.lo).max(k0);
                        if lo >= k1 {
                            continue;
                        }
                        let b = &bufs[c * len + lo..c * len + k1];
                        for (o, x) in out[lo - k0..].iter_mut().zip(b) {
                            *o += *x;
                        }
                    }
                });
                r
            } else {
                let (r, ()) = rayon::join(
                    || other(&mut *grad),
                    || {
                        energies.par_iter_mut().zip(self.chunks.par_iter()).for_each(|(en, ch)| {
                            let mut c = 0.0;
                            let (v, e) = self.eval_rows_dispatch::<false>(pos, &mut [], ch.rows.clone(), &mut c);
                            *en = [c, v, e];
                        })
                    },
                );
                r
            };
            let mut out = NonbondedEnergy::default();
            let mut local = 0.0;
            for en in energies.iter() {
                local += en[0];
                out.vdw += en[1];
                out.electrostatic += en[2];
            }
            *acc += local;
            (out, r)
        })
    }
}

impl NonbondedBackend for ExactNonbonded {
    fn name(&self) -> &str {
        if self.simd { "cpu-simd-exact" } else { "cpu-scalar-exact" }
    }
    fn energy(&self, pos: &[f64]) -> NonbondedEnergy {
        self.eval_into::<false>(pos, &mut [], &mut 0.0)
    }
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> NonbondedEnergy {
        self.eval_into::<true>(pos, grad, &mut 0.0)
    }
}

// ---------------------------------------------------------------------------
// building from extracted contribs (frozen pair set)
// ---------------------------------------------------------------------------

/// Collects pairs from compiled contribs; parameters are deduplicated.
#[derive(Debug, Default)]
pub struct PairCollector {
    pub mmff: Option<bool>,
    pairs: Vec<Pair>,
    table: ParamTable,
    keys: HashMap<[u64; 3], u32>,
}

impl PairCollector {
    pub fn is_empty(&self) -> bool {
        self.mmff.is_none()
    }

    fn key_index(&mut self, key: [u64; 3], push: impl FnOnce(&mut ParamTable)) -> u32 {
        if let Some(&k) = self.keys.get(&key) {
            return k;
        }
        let k = self.keys.len() as u32;
        push(&mut self.table);
        self.keys.insert(key, k);
        k
    }

    /// Adds a UFF `VdwContrib`. Returns `false` if this collector already
    /// holds MMFF pairs.
    pub fn add_uff(&mut self, c: &crate::ff::uff::VdwContrib) -> bool {
        match self.mmff {
            Some(true) => return false,
            None => {
                self.mmff = Some(false);
                self.table = ParamTable::new_uff();
            }
            _ => {}
        }
        let key = [c.xij.to_bits(), c.well_depth.to_bits(), c.thresh.to_bits()];
        let p = self.key_index(key, |t| t.push_uff(c.xij, c.well_depth, c.thresh));
        self.pairs.push(Pair { i: c.idx1 as u32, j: c.idx2 as u32, p, flags: F_VDW, ct: 0.0 });
        true
    }

    /// Adds all pairs of an MMFF `NonbondedContrib`.
    pub fn add_mmff(&mut self, c: &crate::ff::mmff::terms::NonbondedContrib) -> bool {
        use crate::ff::mmff::DielectricModel;
        match self.mmff {
            Some(false) => return false,
            None => {
                self.mmff = Some(true);
                self.table = ParamTable::new_mmff();
                // index 0: dummy entry for pairs without van der Waals part
                self.key_index([u64::MAX; 3], |t| t.push_mmff(1.0, 0.0));
            }
            _ => {}
        }
        for t in 0..c.len() {
            let mut flags = 0;
            let mut p = 0;
            if c.kind[t] & 1 != 0 {
                flags |= F_VDW;
                let (r, e) = (c.r_ij_star[t], c.well_depth[t]);
                p = self.key_index([r.to_bits(), e.to_bits(), 0], |tb| tb.push_mmff(r, e));
            }
            if c.kind[t] & 2 != 0 {
                flags |= F_ELE;
                if c.is_1_4[t] {
                    flags |= F_14;
                }
                if c.diel_model[t] == Some(DielectricModel::Distance) {
                    flags |= F_DIST;
                }
            }
            self.pairs.push(Pair { i: c.at1[t] as u32, j: c.at2[t] as u32, p, flags, ct: c.charge_term[t] });
        }
        true
    }

    /// Builds the backend (pairs stably grouped by their first atom).
    pub fn finish(mut self, n_atoms: usize, grain: usize, simd: bool) -> ExactNonbonded {
        self.pairs.sort_by_key(|p| p.i);
        ExactNonbonded::from_sorted_pairs(self.mmff == Some(true), n_atoms, &self.pairs, Arc::new(self.table), grain, simd)
    }
}

// ---------------------------------------------------------------------------
// pair selection for new geometries (cached typing)
// ---------------------------------------------------------------------------

/// Pair-selection rule of the wrapped builder.
#[derive(Debug, Clone, PartialEq)]
pub enum NbRule {
    /// UFF: typed atoms, not 1-2/1-3, `dist < vdw_thresh · x_ij`.
    Uff { vdw_thresh: f64 },
    /// MMFF: not 1-2/1-3 (1-4 scaled), `!(dist > thresh)`, vdW and/or
    /// electrostatic part present.
    Mmff { thresh: f64, dielectric_constant: f64, distance_model: bool },
}

/// Position-independent nonbonded data of a molecule (typing, charges,
/// topology): everything needed to select pairs exactly like the builder.
#[derive(Debug, Clone)]
pub struct NbTemplate {
    pub rule: NbRule,
    pub n: usize,
    /// Compact type per atom (`u32::MAX` = no parameters).
    pub ty: Vec<u32>,
    pub nt: usize,
    /// Per ordered type pair: vdW parameters present (MMFF) / typed (UFF).
    pub pair_vdw: Vec<bool>,
    /// UFF: `x_ij` per type pair (selection threshold factor).
    pub pair_x: Vec<f64>,
    pub params: Arc<ParamTable>,
    /// MMFF partial charges and `!isDoubleZero(q)`.
    pub q: Vec<f64>,
    pub has_q: Vec<bool>,
    /// Fragment index per atom (only if interfragment pairs are ignored).
    pub frag: Option<Vec<u32>>,
    /// Topological relations (CSR): partner, relation (1 = 1-2, 2 = 1-3, 3 = 1-4).
    pub rel_start: Vec<u32>,
    pub rel_j: Vec<u32>,
    pub rel_v: Vec<u8>,
}

/// Topological neighbors up to `depth` bonds (BFS shortest-path distance),
/// as CSR.
pub(crate) fn topo_relations(mol: &crate::core::Molecule, depth: u8) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    let n = mol.num_atoms();
    let mut start = Vec::with_capacity(n + 1);
    let mut js = Vec::new();
    let mut vs = Vec::new();
    let mut dist = vec![u8::MAX; n];
    let mut frontier = Vec::new();
    let mut next = Vec::new();
    let mut touched = Vec::new();
    start.push(0u32);
    for s in 0..n {
        dist[s] = 0;
        touched.push(s);
        frontier.clear();
        frontier.push(s);
        for d in 1..=depth {
            next.clear();
            for &u in &frontier {
                for v in mol.neighbors(u) {
                    if dist[v] == u8::MAX {
                        dist[v] = d;
                        touched.push(v);
                        next.push(v);
                    }
                }
            }
            std::mem::swap(&mut frontier, &mut next);
        }
        let mut row: Vec<(u32, u8)> = touched.iter().filter(|&&v| v != s).map(|&v| (v as u32, dist[v])).collect();
        row.sort_unstable();
        for (v, d) in row {
            js.push(v);
            vs.push(d);
        }
        start.push(js.len() as u32);
        for &v in &touched {
            dist[v] = u8::MAX;
        }
        touched.clear();
    }
    (start, js, vs)
}

impl NbTemplate {
    /// UFF template from per-atom parameters (`None` = untyped).
    pub fn uff(
        mol: &crate::core::Molecule,
        params: &[Option<&'static crate::ff::uff::AtomicParams>],
        vdw_thresh: f64,
        ignore_interfrag: bool,
    ) -> Self {
        let n = mol.num_atoms();
        let mut types: Vec<&'static crate::ff::uff::AtomicParams> = Vec::new();
        let mut ty = vec![u32::MAX; n];
        for i in 0..n {
            if let Some(p) = params[i] {
                let k = match types.iter().position(|q| std::ptr::eq(*q, p)) {
                    Some(k) => k,
                    None => {
                        types.push(p);
                        types.len() - 1
                    }
                };
                ty[i] = k as u32;
            }
        }
        let nt = types.len();
        let mut table = ParamTable::new_uff();
        let mut pair_x = Vec::with_capacity(nt * nt);
        for a in &types {
            for b in &types {
                let x = crate::ff::uff::nonbonded::calc_nonbonded_minimum(a, b);
                let d = crate::ff::uff::nonbonded::calc_nonbonded_depth(a, b);
                table.push_uff(x, d, 10.0 * x);
                pair_x.push(x);
            }
        }
        let (rel_start, rel_j, rel_v) = topo_relations(mol, 2);
        NbTemplate {
            rule: NbRule::Uff { vdw_thresh },
            n,
            ty,
            nt,
            pair_vdw: vec![true; nt * nt],
            pair_x,
            params: Arc::new(table),
            q: Vec::new(),
            has_q: Vec::new(),
            frag: ignore_interfrag.then(|| fragment_ids(mol)),
            rel_start,
            rel_j,
            rel_v,
        }
    }

    /// MMFF template from the molecule properties.
    pub fn mmff(props: &crate::ff::mmff::MmffMolProperties, b: &crate::ff::mmff::MmffBuilder) -> Self {
        use crate::ff::mmff::DielectricModel;
        let mol = props.typing_molecule();
        let n = mol.num_atoms();
        let types = props.atom_types();
        let mut distinct: Vec<u8> = Vec::new();
        let mut rep: Vec<usize> = Vec::new();
        let mut ty = vec![0u32; n];
        for i in 0..n {
            let k = match distinct.iter().position(|&t| t == types[i]) {
                Some(k) => k,
                None => {
                    distinct.push(types[i]);
                    rep.push(i);
                    distinct.len() - 1
                }
            };
            ty[i] = k as u32;
        }
        let nt = distinct.len();
        let mut table = ParamTable::new_mmff();
        let mut pair_vdw = Vec::with_capacity(nt * nt);
        for a in 0..nt {
            for c in 0..nt {
                match props.vdw_params(rep[a], rep[c]) {
                    Some(v) => {
                        table.push_mmff(v.r_ij_star, v.epsilon);
                        pair_vdw.push(true);
                    }
                    None => {
                        table.push_mmff(1.0, 0.0);
                        pair_vdw.push(false);
                    }
                }
            }
        }
        let q = props.partial_charges().to_vec();
        let has_q = q.iter().map(|&x| !(x < 1.0e-10 && x > -1.0e-10)).collect();
        let (rel_start, rel_j, rel_v) = topo_relations(mol, 3);
        NbTemplate {
            rule: NbRule::Mmff {
                thresh: b.nonbonded_thresh,
                dielectric_constant: b.dielectric_constant,
                distance_model: b.dielectric_model == DielectricModel::Distance,
            },
            n,
            ty,
            nt,
            pair_vdw,
            pair_x: Vec::new(),
            params: Arc::new(table),
            q,
            has_q,
            frag: b.ignore_interfrag_interactions.then(|| fragment_ids(mol)),
            rel_start,
            rel_j,
            rel_v,
        }
    }

    /// Largest distance at which a pair can be selected (`None` = no
    /// finite bound).
    fn max_cutoff(&self) -> Option<f64> {
        let c = match self.rule {
            NbRule::Uff { vdw_thresh } => self.pair_x.iter().map(|&x| vdw_thresh * x).fold(0.0, f64::max),
            NbRule::Mmff { thresh, .. } => thresh,
        };
        (c.is_finite() && c > 0.0).then_some(c)
    }

    /// Tests pair `i < j` (relation `rel`, 0 = none within range) and
    /// returns it if the builder would include it.
    #[inline]
    fn test(&self, coords: &[[f64; 3]], i: usize, j: usize, rel: u8) -> Option<Pair> {
        if let Some(f) = &self.frag {
            if f[i] != f[j] {
                return None;
            }
        }
        match self.rule {
            NbRule::Uff { vdw_thresh } => {
                if rel != 0 {
                    return None;
                }
                let (ti, tj) = (self.ty[i], self.ty[j]);
                if ti == u32::MAX || tj == u32::MAX {
                    return None;
                }
                let p = ti as usize * self.nt + tj as usize;
                let d = [coords[i][0] - coords[j][0], coords[i][1] - coords[j][1], coords[i][2] - coords[j][2]];
                let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                (dist < vdw_thresh * self.pair_x[p]).then_some(Pair { i: i as u32, j: j as u32, p: p as u32, flags: F_VDW, ct: 0.0 })
            }
            NbRule::Mmff { thresh, dielectric_constant, distance_model } => {
                if rel == 1 || rel == 2 {
                    return None;
                }
                let d = [coords[i][0] - coords[j][0], coords[i][1] - coords[j][1], coords[i][2] - coords[j][2]];
                let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if dist > thresh {
                    return None;
                }
                let p = self.ty[i] as usize * self.nt + self.ty[j] as usize;
                let mut flags = 0;
                if self.pair_vdw[p] {
                    flags |= F_VDW;
                }
                let mut ct = 0.0;
                if self.has_q[i] && self.has_q[j] {
                    flags |= F_ELE;
                    ct = self.q[i] * self.q[j] / dielectric_constant;
                    if rel == 3 {
                        flags |= F_14;
                    }
                    if distance_model {
                        flags |= F_DIST;
                    }
                }
                if flags == 0 {
                    return None;
                }
                Some(Pair { i: i as u32, j: j as u32, p: p as u32, flags, ct })
            }
        }
    }

    /// Selects the pairs for `coords` exactly like the builder would.
    pub fn select_pairs(&self, coords: &[[f64; 3]]) -> Vec<Pair> {
        let n = self.n;
        let mut out = Vec::new();
        let mut mark = vec![u32::MAX; n];
        let mut relv = vec![0u8; n];
        let set_rel = |i: usize, mark: &mut [u32], relv: &mut [u8]| {
            for t in self.rel_start[i] as usize..self.rel_start[i + 1] as usize {
                let j = self.rel_j[t] as usize;
                mark[j] = i as u32;
                relv[j] = self.rel_v[t];
            }
        };
        let finite = coords.iter().all(|c| c.iter().all(|x| x.is_finite()));
        let grid = if finite { self.max_cutoff().and_then(|c| Grid::new(coords, c)) } else { None };
        match grid {
            None => {
                for i in 0..n {
                    set_rel(i, &mut mark, &mut relv);
                    for j in i + 1..n {
                        let rel = if mark[j] == i as u32 { relv[j] } else { 0 };
                        if let Some(p) = self.test(coords, i, j, rel) {
                            out.push(p);
                        }
                    }
                }
            }
            Some(g) => {
                let mut row: Vec<Pair> = Vec::new();
                for i in 0..n {
                    set_rel(i, &mut mark, &mut relv);
                    row.clear();
                    g.for_neighbors(i, |j| {
                        if j > i {
                            let rel = if mark[j] == i as u32 { relv[j] } else { 0 };
                            if let Some(p) = self.test(coords, i, j, rel) {
                                row.push(p);
                            }
                        }
                    });
                    row.sort_unstable_by_key(|p| p.j);
                    out.extend_from_slice(&row);
                }
            }
        }
        out
    }

    /// Builds the exact backend for `coords`.
    pub fn select(&self, coords: &[[f64; 3]], grain: usize, simd: bool) -> ExactNonbonded {
        let pairs = self.select_pairs(coords);
        ExactNonbonded::from_sorted_pairs(matches!(self.rule, NbRule::Mmff { .. }), self.n, &pairs, self.params.clone(), grain, simd)
    }
}

fn fragment_ids(mol: &crate::core::Molecule) -> Vec<u32> {
    let mut f = vec![0u32; mol.num_atoms()];
    for (k, members) in mol.fragments().iter().enumerate() {
        for &a in members {
            f[a] = k as u32;
        }
    }
    f
}

/// Uniform cell grid with cell size ≥ the cutoff (neighbors = 27 cells).
struct Grid {
    dims: [usize; 3],
    cell_of: Vec<usize>,
    cell_start: Vec<u32>,
    atoms: Vec<u32>,
}

impl Grid {
    /// `None` if a grid would not prune anything (fewer than 4 cells along
    /// every axis).
    fn new(coords: &[[f64; 3]], cutoff: f64) -> Option<Grid> {
        let n = coords.len();
        if n < 64 {
            return None;
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for c in coords {
            for d in 0..3 {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d]);
            }
        }
        // margin so that rounding can never move a pair closer than the
        // cutoff into non-adjacent cells
        let mut cs = cutoff * (1.0 + 1.0e-9) + 1.0e-9;
        let dims_for = |cs: f64| -> [usize; 3] { [0, 1, 2].map(|d| ((hi[d] - lo[d]) / cs).floor() as usize + 1) };
        let mut dims = dims_for(cs);
        while dims.iter().product::<usize>() > 4 * n + 64 {
            cs *= 1.5;
            dims = dims_for(cs);
        }
        if dims.iter().all(|&d| d < 4) {
            return None;
        }
        let ncell = dims.iter().product::<usize>();
        let cell_of: Vec<usize> = coords
            .iter()
            .map(|c| {
                let ix = [0, 1, 2].map(|d| (((c[d] - lo[d]) / cs).floor() as usize).min(dims[d] - 1));
                (ix[2] * dims[1] + ix[1]) * dims[0] + ix[0]
            })
            .collect();
        let mut cell_start = vec![0u32; ncell + 1];
        for &c in &cell_of {
            cell_start[c + 1] += 1;
        }
        for c in 0..ncell {
            cell_start[c + 1] += cell_start[c];
        }
        let mut fill = cell_start.clone();
        let mut atoms = vec![0u32; n];
        for (i, &c) in cell_of.iter().enumerate() {
            atoms[fill[c] as usize] = i as u32;
            fill[c] += 1;
        }
        Some(Grid { dims, cell_of, cell_start, atoms })
    }

    fn for_neighbors(&self, i: usize, mut f: impl FnMut(usize)) {
        let c = self.cell_of[i];
        let [dx, dy, _] = self.dims;
        let ix = [c % dx, (c / dx) % dy, c / (dx * dy)];
        for z in ix[2].saturating_sub(1)..=(ix[2] + 1).min(self.dims[2] - 1) {
            for y in ix[1].saturating_sub(1)..=(ix[1] + 1).min(self.dims[1] - 1) {
                for x in ix[0].saturating_sub(1)..=(ix[0] + 1).min(self.dims[0] - 1) {
                    let cc = (z * dy + y) * dx + x;
                    for &a in &self.atoms[self.cell_start[cc] as usize..self.cell_start[cc + 1] as usize] {
                        f(a as usize);
                    }
                }
            }
        }
    }
}
