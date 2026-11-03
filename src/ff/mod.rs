//! Force-field abstraction.
//!
//! A force field is a function of the flat coordinate vector `pos`
//! (`[x0,y0,z0,x1,...]`, or 4D for distance geometry) with an analytic
//! gradient. [`ContribForceField`] mirrors RDKit's `ForceFields::ForceField`:
//! a list of independent energy terms ([`Contrib`]) plus fixed points.
//!
//! Port of RDKit `Code/ForceField/ForceField.{h,cpp}` and the constraint
//! contribs (BSD-3-Clause, Copyright (C) 2004-2019 Greg Landrum and others).

pub mod assign;
pub mod constraints;
pub mod custom;
pub mod mmff;
pub mod softcore;
pub mod uff;
pub mod utils;

use crate::core::Molecule;
use crate::error::Result;
use std::ops::Range;
use std::sync::{Arc, Mutex};

pub use assign::{Assignment, ForceFieldKind, assign};
pub use constraints::Constraint;

/// One energy term.
pub trait Contrib: Send + Sync {
    /// Energy of this term.
    fn energy(&self, pos: &[f64]) -> f64;
    /// Adds the gradient of this term to `grad`.
    fn grad(&self, pos: &[f64], grad: &mut [f64]);
    /// Adds the gradient of this term to `grad` and returns its energy.
    /// Must return exactly what [`Contrib::energy`] returns and add exactly
    /// what [`Contrib::grad`] adds (same operations in the same order);
    /// hot contribs override it to compute the shared geometry once.
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> f64 {
        let e = self.energy(pos);
        self.grad(pos, grad);
        e
    }
    /// Category label used for energy breakdowns (e.g. `"BondStretch"`).
    fn label(&self) -> &'static str {
        "Other"
    }
    /// Reports this term's energy for breakdowns, as `(label, energy)`
    /// pairs. The default reports `(self.label(), self.energy(pos))`; a
    /// contrib that combines several kinds of terms (e.g. MMFF's combined
    /// van der Waals + electrostatics pair list) reports each part.
    fn energy_terms(&self, pos: &[f64], out: &mut dyn FnMut(&'static str, f64)) {
        out(self.label(), self.energy(pos));
    }
    /// Downcasting hook used by `crate::fast` to compile known term types
    /// into vectorized tables. `None` = opaque (evaluated as is).
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
    /// Number of terms [`ContribForceField`] may split this contrib into
    /// (for its parallel path and its parallel threshold). 1 for a plain
    /// contrib. Contribs holding many terms report their count in one of
    /// two ways:
    ///
    /// * a [`TermList`] adds every term to the running sum on its own
    ///   ([`Contrib::energy_acc`]), exactly as one contrib per term;
    /// * a *block* contrib (all terms of one kind in one contrib, summed
    ///   inside the contrib as RDKit's MMFF contribs do) implements
    ///   [`Contrib::energy_range`] / [`Contrib::energy_grad_range`]: the
    ///   terms of the range are summed from `0.0` in order and that partial
    ///   sum is added to the running sum. Over the full range this is
    ///   exactly [`Contrib::energy`]; only the parallel path, which cuts the
    ///   block into several ranges, groups the sum differently (rounding
    ///   level).
    fn num_terms(&self) -> usize {
        1
    }
    /// Sum (from `0.0`, in term order) of the energies of the terms `range`
    /// (a subrange of `0..num_terms()`). Block contribs override this.
    fn energy_range(&self, pos: &[f64], range: Range<usize>) -> f64 {
        debug_assert!(range == (0..1));
        self.energy(pos)
    }
    /// Like [`Contrib::energy_range`], also adding the gradients of the
    /// terms `range` to `grad`.
    fn energy_grad_range(&self, pos: &[f64], grad: &mut [f64], range: Range<usize>) -> f64 {
        debug_assert!(range == (0..1));
        self.energy_grad(pos, grad)
    }
    /// Adds the energies of the terms `range` (a subrange of
    /// `0..num_terms()`) to `acc` (a [`TermList`] one term at a time, any
    /// other contrib as the partial sum [`Contrib::energy_range`]).
    fn energy_acc(&self, pos: &[f64], range: Range<usize>, acc: &mut f64) {
        *acc += self.energy_range(pos, range);
    }
    /// Like [`Contrib::energy_acc`], also adding the gradients to `grad`.
    fn energy_grad_acc(&self, pos: &[f64], grad: &mut [f64], range: Range<usize>, acc: &mut f64) {
        *acc += self.energy_grad_range(pos, grad, range);
    }
    /// Calls `f` with the downcasting hook of every term in order (that of
    /// the contrib itself unless it is a [`TermList`]).
    fn visit_terms<'a>(&'a self, f: &mut dyn FnMut(Option<&'a dyn std::any::Any>)) {
        f(self.as_any());
    }
    /// Releases unused capacity of the term storage (built by pushing; see
    /// [`utils::shrink_vec`]). [`ContribForceField::add_contrib`] calls it.
    /// Default: nothing.
    fn shrink_to_fit(&mut self) {}
}

/// Terms of one type evaluated as one contrib, without dynamic dispatch or
/// a separate allocation per term. The list stands for its terms as
/// separate contribs: [`ContribForceField`] adds their energies to its
/// running sum one by one in list order and its parallel split may cut
/// through the list (see [`Contrib::num_terms`]), so the results are the
/// same as with one contrib per term. (Only [`Contrib::energy`] /
/// [`Contrib::energy_grad`] of the list itself return the partial sum.)
#[derive(Debug, Clone)]
pub struct TermList<T> {
    terms: Vec<T>,
}

impl<T> TermList<T> {
    pub fn new(terms: Vec<T>) -> Self {
        TermList { terms }
    }
    pub fn terms(&self) -> &[T] {
        &self.terms
    }
}

impl<T: Contrib + 'static> TermList<T> {
    /// The boxed list, or `None` if `terms` is empty.
    pub fn boxed(terms: Vec<T>) -> Option<Box<dyn Contrib>> {
        (!terms.is_empty()).then(|| Box::new(TermList::new(terms)) as Box<dyn Contrib>)
    }
}

impl<T: Contrib + 'static> Contrib for TermList<T> {
    fn energy(&self, pos: &[f64]) -> f64 {
        let mut e = 0.0;
        self.energy_acc(pos, 0..self.terms.len(), &mut e);
        e
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        for t in &self.terms {
            t.grad(pos, grad);
        }
    }
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> f64 {
        let mut e = 0.0;
        self.energy_grad_acc(pos, grad, 0..self.terms.len(), &mut e);
        e
    }
    fn label(&self) -> &'static str {
        self.terms.first().map_or("Other", |t| t.label())
    }
    fn energy_terms(&self, pos: &[f64], out: &mut dyn FnMut(&'static str, f64)) {
        for t in &self.terms {
            t.energy_terms(pos, out);
        }
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn num_terms(&self) -> usize {
        self.terms.len()
    }
    fn energy_acc(&self, pos: &[f64], range: Range<usize>, acc: &mut f64) {
        for t in &self.terms[range] {
            *acc += t.energy(pos);
        }
    }
    fn energy_grad_acc(&self, pos: &[f64], grad: &mut [f64], range: Range<usize>, acc: &mut f64) {
        for t in &self.terms[range] {
            *acc += t.energy_grad(pos, grad);
        }
    }
    fn visit_terms<'a>(&'a self, f: &mut dyn FnMut(Option<&'a dyn std::any::Any>)) {
        for t in &self.terms {
            t.visit_terms(f);
        }
    }
    fn shrink_to_fit(&mut self) {
        utils::shrink_vec(&mut self.terms);
    }
}

/// A differentiable energy function of the coordinates.
pub trait ForceField: Send + Sync {
    /// Dimension of each point (3, or 4 during distance-geometry embedding).
    fn dimension(&self) -> usize {
        3
    }
    /// Number of points (atoms).
    fn num_points(&self) -> usize;
    /// Total energy (kcal/mol for molecular force fields).
    fn energy(&self, pos: &[f64]) -> f64;
    /// Computes the gradient into `grad` (overwritten) and returns the energy.
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> f64;
    /// Energy split by term category.
    fn energy_breakdown(&self, pos: &[f64]) -> Vec<(String, f64)> {
        vec![("Total".to_string(), self.energy(pos))]
    }
    /// Indices of points that must not move.
    fn fixed_points(&self) -> &[usize] {
        &[]
    }
    /// Called by the minimizers after each accepted step (e.g. to update
    /// neighbor lists). Default: nothing.
    fn on_accepted_step(&self, _pos: &[f64]) {}
}

impl<T: ForceField + ?Sized> ForceField for Box<T> {
    fn dimension(&self) -> usize {
        (**self).dimension()
    }
    fn num_points(&self) -> usize {
        (**self).num_points()
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        (**self).energy(pos)
    }
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> f64 {
        (**self).energy_grad(pos, grad)
    }
    fn energy_breakdown(&self, pos: &[f64]) -> Vec<(String, f64)> {
        (**self).energy_breakdown(pos)
    }
    fn fixed_points(&self) -> &[usize] {
        (**self).fixed_points()
    }
    fn on_accepted_step(&self, pos: &[f64]) {
        (**self).on_accepted_step(pos)
    }
}

impl<T: ForceField + ?Sized> ForceField for Arc<T> {
    fn dimension(&self) -> usize {
        (**self).dimension()
    }
    fn num_points(&self) -> usize {
        (**self).num_points()
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        (**self).energy(pos)
    }
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> f64 {
        (**self).energy_grad(pos, grad)
    }
    fn energy_breakdown(&self, pos: &[f64]) -> Vec<(String, f64)> {
        (**self).energy_breakdown(pos)
    }
    fn fixed_points(&self) -> &[usize] {
        (**self).fixed_points()
    }
    fn on_accepted_step(&self, pos: &[f64]) {
        (**self).on_accepted_step(pos)
    }
}

/// Creates force fields for molecules (UFF, MMFF94, user-defined ...).
pub trait ForceFieldBuilder: Send + Sync {
    /// Short name, e.g. `"UFF"`.
    fn name(&self) -> &str;
    /// Builds a force field for `mol` with additional `constraints`.
    fn build(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<ContribForceField>;
    /// Builds a force field as a trait object. Builders whose force field is
    /// not a [`ContribForceField`] (e.g. the fast path) override this; the
    /// calculations in [`crate::calc`] always call it.
    fn build_dyn(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<Box<dyn ForceField>> {
        Ok(Box::new(self.build(mol, constraints)?))
    }
}

impl<T: ForceFieldBuilder + ?Sized> ForceFieldBuilder for Arc<T> {
    fn name(&self) -> &str {
        (**self).name()
    }
    fn build(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<ContribForceField> {
        (**self).build(mol, constraints)
    }
    fn build_dyn(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<Box<dyn ForceField>> {
        (**self).build_dyn(mol, constraints)
    }
}

/// Number of terms (see [`Contrib::num_terms`]) above which energy/gradient
/// are evaluated in parallel.
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 4096;

/// Maximum number of chunks the terms are split into for parallel evaluation.
const PAR_CHUNKS: usize = 64;

/// Leaf function of [`par_tree`]: evaluates a term range with its chunk buffer.
type Leaf<'a, T> = dyn Fn(Range<usize>, &mut [f64]) -> T + Sync + 'a;
/// Combines two adjacent results; the buffers are those of the first chunk
/// of each half (the first one receives the combined values).
type Combine<'a, T> = dyn Fn(T, T, &mut [f64], &[f64]) -> T + Sync + 'a;

/// Evaluates `leaf` on fixed chunks of `0..total` in parallel and combines
/// the results pairwise in a fixed binary tree. Neither the chunks nor the
/// tree depend on the thread count or on scheduling, so the floating-point
/// sums are the same on every run. Chunk `c` gets `bufs[c·len..(c+1)·len]`
/// (see [`par_tree_chunks`]); the root result is in the first buffer.
fn par_tree<T: Send>(total: usize, len: usize, bufs: &mut [f64], leaf: &Leaf<'_, T>, combine: &Combine<'_, T>) -> T {
    #[allow(clippy::too_many_arguments)]
    fn rec<T: Send>(lo: usize, hi: usize, chunk: usize, total: usize, len: usize, bufs: &mut [f64], leaf: &Leaf<'_, T>, combine: &Combine<'_, T>) -> T {
        if hi - lo == 1 {
            return leaf(lo * chunk..total.min(hi * chunk), &mut bufs[..len]);
        }
        let mid = lo + (hi - lo) / 2;
        let (l, r) = bufs.split_at_mut((mid - lo) * len);
        let (a, b) = rayon::join(|| rec(lo, mid, chunk, total, len, &mut *l, leaf, combine), || rec(mid, hi, chunk, total, len, &mut *r, leaf, combine));
        combine(a, b, &mut l[..len], &r[..len])
    }
    let (chunk, n) = par_tree_chunks(total);
    rec(0, n, chunk, total, len, &mut bufs[..n * len], leaf, combine)
}

/// Chunk size and number of chunks of [`par_tree`].
fn par_tree_chunks(total: usize) -> (usize, usize) {
    let chunk = total.div_ceil(PAR_CHUNKS).max(256);
    (chunk, total.div_ceil(chunk).max(1))
}

/// Per-chunk gradient buffers of the parallel path, kept for reuse so that
/// steady-state evaluations do not allocate: one set per concurrent caller
/// (usually one), taken for a whole evaluation.
#[derive(Default)]
struct GradBufs(Mutex<Vec<Vec<f64>>>);

impl GradBufs {
    /// Runs `f` with a buffer set of at least `n` values (contents
    /// unspecified).
    fn with<R>(&self, n: usize, f: impl FnOnce(&mut [f64]) -> R) -> R {
        let mut v = self.0.lock().unwrap_or_else(|e| e.into_inner()).pop().unwrap_or_default();
        if v.len() < n {
            v.resize(n, 0.0);
        }
        let r = f(&mut v);
        self.0.lock().unwrap_or_else(|e| e.into_inner()).push(v);
        r
    }
}

/// Force field made of a list of contributions (RDKit `ForceField`).
pub struct ContribForceField {
    dim: usize,
    n_points: usize,
    contribs: Vec<Box<dyn Contrib>>,
    /// `term_start[c]`: number of terms before contrib `c` (the extra last
    /// entry is the total).
    term_start: Vec<usize>,
    fixed: Vec<usize>,
    /// Term count above which evaluation uses rayon (0 = never).
    pub parallel_threshold: usize,
    grad_bufs: GradBufs,
}

impl std::fmt::Debug for ContribForceField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContribForceField")
            .field("dim", &self.dim)
            .field("n_points", &self.n_points)
            .field("n_contribs", &self.contribs.len())
            .field("fixed", &self.fixed)
            .finish()
    }
}

impl ContribForceField {
    pub fn new(n_points: usize, dim: usize) -> Self {
        ContribForceField {
            dim,
            n_points,
            contribs: Vec::new(),
            term_start: vec![0],
            fixed: Vec::new(),
            parallel_threshold: DEFAULT_PARALLEL_THRESHOLD,
            grad_bufs: GradBufs::default(),
        }
    }
    pub fn add_contrib(&mut self, mut c: Box<dyn Contrib>) {
        c.shrink_to_fit();
        self.term_start.push(self.num_terms() + c.num_terms());
        self.contribs.push(c);
    }
    pub fn add_contribs(&mut self, cs: impl IntoIterator<Item = Box<dyn Contrib>>) {
        for c in cs {
            self.add_contrib(c);
        }
    }
    pub fn contribs(&self) -> &[Box<dyn Contrib>] {
        &self.contribs
    }
    pub fn num_contribs(&self) -> usize {
        self.contribs.len()
    }
    /// Number of terms (each term of a [`TermList`] or block contrib counts).
    pub fn num_terms(&self) -> usize {
        self.term_start[self.contribs.len()]
    }
    /// Marks a point as fixed (its gradient is zeroed).
    pub fn fix_point(&mut self, i: usize) {
        if !self.fixed.contains(&i) {
            self.fixed.push(i);
        }
    }
    /// Adds constraint terms (see [`Constraint`]).
    pub fn add_constraints(&mut self, mol_coords: &[[f64; 3]], cs: &[Constraint]) {
        for c in cs {
            c.apply(self, mol_coords);
        }
    }

    fn parallel(&self) -> bool {
        self.parallel_threshold > 0 && self.num_terms() >= self.parallel_threshold
    }

    /// Calls `f(contrib, its term range)` for the terms `lo..hi` in order.
    fn for_terms(&self, lo: usize, hi: usize, mut f: impl FnMut(&dyn Contrib, Range<usize>)) {
        let mut c = self.term_start.partition_point(|&s| s <= lo) - 1;
        while c < self.contribs.len() && self.term_start[c] < hi {
            let s = self.term_start[c];
            let r = lo.max(s) - s..hi.min(self.term_start[c + 1]) - s;
            if !r.is_empty() {
                f(&*self.contribs[c], r);
            }
            c += 1;
        }
    }
}

impl ForceField for ContribForceField {
    fn dimension(&self) -> usize {
        self.dim
    }
    fn num_points(&self) -> usize {
        self.n_points
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        // terms summed one by one, starting from -0.0 like `Iterator::sum`
        if self.parallel() {
            par_tree(
                self.num_terms(),
                0,
                &mut [],
                &|r, _| {
                    let mut e = -0.0;
                    self.for_terms(r.start, r.end, |c, r| c.energy_acc(pos, r, &mut e));
                    e
                },
                &|a, b, _, _| a + b,
            )
        } else {
            let mut e = -0.0;
            for c in &self.contribs {
                c.energy_acc(pos, 0..c.num_terms(), &mut e);
            }
            e
        }
    }
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> f64 {
        let n = self.n_points * self.dim;
        grad[..n].iter_mut().for_each(|g| *g = 0.0);
        if self.parallel() {
            let total = self.num_terms();
            let e = self.grad_bufs.with(par_tree_chunks(total).1 * n, |bufs| {
                let e = par_tree(
                    total,
                    n,
                    bufs,
                    &|r, g| {
                        g.fill(0.0);
                        let mut e = 0.0;
                        self.for_terms(r.start, r.end, |c, r| c.energy_grad_acc(pos, g, r, &mut e));
                        e
                    },
                    &|ea, eb, a, b| {
                        a.iter_mut().zip(b).for_each(|(x, y)| *x += *y);
                        ea + eb
                    },
                );
                grad[..n].copy_from_slice(&bufs[..n]);
                e
            });
            for &f in &self.fixed {
                for d in 0..self.dim {
                    grad[self.dim * f + d] = 0.0;
                }
            }
            e
        } else {
            let mut e = 0.0;
            for c in &self.contribs {
                c.energy_grad_acc(pos, grad, 0..c.num_terms(), &mut e);
            }
            for &f in &self.fixed {
                for d in 0..self.dim {
                    grad[self.dim * f + d] = 0.0;
                }
            }
            e
        }
    }
    fn energy_breakdown(&self, pos: &[f64]) -> Vec<(String, f64)> {
        let mut out: Vec<(String, f64)> = Vec::new();
        for c in &self.contribs {
            c.energy_terms(pos, &mut |label, e| match out.iter_mut().find(|(k, _)| k == label) {
                Some((_, v)) => *v += e,
                None => out.push((label.to_string(), e)),
            });
        }
        out
    }
    fn fixed_points(&self) -> &[usize] {
        &self.fixed
    }
}

/// Central finite-difference gradient (for testing).
pub fn numerical_gradient(ff: &dyn ForceField, pos: &[f64], h: f64) -> Vec<f64> {
    let mut p = pos.to_vec();
    let mut g = vec![0.0; pos.len()];
    for i in 0..pos.len() {
        let x = p[i];
        p[i] = x + h;
        let ep = ff.energy(&p);
        p[i] = x - h;
        let em = ff.energy(&p);
        p[i] = x;
        g[i] = (ep - em) / (2.0 * h);
    }
    g
}

/// Maximum absolute difference between analytic and numeric gradients.
pub fn gradient_error(ff: &dyn ForceField, pos: &[f64]) -> f64 {
    let mut ga = vec![0.0; pos.len()];
    ff.energy_grad(pos, &mut ga);
    let gn = numerical_gradient(ff, pos, 1e-5);
    let fixed = ff.fixed_points();
    let dim = ff.dimension();
    ga.iter()
        .zip(&gn)
        .enumerate()
        .filter(|(i, _)| !fixed.contains(&(i / dim)))
        .map(|(_, (a, b))| (a - b).abs())
        .fold(0.0, f64::max)
}
