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
use rayon::prelude::*;
use std::ops::Range;
use std::sync::Arc;

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
    /// Number of independent terms that [`ContribForceField`] sums one by
    /// one, as if each were its own contrib (1 unless this is a
    /// [`TermList`]).
    fn num_terms(&self) -> usize {
        1
    }
    /// Adds the energies of the terms `range` (a subrange of
    /// `0..num_terms()`) to `acc`, one term at a time.
    fn energy_acc(&self, pos: &[f64], range: Range<usize>, acc: &mut f64) {
        debug_assert!(range == (0..1));
        *acc += self.energy(pos);
    }
    /// Like [`Contrib::energy_acc`], also adding the gradients to `grad`.
    fn energy_grad_acc(&self, pos: &[f64], grad: &mut [f64], range: Range<usize>, acc: &mut f64) {
        debug_assert!(range == (0..1));
        *acc += self.energy_grad(pos, grad);
    }
    /// Calls `f` with the downcasting hook of every term in order (that of
    /// the contrib itself unless it is a [`TermList`]).
    fn visit_terms<'a>(&'a self, f: &mut dyn FnMut(Option<&'a dyn std::any::Any>)) {
        f(self.as_any());
    }
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
        }
    }
    pub fn add_contrib(&mut self, c: Box<dyn Contrib>) {
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
    /// Number of terms (each term of a [`TermList`] counts).
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
            (0..self.num_terms())
                .into_par_iter()
                .map(|t| {
                    let mut e = -0.0;
                    self.for_terms(t, t + 1, |c, r| c.energy_acc(pos, r, &mut e));
                    e
                })
                .sum()
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
            let chunk = (total / (rayon::current_num_threads() * 4)).max(256);
            let (g, e) = (0..total.div_ceil(chunk))
                .into_par_iter()
                .map(|k| {
                    let mut g = vec![0.0; n];
                    let mut e = 0.0;
                    self.for_terms(k * chunk, total.min((k + 1) * chunk), |c, r| c.energy_grad_acc(pos, &mut g, r, &mut e));
                    (g, e)
                })
                .reduce(
                    || (vec![0.0; n], 0.0),
                    |(mut a, ea), (b, eb)| {
                        a.iter_mut().zip(&b).for_each(|(x, y)| *x += *y);
                        (a, ea + eb)
                    },
                );
            grad[..n].copy_from_slice(&g);
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
