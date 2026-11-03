//! Force-field abstraction.
//!
//! A force field is a function of the flat coordinate vector `pos`
//! (`[x0,y0,z0,x1,...]`, or 4D for distance geometry) with an analytic
//! gradient. [`ContribForceField`] mirrors RDKit's `ForceFields::ForceField`:
//! a list of independent energy terms ([`Contrib`]) plus fixed points.
//!
//! Port of RDKit `Code/ForceField/ForceField.{h,cpp}` and the constraint
//! contribs (BSD-3-Clause, Copyright (C) 2004-2019 Greg Landrum and others).

pub mod constraints;
pub mod mmff;
pub mod uff;
pub mod utils;

use crate::core::Molecule;
use crate::error::Result;
use rayon::prelude::*;
use std::sync::Arc;

pub use constraints::Constraint;

/// One energy term.
pub trait Contrib: Send + Sync {
    /// Energy of this term.
    fn energy(&self, pos: &[f64]) -> f64;
    /// Adds the gradient of this term to `grad`.
    fn grad(&self, pos: &[f64], grad: &mut [f64]);
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
}

/// Creates force fields for molecules (UFF, MMFF94, user-defined ...).
pub trait ForceFieldBuilder: Send + Sync {
    /// Short name, e.g. `"UFF"`.
    fn name(&self) -> &str;
    /// Builds a force field for `mol` with additional `constraints`.
    fn build(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<ContribForceField>;
}

impl<T: ForceFieldBuilder + ?Sized> ForceFieldBuilder for Arc<T> {
    fn name(&self) -> &str {
        (**self).name()
    }
    fn build(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<ContribForceField> {
        (**self).build(mol, constraints)
    }
}

/// Number of contribs above which energy/gradient are evaluated in parallel.
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 4096;

/// Force field made of a list of contributions (RDKit `ForceField`).
pub struct ContribForceField {
    dim: usize,
    n_points: usize,
    contribs: Vec<Box<dyn Contrib>>,
    fixed: Vec<usize>,
    /// Contrib count above which evaluation uses rayon (0 = never).
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
            fixed: Vec::new(),
            parallel_threshold: DEFAULT_PARALLEL_THRESHOLD,
        }
    }
    pub fn add_contrib(&mut self, c: Box<dyn Contrib>) {
        self.contribs.push(c);
    }
    pub fn add_contribs(&mut self, cs: impl IntoIterator<Item = Box<dyn Contrib>>) {
        self.contribs.extend(cs);
    }
    pub fn contribs(&self) -> &[Box<dyn Contrib>] {
        &self.contribs
    }
    pub fn num_contribs(&self) -> usize {
        self.contribs.len()
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
        self.parallel_threshold > 0 && self.contribs.len() >= self.parallel_threshold
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
        if self.parallel() {
            self.contribs.par_iter().map(|c| c.energy(pos)).sum()
        } else {
            self.contribs.iter().map(|c| c.energy(pos)).sum()
        }
    }
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> f64 {
        let n = self.n_points * self.dim;
        grad[..n].iter_mut().for_each(|g| *g = 0.0);
        if self.parallel() {
            let chunk = (self.contribs.len() / (rayon::current_num_threads() * 4)).max(256);
            let (g, e) = self
                .contribs
                .par_chunks(chunk)
                .map(|cs| {
                    let mut g = vec![0.0; n];
                    let mut e = 0.0;
                    for c in cs {
                        e += c.energy(pos);
                        c.grad(pos, &mut g);
                    }
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
                e += c.energy(pos);
                c.grad(pos, grad);
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
