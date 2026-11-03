//! [`WithCustomTerms`]: adds custom terms to any force-field builder.

use super::kernel::Globals;
use super::terms::{CustomTerm, PreparedTerm, as_custom};
use crate::core::Molecule;
use crate::error::{Error, Result};
use crate::ff::{Constraint, Contrib, ContribForceField, ForceField, ForceFieldBuilder};

/// Wraps a builder and appends custom terms to every force field it builds.
///
/// ```no_run
/// use molecule_optimizer_rs::ff::custom::{CustomExternal, WithCustomTerms};
/// use molecule_optimizer_rs::ff::mmff::MmffBuilder;
/// let wall = CustomExternal::new("wall", "k*max(0, sqrt(x^2+y^2+z^2) - R)^2", &[])
///     .atoms(0..10, &[]);
/// let b = WithCustomTerms::new(MmffBuilder::default())
///     .with_global("k", 10.0)
///     .with_global("R", 5.0)
///     .with(wall)
///     .unwrap();
/// b.set_global("R", 6.0).unwrap(); // affects already built force fields too
/// ```
///
/// Declare all globals before adding the terms that use them.
#[derive(Debug, Clone)]
pub struct WithCustomTerms<B> {
    inner: B,
    globals: Globals,
    terms: Vec<PreparedTerm>,
    name: String,
}

impl<B: ForceFieldBuilder> WithCustomTerms<B> {
    pub fn new(inner: B) -> Self {
        let name = format!("{}+Custom", inner.name());
        WithCustomTerms { inner, globals: Globals::new(), terms: Vec::new(), name }
    }
    pub fn inner(&self) -> &B {
        &self.inner
    }
    /// Declares a global parameter with its initial value.
    pub fn with_global(mut self, name: &str, value: f64) -> Self {
        self.add_global(name, value);
        self
    }
    pub fn add_global(&mut self, name: &str, value: f64) {
        self.globals = self.globals.with(name, value);
    }
    /// Compiles and adds a term (errors on syntax errors / unknown names).
    pub fn add(&mut self, term: impl Into<CustomTerm>) -> Result<()> {
        let t = PreparedTerm::new(term.into(), &self.globals)?;
        self.terms.push(t);
        Ok(())
    }
    pub fn with(mut self, term: impl Into<CustomTerm>) -> Result<Self> {
        self.add(term)?;
        Ok(self)
    }
    pub fn terms(&self) -> &[PreparedTerm] {
        &self.terms
    }
    /// Shared global parameter values (also used by already built force fields).
    pub fn globals(&self) -> &Globals {
        &self.globals
    }
    pub fn set_global(&self, name: &str, value: f64) -> Result<()> {
        self.globals.set(name, value)
    }
    pub fn get_global(&self, name: &str) -> Option<f64> {
        self.globals.get(name)
    }
    /// Creates the custom contribs for `mol`.
    pub fn custom_contribs(&self, mol: &Molecule) -> Result<Vec<Box<dyn Contrib>>> {
        self.terms.iter().map(|t| t.contrib(mol, &self.globals)).collect()
    }
}

impl<B: ForceFieldBuilder> ForceFieldBuilder for WithCustomTerms<B> {
    fn name(&self) -> &str {
        &self.name
    }
    fn build(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<ContribForceField> {
        let mut ff = self.inner.build(mol, constraints)?;
        if ff.dimension() != 3 {
            return Err(Error::ForceField("custom terms require 3D coordinates".into()));
        }
        ff.add_contribs(self.custom_contribs(mol)?);
        Ok(ff)
    }
    fn build_dyn(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<Box<dyn ForceField>> {
        let base = self.inner.build_dyn(mol, constraints)?;
        if self.terms.is_empty() {
            return Ok(base);
        }
        if base.dimension() != 3 {
            return Err(Error::ForceField("custom terms require 3D coordinates".into()));
        }
        Ok(Box::new(CompositeForceField { base, extra: self.custom_contribs(mol)? }))
    }
}

/// A force field plus extra contribs (used by [`WithCustomTerms::build_dyn`]
/// so that custom terms also work on top of builders whose force field is
/// not a [`ContribForceField`]). Fixed points of the base stay fixed.
pub struct CompositeForceField {
    pub base: Box<dyn ForceField>,
    pub extra: Vec<Box<dyn Contrib>>,
}

impl ForceField for CompositeForceField {
    fn dimension(&self) -> usize {
        self.base.dimension()
    }
    fn num_points(&self) -> usize {
        self.base.num_points()
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        self.base.energy(pos) + self.extra.iter().map(|c| c.energy(pos)).sum::<f64>()
    }
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> f64 {
        let mut e = self.base.energy_grad(pos, grad);
        for c in &self.extra {
            e += c.energy(pos);
            c.grad(pos, grad);
        }
        for &f in self.base.fixed_points() {
            grad[3 * f..3 * f + 3].iter_mut().for_each(|g| *g = 0.0);
        }
        e
    }
    fn energy_breakdown(&self, pos: &[f64]) -> Vec<(String, f64)> {
        let mut out = self.base.energy_breakdown(pos);
        for c in &self.extra {
            c.energy_terms(pos, &mut |label, e| match out.iter_mut().find(|(k, _)| k == label) {
                Some((_, v)) => *v += e,
                None => out.push((label.to_string(), e)),
            });
        }
        out
    }
    fn fixed_points(&self) -> &[usize] {
        self.base.fixed_points()
    }
    fn on_accepted_step(&self, pos: &[f64]) {
        self.base.on_accepted_step(pos)
    }
}

/// `dE/d(global)` summed over the custom contribs in `contribs`.
pub fn energy_param_derivative(contribs: &[Box<dyn Contrib>], name: &str, pos: &[f64]) -> f64 {
    contribs.iter().filter_map(|c| as_custom(c.as_ref())).map(|c| c.param_derivative(name, pos)).sum()
}

/// Sets `name` in every global store referenced by the custom contribs.
/// Returns the number of stores updated (0 if no custom term declares it).
fn set_global_in(contribs: &[Box<dyn Contrib>], name: &str, value: f64) -> usize {
    let mut done: Vec<super::Globals> = Vec::new();
    for c in contribs.iter().filter_map(|c| as_custom(c.as_ref())) {
        let g = c.globals();
        if g.index(name).is_some() && !done.iter().any(|d| d.same_store(g)) {
            let _ = g.set(name, value);
            done.push(g.clone());
        }
    }
    done.len()
}

/// Access to the global parameters of custom terms in a built force field.
pub trait CustomTermsExt {
    /// `dE/d(name)` at `pos` (analytic, from the symbolic derivative).
    fn energy_param_derivative(&self, name: &str, pos: &[f64]) -> f64;
    /// Sets a global parameter of the custom terms in this force field
    /// (shared with the builder and other force fields built from it).
    fn set_global(&self, name: &str, value: f64) -> Result<()>;
}

fn set_or_err(n: usize, name: &str) -> Result<()> {
    if n == 0 {
        return Err(Error::InvalidArgument(format!("no custom term declares global parameter '{name}'")));
    }
    Ok(())
}

impl CustomTermsExt for ContribForceField {
    fn energy_param_derivative(&self, name: &str, pos: &[f64]) -> f64 {
        energy_param_derivative(self.contribs(), name, pos)
    }
    fn set_global(&self, name: &str, value: f64) -> Result<()> {
        set_or_err(set_global_in(self.contribs(), name, value), name)
    }
}

impl CustomTermsExt for CompositeForceField {
    fn energy_param_derivative(&self, name: &str, pos: &[f64]) -> f64 {
        energy_param_derivative(&self.extra, name, pos)
    }
    fn set_global(&self, name: &str, value: f64) -> Result<()> {
        set_or_err(set_global_in(&self.extra, name, value), name)
    }
}
