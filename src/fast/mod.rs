//! High-performance evaluation path.
//!
//! [`FastBuilder`] wraps any [`ForceFieldBuilder`] (UFF, MMFF94, ...) and
//! produces a [`FastForceField`] that evaluates the same terms with
//! structure-of-arrays tables, SIMD kernels and deterministic parallel
//! reductions. Two modes:
//!
//! * [`FastMode::Exact`]: identical term set and energy expressions as the
//!   RDKit-compatible path (nonbonded pairs selected at build time exactly
//!   like RDKit). Serial evaluation reproduces the reference contribs'
//!   operation, summation and gradient-scatter order and is bit-identical
//!   to [`ContribForceField`]; parallel evaluation (systems above
//!   [`FastOptions::parallel_grain`]) uses fixed chunks reduced in a fixed
//!   order, so results do not depend on the thread count.
//! * [`FastMode::Approximate`]: cutoff nonbonded interactions with a
//!   buffered (Verlet) cluster-pair list that is rebuilt only when atoms
//!   moved more than half the buffer, optional periodic boundary
//!   conditions and particle-mesh Ewald electrostatics.
//!
//! The nonbonded part is evaluated through the [`NonbondedBackend`] trait
//! so that accelerators (e.g. GPUs) can be added later.
//!
//! ```no_run
//! use molecule_optimizer_rs::fast::{FastBuilder, FastOptions};
//! use molecule_optimizer_rs::ff::{ForceField, ForceFieldBuilder, mmff::MmffBuilder};
//! # fn main() -> molecule_optimizer_rs::Result<()> {
//! let mol = molecule_optimizer_rs::io::read_file("paracetamol.mol")?.remove(0);
//! let fast = FastBuilder::new(MmffBuilder::default(), FastOptions::exact());
//! let ff = fast.build_dyn(&mol, &[])?;
//! let pos = mol.flat_coords();
//! let mut g = vec![0.0; pos.len()];
//! println!("E = {}", ff.energy_grad(&pos, &mut g));
//! # Ok(()) }
//! ```

pub mod approx;
pub mod backend;
pub mod bonded;
pub mod compile;
pub mod exact;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod workspace;

pub use backend::{NonbondedBackend, NonbondedEnergy};
pub use bonded::{Bonded, TableKind};
pub use exact::ExactNonbonded;

use crate::core::Molecule;
use crate::error::Result;
use crate::ff::{Constraint, ContribForceField, ForceField, ForceFieldBuilder};
use compile::{ConstraintTable, Segment, Template};
use rayon::prelude::*;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use workspace::WorkspacePool;

/// Van der Waals truncation for approximate mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VdwModifier {
    /// Shift the potential to zero at the cutoff.
    PotentialShift,
    /// Smoothly switch the potential to zero between `r_switch` and the cutoff.
    Switch { r_switch: f64 },
}

/// Electrostatics treatment for approximate mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Electrostatics {
    /// No cutoff (all pairs, SIMD N²); isolated systems only.
    Full,
    /// Reaction-field cutoff (`eps_rf = f64::INFINITY` = conducting boundary).
    ReactionField { eps_rf: f64 },
    /// Smooth particle-mesh Ewald (periodic systems only).
    Pme { ewald_rtol: f64, grid_spacing: f64, order: usize },
}

/// Options of the approximate mode.
#[derive(Debug, Clone, PartialEq)]
pub struct ApproxOptions {
    /// Interaction cutoff (Å).
    pub cutoff: f64,
    /// Verlet buffer (Å): the pair list uses `cutoff + buffer` and is rebuilt
    /// when any atom moved more than `buffer / 2` since the last build.
    pub buffer: f64,
    pub vdw_modifier: VdwModifier,
    pub electrostatics: Electrostatics,
}

impl Default for ApproxOptions {
    fn default() -> Self {
        ApproxOptions {
            cutoff: 10.0,
            buffer: 1.0,
            vdw_modifier: VdwModifier::PotentialShift,
            electrostatics: Electrostatics::ReactionField { eps_rf: f64::INFINITY },
        }
    }
}

/// Evaluation mode.
#[derive(Debug, Clone, PartialEq)]
pub enum FastMode {
    Exact,
    Approximate(ApproxOptions),
}

/// Options of [`FastBuilder`].
#[derive(Debug, Clone, PartialEq)]
pub struct FastOptions {
    pub mode: FastMode,
    /// Use SIMD kernels (falls back to scalar reference kernels if `false`).
    pub simd: bool,
    /// Minimum number of terms per parallel task (smaller systems run on
    /// one thread to avoid overhead). Results do not depend on the thread count.
    pub parallel_grain: usize,
    /// Cache atom typing / parameter lookup per molecular graph (UFF and
    /// MMFF builders only).
    pub cache: bool,
}

impl FastOptions {
    pub fn exact() -> Self {
        FastOptions { mode: FastMode::Exact, simd: true, parallel_grain: 2048, cache: true }
    }
    pub fn approximate() -> Self {
        FastOptions { mode: FastMode::Approximate(ApproxOptions::default()), simd: true, parallel_grain: 2048, cache: true }
    }
    /// SIMD (`true`) or scalar reference kernels.
    pub fn simd(mut self, on: bool) -> Self {
        self.simd = on;
        self
    }
    pub fn parallel_grain(mut self, grain: usize) -> Self {
        self.parallel_grain = grain.max(1);
        self
    }
    /// Enables / disables the typing cache of [`FastBuilder`].
    pub fn cache(mut self, on: bool) -> Self {
        self.cache = on;
        self
    }
    pub fn cutoff(mut self, rc: f64) -> Self {
        if let FastMode::Approximate(a) = &mut self.mode {
            a.cutoff = rc;
        }
        self
    }
    pub fn buffer(mut self, b: f64) -> Self {
        if let FastMode::Approximate(a) = &mut self.mode {
            a.buffer = b;
        }
        self
    }
    pub fn electrostatics(mut self, e: Electrostatics) -> Self {
        if let FastMode::Approximate(a) = &mut self.mode {
            a.electrostatics = e;
        }
        self
    }
    pub fn vdw_modifier(mut self, m: VdwModifier) -> Self {
        if let FastMode::Approximate(a) = &mut self.mode {
            a.vdw_modifier = m;
        }
        self
    }
}

impl Default for FastOptions {
    fn default() -> Self {
        Self::exact()
    }
}

// ---------------------------------------------------------------------------
// FastForceField
// ---------------------------------------------------------------------------

enum Nb {
    None,
    Exact(ExactNonbonded),
    Other(Box<dyn NonbondedBackend>),
}

impl Nb {
    fn backend(&self) -> Option<&dyn NonbondedBackend> {
        match self {
            Nb::None => None,
            Nb::Exact(e) => Some(e),
            Nb::Other(b) => Some(b.as_ref()),
        }
    }
}

type NbLabels = (Option<&'static str>, Option<&'static str>);

/// Force field evaluated with SoA tables, SIMD kernels and deterministic
/// parallel reductions (built by [`FastBuilder`]).
pub struct FastForceField {
    n: usize,
    bonded: Arc<Bonded>,
    segments: Vec<Segment>,
    constraints: ConstraintTable,
    /// Wrapped force field holding the opaque contribs (if any).
    opaque_src: Option<Arc<ContribForceField>>,
    opaque: Vec<usize>,
    nb: Nb,
    /// Breakdown labels of the nonbonded parts (vdW, electrostatics).
    nb_labels: NbLabels,
    fixed: Vec<usize>,
    simd: bool,
    ws: WorkspacePool,
}

impl std::fmt::Debug for FastForceField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastForceField")
            .field("n", &self.n)
            .field("bonded_terms", &self.bonded.num_terms())
            .field("bonded_chunks", &self.bonded.chunks.len())
            .field("nonbonded", &self.nb.backend().map(|b| b.name().to_string()))
            .field("opaque", &self.opaque.len())
            .field("fixed", &self.fixed)
            .finish()
    }
}

struct Parts {
    n: usize,
    bonded: Arc<Bonded>,
    segments: Vec<Segment>,
    constraints: ConstraintTable,
    opaque_src: Option<Arc<ContribForceField>>,
    opaque: Vec<usize>,
    nb: Nb,
    nb_labels: NbLabels,
    fixed: Vec<usize>,
    simd: bool,
}

impl FastForceField {
    fn assemble(p: Parts) -> Self {
        let ws = WorkspacePool::new(p.bonded.chunks.len(), 3 * p.n);
        FastForceField {
            n: p.n,
            bonded: p.bonded,
            segments: p.segments,
            constraints: p.constraints,
            opaque_src: p.opaque_src,
            opaque: p.opaque,
            nb: p.nb,
            nb_labels: p.nb_labels,
            fixed: p.fixed,
            simd: p.simd,
            ws,
        }
    }

    /// The exact nonbonded backend, if this force field uses one.
    pub fn exact_nonbonded(&self) -> Option<&ExactNonbonded> {
        match &self.nb {
            Nb::Exact(e) => Some(e),
            _ => None,
        }
    }

    /// The nonbonded backend.
    pub fn nonbonded(&self) -> Option<&dyn NonbondedBackend> {
        self.nb.backend()
    }

    /// Bonded tables.
    pub fn bonded(&self) -> &Bonded {
        &self.bonded
    }

    /// Number of contribs evaluated through their own (opaque) code.
    pub fn num_opaque(&self) -> usize {
        self.opaque.len()
    }

    /// Switches between SIMD and scalar reference kernels.
    pub fn set_simd(&mut self, simd: bool) {
        self.simd = simd;
        if let Nb::Exact(e) = &mut self.nb {
            e.set_simd(simd);
        }
    }

    /// Evaluates every term, adding the energies to a running total in the
    /// wrapped force field's order (serial: bit-identical to
    /// [`ContribForceField`]'s serial evaluation, whose summation and scatter
    /// order it reproduces); with a parallel plan the bonded tables are
    /// evaluated first in fixed chunks.
    fn eval<const G: bool>(&self, pos: &[f64], grad: &mut [f64]) -> f64 {
        if let Nb::Exact(e) = &self.nb {
            if e.is_parallel() {
                // overlap the other terms with the parallel pair chunks
                let mut nb_acc = 0.0;
                let (_, rest) = e.eval_overlapped::<G, f64>(pos, grad, &mut nb_acc, |g| self.eval_segments::<G>(pos, g, false));
                return rest + nb_acc;
            }
        }
        self.eval_segments::<G>(pos, grad, true)
    }

    /// Evaluates the segments in order (the nonbonded one only if
    /// `with_nb`).
    fn eval_segments<const G: bool>(&self, pos: &[f64], grad: &mut [f64], with_nb: bool) -> f64 {
        let b = &*self.bonded;
        let simd = self.simd;
        let parallel = !b.chunks.is_empty();
        let mut acc = if parallel { self.bonded_parallel::<G>(pos, grad) } else { 0.0 };
        for seg in &self.segments {
            match *seg {
                Segment::Table(k) => {
                    if !parallel {
                        b.eval_table::<G>(k, pos, grad, simd, &mut acc);
                    }
                }
                Segment::Constraints => self.constraints.eval::<G>(pos, grad, &mut acc),
                Segment::Opaque(i) => {
                    if let Some(src) = &self.opaque_src {
                        let c = &src.contribs()[i];
                        acc += c.energy(pos);
                        if G {
                            c.grad(pos, grad);
                        }
                    }
                }
                Segment::Nonbonded if !with_nb => {}
                Segment::Nonbonded => match &self.nb {
                    Nb::None => {}
                    Nb::Exact(e) => {
                        e.eval_into::<G>(pos, grad, &mut acc);
                    }
                    Nb::Other(o) => acc += if G { o.energy_grad(pos, grad) } else { o.energy(pos) }.total(),
                },
            }
        }
        acc
    }

    fn bonded_parallel<const G: bool>(&self, pos: &[f64], grad: &mut [f64]) -> f64 {
        let b = &*self.bonded;
        let simd = self.simd;
        let len = 3 * self.n;
        self.ws.with(|ws| {
            let (bufs, energies) = ws.split();
            let run = |ch: &Vec<(TableKind, std::ops::Range<usize>)>, buf: &mut [f64]| -> f64 {
                let mut e = 0.0;
                for (k, r) in ch {
                    b.eval::<G>(*k, pos, buf, r.clone(), simd, &mut e);
                }
                e
            };
            if G {
                bufs.par_chunks_mut(len).zip(energies.par_iter_mut()).zip(b.chunks.par_iter()).for_each(
                    |((buf, en), ch)| {
                        buf.fill(0.0);
                        en[0] = run(ch, buf);
                    },
                );
                let nc = b.chunks.len();
                let bufs: &[f64] = bufs;
                workspace::reduce_into(grad, len, |k0, k1, out| {
                    for c in 0..nc {
                        for (o, x) in out.iter_mut().zip(&bufs[c * len + k0..c * len + k1]) {
                            *o += *x;
                        }
                    }
                });
            } else {
                energies.par_iter_mut().zip(b.chunks.par_iter()).for_each(|(en, ch)| {
                    en[0] = run(ch, &mut []);
                });
            }
            let mut e = 0.0;
            for en in energies.iter() {
                e += en[0];
            }
            e
        })
    }

}

impl ForceField for FastForceField {
    fn num_points(&self) -> usize {
        self.n
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        self.eval::<false>(pos, &mut [])
    }
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> f64 {
        let n3 = 3 * self.n;
        grad[..n3].fill(0.0);
        let e = self.eval::<true>(pos, grad);
        for &f in &self.fixed {
            grad[3 * f..3 * f + 3].fill(0.0);
        }
        e
    }
    fn energy_breakdown(&self, pos: &[f64]) -> Vec<(String, f64)> {
        let mut out: Vec<(String, f64)> = Vec::new();
        let mut push = |label: &str, e: f64| match out.iter_mut().find(|(k, _)| k == label) {
            Some((_, v)) => *v += e,
            None => out.push((label.to_string(), e)),
        };
        for seg in &self.segments {
            match *seg {
                Segment::Table(k) => push(k.label(), self.bonded.table_energy(k, pos, self.simd)),
                Segment::Constraints => push("Constraint", self.constraints.energy(pos)),
                Segment::Opaque(i) => {
                    if let Some(src) = &self.opaque_src {
                        src.contribs()[i].energy_terms(pos, &mut |l, e| push(l, e));
                    }
                }
                Segment::Nonbonded => {
                    if let Some(nb) = self.nb.backend() {
                        let e = nb.energy(pos);
                        if let Some(l) = self.nb_labels.0 {
                            push(l, e.vdw);
                        }
                        if let Some(l) = self.nb_labels.1 {
                            push(l, e.electrostatic);
                        }
                    }
                }
            }
        }
        out
    }
    fn fixed_points(&self) -> &[usize] {
        &self.fixed
    }
    fn on_accepted_step(&self, pos: &[f64]) {
        if let Some(nb) = self.nb.backend() {
            nb.on_accepted_step(pos);
        }
    }
}

// ---------------------------------------------------------------------------
// FastBuilder
// ---------------------------------------------------------------------------

/// Maximum number of cached molecular graphs per builder.
const CACHE_CAPACITY: usize = 256;

/// Wraps a [`ForceFieldBuilder`] and builds [`FastForceField`]s
/// ([`ForceFieldBuilder::build_dyn`]); [`ForceFieldBuilder::build`] still
/// returns the wrapped builder's [`ContribForceField`] unchanged.
///
/// For [`crate::ff::uff::UffBuilder`] and [`crate::ff::mmff::MmffBuilder`]
/// the expensive per-molecule work (typing, charges, parameter lookup,
/// bonded tables) is cached per molecular graph, so that scan points and
/// conformers of one molecule only redo the position-dependent nonbonded
/// pair selection and the constraints.
pub struct FastBuilder<B: ForceFieldBuilder + 'static> {
    inner: B,
    opts: FastOptions,
    name: String,
    cache: Mutex<HashMap<Vec<u64>, Arc<Template>>>,
}

impl<B: ForceFieldBuilder + 'static> std::fmt::Debug for FastBuilder<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastBuilder").field("name", &self.name).field("opts", &self.opts).finish()
    }
}

impl<B: ForceFieldBuilder + 'static> FastBuilder<B> {
    pub fn new(inner: B, opts: FastOptions) -> Self {
        let name = format!("{}/fast", inner.name());
        FastBuilder { inner, opts, name, cache: Mutex::new(HashMap::new()) }
    }
    pub fn inner(&self) -> &B {
        &self.inner
    }
    pub fn options(&self) -> &FastOptions {
        &self.opts
    }
    /// Number of cached molecular graphs.
    pub fn cache_len(&self) -> usize {
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
    pub fn clear_cache(&self) {
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    fn inner_any(&self) -> &dyn Any {
        &self.inner
    }

    fn template(&self, mol: &Molecule) -> Result<Option<Arc<Template>>> {
        if !self.opts.cache || !compile::is_cacheable_builder(self.inner_any()) {
            return Ok(None);
        }
        let key = compile::graph_key(mol);
        let hit = self.cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key).cloned();
        if let Some(t) = hit {
            if mol.coords.len() != mol.num_atoms() {
                return Err(crate::Error::InvalidArgument("molecule has no coordinates".into()));
            }
            return Ok(Some(t));
        }
        let Some(t) = compile::make_template(self.inner_any(), mol, self.opts.parallel_grain)? else { return Ok(None) };
        let t = Arc::new(t);
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if c.len() >= CACHE_CAPACITY {
            c.clear();
        }
        c.insert(key, t.clone());
        Ok(Some(t))
    }

    fn approx_backend(&self, mol: &Molecule) -> Result<Option<Nb>> {
        match &self.opts.mode {
            FastMode::Exact => Ok(None),
            FastMode::Approximate(a) => {
                let nb = approx::build_approx_nonbonded(mol, self.inner_any(), a, self.opts.simd)?;
                Ok(Some(Nb::Other(Box::new(nb))))
            }
        }
    }

    fn approx_labels(&self) -> NbLabels {
        let uff = self.inner_any().is::<crate::ff::uff::UffBuilder>();
        (Some("VdW"), if uff { None } else { Some("Electrostatic") })
    }

    /// Builds the fast force field.
    pub fn build_fast(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<FastForceField> {
        let n = mol.num_atoms();
        let (simd, grain) = (self.opts.simd, self.opts.parallel_grain);
        if let Some(t) = self.template(mol)? {
            let (nb, nb_labels) = match self.approx_backend(mol)? {
                Some(nb) => (nb, self.approx_labels()),
                None => {
                    let e = t.nb.select(&mol.coords, grain, simd);
                    let labels = (e.has_vdw().then_some("VdW"), e.has_electrostatics().then_some("Electrostatic"));
                    (Nb::Exact(e), labels)
                }
            };
            let mut segments = t.before.clone();
            segments.push(Segment::Nonbonded);
            segments.extend_from_slice(&t.after);
            let mut cff = ContribForceField::new(n, 3);
            cff.add_constraints(&mol.coords, constraints);
            let c = compile::compile(cff.contribs(), false);
            debug_assert!(c.opaque.is_empty());
            if !c.constraints.is_empty() {
                segments.push(Segment::Constraints);
            }
            let fixed = cff.fixed_points().to_vec();
            return Ok(FastForceField::assemble(Parts {
                n,
                bonded: t.bonded.clone(),
                segments,
                constraints: c.constraints,
                opaque_src: None,
                opaque: Vec::new(),
                nb,
                nb_labels,
                fixed,
                simd,
            }));
        }
        // generic path: compile the wrapped force field
        let ff = self.inner.build(mol, constraints)?;
        let approx = self.approx_backend(mol)?;
        let mut c = compile::compile(ff.contribs(), approx.is_some());
        c.bonded.plan(grain);
        let (nb, nb_labels) = match approx {
            Some(nb) => (nb, self.approx_labels()),
            None if c.nb.is_empty() => (Nb::None, (None, None)),
            None => {
                let e = std::mem::take(&mut c.nb).finish(n, grain, simd);
                let labels = (e.has_vdw().then_some("VdW"), e.has_electrostatics().then_some("Electrostatic"));
                (Nb::Exact(e), labels)
            }
        };
        let fixed = ff.fixed_points().to_vec();
        let opaque_src = if c.opaque.is_empty() { None } else { Some(Arc::new(ff)) };
        Ok(FastForceField::assemble(Parts {
            n,
            bonded: Arc::new(c.bonded),
            segments: c.segments,
            constraints: c.constraints,
            opaque_src,
            opaque: c.opaque,
            nb,
            nb_labels,
            fixed,
            simd,
        }))
    }
}

impl<B: ForceFieldBuilder + 'static> ForceFieldBuilder for FastBuilder<B> {
    fn name(&self) -> &str {
        &self.name
    }
    fn build(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<ContribForceField> {
        self.inner.build(mol, constraints)
    }
    fn build_dyn(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<Box<dyn ForceField>> {
        Ok(Box::new(self.build_fast(mol, constraints)?))
    }
}
