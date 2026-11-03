//! Conformer search: generate starting geometries in parallel (ETKDG by
//! default), minimize each with a force field, and keep the unique
//! low-energy conformers.

use super::{Frame, SharedBuilder, to_flat, to_points};
use crate::core::geometry::{V3, kabsch_rmsd};
use crate::core::{BondOrder, Molecule};
use crate::error::{Error, Result};
use crate::ff::ForceFieldBuilder;
use crate::optim::{self, MinimizeOptions, MinimizeStatus, Observer};
use crate::tasks::{CtxObserver, Engine, JobContext, JobHandle};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Produces starting geometries. `conf_id` selects the (deterministic)
/// random stream, so results do not depend on scheduling.
pub trait ConformerGenerator: Send + Sync {
    fn name(&self) -> &str;
    /// Generates conformer `conf_id`. The observer sees internal
    /// minimization steps (positions may be 4D); it may stop the run.
    fn generate(&self, mol: &Molecule, conf_id: usize, observer: &mut dyn Observer) -> Result<Vec<V3>>;
}

/// Simple generator: random rotations about rotatable bonds of the input
/// geometry (no ring sampling). Useful when the input geometry is good.
#[derive(Debug, Clone)]
pub struct RandomTorsionGenerator {
    pub seed: u64,
}

impl RandomTorsionGenerator {
    /// Rotatable bonds: acyclic single bonds between atoms with degree > 1.
    pub fn rotatable_bonds(mol: &mut Molecule) -> Vec<(usize, usize)> {
        let ri = mol.rings().clone();
        (0..mol.num_bonds())
            .filter(|&b| {
                let bd = mol.bond(b);
                bd.order == BondOrder::Single
                    && !ri.is_bond_in_ring(b)
                    && mol.degree(bd.begin) > 1
                    && mol.degree(bd.end) > 1
            })
            .map(|b| (mol.bond(b).begin, mol.bond(b).end))
            .collect()
    }
}

impl ConformerGenerator for RandomTorsionGenerator {
    fn name(&self) -> &str {
        "RandomTorsion"
    }
    fn generate(&self, mol: &Molecule, conf_id: usize, _observer: &mut dyn Observer) -> Result<Vec<V3>> {
        let mut m = mol.clone();
        let rot = Self::rotatable_bonds(&mut m);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed ^ (conf_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut c = mol.coords.clone();
        if conf_id == 0 {
            return Ok(c);
        }
        for (a, b) in rot {
            let Some(side) = mol.fragment_side(b, a) else { continue };
            let theta: f64 = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
            let (o, axis) = (c[a], crate::core::geometry::sub(c[b], c[a]));
            for i in side {
                c[i] = crate::core::geometry::rotate_about_axis(c[i], o, axis, theta);
            }
        }
        Ok(c)
    }
}

/// Conformer search options.
#[derive(Clone)]
pub struct ConformerSearchOptions {
    /// Number of starting geometries.
    pub num_conformers: usize,
    pub generator: Arc<dyn ConformerGenerator>,
    pub minimize: MinimizeOptions,
    /// Keep conformers within this window above the minimum (kcal/mol).
    pub energy_window: f64,
    /// Two conformers are duplicates if their RMSD is below this (Å).
    pub rmsd_threshold: f64,
    /// ... and their energies differ by less than this (kcal/mol).
    pub energy_tolerance: f64,
    /// Use hydrogens in the RMSD.
    pub rmsd_with_hydrogens: bool,
    /// Maximum number of conformers to return.
    pub max_conformers: Option<usize>,
}

impl std::fmt::Debug for ConformerSearchOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConformerSearchOptions")
            .field("num_conformers", &self.num_conformers)
            .field("generator", &self.generator.name())
            .field("minimize", &self.minimize)
            .field("energy_window", &self.energy_window)
            .field("rmsd_threshold", &self.rmsd_threshold)
            .finish()
    }
}

impl ConformerSearchOptions {
    pub fn new(generator: Arc<dyn ConformerGenerator>) -> Self {
        ConformerSearchOptions {
            num_conformers: 50,
            generator,
            minimize: MinimizeOptions { max_iters: 2000, ..Default::default() },
            energy_window: f64::INFINITY,
            rmsd_threshold: 0.125,
            energy_tolerance: 1e-2,
            rmsd_with_hydrogens: false,
            max_conformers: None,
        }
    }

    /// Options using the ETKDG generator ([`crate::distgeom::EtkdgGenerator`]):
    /// starting geometry `i` equals conformer `i` of RDKit
    /// `EmbedMultipleConfs(mol, num_conformers, params)`.
    pub fn etkdg(params: crate::distgeom::EmbedParams, num_conformers: usize) -> Self {
        let mut o = Self::new(Arc::new(crate::distgeom::EtkdgGenerator::new(params)));
        o.num_conformers = num_conformers;
        o
    }
}

/// An optimized conformer.
#[derive(Debug, Clone, PartialEq)]
pub struct Conformer {
    /// Index of the starting geometry it came from.
    pub id: usize,
    pub coords: Vec<V3>,
    pub energy: f64,
    pub status: MinimizeStatus,
    pub iterations: usize,
}

/// Stage of one in-flight conformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Generating,
    Optimizing,
}

/// Snapshot of a running conformer search.
#[derive(Debug, Clone, Default)]
pub struct ConformerSearchSnapshot {
    /// Conformers currently being generated/optimized (by id) with their
    /// current coordinates (3D projection during 4D embedding).
    pub in_flight: BTreeMap<usize, (Stage, Frame)>,
    /// Unique optimized conformers found so far, sorted by energy.
    pub found: Vec<Conformer>,
    pub completed: usize,
    pub failed: usize,
    pub total: usize,
}

impl ConformerSearchSnapshot {
    /// Lowest-energy conformer found so far.
    pub fn best(&self) -> Option<&Conformer> {
        self.found.first()
    }
}

/// Result of a conformer search.
#[derive(Debug, Clone)]
pub struct ConformerSearchResult {
    /// Unique conformers sorted by energy (lowest first).
    pub conformers: Vec<Conformer>,
    /// All optimized conformers before deduplication, by id.
    pub all: Vec<Conformer>,
    pub failed: usize,
    pub cancelled: bool,
}

impl ConformerSearchResult {
    pub fn global_minimum(&self) -> Option<&Conformer> {
        self.conformers.first()
    }
}

fn rmsd_atoms(mol: &Molecule, with_h: bool) -> Vec<usize> {
    if with_h { (0..mol.num_atoms()).collect() } else { mol.heavy_atoms() }
}

fn rmsd(a: &[V3], b: &[V3], atoms: &[usize]) -> f64 {
    let pa: Vec<V3> = atoms.iter().map(|&i| a[i]).collect();
    let pb: Vec<V3> = atoms.iter().map(|&i| b[i]).collect();
    kabsch_rmsd(&pa, &pb)
}

/// Deduplicates conformers (deterministic: by energy, then id).
pub fn deduplicate(mut confs: Vec<Conformer>, mol: &Molecule, opts: &ConformerSearchOptions) -> Vec<Conformer> {
    confs.sort_by(|a, b| a.energy.total_cmp(&b.energy).then(a.id.cmp(&b.id)));
    let atoms = rmsd_atoms(mol, opts.rmsd_with_hydrogens);
    let emin = confs.first().map(|c| c.energy).unwrap_or(0.0);
    let mut out: Vec<Conformer> = Vec::new();
    for c in confs {
        if c.energy - emin > opts.energy_window {
            break;
        }
        let dup = out
            .iter()
            .any(|o| (o.energy - c.energy).abs() < opts.energy_tolerance && rmsd(&o.coords, &c.coords, &atoms) < opts.rmsd_threshold);
        if !dup {
            out.push(c);
        }
        if let Some(m) = opts.max_conformers {
            if out.len() >= m {
                break;
            }
        }
    }
    out
}

fn insert_found(found: &mut Vec<Conformer>, c: Conformer, atoms: &[usize], opts: &ConformerSearchOptions) {
    if let Some(pos) = found
        .iter()
        .position(|o| (o.energy - c.energy).abs() < opts.energy_tolerance && rmsd(&o.coords, &c.coords, atoms) < opts.rmsd_threshold)
    {
        if c.energy < found[pos].energy {
            found[pos] = c;
        } else {
            return;
        }
    } else {
        found.push(c);
    }
    found.sort_by(|a, b| a.energy.total_cmp(&b.energy).then(a.id.cmp(&b.id)));
}

fn run_one(
    mol: &Molecule,
    builder: &dyn ForceFieldBuilder,
    ff: &dyn crate::ff::ForceField,
    opts: &ConformerSearchOptions,
    id: usize,
    job: Option<&JobContext<ConformerSearchSnapshot>>,
) -> Result<Conformer> {
    let _ = builder;
    let coords = match job {
        Some(ctx) => {
            ctx.update_silent(|s| {
                s.in_flight.insert(id, (Stage::Generating, Frame::default()));
            });
            let mut obs = CtxObserver::new(ctx, |ctx, info| {
                let dim = info.pos.len() / mol.num_atoms().max(1);
                let coords: Vec<V3> = (0..mol.num_atoms())
                    .map(|i| [info.pos[dim * i], info.pos[dim * i + 1], info.pos[dim * i + 2]])
                    .collect();
                let (e, it) = (info.energy, info.iteration);
                ctx.update(|s| {
                    s.in_flight.insert(id, (Stage::Generating, Frame { coords, energy: e, iteration: it }));
                });
            });
            opts.generator.generate(mol, id, &mut obs)?
        }
        None => opts.generator.generate(mol, id, &mut optim::NoObserver)?,
    };
    if let Some(ctx) = job {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
    }
    let mut pos = to_flat(&coords);
    let r = match job {
        Some(ctx) => {
            ctx.update(|s| {
                s.in_flight.insert(id, (Stage::Optimizing, Frame { coords: coords.clone(), energy: f64::NAN, iteration: 0 }));
            });
            let mut obs = CtxObserver::new(ctx, |ctx, info| {
                let coords = to_points(info.pos);
                let (e, it) = (info.energy, info.iteration);
                ctx.update(|s| {
                    s.in_flight.insert(id, (Stage::Optimizing, Frame { coords, energy: e, iteration: it }));
                });
            });
            optim::minimize(ff, &mut pos, &opts.minimize, &mut obs)
        }
        None => optim::minimize(ff, &mut pos, &opts.minimize, &mut optim::NoObserver),
    };
    if r.status == MinimizeStatus::Cancelled {
        return Err(Error::Cancelled);
    }
    Ok(Conformer { id, coords: to_points(&pos), energy: r.energy, status: r.status, iterations: r.iterations })
}

fn search(
    mol: &Molecule,
    builder: &dyn ForceFieldBuilder,
    opts: &ConformerSearchOptions,
    job: Option<&JobContext<ConformerSearchSnapshot>>,
) -> Result<ConformerSearchResult> {
    let ff = builder.build_dyn(mol, &[])?;
    let atoms = rmsd_atoms(mol, opts.rmsd_with_hydrogens);
    let results: Vec<Result<Conformer>> = (0..opts.num_conformers)
        .into_par_iter()
        .map(|id| {
            let r = run_one(mol, builder, &ff, opts, id, job);
            if let Some(ctx) = job {
                ctx.update(|s| {
                    s.in_flight.remove(&id);
                    match &r {
                        Ok(c) => {
                            s.completed += 1;
                            insert_found(&mut s.found, c.clone(), &atoms, opts);
                        }
                        Err(Error::Cancelled) => {}
                        Err(_) => s.failed += 1,
                    }
                });
                ctx.inc_progress();
            }
            r
        })
        .collect();
    let mut all = Vec::new();
    let mut failed = 0;
    let mut cancelled = false;
    for r in results {
        match r {
            Ok(c) => all.push(c),
            Err(Error::Cancelled) => cancelled = true,
            Err(_) => failed += 1,
        }
    }
    if all.is_empty() && !cancelled {
        return Err(Error::Embedding("no conformer could be generated".into()));
    }
    let conformers = deduplicate(all.clone(), mol, opts);
    Ok(ConformerSearchResult { conformers, all, failed, cancelled })
}

/// Synchronous conformer search (parallel on the current rayon pool).
pub fn conformer_search(mol: &Molecule, builder: &dyn ForceFieldBuilder, opts: &ConformerSearchOptions) -> Result<ConformerSearchResult> {
    search(mol, builder, opts, None)
}

impl Engine {
    /// Asynchronous conformer search. The snapshot shows every in-flight
    /// conformer's current coordinates and the unique conformers found so far.
    pub fn conformer_search(
        &self,
        mol: Arc<Molecule>,
        builder: SharedBuilder,
        opts: ConformerSearchOptions,
    ) -> JobHandle<ConformerSearchResult, ConformerSearchSnapshot> {
        let n = opts.num_conformers;
        self.spawn(ConformerSearchSnapshot { total: n, ..Default::default() }, n, move |ctx| {
            search(&mol, builder.as_ref(), &opts, Some(&ctx))
        })
    }
}
