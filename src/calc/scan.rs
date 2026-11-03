//! Coordinate scans (relaxed / independent, 1D or multi-dimensional).

use super::{Frame, SharedBuilder, to_flat, to_points};
use crate::core::Molecule;
use crate::core::geometry::{V3, add, angle, cross, dihedral, dist, normalize, rotate_about_axis, scale, sub};
use crate::error::{Error, Result};
use crate::ff::{Constraint, ForceField, ForceFieldBuilder};
use crate::optim::{self, MinimizeOptions, MinimizeStatus};
use crate::tasks::{CtxObserver, Engine, JobContext, JobHandle};
use rayon::prelude::*;
use std::sync::Arc;

/// A scanned internal coordinate with the values to visit
/// (Å for distances, degrees for angles and dihedrals).
#[derive(Debug, Clone, PartialEq)]
pub enum ScanCoordinate {
    Distance { i: usize, j: usize, values: Vec<f64> },
    Angle { i: usize, j: usize, k: usize, values: Vec<f64> },
    Dihedral { i: usize, j: usize, k: usize, l: usize, values: Vec<f64> },
}

impl ScanCoordinate {
    /// Evenly spaced values `start, start+step, ...` (`n` values).
    pub fn steps(start: f64, step: f64, n: usize) -> Vec<f64> {
        (0..n).map(|k| start + step * k as f64).collect()
    }
    pub fn values(&self) -> &[f64] {
        match self {
            ScanCoordinate::Distance { values, .. }
            | ScanCoordinate::Angle { values, .. }
            | ScanCoordinate::Dihedral { values, .. } => values,
        }
    }
    /// Current value of the coordinate.
    pub fn measure(&self, c: &[V3]) -> f64 {
        match *self {
            ScanCoordinate::Distance { i, j, .. } => dist(c[i], c[j]),
            ScanCoordinate::Angle { i, j, k, .. } => angle(c[i], c[j], c[k]).to_degrees(),
            ScanCoordinate::Dihedral { i, j, k, l, .. } => dihedral(c[i], c[j], c[k], c[l]).to_degrees(),
        }
    }
    fn constraint(&self, value: f64, force: f64) -> Constraint {
        match *self {
            ScanCoordinate::Distance { i, j, .. } => Constraint::distance(i, j, value, force),
            ScanCoordinate::Angle { i, j, k, .. } => Constraint::angle(i, j, k, value, force),
            ScanCoordinate::Dihedral { i, j, k, l, .. } => Constraint::dihedral(i, j, k, l, value, force),
        }
    }
    fn atoms(&self) -> Vec<usize> {
        match *self {
            ScanCoordinate::Distance { i, j, .. } => vec![i, j],
            ScanCoordinate::Angle { i, j, k, .. } => vec![i, j, k],
            ScanCoordinate::Dihedral { i, j, k, l, .. } => vec![i, j, k, l],
        }
    }

    /// Moves atoms so that the coordinate takes `value` (rigid rotation /
    /// translation of the moving fragment). Returns `false` if the
    /// coordinate is part of a ring and was left unchanged.
    pub fn set_value(&self, mol: &Molecule, c: &mut [V3], value: f64) -> bool {
        match *self {
            ScanCoordinate::Distance { i, j, .. } => {
                let side = if mol.bond_between(i, j).is_some() {
                    mol.fragment_side(j, i)
                } else {
                    let frags = mol.fragments();
                    let fi = frags.iter().position(|f| f.contains(&i));
                    let fj = frags.iter().position(|f| f.contains(&j));
                    if fi != fj { fj.map(|k| frags[k].clone()) } else { Some(vec![j]) }
                };
                let Some(side) = side else { return false };
                if side.contains(&i) {
                    return false;
                }
                let d = dist(c[i], c[j]);
                let u = normalize(sub(c[j], c[i]));
                let shift = scale(u, value - d);
                for a in side {
                    c[a] = add(c[a], shift);
                }
                true
            }
            ScanCoordinate::Angle { i, j, k, .. } => {
                let Some(side) = mol.fragment_side(k, j) else { return false };
                if side.contains(&i) || side.contains(&j) {
                    return false;
                }
                let cur = angle(c[i], c[j], c[k]).to_degrees();
                let mut n = cross(sub(c[i], c[j]), sub(c[k], c[j]));
                if crate::core::geometry::norm(n) < 1e-8 {
                    n = crate::core::geometry::cross(sub(c[i], c[j]), [0.3, 0.5, 0.8]);
                }
                let delta = (value - cur).to_radians();
                let origin = c[j];
                for a in side {
                    c[a] = rotate_about_axis(c[a], origin, n, delta);
                }
                true
            }
            ScanCoordinate::Dihedral { i, j, k, l, .. } => {
                let Some(side) = mol.fragment_side(k, j) else { return false };
                if side.contains(&i) || side.contains(&j) {
                    return false;
                }
                let cur = dihedral(c[i], c[j], c[k], c[l]).to_degrees();
                let delta = (value - cur).to_radians();
                let (origin, axis) = (c[j], sub(c[k], c[j]));
                for a in side {
                    c[a] = rotate_about_axis(c[a], origin, axis, delta);
                }
                true
            }
        }
    }
}

/// How scan points are started.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScanMode {
    /// Each point starts from the optimized neighbor point (warm start).
    /// For multi-dimensional scans, lines along later axes run in parallel.
    #[default]
    Relaxed,
    /// Every point starts from the input geometry; all points run in parallel.
    Independent,
}

/// Scan options.
#[derive(Debug, Clone)]
pub struct ScanOptions {
    pub coordinates: Vec<ScanCoordinate>,
    pub mode: ScanMode,
    /// Restraint force constant (RDKit units: kcal/mol/Å² for distances,
    /// kcal/mol/deg² for angles/dihedrals).
    pub distance_force: f64,
    pub angle_force: f64,
    pub minimize: MinimizeOptions,
    /// Additional constraints applied at every point.
    pub extra_constraints: Vec<Constraint>,
    /// Rigidly pre-set the coordinate before minimizing (recommended).
    pub preset_geometry: bool,
}

impl Default for ScanOptions {
    fn default() -> Self {
        ScanOptions {
            coordinates: Vec::new(),
            mode: ScanMode::Relaxed,
            distance_force: 1.0e4,
            angle_force: 1.0e3,
            minimize: MinimizeOptions { max_iters: 2000, ..Default::default() },
            extra_constraints: Vec::new(),
            preset_geometry: true,
        }
    }
}

/// One computed scan point.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanPoint {
    /// Grid index per scan coordinate.
    pub index: Vec<usize>,
    /// Target values.
    pub values: Vec<f64>,
    /// Values actually reached after optimization.
    pub achieved: Vec<f64>,
    pub coords: Vec<[f64; 3]>,
    /// Force-field energy without restraint terms.
    pub energy: f64,
    /// Energy including restraint terms.
    pub restrained_energy: f64,
    pub status: MinimizeStatus,
    pub iterations: usize,
}

/// State of a scan point inside a snapshot.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum PointState {
    #[default]
    Pending,
    Running(Frame),
    Done(Frame),
}

/// Snapshot of a running scan.
#[derive(Debug, Clone, Default)]
pub struct ScanSnapshot {
    /// One entry per grid point (row-major order over the scan coordinates).
    pub points: Vec<PointState>,
    pub shape: Vec<usize>,
}

/// Scan result.
#[derive(Debug, Clone)]
pub struct ScanResult {
    pub shape: Vec<usize>,
    /// Computed points sorted in row-major grid order.
    pub points: Vec<ScanPoint>,
    pub cancelled: bool,
}

impl ScanResult {
    /// Energies relative to the minimum (kcal/mol), in point order.
    pub fn relative_energies(&self) -> Vec<f64> {
        let m = self.points.iter().map(|p| p.energy).fold(f64::INFINITY, f64::min);
        self.points.iter().map(|p| p.energy - m).collect()
    }
}

fn flat_index(idx: &[usize], shape: &[usize]) -> usize {
    idx.iter().zip(shape).fold(0, |acc, (i, n)| acc * n + i)
}

struct ScanCtx<'a> {
    mol: &'a Molecule,
    builder: &'a dyn ForceFieldBuilder,
    opts: &'a ScanOptions,
    shape: Vec<usize>,
}

impl ScanCtx<'_> {
    fn compute_point(
        &self,
        idx: &[usize],
        start: &[V3],
        job: Option<&JobContext<ScanSnapshot>>,
    ) -> Result<ScanPoint> {
        let values: Vec<f64> = idx.iter().zip(&self.opts.coordinates).map(|(&k, c)| c.values()[k]).collect();
        let mut c = start.to_vec();
        if self.opts.preset_geometry {
            for (sc, &v) in self.opts.coordinates.iter().zip(&values) {
                sc.set_value(self.mol, &mut c, v);
            }
        }
        let mut cons = self.opts.extra_constraints.clone();
        for (sc, &v) in self.opts.coordinates.iter().zip(&values) {
            let f = match sc {
                ScanCoordinate::Distance { .. } => self.opts.distance_force,
                _ => self.opts.angle_force,
            };
            cons.push(sc.constraint(v, f));
        }
        let ff = self.builder.build_dyn(self.mol, &cons)?;
        let ff_plain = self.builder.build_dyn(self.mol, &self.opts.extra_constraints)?;
        let mut pos = to_flat(&c);
        let fi = flat_index(idx, &self.shape);
        let r = match job {
            Some(ctx) => {
                if ctx.is_cancelled() {
                    return Err(Error::Cancelled);
                }
                let mut obs = CtxObserver::new(ctx, |ctx, info| {
                    let coords = to_points(info.pos);
                    let (e, it) = (info.energy, info.iteration);
                    ctx.update(|s| s.points[fi] = PointState::Running(Frame { coords, energy: e, iteration: it }));
                });
                ctx.update_silent(|s| s.points[fi] = PointState::Running(Frame { coords: c.clone(), energy: f64::NAN, iteration: 0 }));
                optim::minimize_with(&ff, &mut pos, &self.opts.minimize, false, &mut obs)
            }
            None => optim::minimize_with(&ff, &mut pos, &self.opts.minimize, false, &mut optim::NoObserver),
        };
        if r.status == MinimizeStatus::Cancelled {
            return Err(Error::Cancelled);
        }
        let coords = to_points(&pos);
        let energy = ff_plain.energy(&pos);
        let achieved = self.opts.coordinates.iter().map(|sc| sc.measure(&coords)).collect();
        let pt = ScanPoint {
            index: idx.to_vec(),
            values,
            achieved,
            coords: coords.clone(),
            energy,
            restrained_energy: r.energy,
            status: r.status,
            iterations: r.iterations,
        };
        if let Some(ctx) = job {
            ctx.update(|s| s.points[fi] = PointState::Done(Frame { coords, energy, iteration: r.iterations }));
            ctx.inc_progress();
        }
        Ok(pt)
    }

    /// All points with the given prefix; `first` is the already computed
    /// point `prefix + [0, 0, ...]`.
    fn line(&self, axis: usize, prefix: Vec<usize>, first: ScanPoint, job: Option<&JobContext<ScanSnapshot>>) -> Vec<Result<ScanPoint>> {
        let d = self.shape.len();
        let mut anchors: Vec<Result<ScanPoint>> = vec![Ok(first)];
        for v in 1..self.shape[axis] {
            let start = match anchors.iter().rev().find_map(|a| a.as_ref().ok()) {
                Some(p) => p.coords.clone(),
                None => self.mol.coords.clone(),
            };
            let mut idx = prefix.clone();
            idx.push(v);
            idx.resize(d, 0);
            anchors.push(self.compute_point(&idx, &start, job));
        }
        if axis + 1 == d {
            return anchors;
        }
        anchors
            .into_par_iter()
            .enumerate()
            .flat_map_iter(|(v, a)| {
                let mut p = prefix.clone();
                p.push(v);
                match a {
                    Ok(pt) => self.line(axis + 1, p, pt, job),
                    Err(e) => vec![Err(e)],
                }
            })
            .collect()
    }

    fn run(&self, job: Option<&JobContext<ScanSnapshot>>) -> Result<ScanResult> {
        let d = self.shape.len();
        let total: usize = self.shape.iter().product();
        let mut results: Vec<Result<ScanPoint>> = match self.opts.mode {
            ScanMode::Independent => (0..total)
                .into_par_iter()
                .map(|f| {
                    let mut idx = vec![0; d];
                    let mut rem = f;
                    for a in (0..d).rev() {
                        idx[a] = rem % self.shape[a];
                        rem /= self.shape[a];
                    }
                    self.compute_point(&idx, &self.mol.coords, job)
                })
                .collect(),
            ScanMode::Relaxed => {
                let first = self.compute_point(&vec![0; d], &self.mol.coords, job)?;
                self.line(0, Vec::new(), first, job)
            }
        };
        let mut cancelled = false;
        let mut points = Vec::new();
        for r in results.drain(..) {
            match r {
                Ok(p) => points.push(p),
                Err(Error::Cancelled) => cancelled = true,
                Err(e) => return Err(e),
            }
        }
        points.sort_by_key(|p| flat_index(&p.index, &self.shape));
        Ok(ScanResult { shape: self.shape.clone(), points, cancelled })
    }
}

fn validate(mol: &Molecule, opts: &ScanOptions) -> Result<Vec<usize>> {
    if opts.coordinates.is_empty() {
        return Err(Error::InvalidArgument("no scan coordinates".into()));
    }
    for c in &opts.coordinates {
        if c.values().is_empty() {
            return Err(Error::InvalidArgument("scan coordinate without values".into()));
        }
        if c.atoms().iter().any(|&a| a >= mol.num_atoms()) {
            return Err(Error::InvalidArgument("scan atom index out of range".into()));
        }
    }
    Ok(opts.coordinates.iter().map(|c| c.values().len()).collect())
}

/// Runs a scan synchronously (uses the current rayon pool for parallelism).
pub fn scan(mol: &Molecule, builder: &dyn ForceFieldBuilder, opts: &ScanOptions) -> Result<ScanResult> {
    let shape = validate(mol, opts)?;
    ScanCtx { mol, builder, opts, shape }.run(None)
}

impl Engine {
    /// Runs a coordinate scan asynchronously. Completed and in-progress
    /// points (with their current coordinates) are visible in the snapshot.
    pub fn scan(&self, mol: Arc<Molecule>, builder: SharedBuilder, opts: ScanOptions) -> JobHandle<ScanResult, ScanSnapshot> {
        let shape = validate(&mol, &opts).unwrap_or_default();
        let total: usize = shape.iter().product();
        let init = ScanSnapshot { points: vec![PointState::Pending; total], shape: shape.clone() };
        self.spawn(init, total, move |ctx| {
            let shape = validate(&mol, &opts)?;
            ScanCtx { mol: &mol, builder: builder.as_ref(), opts: &opts, shape }.run(Some(&ctx))
        })
    }
}
