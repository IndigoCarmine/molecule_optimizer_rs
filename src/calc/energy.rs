//! Single-point energies.

use super::{SharedBuilder, to_flat, to_points};
use crate::core::Molecule;
use crate::error::Result;
use crate::ff::{ForceField, ForceFieldBuilder};
use crate::tasks::{Engine, JobHandle};
use rayon::prelude::*;
use std::sync::Arc;

/// Energy with breakdown and gradient.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyResult {
    /// Total energy (kcal/mol).
    pub total: f64,
    /// Energy per term category (e.g. "BondStretch").
    pub breakdown: Vec<(String, f64)>,
    /// dE/dx per atom (kcal/mol/Å).
    pub gradient: Vec<[f64; 3]>,
    pub grad_rms: f64,
}

/// Energy of `mol` at its current coordinates.
pub fn energy(mol: &Molecule, builder: &dyn ForceFieldBuilder) -> Result<EnergyResult> {
    energy_at(mol, &mol.coords, builder)
}

/// Energy of `mol` at the given coordinates.
pub fn energy_at(mol: &Molecule, coords: &[[f64; 3]], builder: &dyn ForceFieldBuilder) -> Result<EnergyResult> {
    let ff = builder.build(mol, &[])?;
    Ok(evaluate(&ff, coords))
}

/// Evaluates an existing force field.
pub fn evaluate(ff: &dyn ForceField, coords: &[[f64; 3]]) -> EnergyResult {
    let pos = to_flat(coords);
    let mut g = vec![0.0; pos.len()];
    let total = ff.energy_grad(&pos, &mut g);
    let grad_rms = (g.iter().map(|x| x * x).sum::<f64>() / g.len().max(1) as f64).sqrt();
    EnergyResult { total, breakdown: ff.energy_breakdown(&pos), gradient: to_points(&g), grad_rms }
}

/// Snapshot of an energy batch: results computed so far (by input index).
#[derive(Debug, Clone, Default)]
pub struct EnergyBatchSnapshot {
    pub results: Vec<Option<f64>>,
}

impl Engine {
    /// Evaluates many structures in parallel. Each item is a molecule with
    /// optional replacement coordinates (e.g. conformers of one molecule).
    pub fn energy_batch(
        &self,
        items: Vec<super::StructureItem>,
        builder: SharedBuilder,
    ) -> JobHandle<Vec<Result<EnergyResult>>, EnergyBatchSnapshot> {
        let n = items.len();
        self.spawn(EnergyBatchSnapshot { results: vec![None; n] }, n, move |ctx| {
            let out: Vec<Result<EnergyResult>> = items
                .par_iter()
                .enumerate()
                .map(|(i, (mol, coords))| {
                    if ctx.is_cancelled() {
                        return Err(crate::error::Error::Cancelled);
                    }
                    let r = energy_at(mol, coords.as_deref().unwrap_or(&mol.coords), builder.as_ref());
                    if let Ok(e) = &r {
                        let t = e.total;
                        ctx.update_silent(|s| s.results[i] = Some(t));
                    }
                    ctx.inc_progress();
                    r
                })
                .collect();
            Ok(out)
        })
    }

    /// Asynchronous single-point energy.
    pub fn energy(&self, mol: Arc<Molecule>, builder: SharedBuilder) -> JobHandle<EnergyResult, ()> {
        self.spawn((), 1, move |_ctx| energy(&mol, builder.as_ref()))
    }
}
