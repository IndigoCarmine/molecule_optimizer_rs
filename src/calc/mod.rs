//! High-level calculations: energy, optimization, conformer search and
//! coordinate scans. Every calculation has a synchronous function and an
//! asynchronous method on [`crate::tasks::Engine`] that returns a
//! [`crate::tasks::JobHandle`] whose snapshot holds the current coordinates.

pub mod conformer;
pub mod energy;
pub mod optimize;
pub mod scan;

use crate::ff::ForceFieldBuilder;
use std::sync::Arc;

/// Shareable force-field builder.
pub type SharedBuilder = Arc<dyn ForceFieldBuilder>;

/// Coordinates with their energy, as reported in snapshots.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Frame {
    pub coords: Vec<[f64; 3]>,
    pub energy: f64,
    pub iteration: usize,
}

pub(crate) fn to_points(flat: &[f64]) -> Vec<[f64; 3]> {
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

pub(crate) fn to_flat(pts: &[[f64; 3]]) -> Vec<f64> {
    pts.iter().flat_map(|p| p.iter().copied()).collect()
}

pub use conformer::{Conformer, ConformerSearchOptions, ConformerSearchResult, ConformerSearchSnapshot};
pub use energy::EnergyResult;
pub use optimize::{OptimizeOptions, OptimizeResult, OptimizeSnapshot};
pub use scan::{ScanCoordinate, ScanMode, ScanOptions, ScanPoint, ScanResult, ScanSnapshot};
