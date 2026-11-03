//! # molecule_optimizer_rs
//!
//! Pure-Rust molecular mechanics library.
//!
//! * energy evaluation, geometry optimization, conformer search and
//!   coordinate scans ([`calc`])
//! * every calculation runs asynchronously on a dedicated rayon pool and can
//!   be observed while it runs: [`tasks::JobHandle::snapshot`] returns the
//!   current coordinates at any time ([`tasks`])
//! * force fields are pluggable through [`ff::ForceFieldBuilder`]; UFF and
//!   MMFF94 are provided as ports of the RDKit implementations
//!   ([`ff::uff`], [`ff::mmff`])
//! * initial conformers are generated with a port of RDKit's ETKDG
//!   ([`distgeom`])
//!
//! Large parts of this crate are ports of the RDKit
//! (<https://github.com/rdkit/rdkit>), which is distributed under the
//! BSD 3-Clause license.  See `LICENSE` and `LICENSE-RDKIT.txt`.

// The ported numerical / graph code deliberately mirrors RDKit's C++ control
// flow (index loops, nested conditions) to keep the port reviewable against
// the original and floating-point operation order identical.
#![allow(clippy::needless_range_loop, clippy::collapsible_if, clippy::collapsible_match)]

pub mod calc;
pub mod core;
pub mod depict;
pub mod distgeom;
pub mod error;
pub mod fast;
pub mod ff;
pub mod io;
pub mod optim;
pub mod perception;
pub mod smarts;
pub mod tasks;

pub use crate::core::{Atom, Bond, BondOrder, Molecule};
pub use crate::error::{Error, Result};
