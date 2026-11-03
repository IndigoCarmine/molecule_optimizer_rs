//! Distance geometry and ETKDG conformer embedding – a port of RDKit
//! `Code/DistGeom`, `Code/GraphMol/DistGeomHelpers` and
//! `Code/GraphMol/ForceFieldHelpers/CrystalFF` (RDKit 2026.09.1,
//! BSD-3-Clause, Copyright (C) 2004-2026 Greg Landrum, Sereina Riniker and
//! other RDKit contributors).
//!
//! The random-number stream (boost `minstd_rand` + `uniform_real`), the
//! power eigen solver, the force-field terms and the BFGS minimizer follow
//! RDKit operation by operation, so for the same input molecule (same atom
//! and bond order, same stereo) and the same `random_seed` the embedded
//! coordinates reproduce RDKit's `EmbedMolecule` / `EmbedMultipleConfs`.
//!
//! ```no_run
//! use molecule_optimizer_rs::{io, distgeom::{self, EmbedParams}, optim::NoObserver};
//! let mol = io::read_file("mol.sdf").unwrap().remove(0);
//! let mut params = EmbedParams::etkdg_v3();
//! params.random_seed = 42;
//! let coords = distgeom::embed_molecule(&mol, &params, &mut NoObserver).unwrap();
//! ```
//!
//! Not ported: coordinate maps (`coordMap`), user bounds matrices, CPCI
//! electrostatics, the "all-in-one" ETKDGv4 minimization, MMFF-based 1-2/1-3
//! bounds and the internal-coordinate initial embedding.

pub mod bounds;
pub mod bounds_builder;
pub mod contribs;
pub mod dg_utils;
pub mod embedder;
pub mod force_fields;
pub mod generator;
pub mod point;
pub mod rng;
pub mod torsion_prefs;
pub mod uff_params;

pub use bounds::BoundsMatrix;
pub use embedder::{FragmentSetup, StopAfter, conformer_seed};
pub use generator::EtkdgGenerator;

use crate::core::Molecule;
use crate::error::{Error, Result};
use crate::optim::{NoObserver, Observer};

/// Embedding parameters (RDKit `EmbedParameters`). Use the constructors
/// [`EmbedParams::etkdg_v3`] (default), [`EmbedParams::etkdg`],
/// [`EmbedParams::kdg`], [`EmbedParams::etdg`], [`EmbedParams::dg`].
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedParams {
    /// Max. attempts per conformer; 0 = 10 × number of atoms.
    pub max_iterations: usize,
    /// Ignored (RDKit thread count); callers parallelize via
    /// [`Embedding::embed_conformer`].
    pub num_threads: i32,
    /// `-1` = unseeded.
    pub random_seed: i32,
    pub use_random_coords: bool,
    pub box_size_mult: f64,
    pub rand_neg_eig: bool,
    pub num_zero_fail: u32,
    pub optimizer_force_tol: f64,
    pub ignore_smoothing_failures: bool,
    pub enforce_chirality: bool,
    pub use_exp_torsion_angle_prefs: bool,
    pub use_basic_knowledge: bool,
    pub basin_thresh: f64,
    /// RMS pruning threshold (heavy atoms, symmetry aware); `<= 0` = off.
    pub prune_rms_thresh: f64,
    pub only_heavy_atoms_for_rms: bool,
    pub use_symmetry_for_pruning: bool,
    /// ETKDG torsion table version (1 or 2).
    pub et_version: u32,
    pub embed_fragments_separately: bool,
    pub use_small_ring_torsions: bool,
    pub use_macrocycle_torsions: bool,
    pub use_macrocycle14config: bool,
    pub force_trans_amides: bool,
    pub bounds_mat_force_scaling: f64,
    pub enable_sequential_random_seeds: bool,
    /// Assign stereo from the input 3D coordinates with RDKit's (legacy)
    /// CIP-based perception before embedding, so that exactly the
    /// stereocenters RDKit would see are enforced. When `false` the chiral
    /// tags / bond stereo on the input molecule are used as they are.
    pub rdkit_stereo_from_3d: bool,
}

impl Default for EmbedParams {
    fn default() -> Self {
        Self::etkdg_v3()
    }
}

impl EmbedParams {
    /// Plain distance geometry (RDKit `DG`).
    pub fn dg() -> Self {
        EmbedParams {
            max_iterations: 0,
            num_threads: 1,
            random_seed: -1,
            use_random_coords: false,
            box_size_mult: 2.0,
            rand_neg_eig: true,
            num_zero_fail: 1,
            optimizer_force_tol: 1e-3,
            ignore_smoothing_failures: false,
            enforce_chirality: true,
            use_exp_torsion_angle_prefs: false,
            use_basic_knowledge: false,
            basin_thresh: 5.0,
            prune_rms_thresh: -1.0,
            only_heavy_atoms_for_rms: true,
            use_symmetry_for_pruning: true,
            et_version: 2,
            embed_fragments_separately: true,
            use_small_ring_torsions: false,
            use_macrocycle_torsions: false,
            use_macrocycle14config: false,
            force_trans_amides: true,
            bounds_mat_force_scaling: 1.0,
            enable_sequential_random_seeds: false,
            rdkit_stereo_from_3d: true,
        }
    }
    /// RDKit `ETKDGv3` (default).
    pub fn etkdg_v3() -> Self {
        EmbedParams {
            use_exp_torsion_angle_prefs: true,
            use_basic_knowledge: true,
            et_version: 2,
            use_macrocycle_torsions: true,
            use_macrocycle14config: true,
            ..Self::dg()
        }
    }
    /// RDKit `srETKDGv3`.
    pub fn sr_etkdg_v3() -> Self {
        EmbedParams { use_small_ring_torsions: true, use_macrocycle_torsions: false, use_macrocycle14config: false, ..Self::etkdg_v3() }
    }
    /// RDKit `ETKDGv2`.
    pub fn etkdg_v2() -> Self {
        EmbedParams { use_macrocycle_torsions: false, use_macrocycle14config: false, ..Self::etkdg_v3() }
    }
    /// RDKit `ETKDG` (version 1 torsions).
    pub fn etkdg() -> Self {
        EmbedParams { et_version: 1, ..Self::etkdg_v2() }
    }
    /// RDKit `KDG`.
    pub fn kdg() -> Self {
        EmbedParams { use_exp_torsion_angle_prefs: false, use_basic_knowledge: true, et_version: 1, ..Self::dg() }
    }
    /// RDKit `ETDG`.
    pub fn etdg() -> Self {
        EmbedParams { use_exp_torsion_angle_prefs: true, use_basic_knowledge: false, et_version: 1, ..Self::dg() }
    }

    fn validate(&self) -> Result<()> {
        if !(self.et_version == 1 || self.et_version == 2) {
            return Err(Error::InvalidArgument("only ETversion 1 and 2 are supported".into()));
        }
        Ok(())
    }
}

/// Everything precomputed for a molecule (bounds, torsion terms, chiral
/// sets) – build once with [`Embedding::new`], then embed any number of
/// conformers, possibly in parallel (`Embedding` is `Send + Sync`).
#[derive(Debug, Clone)]
pub struct Embedding {
    pub params: EmbedParams,
    pub num_atoms: usize,
    pub fragments: Vec<FragmentSetup>,
    /// The molecule with the stereo actually enforced.
    pub mol: Molecule,
}

/// Prepares the molecule for embedding: stereo, fragments.
fn prepared_molecule(mol: &Molecule, p: &EmbedParams) -> Molecule {
    let mut m = mol.clone();
    if m.ring_info().is_none() {
        m.rings();
    }
    if p.rdkit_stereo_from_3d && crate::perception::has_3d_coords(&m) {
        crate::perception::legacy_stereo::assign_stereo_legacy_from_3d(&mut m);
    }
    m
}

impl Embedding {
    pub fn new(mol: &Molecule, params: &EmbedParams) -> Result<Embedding> {
        params.validate()?;
        if mol.num_atoms() == 0 {
            return Err(Error::InvalidArgument("molecule has no atoms".into()));
        }
        let m = prepared_molecule(mol, params);
        let mut fragments = Vec::new();
        for (atoms, fm) in embedder::split_fragments(&m, params.embed_fragments_separately) {
            fragments.push(embedder::setup_fragment(atoms, fm, params)?);
        }
        Ok(Embedding { params: params.clone(), num_atoms: mol.num_atoms(), fragments, mol: m })
    }

    /// Embeds conformer number `conf_idx` exactly as RDKit's
    /// `EmbedMultipleConfs` would (seed derived from `random_seed` and
    /// `conf_idx`). Returns `None` if embedding failed.
    pub fn embed_conformer(&self, conf_idx: usize, observer: &mut dyn Observer) -> Result<Option<Vec<[f64; 3]>>> {
        let seed = conformer_seed(self.params.random_seed, conf_idx, self.params.enable_sequential_random_seeds);
        self.embed_with_seed(seed, StopAfter::default(), observer)
    }

    /// Embeds with an explicit per-conformer seed.
    pub fn embed_with_seed(&self, seed: i32, stop: StopAfter, observer: &mut dyn Observer) -> Result<Option<Vec<[f64; 3]>>> {
        let mut out = vec![[0.0; 3]; self.num_atoms];
        for f in &self.fragments {
            match embedder::embed_points(f, &self.params, seed, stop, observer)? {
                Some((pos, dim)) => {
                    for (k, &a) in f.atoms.iter().enumerate() {
                        out[a] = [pos[k * dim], pos[k * dim + 1], pos[k * dim + 2]];
                    }
                }
                None => return Ok(None),
            }
        }
        Ok(Some(out))
    }

    /// Heavy-atom (symmetry-aware) atom maps used for RMS pruning.
    pub fn pruning_matches(&self) -> Vec<Vec<usize>> {
        pruning::self_matches(&self.mol, &self.params)
    }
}

/// Embeds a single conformer (RDKit `EmbedMolecule`). The observer sees
/// every minimizer iteration (positions may be 4D in the first stages) and
/// can cancel.
pub fn embed_molecule(mol: &Molecule, params: &EmbedParams, observer: &mut dyn Observer) -> Result<Vec<[f64; 3]>> {
    let e = Embedding::new(mol, params)?;
    e.embed_conformer(0, observer)?.ok_or_else(|| Error::Embedding("embedding failed".into()))
}

/// Embeds `n` conformers (RDKit `EmbedMultipleConfs` semantics: per-
/// conformer seeds, failed conformers dropped, then greedy RMS pruning).
pub fn embed_multiple(mol: &Molecule, n: usize, params: &EmbedParams) -> Result<Vec<Vec<[f64; 3]>>> {
    let e = Embedding::new(mol, params)?;
    let mut confs = Vec::new();
    for ci in 0..n {
        if let Some(c) = e.embed_conformer(ci, &mut NoObserver)? {
            confs.push(c);
        }
    }
    Ok(prune_conformers(&e, confs))
}

/// RDKit's greedy RMS pruning (`_isConfFarFromRest`) of conformers in order.
pub fn prune_conformers(e: &Embedding, confs: Vec<Vec<[f64; 3]>>) -> Vec<Vec<[f64; 3]>> {
    if e.params.prune_rms_thresh <= 0.0 {
        return confs;
    }
    let matches = e.pruning_matches();
    let mut kept: Vec<Vec<[f64; 3]>> = Vec::new();
    for c in confs {
        if pruning::is_far_from_rest(&c, &kept, e.params.prune_rms_thresh, &matches) {
            kept.push(c);
        }
    }
    kept
}

/// Bounds matrix as used by the embedder (after triangle smoothing) for a
/// single-fragment molecule (RDKit `setTopolBounds` + `triangleSmoothBounds`).
pub fn bounds_matrix(mol: &Molecule, params: &EmbedParams) -> Result<BoundsMatrix> {
    let m = prepared_molecule(mol, params);
    Ok(embedder::setup_bounds(&m, params)?.0)
}

/// Unsmoothed topological bounds (RDKit `GetMoleculeBoundsMatrix(...,
/// doTriangleSmoothing=False)`).
pub fn raw_bounds_matrix(mol: &Molecule, params: &EmbedParams) -> BoundsMatrix {
    let m = prepared_molecule(mol, params);
    let mut b = BoundsMatrix::new_initialized(m.num_atoms());
    bounds_builder::set_topol_bounds(
        &m,
        &mut b,
        &bounds_builder::TopolOptions {
            use_macrocycle14config: params.use_macrocycle14config,
            force_trans_amides: params.force_trans_amides,
            ..Default::default()
        },
    );
    b
}

pub mod pruning;
