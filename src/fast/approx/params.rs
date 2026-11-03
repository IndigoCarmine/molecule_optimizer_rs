//! Per-atom and per-type-pair nonbonded parameters extracted from the
//! wrapped force-field builder (MMFF94 / UFF).
//!
//! * van der Waals parameters depend only on the atom types, so they are
//!   stored as `ntypes × ntypes` tables;
//! * topological relations (1-2, 1-3, 1-4) come from a depth-limited
//!   breadth-first search of the bond graph (shortest-path relation, as in
//!   the RDKit-compatible builders);
//! * MMFF: 1-2 and 1-3 pairs are excluded, 1-4 pairs interact with full
//!   van der Waals and electrostatics scaled by 0.75 (they are evaluated in
//!   an explicit pair list, not in the cluster kernels);
//! * UFF: 1-2 and 1-3 pairs are excluded, 1-4 pairs are ordinary pairs (UFF
//!   applies no 1-4 scaling); UFF has no electrostatics;
//! * `ignore_interfrag_interactions` removes all pairs between different
//!   connected components.

use crate::core::Molecule;
use crate::error::{Error, Result};
use crate::ff::mmff::{DielectricModel, MmffBuilder, MmffMolProperties};
use crate::ff::uff::UffBuilder;
use crate::ff::uff::atom_typer::atom_types_and_params;
use crate::ff::uff::nonbonded::{calc_nonbonded_depth, calc_nonbonded_minimum};
use std::collections::HashMap;

/// Coulomb constant used by MMFF94 (kcal Å / (mol e²)).
pub const MMFF_COULOMB: f64 = 332.0716;
/// MMFF buffering distance of the electrostatic term (Å).
pub const MMFF_ELE_BUFFER: f64 = 0.05;
/// MMFF 1-4 electrostatic scale factor.
pub const MMFF_SCALE_14: f64 = 0.75;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VdwKind {
    /// MMFF buffered 14-7: tables hold `R*_ij`, `eps_ij`, `R*_ij^7`.
    Mmff,
    /// UFF Lennard-Jones 12-6 `D (x/r)^12 - 2 D (x/r)^6`: tables hold
    /// `x_ij²`, `D_ij`.
    Uff,
}

/// Static nonbonded description of a system.
#[derive(Debug, Clone)]
pub(crate) struct NbParams {
    pub n: usize,
    pub kind: VdwKind,
    pub ntypes: usize,
    /// vdW type index of every atom.
    pub atom_type: Vec<u32>,
    /// `ntypes²` tables (row-major `ti * ntypes + tj`).
    pub p1: Vec<f64>,
    pub p2: Vec<f64>,
    pub p3: Vec<f64>,
    /// Partial charges (exact zeros for |q| < 1e-10, like RDKit).
    pub q: Vec<f64>,
    /// `true` if any charge is non-zero.
    pub has_charges: bool,
    /// Electrostatic prefactor `332.0716 / D`.
    pub coulomb: f64,
    /// Distance-dependent dielectric (`1/r²` instead of `1/r`).
    pub distance_dielectric: bool,
    /// Buffering distance of the electrostatic term (MMFF 0.05 Å).
    pub ele_buffer: f64,
    /// Fragment index of every atom.
    pub frag: Vec<u32>,
    pub num_frags: usize,
    pub ignore_interfrag: bool,
    /// Pairs not evaluated by the cluster kernels (sorted per atom,
    /// symmetric): 1-2, 1-3 and (MMFF) 1-4 pairs.
    pub masked: Vec<Vec<u32>>,
    /// MMFF 1-4 pairs `(i, j)`, `i < j`, evaluated explicitly.
    pub pairs14: Vec<(u32, u32)>,
    /// 1-2 / 1-3 pairs `(i, j)`, `i < j` (needed for Ewald exclusion
    /// corrections).
    pub excluded: Vec<(u32, u32)>,
    /// Electrostatic scale of 1-4 pairs.
    pub scale14: f64,
    /// Human-readable force-field name.
    pub ff_name: &'static str,
}

impl NbParams {
    #[inline]
    pub fn is_masked(&self, i: usize, j: usize) -> bool {
        if self.ignore_interfrag && self.frag[i] != self.frag[j] {
            return true;
        }
        self.masked[i].contains(&(j as u32))
    }
}

/// Shortest-path relations up to 1-4: returns per-atom lists of (partner,
/// distance in bonds) with distance 1, 2 or 3.
fn relations(mol: &Molecule) -> Vec<Vec<(u32, u8)>> {
    let n = mol.num_atoms();
    let mut stamp = vec![usize::MAX; n];
    let mut out = Vec::with_capacity(n);
    let mut cur: Vec<usize> = Vec::new();
    let mut next: Vec<usize> = Vec::new();
    for i in 0..n {
        let mut rel = Vec::new();
        stamp[i] = i;
        cur.clear();
        cur.push(i);
        for depth in 1..=3u8 {
            next.clear();
            for &u in &cur {
                for v in mol.neighbors(u) {
                    if stamp[v] != i {
                        stamp[v] = i;
                        next.push(v);
                        rel.push((v as u32, depth));
                    }
                }
            }
            std::mem::swap(&mut cur, &mut next);
        }
        rel.sort_unstable();
        out.push(rel);
    }
    out
}

fn fragments(mol: &Molecule) -> (Vec<u32>, usize) {
    let mut frag = vec![0u32; mol.num_atoms()];
    let fr = mol.fragments();
    for (k, f) in fr.iter().enumerate() {
        for &a in f {
            frag[a] = k as u32;
        }
    }
    (frag, fr.len())
}

fn exact_zero(q: f64) -> f64 {
    if q < 1.0e-10 && q > -1.0e-10 { 0.0 } else { q }
}

/// Extracts the parameters for `mol` from `inner_builder` (`MmffBuilder`
/// or `UffBuilder`).
pub(crate) fn extract(mol: &Molecule, inner_builder: &dyn std::any::Any) -> Result<NbParams> {
    if mol.num_atoms() == 0 {
        return Err(Error::ForceField("empty molecule".into()));
    }
    if mol.num_atoms() >= u32::MAX as usize / 2 {
        return Err(Error::ForceField("too many atoms".into()));
    }
    if let Some(b) = inner_builder.downcast_ref::<MmffBuilder>() {
        extract_mmff(mol, b)
    } else if let Some(b) = inner_builder.downcast_ref::<UffBuilder>() {
        extract_uff(mol, b)
    } else {
        Err(Error::ForceField("approximate nonbonded mode supports MmffBuilder and UffBuilder only".into()))
    }
}

type PairVec = Vec<(u32, u32)>;

/// Returns (masked partners per atom, 1-4 pairs, 1-2/1-3 pairs).
fn topology(mol: &Molecule, mask_14: bool) -> (Vec<Vec<u32>>, PairVec, PairVec) {
    let rel = relations(mol);
    let mut masked = Vec::with_capacity(rel.len());
    let mut p14 = Vec::new();
    let mut excl = Vec::new();
    for (i, r) in rel.iter().enumerate() {
        let mut m = Vec::new();
        for &(j, d) in r {
            if d <= 2 || mask_14 {
                m.push(j);
            }
            if (j as usize) > i {
                if d <= 2 {
                    excl.push((i as u32, j));
                } else {
                    p14.push((i as u32, j));
                }
            }
        }
        masked.push(m);
    }
    (masked, p14, excl)
}

fn extract_mmff(mol: &Molecule, b: &MmffBuilder) -> Result<NbParams> {
    let props = MmffMolProperties::new(mol, b.variant)?;
    if !props.is_valid() {
        return Err(Error::MissingParameters(format!("{}: missing MMFF atom types", b.variant.name())));
    }
    let n = mol.num_atoms();
    let types = props.atom_types();
    let mut index: HashMap<u8, u32> = HashMap::new();
    let mut rep: Vec<usize> = Vec::new();
    let mut atom_type = Vec::with_capacity(n);
    for (i, &t) in types.iter().enumerate() {
        let k = *index.entry(t).or_insert_with(|| {
            rep.push(i);
            (rep.len() - 1) as u32
        });
        atom_type.push(k);
    }
    let nt = rep.len();
    let mut p1 = vec![0.0; nt * nt];
    let mut p2 = vec![0.0; nt * nt];
    let mut p3 = vec![0.0; nt * nt];
    for a in 0..nt {
        for c in 0..nt {
            let v = props
                .vdw_params(rep[a], rep[c])
                .ok_or_else(|| Error::MissingParameters("MMFF van der Waals parameters".into()))?;
            let r = v.r_ij_star;
            let r2 = r * r;
            p1[a * nt + c] = r;
            p2[a * nt + c] = v.epsilon;
            p3[a * nt + c] = r2 * r2 * r2 * r;
        }
    }
    let q: Vec<f64> = props.partial_charges().iter().map(|&x| exact_zero(x)).collect();
    let has_charges = q.iter().any(|&x| x != 0.0);
    let (frag, num_frags) = fragments(mol);
    let (masked, pairs14, excluded) = topology(mol, true);
    Ok(NbParams {
        n,
        kind: VdwKind::Mmff,
        ntypes: nt,
        atom_type,
        p1,
        p2,
        p3,
        q,
        has_charges,
        coulomb: MMFF_COULOMB / b.dielectric_constant,
        distance_dielectric: b.dielectric_model == DielectricModel::Distance,
        ele_buffer: MMFF_ELE_BUFFER,
        frag,
        num_frags,
        ignore_interfrag: b.ignore_interfrag_interactions,
        masked,
        pairs14,
        excluded,
        scale14: MMFF_SCALE_14,
        ff_name: b.variant.name(),
    })
}

fn extract_uff(mol: &Molecule, b: &UffBuilder) -> Result<NbParams> {
    let n = mol.num_atoms();
    let (labels, params, _) = atom_types_and_params(mol);
    // atoms without parameters get their own type with zero well depth
    let mut index: HashMap<String, u32> = HashMap::new();
    let mut reps = Vec::new();
    let mut atom_type = Vec::with_capacity(n);
    for i in 0..n {
        let key = if params[i].is_some() { labels[i].clone() } else { String::from("\u{0}none") };
        let k = *index.entry(key).or_insert_with(|| {
            reps.push(params[i]);
            (reps.len() - 1) as u32
        });
        atom_type.push(k);
    }
    let nt = reps.len();
    let mut p1 = vec![1.0; nt * nt];
    let mut p2 = vec![0.0; nt * nt];
    for a in 0..nt {
        for c in 0..nt {
            if let (Some(pa), Some(pc)) = (reps[a], reps[c]) {
                let x = calc_nonbonded_minimum(pa, pc);
                p1[a * nt + c] = x * x;
                p2[a * nt + c] = calc_nonbonded_depth(pa, pc);
            }
        }
    }
    let (frag, num_frags) = fragments(mol);
    let (masked, _p14, excluded) = topology(mol, false);
    Ok(NbParams {
        n,
        kind: VdwKind::Uff,
        ntypes: nt,
        atom_type,
        p3: vec![0.0; nt * nt],
        p1,
        p2,
        q: vec![0.0; n],
        has_charges: false,
        coulomb: 0.0,
        distance_dielectric: false,
        ele_buffer: 0.0,
        frag,
        num_frags,
        ignore_interfrag: b.ignore_interfrag_interactions,
        masked,
        pairs14: Vec::new(),
        excluded,
        scale14: 1.0,
        ff_name: "UFF",
    })
}
