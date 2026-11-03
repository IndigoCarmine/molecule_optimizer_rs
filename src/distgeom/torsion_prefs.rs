//! Experimental torsion-angle preferences ("ET") and basic-knowledge terms
//! ("K") of ETKDG.
//!
//! Port of RDKit `Code/GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.{h,cpp}`
//! (cosine-series parameters, ETversion 1 and 2, small-ring and macrocycle
//! tables). The SMARTS tables in `data/torsion_prefs_*.txt` are extracted
//! verbatim from RDKit `torsionData/*.in` by `tools/extract_torsion_data.py`.
//! RDKit: Copyright (C) 2017-2026 Sereina Riniker and other RDKit
//! contributors, BSD-3-Clause.

use super::bounds_builder::{Path14, TorsionType};
use crate::core::{Hybridization, Molecule};
use crate::error::{Error, Result};
use crate::perception::rings::RingInfo;
use crate::smarts::Smarts;
use std::sync::OnceLock;

const MIN_MACROCYCLE_SIZE: usize = 9;

/// Raw table text: ETKDG version 1 torsion preferences.
pub const TORSION_PREFS_V1: &str = include_str!("data/torsion_prefs_v1.txt");
/// Raw table text: ETKDG version 2 torsion preferences.
pub const TORSION_PREFS_V2: &str = include_str!("data/torsion_prefs_v2.txt");
/// Raw table text: small-ring torsions (srETKDG).
pub const TORSION_PREFS_SMALLRINGS: &str = include_str!("data/torsion_prefs_smallrings.txt");
/// Raw table text: macrocycle torsions (ETKDGv3).
pub const TORSION_PREFS_MACROCYCLES: &str = include_str!("data/torsion_prefs_macrocycles.txt");

/// One torsion-preference pattern (RDKit `ExpTorsionAngle`).
#[derive(Debug)]
pub struct ExpTorsionAngle {
    pub smarts: String,
    pub v: [f64; 6],
    pub signs: [i32; 6],
    pub pattern: Smarts,
    /// query atom indices of map numbers 1..4
    pub idx: [usize; 4],
}

/// Parses a torsion-preference table (RDKit `ExpTorsionAngleCollection`).
pub fn parse_table(text: &str) -> Result<Vec<ExpTorsionAngle>> {
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tok: Vec<&str> = line.split(' ').filter(|t| !t.is_empty()).collect();
        if tok.len() < 13 {
            return Err(Error::parse(ln + 1, format!("bad torsion preference line: {line}")));
        }
        let mut v = [0.0; 6];
        let mut signs = [0; 6];
        for k in 0..6 {
            signs[k] = tok[1 + 2 * k].parse().map_err(|_| Error::parse(ln + 1, "bad sign"))?;
            v[k] = tok[2 + 2 * k].parse().map_err(|_| Error::parse(ln + 1, "bad V"))?;
        }
        let pattern = Smarts::parse(tok[0])?;
        let mut idx = [0usize; 4];
        for (k, slot) in idx.iter_mut().enumerate() {
            *slot = pattern
                .atom_with_map_number(k as u32 + 1)
                .ok_or_else(|| Error::parse(ln + 1, format!("missing map number {} in {}", k + 1, tok[0])))?;
        }
        out.push(ExpTorsionAngle { smarts: tok[0].to_string(), v, signs, pattern, idx });
    }
    Ok(out)
}

fn cached(cell: &'static OnceLock<Vec<ExpTorsionAngle>>, text: &str) -> &'static [ExpTorsionAngle] {
    cell.get_or_init(|| parse_table(text).expect("built-in torsion table must parse"))
}

/// The built-in parameter list for the given options (RDKit `getParams`).
pub fn builtin_params(version: u32, small_rings: bool, macrocycles: bool) -> Result<Vec<&'static ExpTorsionAngle>> {
    static V1: OnceLock<Vec<ExpTorsionAngle>> = OnceLock::new();
    static V2: OnceLock<Vec<ExpTorsionAngle>> = OnceLock::new();
    static SR: OnceLock<Vec<ExpTorsionAngle>> = OnceLock::new();
    static MC: OnceLock<Vec<ExpTorsionAngle>> = OnceLock::new();
    let mut out: Vec<&'static ExpTorsionAngle> = match version {
        1 => cached(&V1, TORSION_PREFS_V1).iter().collect(),
        2 => cached(&V2, TORSION_PREFS_V2).iter().collect(),
        _ => return Err(Error::InvalidArgument("ETversion must be 1 or 2".into())),
    };
    if small_rings {
        out.extend(cached(&SR, TORSION_PREFS_SMALLRINGS).iter());
    }
    if macrocycles {
        out.extend(cached(&MC, TORSION_PREFS_MACROCYCLES).iter());
    }
    Ok(out)
}

/// Force constants (RDKit `ETKDGForceConsts::SEQ::Cosine`).
#[derive(Debug, Clone, Copy)]
pub struct ForceConsts {
    pub distance: f64,
    pub fourth_dim: f64,
    pub chiral: f64,
    pub k_term_angle: f64,
    pub k_term_improper: f64,
    pub k_term_torsion: f64,
    pub et_term_scaling: f64,
}

impl Default for ForceConsts {
    fn default() -> Self {
        ForceConsts {
            distance: 1.0,
            fourth_dim: 1.0,
            chiral: 1.0,
            k_term_angle: 1.0,
            k_term_improper: 10.0,
            k_term_torsion: 100.0,
            et_term_scaling: 1.0,
        }
    }
}

/// One torsion term chosen for a molecule.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpTorsion {
    pub atoms: [usize; 4],
    pub v: [f64; 6],
    pub signs: [i32; 6],
    /// SMARTS of the pattern that produced the term (empty for basic-knowledge terms).
    pub smarts: String,
}

/// RDKit `CrystalFFDetails` (cosine form).
#[derive(Debug, Clone, Default)]
pub struct EtkdgDetails {
    pub exp_torsions: Vec<ExpTorsion>,
    /// `[n0, center, n1, n2, atomicNum(center), isCBoundToSP2O]`
    pub improper_atoms: Vec<[usize; 6]>,
    pub bonds: Vec<(usize, usize)>,
    pub angles: Vec<[usize; 4]>,
    pub atom_nums: Vec<u8>,
    pub bounds_mat_force_scaling: f64,
    pub force_consts: ForceConsts,
    pub path14: Vec<Path14>,
}

/// Which terms to add.
#[derive(Debug, Clone, Copy)]
pub struct TorsionOptions {
    pub use_exp_torsions: bool,
    pub use_small_ring_torsions: bool,
    pub use_macrocycle_torsions: bool,
    pub use_basic_knowledge: bool,
    pub version: u32,
}

fn ring_info(mol: &Molecule) -> std::borrow::Cow<'_, RingInfo> {
    match mol.ring_info() {
        Some(r) => std::borrow::Cow::Borrowed(r),
        None => std::borrow::Cow::Owned(RingInfo::compute(mol)),
    }
}

/// RDKit `getExperimentalTorsions` (fills `exp_torsions` and
/// `improper_atoms`; `details.path14` must already be set).
pub fn get_experimental_torsions(mol: &Molecule, details: &mut EtkdgDetails, opts: &TorsionOptions) -> Result<()> {
    let nb = mol.num_bonds();
    let na = mol.num_atoms();
    if na == 0 {
        return Err(Error::InvalidArgument("molecule has no atoms".into()));
    }
    details.exp_torsions.clear();
    details.improper_atoms.clear();
    let ri = ring_info(mol);
    let bond_rings = &ri.bond_rings;
    let mut excluded = vec![false; nb];
    for (ii, ri_) in bond_rings.iter().enumerate() {
        let mut rs1 = vec![false; nb];
        for &b in ri_ {
            rs1[b] = true;
        }
        for rj in &bond_rings[ii + 1..] {
            if ri_.len() >= MIN_MACROCYCLE_SIZE && rj.len() >= MIN_MACROCYCLE_SIZE {
                continue;
            }
            let mut n_common = 0;
            for &b in rj {
                if rs1[b] {
                    n_common += 1;
                    if n_common > 1 {
                        break;
                    }
                }
            }
            if n_common > 1 {
                if ri_.len() < MIN_MACROCYCLE_SIZE {
                    for &b in ri_ {
                        excluded[b] = true;
                    }
                }
                if rj.len() < MIN_MACROCYCLE_SIZE {
                    for &b in rj {
                        excluded[b] = true;
                    }
                }
            }
        }
    }
    let mut done = vec![false; nb];
    if opts.use_basic_knowledge {
        for cfg in &details.path14 {
            if !cfg.value.is_forced || !matches!(cfg.value.kind, TorsionType::Trans | TorsionType::Cis) {
                continue;
            }
            let (i, j, k, l) = (cfg.aid1, cfg.aid2, cfg.aid3, cfg.aid4);
            let b = mol.bond_between(j, k).expect("bond");
            if excluded[b] || ri.num_bond_rings(b) > 3 {
                done[b] = true;
            }
            if done[b] {
                continue;
            }
            let is_cis = cfg.value.kind == TorsionType::Cis;
            let mut v = [0.0; 6];
            let mut s = [1; 6];
            v[0] = 75.0;
            if is_cis {
                s[0] = -1;
            }
            details.exp_torsions.push(ExpTorsion { atoms: [i, j, k, l], v, signs: s, smarts: String::new() });
        }
    }
    if opts.use_exp_torsions {
        let params = builtin_params(opts.version, opts.use_small_ring_torsions, opts.use_macrocycle_torsions)?;
        for p in params {
            for m in p.pattern.find_matches(mol, false) {
                let (a1, a2, a3, a4) = (m[p.idx[0]], m[p.idx[1]], m[p.idx[2]], m[p.idx[3]]);
                let b = mol
                    .bond_between(a2, a3)
                    .ok_or_else(|| Error::Embedding("bond between central atoms not found".into()))?;
                if excluded[b] || ri.num_bond_rings(b) > 3 {
                    done[b] = true;
                }
                if done[b] {
                    continue;
                }
                done[b] = true;
                let mut v = p.v;
                for x in v.iter_mut() {
                    *x *= details.force_consts.et_term_scaling;
                }
                details.exp_torsions.push(ExpTorsion { atoms: [a1, a2, a3, a4], v, signs: p.signs, smarts: p.smarts.clone() });
            }
        }
    }
    if opts.use_basic_knowledge {
        for aid2 in 0..na {
            let at = mol.atom(aid2);
            let an = at.atomic_num;
            if (an == 6 || an == 7 || an == 8) && at.hybridization == Hybridization::SP2 && mol.degree(aid2) == 3 {
                let mut atoms = [0usize; 4];
                atoms[1] = aid2;
                let mut i = 0;
                let mut bound_o = false;
                for x in mol.neighbors(aid2) {
                    atoms[i] = x;
                    if !bound_o {
                        bound_o = an == 6 && mol.atom(x).atomic_num == 8 && mol.atom(x).hybridization == Hybridization::SP2;
                    }
                    if i == 0 {
                        i += 1;
                    }
                    i += 1;
                }
                details.improper_atoms.push([atoms[0], atoms[1], atoms[2], atoms[3], an as usize, bound_o as usize]);
            }
        }
        for ring in &ri.atom_rings {
            let rs = ring.len();
            if !(4..=6).contains(&rs) {
                continue;
            }
            for i in 0..rs {
                let a1 = ring[i];
                let a2 = ring[(i + 1) % rs];
                let a3 = ring[(i + 2) % rs];
                let a4 = ring[(i + 3) % rs];
                let b = mol.bond_between(a2, a3).expect("ring bond");
                let sp2 = |a: usize| mol.atom(a).hybridization == Hybridization::SP2;
                if !done[b] && sp2(a1) && sp2(a2) && sp2(a3) && sp2(a4) {
                    done[b] = true;
                    let mut s = [1; 6];
                    s[1] = -1;
                    let mut v = [0.0; 6];
                    v[1] = details.force_consts.k_term_torsion;
                    details.exp_torsions.push(ExpTorsion { atoms: [a1, a2, a3, a4], v, signs: s, smarts: String::new() });
                }
            }
        }
    }
    Ok(())
}
