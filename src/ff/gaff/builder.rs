//! Parameter assignment and construction of the GAFF force field.

use super::estimate;
use super::params::{AngleParam, BondParam, GaffParams, GaffVariant, TorsionTerm, VdwParam, WILD};
use super::terms::{CoulombContrib, FourierTorsionContrib, HarmonicAngleContrib};
use super::typer::{self, basic_type, paired_type};
use super::GaffBuilder;
use crate::core::Molecule;
use crate::core::geometry::{angle, dist};
use crate::error::{Error, Result};
use crate::ff::assign::normalized;
use crate::ff::uff::{BondStretchContrib, VdwContrib};
use crate::ff::utils::BondDistances;
use crate::ff::{Contrib, TermList};

/// Atom types and the parameter coverage of one molecule.
#[derive(Debug, Clone, PartialEq)]
pub struct GaffAssignment {
    pub variant: GaffVariant,
    /// GAFF atom type per atom; empty = no type.
    pub types: Vec<String>,
    /// Atoms without a GAFF type or without van der Waals parameters. They
    /// are left out of every term.
    pub untyped: Vec<usize>,
    /// Bonds whose force constant comes from eq. (3) of the paper, with the
    /// bond length of the input structure as equilibrium length.
    pub estimated_bonds: Vec<[usize; 2]>,
    /// Angles whose parameters come from the empirical rules of the paper
    /// (equilibrium angle from the `A-B-A` / `C-B-C` entries or from the
    /// input structure, force constant from eq. 5).
    pub estimated_angles: Vec<[usize; 3]>,
    /// Torsions taken from the entry of the equivalent paired types or of
    /// the basic types of the central atoms.
    pub fallback_torsions: Vec<[usize; 4]>,
    /// Bonds / angles / torsions without parameters (no term is created).
    pub missing_bonds: Vec<[usize; 2]>,
    pub missing_angles: Vec<[usize; 3]>,
    pub missing_torsions: Vec<[usize; 4]>,
    /// Human-readable notes: typing problems, zero charges ...
    pub warnings: Vec<String>,
}

impl GaffAssignment {
    /// `true` if every atom is typed and every bond, angle and torsion has
    /// parameters (tabulated, estimated or from a fallback entry).
    pub fn is_complete(&self) -> bool {
        self.untyped.is_empty() && self.missing_bonds.is_empty() && self.missing_angles.is_empty() && self.missing_torsions.is_empty()
    }
}

/// Parameters of every term of one molecule.
pub(crate) struct Terms {
    pub bonds: Vec<([usize; 2], BondParam)>,
    pub angles: Vec<([usize; 3], AngleParam)>,
    pub torsions: Vec<([usize; 4], TorsionTerm)>,
    pub impropers: Vec<([usize; 4], TorsionTerm)>,
    pub vdw: Vec<Option<VdwParam>>,
}

/// Torsion terms for the types `key`: the entry with all four types, the
/// generic `X-j-k-X` entry, then (flagged as fallback) the same two for the
/// equivalent paired types (Table 5: "X-c-ne-X or X-c-nf-X") and the
/// generic entry of the basic types of the central atoms.
fn torsion_lookup(p: &GaffParams, key: [u16; 4]) -> Option<(&[TorsionTerm], bool)> {
    let generic = |j: u16, k: u16| p.torsion([WILD, j, k, WILD]);
    if let Some(t) = p.torsion(key).or_else(|| generic(key[1], key[2])) {
        return Some((t, false));
    }
    let map = |t: u16, f: fn(&str) -> Option<&'static str>| f(p.name(t)).and_then(|n| p.id(n));
    let swapped = key.map(|t| map(t, paired_type).unwrap_or(t));
    if swapped != key {
        if let Some(t) = p.torsion(swapped).or_else(|| generic(swapped[1], swapped[2])) {
            return Some((t, true));
        }
    }
    let (bj, bk) = (map(key[1], basic_type), map(key[2], basic_type));
    for (j, k) in [(bj, Some(key[2])), (Some(key[1]), bk), (bj, bk)] {
        if let (Some(j), Some(k)) = (j, k) {
            if let Some(t) = generic(j, k) {
                return Some((t, true));
            }
        }
    }
    None
}

/// The improper entry for the centre `c` with the neighbours `nb`: the
/// matching entry with the fewest wildcards (the first one in the file
/// among equals). Atoms are placed on the typed positions first; wildcard
/// positions take the remaining atoms in index order.
fn improper_lookup(p: &GaffParams, types: &[Option<u16>], c: usize, nb: [usize; 3]) -> Option<([usize; 4], TorsionTerm)> {
    let tc = types[c]?;
    let mut nb = nb;
    nb.sort_unstable();
    const PERMS: [[usize; 3]; 6] = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    let mut best: Option<(usize, [usize; 4], TorsionTerm)> = None;
    for &(key, term) in p.impropers() {
        if key[2] != tc {
            continue;
        }
        let slots = [key[0], key[1], key[3]];
        let score = slots.iter().filter(|&&s| s != WILD).count();
        if best.as_ref().is_some_and(|b| b.0 >= score) {
            continue;
        }
        let hit = PERMS.iter().find(|perm| (0..3).all(|s| slots[s] == WILD || Some(slots[s]) == types[nb[perm[s]]]));
        if let Some(perm) = hit {
            best = Some((score, [nb[perm[0]], nb[perm[1]], c, nb[perm[2]]], term));
        }
    }
    best.map(|(_, idx, term)| (idx, term))
}

/// Assigns atom types and the parameters of every term of `mol`.
pub(crate) fn parameterize(mol: &Molecule, variant: GaffVariant) -> Result<(GaffAssignment, Terms)> {
    let p = GaffParams::get(variant);
    let norm = normalized(mol)?;
    let typing = typer::assign_types(&norm, p)?;
    let n = mol.num_atoms();
    let has_coords = mol.coords.len() == n;
    let ids: Vec<Option<u16>> = typing.types.iter().map(|t| if t.is_empty() { None } else { p.id(t) }).collect();
    let vdw: Vec<Option<VdwParam>> = ids.iter().map(|t| t.and_then(|t| p.vdw(t))).collect();
    let z = |i: usize| mol.atom(i).atomic_num;

    let mut a = GaffAssignment {
        variant,
        types: typing.types.iter().map(|t| t.to_string()).collect(),
        untyped: (0..n).filter(|&i| vdw[i].is_none()).collect(),
        estimated_bonds: Vec::new(),
        estimated_angles: Vec::new(),
        fallback_torsions: Vec::new(),
        missing_bonds: Vec::new(),
        missing_angles: Vec::new(),
        missing_torsions: Vec::new(),
        warnings: typing.warnings,
    };
    let mut terms = Terms { bonds: Vec::new(), angles: Vec::new(), torsions: Vec::new(), impropers: Vec::new(), vdw };

    // bonds
    let mut bond_r0: Vec<Option<f64>> = vec![None; mol.num_bonds()];
    for (b, bond) in mol.bonds().iter().enumerate() {
        let (i, j) = (bond.begin, bond.end);
        let (Some(ti), Some(tj)) = (ids[i], ids[j]) else { continue };
        let param = p.bond(ti, tj).or_else(|| {
            let r = if has_coords { dist(mol.coords[i], mol.coords[j]) } else { return None };
            let k = estimate::bond_force_constant(z(i), z(j), r)?;
            a.estimated_bonds.push([i, j]);
            Some(BondParam { k, r0: r })
        });
        match param {
            Some(bp) => {
                bond_r0[b] = Some(bp.r0);
                terms.bonds.push(([i, j], bp));
            }
            None => a.missing_bonds.push([i, j]),
        }
    }

    // angles
    for j in 0..n {
        let Some(tj) = ids[j] else { continue };
        let nbrs = mol.atom_bonds(j);
        for (x, &(i, bi)) in nbrs.iter().enumerate() {
            for &(k, bk) in &nbrs[x + 1..] {
                let (Some(ti), Some(tk)) = (ids[i], ids[k]) else { continue };
                let param = p.angle(ti, tj, tk).or_else(|| {
                    let theta0 = match (p.angle(ti, tj, ti), p.angle(tk, tj, tk)) {
                        (Some(a1), Some(a2)) => 0.5 * (a1.theta0 + a2.theta0),
                        _ if has_coords => angle(mol.coords[i], mol.coords[j], mol.coords[k]).to_degrees(),
                        _ => return None,
                    };
                    let k_theta = estimate::angle_force_constant(z(i), z(j), z(k), bond_r0[bi]?, bond_r0[bk]?, theta0, p.angle_scale)?;
                    a.estimated_angles.push([i, j, k]);
                    Some(AngleParam { k: k_theta, theta0 })
                });
                match param {
                    Some(ap) => terms.angles.push(([i, j, k], ap)),
                    None => a.missing_angles.push([i, j, k]),
                }
            }
        }
    }

    // torsions about every bond
    for bond in mol.bonds() {
        let (j, k) = (bond.begin, bond.end);
        let (Some(tj), Some(tk)) = (ids[j], ids[k]) else { continue };
        for &(i, _) in mol.atom_bonds(j) {
            if i == k {
                continue;
            }
            for &(l, _) in mol.atom_bonds(k) {
                if l == j || l == i {
                    continue;
                }
                let (Some(ti), Some(tl)) = (ids[i], ids[l]) else { continue };
                match torsion_lookup(p, [ti, tj, tk, tl]) {
                    Some((list, fallback)) => {
                        if fallback {
                            a.fallback_torsions.push([i, j, k, l]);
                        }
                        terms.torsions.extend(list.iter().filter(|t| t.k != 0.0).map(|&t| ([i, j, k, l], t)));
                    }
                    None => a.missing_torsions.push([i, j, k, l]),
                }
            }
        }
    }

    // impropers of three-coordinate atoms
    for c in 0..n {
        if mol.degree(c) != 3 {
            continue;
        }
        let mut nb = [0usize; 3];
        for (slot, x) in nb.iter_mut().zip(mol.neighbors(c)) {
            *slot = x;
        }
        if let Some((idx, term)) = improper_lookup(p, &ids, c, nb) {
            if term.k != 0.0 {
                terms.impropers.push((idx, term));
            }
        }
    }
    Ok((a, terms))
}

/// `cos(phase)` of a torsion term; GAFF only uses 0° and 180°.
fn cos_phase(t: &TorsionTerm) -> Result<f64> {
    if t.phase.abs() < 1e-6 {
        Ok(1.0)
    } else if (t.phase.abs() - 180.0).abs() < 1e-6 {
        Ok(-1.0)
    } else {
        Err(Error::ForceField(format!("GAFF: torsion phase {} is not 0 or 180 degrees", t.phase)))
    }
}

/// Builds the contribs of `mol`: bonds, angles, torsions, impropers, van
/// der Waals, electrostatics (empty kinds are left out).
pub(crate) fn build_contribs(mol: &Molecule, terms: &Terms, opts: &GaffBuilder) -> Result<Vec<Box<dyn Contrib>>> {
    let n = mol.num_atoms();
    let bonds: Vec<BondStretchContrib> = terms
        .bonds
        .iter()
        .map(|&([i, j], bp)| BondStretchContrib { idx1: i, idx2: j, rest_len: bp.r0, force_constant: 2.0 * bp.k })
        .collect();
    let angles: Vec<HarmonicAngleContrib> = terms
        .angles
        .iter()
        .map(|&(idx, ap)| HarmonicAngleContrib { idx, force_constant: ap.k, theta0: ap.theta0.to_radians() })
        .collect();
    let fourier = |list: &[([usize; 4], TorsionTerm)], improper: bool| -> Result<Vec<FourierTorsionContrib>> {
        list.iter()
            .map(|(idx, t)| {
                Ok(FourierTorsionContrib { idx: *idx, force_constant: t.k, periodicity: t.n, cos_phase: cos_phase(t)?, improper })
            })
            .collect()
    };
    let torsions = fourier(&terms.torsions, false)?;
    let impropers = fourier(&terms.impropers, true)?;

    // non-bonded pairs: 1-2 and 1-3 pairs excluded, 1-4 pairs scaled
    let mut frag = vec![0usize; n];
    if opts.ignore_interfrag_interactions {
        for (f, members) in mol.fragments().iter().enumerate() {
            for &x in members {
                frag[x] = f;
            }
        }
    }
    let q = opts.charges.as_deref();
    let mut vdw: Vec<VdwContrib> = Vec::new();
    let mut ele: Vec<CoulombContrib> = Vec::new();
    let mut rel = BondDistances::new(n, 3);
    for i in 0..n {
        rel.compute(mol, i);
        for j in i + 1..n {
            if opts.ignore_interfrag_interactions && frag[i] != frag[j] {
                continue;
            }
            let (s_vdw, s_ele) = match rel.get(j) {
                1 | 2 => continue,
                3 => (opts.scale14_vdw, opts.scale14_ele),
                _ => (1.0, 1.0),
            };
            if let (Some(vi), Some(vj)) = (terms.vdw[i], terms.vdw[j]) {
                let well_depth = (vi.eps * vj.eps).sqrt() * s_vdw;
                if well_depth != 0.0 {
                    vdw.push(VdwContrib { idx1: i, idx2: j, xij: vi.r_star + vj.r_star, well_depth, thresh: f64::INFINITY });
                }
            }
            if let Some(q) = q {
                let qq = opts.coulomb_constant * q[i] * q[j] * s_ele / opts.dielectric_constant;
                if qq != 0.0 {
                    ele.push(CoulombContrib { idx1: i, idx2: j, qq });
                }
            }
        }
    }
    Ok([
        TermList::boxed(bonds),
        TermList::boxed(angles),
        TermList::boxed(torsions),
        TermList::boxed(impropers),
        TermList::boxed(vdw),
        TermList::boxed(ele),
    ]
    .into_iter()
    .flatten()
    .collect())
}
