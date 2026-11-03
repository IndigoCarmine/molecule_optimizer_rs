//! Compilation of a built [`ContribForceField`] into structure-of-arrays
//! tables, and the per-molecule typing/parameter templates used by the
//! [`super::FastBuilder`] cache.
//!
//! Known contribs (every UFF and MMFF term type and the constraint contribs)
//! are recognized through [`Contrib::as_any`]; anything else is kept as an
//! opaque contrib that is evaluated as is. The order in which the term
//! categories first appear is recorded so that energy breakdowns report the
//! same labels in the same order as the wrapped force field.

use super::bonded::{Bonded, TableKind};
use super::exact::{NbTemplate, PairCollector};
use crate::core::{Hybridization, Molecule};
use crate::error::{Error, Result};
use crate::ff::constraints::{
    AngleConstraintContrib, DistanceConstraintContrib, PositionConstraintContrib, TorsionConstraintContrib,
};
use crate::ff::mmff::{self, MmffBuilder, MmffMolProperties};
use crate::ff::uff::{self, UffBuilder};
use crate::ff::{Contrib, ContribForceField};
use std::any::Any;
use std::sync::Arc;

/// One entry of the breakdown order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    Table(TableKind),
    Nonbonded,
    /// Index into the opaque contribs of the wrapped force field.
    Opaque(usize),
    Constraints,
}

/// One constraint term.
#[derive(Debug, Clone)]
pub enum ConstraintTerm {
    Distance(DistanceConstraintContrib),
    Angle(AngleConstraintContrib),
    Torsion(TorsionConstraintContrib),
    Position(PositionConstraintContrib),
}

impl ConstraintTerm {
    #[inline]
    fn as_contrib(&self) -> &dyn Contrib {
        match self {
            ConstraintTerm::Distance(c) => c,
            ConstraintTerm::Angle(c) => c,
            ConstraintTerm::Torsion(c) => c,
            ConstraintTerm::Position(c) => c,
        }
    }
}

/// Constraint contribs in their original order (few terms; evaluated with
/// their own scalar code).
#[derive(Debug, Clone, Default)]
pub struct ConstraintTable {
    pub terms: Vec<ConstraintTerm>,
}

impl ConstraintTable {
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
    /// Adds the energies to `acc` term by term (and the gradients if `G`).
    pub fn eval<const G: bool>(&self, pos: &[f64], grad: &mut [f64], acc: &mut f64) {
        for t in &self.terms {
            let c = t.as_contrib();
            *acc += if G { c.energy_grad(pos, grad) } else { c.energy(pos) };
        }
    }
    pub fn energy(&self, pos: &[f64]) -> f64 {
        let mut e = 0.0;
        self.eval::<false>(pos, &mut [], &mut e);
        e
    }
}

/// Result of [`compile`].
#[derive(Debug, Default)]
pub struct Compiled {
    pub bonded: Bonded,
    pub segments: Vec<Segment>,
    pub constraints: ConstraintTable,
    /// Indices of opaque contribs.
    pub opaque: Vec<usize>,
    /// Nonbonded pairs (empty if none or skipped).
    pub nb: PairCollector,
}

fn idx(i: usize) -> u32 {
    u32::try_from(i).expect("atom index exceeds u32")
}

/// Compiles `contribs`. With `skip_nonbonded` the UFF/MMFF nonbonded
/// contribs are dropped (their breakdown position is still recorded).
pub fn compile(contribs: &[Box<dyn Contrib>], skip_nonbonded: bool) -> Compiled {
    let mut out = Compiled::default();
    let mut seen_tables: Vec<TableKind> = Vec::new();
    let (mut seen_nb, mut seen_cons) = (false, false);
    let b = &mut out.bonded;
    let mut terms: Vec<Option<&dyn Any>> = Vec::new();
    for (ci, c) in contribs.iter().enumerate() {
        // a term list is compiled term by term (an opaque one is recorded once)
        terms.clear();
        c.visit_terms(&mut |a| terms.push(a));
        for &any in &terms {
            let mut table = |k: TableKind, segs: &mut Vec<Segment>| {
                if !seen_tables.contains(&k) {
                    seen_tables.push(k);
                    segs.push(Segment::Table(k));
                }
            };
            macro_rules! is {
                ($t:ty) => {
                    any.and_then(|a| a.downcast_ref::<$t>())
                };
            }
            if let Some(t) = is!(uff::BondStretchContrib) {
                let u = &mut b.uff_bonds;
                u.a.push(idx(t.idx1));
                u.b.push(idx(t.idx2));
                u.r0.push(t.rest_len);
                u.k.push(t.force_constant);
                table(TableKind::UffBond, &mut out.segments);
            } else if let Some(t) = is!(uff::AngleBendContrib) {
                let u = &mut b.uff_angles;
                u.a.push(idx(t.idx1));
                u.b.push(idx(t.idx2));
                u.c.push(idx(t.idx3));
                u.fc.push(t.force_constant);
                u.c0.push(t.c0);
                u.c1.push(t.c1);
                u.c2.push(t.c2);
                u.theta0.push(t.theta0);
                u.ord.push(t.order as f64);
                u.ord_sq.push((t.order * t.order) as f64);
                u.fc_ord.push(t.force_constant / t.order as f64);
                table(TableKind::UffAngle, &mut out.segments);
            } else if let Some(t) = is!(uff::TorsionAngleContrib) {
                let u = &mut b.uff_torsions;
                u.a.push(idx(t.idx[0]));
                u.b.push(idx(t.idx[1]));
                u.c.push(idx(t.idx[2]));
                u.d.push(idx(t.idx[3]));
                u.half_fc.push(t.force_constant / 2.0);
                u.cos_term.push(t.cos_term);
                u.ord.push(t.order as f64);
                u.dfac.push(-(t.force_constant / 2.0 * t.cos_term) * t.order as f64);
                table(TableKind::UffTorsion, &mut out.segments);
            } else if let Some(t) = is!(uff::InversionContrib) {
                let u = &mut b.uff_inversions;
                u.a.push(idx(t.idx[0]));
                u.b.push(idx(t.idx[1]));
                u.c.push(idx(t.idx[2]));
                u.d.push(idx(t.idx[3]));
                u.fc.push(t.force_constant);
                u.c0.push(t.c0);
                u.c1.push(t.c1);
                u.c2.push(t.c2);
                table(TableKind::UffInversion, &mut out.segments);
            } else if let Some(t) = is!(mmff::terms::BondStretchContrib) {
                let m = &mut b.mmff_bonds;
                for k in 0..t.len() {
                    m.a.push(t.at1[k]);
                    m.b.push(t.at2[k]);
                    m.r0.push(t.r0[k]);
                    m.kb.push(t.kb[k]);
                }
                table(TableKind::MmffBond, &mut out.segments);
            } else if let Some(t) = is!(mmff::terms::AngleBendContrib) {
                let m = &mut b.mmff_angles;
                for k in 0..t.len() {
                    m.a.push(t.at[k][0]);
                    m.b.push(t.at[k][1]);
                    m.c.push(t.at[k][2]);
                    m.theta0.push(t.theta0[k]);
                    m.ka.push(t.ka[k]);
                    m.lin.push(if t.is_linear[k] { 1.0 } else { 0.0 });
                }
                table(TableKind::MmffAngle, &mut out.segments);
            } else if let Some(t) = is!(mmff::terms::StretchBendContrib) {
                let m = &mut b.mmff_stbn;
                for k in 0..t.len() {
                    m.a.push(t.at[k][0]);
                    m.b.push(t.at[k][1]);
                    m.c.push(t.at[k][2]);
                    m.r1.push(t.rest_len1[k]);
                    m.r2.push(t.rest_len2[k]);
                    m.theta0.push(t.theta0[k]);
                    m.k1.push(t.k1[k]);
                    m.k2.push(t.k2[k]);
                }
                table(TableKind::MmffStretchBend, &mut out.segments);
            } else if let Some(t) = is!(mmff::terms::OopBendContrib) {
                let m = &mut b.mmff_oops;
                for k in 0..t.len() {
                    m.a.push(t.at[k][0]);
                    m.b.push(t.at[k][1]);
                    m.c.push(t.at[k][2]);
                    m.d.push(t.at[k][3]);
                    m.koop.push(t.koop[k]);
                }
                table(TableKind::MmffOop, &mut out.segments);
            } else if let Some(t) = is!(mmff::terms::TorsionAngleContrib) {
                let m = &mut b.mmff_torsions;
                for k in 0..t.len() {
                    m.a.push(t.at[k][0]);
                    m.b.push(t.at[k][1]);
                    m.c.push(t.at[k][2]);
                    m.d.push(t.at[k][3]);
                    m.v1.push(t.v1[k]);
                    m.v2.push(t.v2[k]);
                    m.v3.push(t.v3[k]);
                }
                table(TableKind::MmffTorsion, &mut out.segments);
            } else if let Some(t) = is!(uff::VdwContrib) {
                if skip_nonbonded || out.nb.add_uff(t) {
                    if !seen_nb {
                        seen_nb = true;
                        out.segments.push(Segment::Nonbonded);
                    }
                } else {
                    if out.opaque.last() != Some(&ci) {
                        out.opaque.push(ci);
                        out.segments.push(Segment::Opaque(ci));
                    }
                }
            } else if let Some(t) = is!(mmff::terms::NonbondedContrib) {
                if skip_nonbonded || out.nb.add_mmff(t) {
                    if !seen_nb {
                        seen_nb = true;
                        out.segments.push(Segment::Nonbonded);
                    }
                } else {
                    if out.opaque.last() != Some(&ci) {
                        out.opaque.push(ci);
                        out.segments.push(Segment::Opaque(ci));
                    }
                }
            } else {
                let cons = &mut out.constraints.terms;
                let mut is_cons = true;
                if let Some(t) = is!(DistanceConstraintContrib) {
                    cons.push(ConstraintTerm::Distance(t.clone()));
                } else if let Some(t) = is!(AngleConstraintContrib) {
                    cons.push(ConstraintTerm::Angle(t.clone()));
                } else if let Some(t) = is!(TorsionConstraintContrib) {
                    cons.push(ConstraintTerm::Torsion(t.clone()));
                } else if let Some(t) = is!(PositionConstraintContrib) {
                    cons.push(ConstraintTerm::Position(t.clone()));
                } else {
                    is_cons = false;
                }
                if is_cons {
                    if !seen_cons {
                        seen_cons = true;
                        out.segments.push(Segment::Constraints);
                    }
                } else {
                    if out.opaque.last() != Some(&ci) {
                        out.opaque.push(ci);
                        out.segments.push(Segment::Opaque(ci));
                    }
                }
            }
        }
    }
    out
}

/// Position-independent part of a fast force field for one molecular
/// graph: bonded tables (typing and parameter lookup done) plus the
/// nonbonded typing/topology needed to select pairs.
#[derive(Debug)]
pub struct Template {
    pub bonded: Arc<Bonded>,
    /// Breakdown order before / after the nonbonded slot.
    pub before: Vec<Segment>,
    pub after: Vec<Segment>,
    pub nb: Arc<NbTemplate>,
    pub mmff: bool,
}

fn check_coords(mol: &Molecule) -> Result<()> {
    if mol.coords.len() != mol.num_atoms() {
        return Err(Error::InvalidArgument("molecule has no coordinates".into()));
    }
    Ok(())
}

/// `true` if the wrapped builder is one whose typing can be cached.
pub fn is_cacheable_builder(inner: &dyn Any) -> bool {
    inner.is::<UffBuilder>() || inner.is::<MmffBuilder>()
}

/// Builds the template for `mol` if `inner` is a UFF or MMFF builder and the
/// molecule's bonded terms do not depend on the geometry (`Ok(None)`
/// otherwise: UFF trigonal-bipyramidal centres choose their axial ligands
/// from the coordinates).
pub fn make_template(inner: &dyn Any, mol: &Molecule, grain: usize) -> Result<Option<Template>> {
    if let Some(b) = inner.downcast_ref::<UffBuilder>() {
        check_coords(mol)?;
        if (0..mol.num_atoms()).any(|i| mol.atom(i).hybridization == Hybridization::SP3D && mol.degree(i) == 5) {
            return Ok(None);
        }
        let (_, params, _) = uff::atom_typer::atom_types_and_params(mol);
        // vdw_thresh 0: no nonbonded pairs (they are selected per geometry)
        let ff = uff::builder::construct_force_field(mol, &params, 0.0, b.ignore_interfrag_interactions)?;
        let mut c = compile(ff.contribs(), true);
        c.bonded.plan(grain);
        debug_assert!(c.opaque.is_empty() && c.constraints.is_empty());
        let split = c
            .segments
            .iter()
            .position(|s| matches!(s, Segment::Table(TableKind::UffTorsion) | Segment::Table(TableKind::UffInversion)))
            .unwrap_or(c.segments.len());
        let after = c.segments.split_off(split);
        let nb = NbTemplate::uff(mol, &params, b.vdw_thresh, b.ignore_interfrag_interactions);
        return Ok(Some(Template { bonded: Arc::new(c.bonded), before: c.segments, after, nb: Arc::new(nb), mmff: false }));
    }
    if let Some(b) = inner.downcast_ref::<MmffBuilder>() {
        check_coords(mol)?;
        let props = MmffMolProperties::new(mol, b.variant)?;
        // threshold -inf: no nonbonded pairs
        let b0 = MmffBuilder { nonbonded_thresh: f64::NEG_INFINITY, ..b.clone() };
        let ff: ContribForceField = b0.build_with_properties(mol, &props, &[])?;
        let mut c = compile(ff.contribs(), true);
        c.bonded.plan(grain);
        debug_assert!(c.opaque.is_empty() && c.constraints.is_empty());
        let nb = NbTemplate::mmff(&props, b);
        return Ok(Some(Template { bonded: Arc::new(c.bonded), before: c.segments, after: Vec::new(), nb: Arc::new(nb), mmff: true }));
    }
    Ok(None)
}

/// Cache key: the molecular graph (atoms with charges, hydrogens,
/// hybridization, aromaticity; bonds with orders, in order).
pub fn graph_key(mol: &Molecule) -> Vec<u64> {
    let mut k = Vec::with_capacity(2 + mol.num_atoms() + 2 * mol.num_bonds());
    k.push(mol.num_atoms() as u64);
    k.push(mol.num_bonds() as u64);
    for a in mol.atoms() {
        k.push(
            a.atomic_num as u64
                | (a.formal_charge as u8 as u64) << 8
                | (a.isotope as u64) << 16
                | (a.num_explicit_hs as u64) << 32
                | (a.implicit_hs as u64) << 40
                | (a.no_implicit as u64) << 48
                | (a.is_aromatic as u64) << 49
                | ((a.num_radical_electrons as u64) & 0x3f) << 50
                | (a.hybridization as u64) << 56,
        );
    }
    for b in mol.bonds() {
        k.push(b.begin as u64 | (b.end as u64) << 32);
        k.push(b.order as u64 | (b.is_aromatic as u64) << 8 | (b.is_conjugated as u64) << 9);
    }
    k
}
