//! Atom and bond query primitives.
//!
//! Port of the subset of RDKit `Code/GraphMol/QueryOps.{h,cpp}` (and the
//! `Code/Query/*` comparison semantics) needed by the SMARTS matcher.
//!
//! Copyright (C) 2003-2026 Greg Landrum and other RDKit contributors,
//! BSD-3-Clause (see `LICENSE-RDKIT.txt`).

use crate::core::{BondOrder, Hybridization, Molecule};
use crate::perception::rings::RingInfo;

/// Atom property evaluated by a query ("data function" in RDKit terms).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AtomProp {
    /// `queryAtomNum`
    AtomicNum,
    /// `queryAtomType`: atomic number + 1000 * aromatic
    AtomType,
    /// `queryAtomAromatic`
    Aromatic,
    /// `queryAtomAliphatic`
    Aliphatic,
    /// `queryAtomExplicitDegree` (`D`)
    ExplicitDegree,
    /// `queryAtomNonHydrogenDegree` (`d`)
    NonHydrogenDegree,
    /// `queryAtomTotalDegree` (`X`)
    TotalDegree,
    /// `queryAtomTotalValence` (`v`)
    TotalValence,
    /// `queryAtomHCount` (`H`): `getTotalNumHs(true)`
    HCount,
    /// `queryAtomImplicitHCount` (`h<n>`): `getTotalNumHs(false)`
    ImplicitHCount,
    /// `queryAtomHasImplicitH` (`h`)
    HasImplicitH,
    /// `queryAtomRingBondCount` (`x<n>`)
    RingBondCount,
    /// `queryAtomHasRingBond` (`x`)
    HasRingBond,
    /// `queryAtomRingMembership` (`R`)
    NumRings,
    /// `queryIsAtomInRing` (`r`, `k`)
    InRing,
    /// `queryAtomMinRingSize` (`r<n>`, `r{..}`)
    MinRingSize,
    /// `queryAtomIsInRingOfSize(at, tgt)` (`k<n>`)
    InRingOfSize(i32),
    /// `queryAtomIsInRingOfSize(at, lower, upper)` (`k{..}`); -1 = unbounded
    RingSizeInRange(i32, i32),
    /// `queryAtomIsotope`
    Isotope,
    /// `queryAtomFormalCharge`
    FormalCharge,
    /// `queryAtomNegativeFormalCharge`
    NegativeFormalCharge,
    /// `queryAtomHybridization` (RDKit enum value)
    Hybridization,
    /// `queryAtomNumHeteroatomNbrs` (`z<n>`)
    NumHeteroatomNbrs,
    /// `queryAtomHasHeteroatomNbrs` (`z`)
    HasHeteroatomNbrs,
    /// `queryAtomNumAliphaticHeteroatomNbrs` (`Z<n>`)
    NumAliphaticHeteroatomNbrs,
    /// `queryAtomHasAliphaticHeteroatomNbrs` (`Z`)
    HasAliphaticHeteroatomNbrs,
}

/// Comparison between the query value and the atom data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cmp {
    /// `EqualityQuery`: data == v
    Eq(i32),
    /// `AtomRingQuery` with a negative value: data != 0
    NonZero,
    /// `LessEqualQuery(v)` (SMARTS `{v-}`): v <= data
    AtLeast(i32),
    /// `GreaterEqualQuery(v)` (SMARTS `{-v}`): v >= data
    AtMost(i32),
    /// closed `RangeQuery` (SMARTS `{lo-hi}`)
    Range(i32, i32),
}

impl Cmp {
    fn test(self, data: i32) -> bool {
        match self {
            Cmp::Eq(v) => data == v,
            Cmp::NonZero => data != 0,
            Cmp::AtLeast(v) => v <= data,
            Cmp::AtMost(v) => v >= data,
            Cmp::Range(lo, hi) => lo <= data && data <= hi,
        }
    }
}

/// Atom query expression tree.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AtomExpr {
    /// `makeAtomNullQuery`: matches every atom
    True,
    Prim(AtomProp, Cmp),
    /// `RecursiveStructureQuery`, indexing the owning pattern's recursive list
    Recursive(usize),
    Not(Box<AtomExpr>),
    And(Box<AtomExpr>, Box<AtomExpr>),
    Or(Box<AtomExpr>, Box<AtomExpr>),
}

impl AtomExpr {
    pub(crate) fn prim(p: AtomProp, c: Cmp) -> AtomExpr {
        AtomExpr::Prim(p, c)
    }
    pub(crate) fn and(a: AtomExpr, b: AtomExpr) -> AtomExpr {
        AtomExpr::And(Box::new(a), Box::new(b))
    }
    pub(crate) fn or(a: AtomExpr, b: AtomExpr) -> AtomExpr {
        AtomExpr::Or(Box::new(a), Box::new(b))
    }
    pub(crate) fn negate(a: AtomExpr) -> AtomExpr {
        match a {
            AtomExpr::Not(inner) => *inner,
            other => AtomExpr::Not(Box::new(other)),
        }
    }
    /// `makeAtomTypeQuery(num, aromatic)`
    pub(crate) fn atom_type(num: u8, aromatic: bool) -> AtomExpr {
        AtomExpr::Prim(
            AtomProp::AtomType,
            Cmp::Eq(num as i32 + 1000 * aromatic as i32),
        )
    }
    /// `makeAtomNumQuery(num)`
    pub(crate) fn atomic_num(num: u8) -> AtomExpr {
        AtomExpr::Prim(AtomProp::AtomicNum, Cmp::Eq(num as i32))
    }

    pub(crate) fn matches(&self, t: &Target, atom: usize, rec: &[Vec<bool>]) -> bool {
        match self {
            AtomExpr::True => true,
            AtomExpr::Prim(p, c) => c.test(t.atom_data(*p, atom)),
            AtomExpr::Recursive(id) => rec[*id][atom],
            AtomExpr::Not(e) => !e.matches(t, atom, rec),
            AtomExpr::And(a, b) => a.matches(t, atom, rec) && b.matches(t, atom, rec),
            AtomExpr::Or(a, b) => a.matches(t, atom, rec) || b.matches(t, atom, rec),
        }
    }
}

/// Bond query expression tree.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BondExpr {
    /// `makeBondNullQuery` (`~`)
    True,
    /// `makeBondOrderEqualsQuery`; `None` = a bond type we never store
    /// (quadruple), which therefore never matches.
    Order(Option<BondOrder>),
    /// `makeSingleOrAromaticBondQuery` (implicit bond, `/`, `\`)
    SingleOrAromatic,
    /// `makeBondIsInRingQuery` (`@`)
    InRing,
    Not(Box<BondExpr>),
    And(Box<BondExpr>, Box<BondExpr>),
    Or(Box<BondExpr>, Box<BondExpr>),
}

impl BondExpr {
    pub(crate) fn matches(&self, t: &Target, bond: usize) -> bool {
        match self {
            BondExpr::True => true,
            BondExpr::Order(o) => Some(t.mol.bond(bond).order) == *o,
            BondExpr::SingleOrAromatic => {
                matches!(
                    t.mol.bond(bond).order,
                    BondOrder::Single | BondOrder::Aromatic
                )
            }
            BondExpr::InRing => t.rings.num_bond_rings(bond) != 0,
            BondExpr::Not(e) => !e.matches(t, bond),
            BondExpr::And(a, b) => a.matches(t, bond) && b.matches(t, bond),
            BondExpr::Or(a, b) => a.matches(t, bond) || b.matches(t, bond),
        }
    }
}

/// RDKit `Atom::HybridizationType` enum value.
fn rdkit_hybridization(h: Hybridization) -> i32 {
    match h {
        Hybridization::Unspecified => 0,
        Hybridization::S => 1,
        Hybridization::SP => 2,
        Hybridization::SP2 => 3,
        Hybridization::SP3 => 4,
        Hybridization::SP2D => 5,
        Hybridization::SP3D => 6,
        Hybridization::SP3D2 => 7,
        Hybridization::Other => 8,
    }
}

/// `Hybridization` from the SMARTS `^n` token (see `smarts.ll`).
pub(crate) fn hybridization_token(d: u8) -> Option<i32> {
    let h = match d {
        0 => Hybridization::S,
        1 => Hybridization::SP,
        2 => Hybridization::SP2,
        3 => Hybridization::SP3,
        4 => Hybridization::SP3D,
        5 => Hybridization::SP3D2,
        _ => return None,
    };
    Some(rdkit_hybridization(h))
}

/// Target molecule plus the ring information the queries need.
pub(crate) struct Target<'a> {
    pub(crate) mol: &'a Molecule,
    pub(crate) rings: std::borrow::Cow<'a, RingInfo>,
}

impl<'a> Target<'a> {
    pub(crate) fn new(mol: &'a Molecule) -> Target<'a> {
        let rings = match mol.ring_info() {
            Some(r) => std::borrow::Cow::Borrowed(r),
            None => std::borrow::Cow::Owned(RingInfo::compute(mol)),
        };
        Target { mol, rings }
    }

    fn ring_bond_count(&self, i: usize) -> i32 {
        self.mol
            .atom_bonds(i)
            .iter()
            .filter(|&&(_, b)| self.rings.num_bond_rings(b) != 0)
            .count() as i32
    }

    fn hetero_nbrs(&self, i: usize, aliphatic_only: bool) -> i32 {
        self.mol
            .neighbors(i)
            .filter(|&n| {
                let a = self.mol.atom(n);
                a.atomic_num != 6 && a.atomic_num != 1 && !(aliphatic_only && a.is_aromatic)
            })
            .count() as i32
    }

    /// `queryAtomIsInRingOfSize(at, lower, upper)` with closed ends.
    fn ring_size_in_range(&self, i: usize, lower: i32, upper: i32) -> i32 {
        for &r in self.rings.atom_ring_ids(i) {
            let sz = self.rings.atom_rings[r].len() as i32;
            if sz >= lower && (upper < 0 || sz <= upper) {
                return sz;
            }
        }
        if lower > -1 {
            -1
        } else if upper > -1 {
            i32::MAX
        } else {
            0
        }
    }

    pub(crate) fn atom_data(&self, p: AtomProp, i: usize) -> i32 {
        let m = self.mol;
        let a = m.atom(i);
        match p {
            AtomProp::AtomicNum => a.atomic_num as i32,
            AtomProp::AtomType => a.atomic_num as i32 + 1000 * a.is_aromatic as i32,
            AtomProp::Aromatic => a.is_aromatic as i32,
            AtomProp::Aliphatic => !a.is_aromatic as i32,
            AtomProp::ExplicitDegree => m.degree(i) as i32,
            AtomProp::NonHydrogenDegree => m
                .neighbors(i)
                .filter(|&n| m.atom(n).atomic_num != 1 || m.atom(n).isotope > 1)
                .count() as i32,
            AtomProp::TotalDegree => m.total_degree(i) as i32,
            AtomProp::TotalValence => m.total_valence(i),
            AtomProp::HCount => m.total_num_hs(i) as i32,
            AtomProp::ImplicitHCount => a.total_hidden_hs() as i32,
            AtomProp::HasImplicitH => (a.total_hidden_hs() > 0) as i32,
            AtomProp::RingBondCount => self.ring_bond_count(i),
            AtomProp::HasRingBond => (self.ring_bond_count(i) > 0) as i32,
            AtomProp::NumRings => self.rings.num_atom_rings(i) as i32,
            AtomProp::InRing => self.rings.is_atom_in_ring(i) as i32,
            AtomProp::MinRingSize => self.rings.min_atom_ring_size(i) as i32,
            AtomProp::InRingOfSize(tgt) => {
                if tgt >= 0 && self.rings.is_atom_in_ring_of_size(i, tgt as usize) {
                    tgt
                } else {
                    0
                }
            }
            AtomProp::RingSizeInRange(lo, hi) => self.ring_size_in_range(i, lo, hi),
            AtomProp::Isotope => a.isotope as i32,
            AtomProp::FormalCharge => a.formal_charge as i32,
            AtomProp::NegativeFormalCharge => -(a.formal_charge as i32),
            AtomProp::Hybridization => rdkit_hybridization(a.hybridization),
            AtomProp::NumHeteroatomNbrs => self.hetero_nbrs(i, false),
            AtomProp::HasHeteroatomNbrs => (self.hetero_nbrs(i, false) > 0) as i32,
            AtomProp::NumAliphaticHeteroatomNbrs => self.hetero_nbrs(i, true),
            AtomProp::HasAliphaticHeteroatomNbrs => (self.hetero_nbrs(i, true) > 0) as i32,
        }
    }
}
