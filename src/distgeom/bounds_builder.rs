//! Topological distance bounds (1-2, 1-3, 1-4, 1-5 and van der Waals lower
//! bounds).
//!
//! Port of RDKit `Code/GraphMol/DistGeomHelpers/BoundsMatrixBuilder.cpp`
//! and `BoundsMatrixBuilderDetails.h` (UFF 1-2 distances; the MMFF variant
//! and the internal-coordinate embedding are not ported), plus
//! `RDGeom::compute13Dist` / `compute14Dist*` from `Code/Geometry/Utils.h`.
//! RDKit: Copyright (C) 2004-2026 Greg Landrum and other RDKit contributors,
//! BSD-3-Clause.

use super::bounds::BoundsMatrix;
use super::uff_params::{calc_bond_rest_length, uff_atom_params};
use crate::core::{BondOrder, BondStereo, Hybridization, Molecule, element};
use crate::perception::rings::RingInfo;
use std::collections::{BTreeMap, HashSet};
use std::f64::consts::PI;

const DIST12_DELTA: f64 = 0.01;
const ANGLE_DELTA: f64 = 0.035;
const GEN_DIST_TOL: f64 = 0.06;
const DIST15_TOL: f64 = 0.08;
const VDW_SCALE_15: f64 = 0.7;
const H_BOND_LENGTH: f64 = 1.8;
const MAX_UPPER: f64 = 1000.0;
const MIN_MACROCYCLE_RING_SIZE: usize = 9;

/// Kind of 1-4 torsion (RDKit `TorsionType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorsionType {
    Cis,
    Trans,
    Flexible,
    Custom,
    CisTrans,
}

/// RDKit `TorsionValue`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TorsionValue {
    pub kind: TorsionType,
    pub value: Option<f64>,
    pub extra_dist: Option<f64>,
    pub is_forced: bool,
}

impl TorsionValue {
    fn of(kind: TorsionType) -> Self {
        TorsionValue { kind, value: None, extra_dist: None, is_forced: false }
    }
    fn forced(kind: TorsionType) -> Self {
        TorsionValue { kind, value: None, extra_dist: None, is_forced: true }
    }
}

/// RDKit `Type14`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type14 {
    InChain,
    InRing,
    TwoInSameRing,
    TwoInDiffRing,
    ShareRingBond,
    MacrocycleTwoInSameRing,
    MacrocycleAllInSameRing,
}

/// RDKit `Path14Configuration`.
#[derive(Debug, Clone, PartialEq)]
pub struct Path14 {
    pub bid1: usize,
    pub bid2: usize,
    pub bid3: usize,
    pub aid1: usize,
    pub aid2: usize,
    pub aid3: usize,
    pub aid4: usize,
    pub value: TorsionValue,
    pub type14: Type14,
    pub ring_size: usize,
}

/// Options for [`set_topol_bounds`].
#[derive(Debug, Clone, Copy)]
pub struct TopolOptions {
    pub scale_vdw: bool,
    pub set15bounds: bool,
    pub set14bounds: bool,
    pub set13bounds: bool,
    pub use_macrocycle14config: bool,
    pub force_trans_amides: bool,
}

impl Default for TopolOptions {
    fn default() -> Self {
        TopolOptions {
            scale_vdw: false,
            set15bounds: true,
            set14bounds: true,
            set13bounds: true,
            use_macrocycle14config: false,
            force_trans_amides: true,
        }
    }
}

/// Packed symmetric matrix including the diagonal (RDKit `SymmMatrix`).
struct Symm<T: Copy> {
    data: Vec<T>,
}
impl<T: Copy> Symm<T> {
    fn new(n: usize, v: T) -> Self {
        Symm { data: vec![v; n * (n + 1) / 2] }
    }
    fn idx(i: usize, j: usize) -> usize {
        if j <= i { i * (i + 1) / 2 + j } else { j * (j + 1) / 2 + i }
    }
    fn get(&self, i: usize, j: usize) -> T {
        self.data[Self::idx(i, j)]
    }
    fn set(&mut self, i: usize, j: usize, v: T) {
        self.data[Self::idx(i, j)] = v;
    }
}

struct ComputedData {
    bond_lengths: Vec<f64>,
    bond_adj: Symm<i64>,
    bond_angles: Symm<f64>,
    paths14: Vec<Path14>,
    cis_paths: HashSet<usize>,
    trans_paths: HashSet<usize>,
    set15_atoms: Vec<bool>,
    visited12: Vec<bool>,
    visited13: Vec<bool>,
    visited14: Vec<bool>,
}

#[derive(PartialEq, PartialOrd, Clone, Copy)]
enum DistType {
    D12,
    D13,
    D14,
}

impl ComputedData {
    fn new(na: usize, nb: usize) -> Self {
        ComputedData {
            bond_lengths: vec![0.0; nb],
            bond_adj: Symm::new(nb, -1),
            bond_angles: Symm::new(nb, -1.0),
            paths14: Vec::new(),
            cis_paths: HashSet::new(),
            trans_paths: HashSet::new(),
            set15_atoms: vec![false; na * na],
            visited12: vec![false; na * na],
            visited13: vec![false; na * na],
            visited14: vec![false; na * na],
        }
    }
    fn visited_bound(&self, pid: usize, max: DistType) -> bool {
        (max >= DistType::D12 && self.visited12[pid])
            || (max >= DistType::D13 && self.visited13[pid])
            || (max >= DistType::D14 && self.visited14[pid])
    }
}

#[inline]
fn uid2(a: usize, b: usize, n: usize) -> usize {
    if a < b { a * n + b } else { b * n + a }
}
#[inline]
fn uid3(a: usize, b: usize, c: usize, n: usize) -> usize {
    if a < c { a * n * n + b * n + c } else { c * n * n + b * n + a }
}

/// `RDGeom::compute13Dist`.
#[inline]
pub fn compute13_dist(d1: f64, d2: f64, angle: f64) -> f64 {
    (d1 * d1 + d2 * d2 - 2.0 * d1 * d2 * angle.cos()).sqrt()
}
fn compute14_dist_cis(d1: f64, d2: f64, d3: f64, a12: f64, a23: f64) -> f64 {
    let dx = d2 - d3 * a23.cos() - d1 * a12.cos();
    let dy = d3 * a23.sin() - d1 * a12.sin();
    (dx * dx + dy * dy).sqrt()
}
fn compute14_dist_trans(d1: f64, d2: f64, d3: f64, a12: f64, a23: f64) -> f64 {
    let dx = d2 - d3 * a23.cos() - d1 * a12.cos();
    let dy = d3 * a23.sin() + d1 * a12.sin();
    (dx * dx + dy * dy).sqrt()
}
fn compute14_dist_3d(d1: f64, d2: f64, d3: f64, a12: f64, a23: f64, tor: f64) -> f64 {
    let p1 = [d1 * a12.cos(), d1 * a12.sin(), 0.0];
    let p4 = [d2 - d3 * a23.cos(), d3 * a23.sin(), 0.0];
    let (c, s) = (tor.cos(), tor.sin());
    // Transform3D rotation about x applied to p4 (z = 0)
    let x = 1.0 * p4[0] + 0.0 * p4[1] + 0.0 * p4[2] + 0.0;
    let y = 0.0 * p4[0] + c * p4[1] + (-s) * p4[2] + 0.0;
    let z = 0.0 * p4[0] + s * p4[1] + c * p4[2] + 0.0;
    let (dx, dy, dz) = (x - p1[0], y - p1[1], z - p1[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn check_and_set_bounds(i: usize, j: usize, lb: f64, ub: f64, mmat: &mut BoundsMatrix) {
    let clb = mmat.lower(i, j);
    let cub = mmat.upper(i, j);
    // conservative bound setting
    if clb <= DIST12_DELTA || (lb < clb && lb > DIST12_DELTA) {
        mmat.set_lower(i, j, lb);
    }
    if cub >= MAX_UPPER || (ub > cub && ub < MAX_UPPER) {
        mmat.set_upper(i, j, ub);
    }
}

struct Ctx<'a> {
    mol: &'a Molecule,
    ri: &'a RingInfo,
    na: usize,
    nb: usize,
}

impl Ctx<'_> {
    fn hyb(&self, a: usize) -> Hybridization {
        self.mol.atom(a).hybridization
    }
    fn anum(&self, a: usize) -> u8 {
        self.mol.atom(a).atomic_num
    }
    fn bond_other(&self, b: usize, a: usize) -> usize {
        self.mol.bond(b).other(a)
    }
    fn order(&self, b: usize) -> BondOrder {
        self.mol.bond(b).order
    }
    fn total_hs(&self, a: usize) -> usize {
        self.mol.total_num_hs(a)
    }
    fn degree(&self, a: usize) -> usize {
        self.mol.degree(a)
    }
}

fn squish_bond(cx: &Ctx, b: usize) -> bool {
    let bond = cx.mol.bond(b);
    bond.is_conjugated && (cx.anum(bond.begin) > 10 || cx.anum(bond.end) > 10) && cx.ri.is_bond_in_ring_of_size(b, 5)
}

fn set12_bounds(cx: &Ctx, mmat: &mut BoundsMatrix, acc: &mut ComputedData) -> bool {
    let (params, _found_all) = uff_atom_params(cx.mol);
    let mut squish = vec![false; cx.na];
    for b in 0..cx.nb {
        if squish_bond(cx, b) {
            squish[cx.mol.bond(b).begin] = true;
            squish[cx.mol.bond(b).end] = true;
        }
    }
    let mut success = true;
    for (bi, bond) in cx.mol.bonds().iter().enumerate() {
        let (beg, end) = (bond.begin, bond.end);
        let order = bond.order.as_f64();
        let bl = match (&params[beg], &params[end]) {
            (Some(a), Some(b)) if order > 0.0 => calc_bond_rest_length(order, a, b),
            _ => -1.0,
        };
        if bl > -1.0 {
            let extra = if squish[beg] && squish[end] { 0.2 } else { 0.0 };
            acc.bond_lengths[bi] = bl;
            mmat.set_upper(beg, end, bl + extra + DIST12_DELTA);
            mmat.set_lower(beg, end, bl - extra - DIST12_DELTA);
        } else {
            let e1 = element(cx.anum(beg));
            let e2 = element(cx.anum(end));
            let mut bl = e1.r_cov + e2.r_cov;
            let (mut us, mut ls) = (1.1, 0.9);
            if matches!(bond.order, BondOrder::Zero | BondOrder::Dative) {
                bl = (e1.r_vdw + e2.r_vdw) / 2.0;
                us = 1.5;
                ls = 0.75;
            } else {
                const PAULING_LAMBDA: f64 = 0.1332;
                bl -= PAULING_LAMBDA * order.ln() * bl;
            }
            acc.bond_lengths[bi] = bl;
            mmat.set_upper(beg, end, us * bl);
            mmat.set_lower(beg, end, ls * bl);
            success = false;
        }
        acc.visited12[uid2(beg, end, cx.na)] = true;
    }
    success
}

fn is_larger_sp2(cx: &Ctx, a: usize) -> bool {
    cx.anum(a) > 13 && cx.hyb(a) == Hybridization::SP2 && cx.ri.num_atom_rings(a) > 0
}

fn scale_tol_large_sp2(cx: &Ctx, a1: usize, a2: usize, a3: usize) -> f64 {
    let mut t = ANGLE_DELTA;
    for a in [a1, a2, a3] {
        if is_larger_sp2(cx, a) {
            t *= 2.0;
        }
    }
    t
}

fn angle_range(angle: f64, tol: f64) -> (f64, f64) {
    if angle + tol >= PI {
        return (PI - tol, PI);
    }
    if angle - tol <= 0.0 {
        return (0.0, tol);
    }
    (angle - tol, angle + tol)
}

fn set13_helper(cx: &Ctx, a1: usize, a: usize, a3: usize, angle: f64, mmat: &mut BoundsMatrix) {
    let tol = scale_tol_large_sp2(cx, a1, a, a3);
    let (al, au) = angle_range(angle, tol);
    let du = compute13_dist(mmat.upper(a1, a), mmat.upper(a, a3), au);
    let dl = compute13_dist(mmat.lower(a1, a), mmat.lower(a, a3), al);
    check_and_set_bounds(a1, a3, dl, du, mmat);
}

fn ring_angle(cx: &Ctx, a: usize, rs: usize) -> f64 {
    let h = cx.hyb(a);
    if (h == Hybridization::SP2 && rs <= 8) || rs == 3 || rs == 4 {
        PI * (1.0 - 2.0 / rs as f64)
    } else if h == Hybridization::SP3 {
        if rs == 5 { 104.0 * PI / 180.0 } else { 109.5 * PI / 180.0 }
    } else if h == Hybridization::SP3D {
        105.0 * PI / 180.0
    } else if h == Hybridization::SP3D2 {
        90.0 * PI / 180.0
    } else {
        120.0 * PI / 180.0
    }
}

fn set13_bounds(cx: &Ctx, mmat: &mut BoundsMatrix, acc: &mut ComputedData) {
    let npt = cx.na;
    let nb = cx.nb;
    let mut atom_rings = cx.ri.atom_rings.clone();
    atom_rings.sort_by_key(|r| r.len());
    let mut visited = vec![0usize; npt];
    let mut angle_taken = vec![0.0f64; npt];
    let mut done_paths = vec![false; nb * nb];
    for ring in &atom_rings {
        let rs = ring.len();
        let mut aid1 = ring[rs - 1];
        for i in 0..rs {
            let aid2 = ring[i];
            let aid3 = if i == rs - 1 { ring[0] } else { ring[i + 1] };
            let bid1 = cx.mol.bond_between(aid1, aid2).expect("ring bond");
            let bid2 = cx.mol.bond_between(aid2, aid3).expect("ring bond");
            let bp = uid2(bid1, bid2, nb);
            if !done_paths[bp] {
                let angle = ring_angle(cx, aid2, rs);
                let pid = uid2(aid1, aid3, npt);
                if !acc.visited_bound(pid, DistType::D12) {
                    set13_helper(cx, aid1, aid2, aid3, angle, mmat);
                    acc.visited13[pid] = true;
                }
                acc.bond_angles.set(bid1, bid2, angle);
                acc.bond_adj.set(bid1, bid2, aid2 as i64);
                visited[aid2] += 1;
                angle_taken[aid2] += angle;
                done_paths[bp] = true;
            }
            aid1 = aid2;
        }
    }
    for aid2 in 0..npt {
        let deg = cx.degree(aid2);
        let n13 = if deg == 0 { 0 } else { deg * (deg - 1) / 2 };
        if n13 == visited[aid2] {
            continue;
        }
        let ahyb = cx.hyb(aid2);
        let bonds: Vec<(usize, usize)> = cx.mol.atom_bonds(aid2).to_vec();
        if visited[aid2] >= 1 {
            for (k1, &(aid1, bid1)) in bonds.iter().enumerate() {
                for &(aid3, bid2) in &bonds[..k1] {
                    if acc.bond_angles.get(bid1, bid2) < 0.0 {
                        let angle = if ahyb == Hybridization::SP2 {
                            (2.0 * PI - angle_taken[aid2]) / ((n13 - visited[aid2]) as f64)
                        } else if ahyb == Hybridization::SP3 {
                            if cx.ri.is_atom_in_ring_of_size(aid2, 3) {
                                116.0 * PI / 180.0
                            } else if cx.ri.is_atom_in_ring_of_size(aid2, 4) {
                                112.0 * PI / 180.0
                            } else {
                                109.5 * PI / 180.0
                            }
                        } else if deg == 5 {
                            105.0 * PI / 180.0
                        } else if deg == 6 {
                            135.0 * PI / 180.0
                        } else {
                            120.0 * PI / 180.0
                        };
                        let pid = uid2(aid1, aid3, npt);
                        if !acc.visited_bound(pid, DistType::D12) {
                            set13_helper(cx, aid1, aid2, aid3, angle, mmat);
                            acc.visited13[pid] = true;
                        }
                        acc.bond_angles.set(bid1, bid2, angle);
                        acc.bond_adj.set(bid1, bid2, aid2 as i64);
                        angle_taken[aid2] += angle;
                        visited[aid2] += 1;
                    }
                }
            }
        } else {
            for (k1, &(aid1, bid1)) in bonds.iter().enumerate() {
                for &(aid3, bid2) in &bonds[..k1] {
                    let angle = match ahyb {
                        Hybridization::SP => PI,
                        Hybridization::SP2 => 2.0 * PI / 3.0,
                        Hybridization::SP3 => 109.5 * PI / 180.0,
                        Hybridization::SP3D => 105.0 * PI / 180.0,
                        Hybridization::SP3D2 => 135.0 * PI / 180.0,
                        _ => 120.0 * PI / 180.0,
                    };
                    let pid = uid2(aid1, aid3, npt);
                    if !acc.visited_bound(pid, DistType::D12) {
                        if deg <= 4 {
                            set13_helper(cx, aid1, aid2, aid3, angle, mmat);
                        } else {
                            let dmax = acc.bond_lengths[bid1] + acc.bond_lengths[bid2];
                            check_and_set_bounds(aid1, aid3, 1.0, dmax * 1.2, mmat);
                        }
                        acc.visited13[pid] = true;
                    }
                    acc.bond_angles.set(bid1, bid2, angle);
                    acc.bond_adj.set(bid1, bid2, aid2 as i64);
                    angle_taken[aid2] += angle;
                    visited[aid2] += 1;
                }
            }
        }
    }
}

/// Stereo of `bnd` relative to (aid1, aid4) (RDKit `_getAtomStereo`).
fn atom_stereo(cx: &Ctx, b: usize, aid1: usize, aid4: usize) -> BondStereo {
    let bond = cx.mol.bond(b);
    let st = bond.stereo;
    if st != BondStereo::None {
        if let Some((s0, s1)) = bond.stereo_atoms {
            if (s0 != aid1) ^ (s1 != aid4) {
                return match st {
                    BondStereo::Cis => BondStereo::Trans,
                    BondStereo::Trans => BondStereo::Cis,
                    BondStereo::None => BondStereo::None,
                };
            }
        }
    }
    st
}

fn in_ring14_type(cx: &Ctx, b2: usize, a1: usize, a2: usize, a3: usize, a4: usize, ring_size: usize) -> TorsionValue {
    let (h2, h3) = (cx.hyb(a2), cx.hyb(a3));
    let st = atom_stereo(cx, b2, a1, a4);
    let sp2 = h2 == Hybridization::SP2 && h3 == Hybridization::SP2;
    if (ring_size <= 8 && sp2 && st != BondStereo::Trans) || st == BondStereo::Cis {
        TorsionValue::of(TorsionType::Cis)
    } else if st == BondStereo::Trans {
        TorsionValue::of(TorsionType::Trans)
    } else if sp2 {
        TorsionValue::of(TorsionType::CisTrans)
    } else {
        TorsionValue::of(TorsionType::Flexible)
    }
}

fn two_in_same_ring14_type(cx: &Ctx, b2: usize, a1: usize, a2: usize, a3: usize, a4: usize, prefer_trans: bool) -> TorsionValue {
    let (h2, h3) = (cx.hyb(a2), cx.hyb(a3));
    let st = atom_stereo(cx, b2, a1, a4);
    let sp2 = h2 == Hybridization::SP2 && h3 == Hybridization::SP2;
    if (prefer_trans && sp2 && st != BondStereo::Cis) || st == BondStereo::Trans {
        TorsionValue::of(TorsionType::Trans)
    } else if st == BondStereo::Cis {
        TorsionValue::of(TorsionType::Cis)
    } else if sp2 {
        TorsionValue::of(TorsionType::CisTrans)
    } else {
        TorsionValue::of(TorsionType::Flexible)
    }
}

fn check_amide_ester14(cx: &Ctx, b1: usize, b3: usize, a2: usize, a3: usize, a4: usize) -> bool {
    let (n2, n3, n4) = (cx.anum(a2), cx.anum(a3), cx.anum(a4));
    n3 == 6
        && cx.order(b3) == BondOrder::Double
        && (n4 == 8 || n4 == 7)
        && cx.order(b1) == BondOrder::Single
        && (n2 == 8 || (n2 == 7 && cx.total_hs(a2) == 1))
}

fn check_macrocycle_all_in_same_ring_amide_ester14(cx: &Ctx, a1: usize, a2: usize, a3: usize, a4: usize) -> bool {
    let (n2, n3) = (cx.anum(a2), cx.anum(a3));
    if n3 != 6 {
        return false;
    }
    if (n2 == 7 || n2 == 8) && cx.degree(a2) == 3 && cx.degree(a3) == 3 {
        for nbr in cx.mol.neighbors(a2) {
            if nbr != a1 && nbr != a3 {
                let rb = cx.mol.bond_between(a2, nbr).unwrap();
                if (cx.anum(nbr) != 6 && cx.anum(nbr) != 1) || cx.order(rb) != BondOrder::Single {
                    return false;
                }
                break;
            }
        }
        for nbr in cx.mol.neighbors(a3) {
            if nbr != a2 && nbr != a4 {
                let rb = cx.mol.bond_between(a3, nbr).unwrap();
                if cx.anum(nbr) != 8 || cx.order(rb) != BondOrder::Double {
                    return false;
                }
                break;
            }
        }
        return true;
    }
    false
}

fn is_carbonyl(cx: &Ctx, a: usize) -> bool {
    if cx.anum(a) == 6 && cx.degree(a) > 2 {
        for &(nbr, b) in cx.mol.atom_bonds(a) {
            let n = cx.anum(nbr);
            if (n == 8 || n == 7) && cx.order(b) == BondOrder::Double {
                return true;
            }
        }
    }
    false
}

fn check_amide_ester15(cx: &Ctx, b1: usize, b3: usize, a2: usize, a3: usize) -> bool {
    let n2 = cx.anum(a2);
    (n2 == 8 || (n2 == 7 && cx.total_hs(a2) == 1))
        && cx.order(b1) == BondOrder::Single
        && cx.anum(a3) == 6
        && cx.order(b3) == BondOrder::Single
        && is_carbonyl(cx, a3)
}

fn is_sec_amide_h(cx: &Ctx, h: usize, n: usize) -> bool {
    cx.anum(h) == 1 && cx.anum(n) == 7 && cx.degree(n) == 3 && cx.total_hs(n) == 1
}

#[allow(clippy::too_many_arguments)]
fn chain14_type(cx: &Ctx, b1: usize, b2: usize, b3: usize, a1: usize, a2: usize, a3: usize, a4: usize, force_trans_amides: bool) -> TorsionValue {
    match cx.order(b2) {
        BondOrder::Double => {
            if cx.order(b1) == BondOrder::Double || cx.order(b3) == BondOrder::Double {
                TorsionValue::of(TorsionType::Cis)
            } else if cx.mol.bond(b2).stereo != BondStereo::None {
                if atom_stereo(cx, b2, a1, a4) == BondStereo::Cis {
                    TorsionValue::of(TorsionType::Cis)
                } else {
                    TorsionValue::of(TorsionType::Trans)
                }
            } else if cx.hyb(a2) == Hybridization::SP2 && cx.hyb(a3) == Hybridization::SP2 {
                TorsionValue::of(TorsionType::CisTrans)
            } else {
                TorsionValue::of(TorsionType::Flexible)
            }
        }
        BondOrder::Single => {
            if cx.anum(a2) == 16 && cx.anum(a3) == 16 && cx.degree(a2) == 2 && cx.degree(a3) == 2 {
                return TorsionValue { kind: TorsionType::Custom, value: Some(PI / 2.0), extra_dist: None, is_forced: false };
            }
            if check_amide_ester14(cx, b1, b3, a2, a3, a4) || check_amide_ester14(cx, b3, b1, a3, a2, a1) {
                if force_trans_amides {
                    if is_sec_amide_h(cx, a1, a2) || is_sec_amide_h(cx, a4, a3) {
                        return TorsionValue::forced(TorsionType::Trans);
                    } else {
                        return TorsionValue::forced(TorsionType::Cis);
                    }
                } else {
                    return TorsionValue::of(TorsionType::CisTrans);
                }
            }
            if check_amide_ester15(cx, b1, b3, a2, a3) || check_amide_ester15(cx, b3, b1, a3, a2) {
                if force_trans_amides {
                    if is_sec_amide_h(cx, a1, a2) || is_sec_amide_h(cx, a4, a3) {
                        return TorsionValue::forced(TorsionType::Cis);
                    } else {
                        return TorsionValue::forced(TorsionType::Trans);
                    }
                } else {
                    return TorsionValue::of(TorsionType::CisTrans);
                }
            }
            TorsionValue::of(TorsionType::Flexible)
        }
        _ => TorsionValue::of(TorsionType::Flexible),
    }
}

fn check_macrocycle_two_in_same_ring_amide_ester14(cx: &Ctx, b1: usize, b3: usize, a1: usize, a2: usize, a3: usize, a4: usize) -> bool {
    let (n1, n2, n3, n4) = (cx.anum(a1), cx.anum(a2), cx.anum(a3), cx.anum(a4));
    n1 != 1
        && n3 == 6
        && cx.order(b3) == BondOrder::Double
        && (n4 == 8 || n4 == 7)
        && cx.order(b1) == BondOrder::Single
        && (n2 == 8 || n2 == 7)
}

#[allow(clippy::too_many_arguments)]
fn macro_two_in_same_ring14_type(cx: &Ctx, b1: usize, b2: usize, b3: usize, a1: usize, a2: usize, a3: usize, a4: usize, use_mc: bool) -> TorsionValue {
    let st = atom_stereo(cx, b2, a1, a4);
    if check_macrocycle_two_in_same_ring_amide_ester14(cx, b1, b3, a1, a2, a3, a4)
        || check_macrocycle_two_in_same_ring_amide_ester14(cx, b3, b1, a4, a3, a2, a1)
    {
        return if use_mc { TorsionValue::forced(TorsionType::Cis) } else { TorsionValue::of(TorsionType::CisTrans) };
    }
    if st == BondStereo::Cis {
        TorsionValue::of(TorsionType::Cis)
    } else if st == BondStereo::Trans {
        TorsionValue::of(TorsionType::Trans)
    } else if cx.hyb(a2) == Hybridization::SP2 && cx.hyb(a3) == Hybridization::SP2 {
        TorsionValue::of(TorsionType::CisTrans)
    } else {
        TorsionValue::of(TorsionType::Flexible)
    }
}

#[allow(clippy::too_many_arguments)]
fn macro_all_in_same_ring14_type(cx: &Ctx, b1: usize, b2: usize, b3: usize, a1: usize, a2: usize, a3: usize, a4: usize, use_mc: bool) -> TorsionValue {
    match cx.order(b2) {
        BondOrder::Double => {
            if cx.order(b1) == BondOrder::Double || cx.order(b3) == BondOrder::Double {
                TorsionValue::of(TorsionType::Cis)
            } else if cx.mol.bond(b2).stereo != BondStereo::None {
                if atom_stereo(cx, b2, a1, a4) == BondStereo::Cis {
                    TorsionValue::of(TorsionType::Cis)
                } else {
                    TorsionValue::of(TorsionType::Trans)
                }
            } else if cx.hyb(a2) == Hybridization::SP2 && cx.hyb(a3) == Hybridization::SP2 {
                TorsionValue::of(TorsionType::CisTrans)
            } else {
                TorsionValue::of(TorsionType::Flexible)
            }
        }
        BondOrder::Single => {
            if cx.anum(a2) == 16 && cx.anum(a3) == 16 && cx.degree(a2) == 2 && cx.degree(a3) == 2 {
                return TorsionValue { kind: TorsionType::Custom, value: Some(PI / 2.0), extra_dist: None, is_forced: false };
            }
            if check_macrocycle_all_in_same_ring_amide_ester14(cx, a1, a2, a3, a4)
                || check_macrocycle_all_in_same_ring_amide_ester14(cx, a4, a3, a2, a1)
            {
                return if use_mc {
                    TorsionValue { kind: TorsionType::Trans, value: None, extra_dist: Some(0.1), is_forced: true }
                } else {
                    TorsionValue::of(TorsionType::CisTrans)
                };
            }
            if check_amide_ester15(cx, b1, b3, a2, a3) || check_amide_ester15(cx, b3, b1, a3, a2) {
                if use_mc {
                    if cx.anum(a2) == 7 && cx.degree(a2) == 3 && cx.anum(a1) == 1 && cx.total_hs(a2) == 1 {
                        return TorsionValue::forced(TorsionType::Cis);
                    } else {
                        return TorsionValue::forced(TorsionType::Trans);
                    }
                } else {
                    return TorsionValue::of(TorsionType::CisTrans);
                }
            }
            TorsionValue::of(TorsionType::Flexible)
        }
        _ => TorsionValue::of(TorsionType::Flexible),
    }
}

#[derive(Debug, Clone, Copy)]
struct Bounds14 {
    lower: f64,
    upper: f64,
    aid1: usize,
    aid4: usize,
}

/// RDKit `merge(std::vector<Bounds>)`.
fn merge_bounds(mut b: Vec<Bounds14>) -> Bounds14 {
    b.sort_by(|x, y| x.lower.partial_cmp(&y.lower).unwrap_or(std::cmp::Ordering::Equal));
    let mut cur = b[0];
    let mut comp_upper = cur.upper;
    let mut result_lower: Option<f64> = None;
    for bd in &b[1..] {
        if bd.lower <= cur.upper {
            cur.lower = bd.lower;
            cur.upper = cur.upper.min(bd.upper);
        } else {
            if result_lower.is_none() {
                result_lower = Some(cur.lower);
            }
            cur.lower = bd.lower;
            cur.upper = if bd.lower <= comp_upper { comp_upper.min(bd.upper) } else { bd.upper };
        }
        comp_upper = comp_upper.max(bd.upper);
    }
    Bounds14 { lower: result_lower.unwrap_or(cur.lower), upper: cur.upper, aid1: b[0].aid1, aid4: b[0].aid4 }
}

#[derive(Default, Clone, Copy)]
struct Info14 {
    force_trans_amides: bool,
    use_macrocycle14config: bool,
    ring_size: usize,
    prefer_trans: bool,
}

#[allow(clippy::too_many_arguments)]
fn collect14(
    cx: &Ctx,
    b1: usize,
    b2: usize,
    b3: usize,
    ty: Type14,
    acc: &mut ComputedData,
    mmat: &BoundsMatrix,
    dmat: &[f64],
    info: Info14,
    collected: &mut BTreeMap<usize, Vec<Bounds14>>,
) {
    let a2 = acc.bond_adj.get(b1, b2) as usize;
    let a3 = acc.bond_adj.get(b2, b3) as usize;
    let a1 = cx.bond_other(b1, a2);
    let a4 = cx.bond_other(b3, a3);
    let na = cx.na;
    let pid = a1.min(a4) * na + a1.max(a4);
    if acc.visited_bound(pid, DistType::D13) || dmat[a1.max(a4) * na + a1.min(a4)] < 2.9 {
        return;
    }
    let (bl_u1, bl_u2, bl_u3) = (mmat.upper(a1, a2), mmat.upper(a2, a3), mmat.upper(a3, a4));
    let (bl_l1, bl_l2, bl_l3) = (mmat.lower(a1, a2), mmat.lower(a2, a3), mmat.lower(a3, a4));
    let ba12 = acc.bond_angles.get(b1, b2);
    let ba23 = acc.bond_angles.get(b2, b3);
    if ty != Type14::InChain && (cx.mol.bond_between(a1, a3).is_some() || cx.mol.bond_between(a4, a2).is_some()) {
        return;
    }
    let tv = match ty {
        Type14::InChain => chain14_type(cx, b1, b2, b3, a1, a2, a3, a4, info.force_trans_amides),
        Type14::InRing => in_ring14_type(cx, b2, a1, a2, a3, a4, info.ring_size),
        Type14::MacrocycleAllInSameRing => macro_all_in_same_ring14_type(cx, b1, b2, b3, a1, a2, a3, a4, info.use_macrocycle14config),
        Type14::MacrocycleTwoInSameRing => macro_two_in_same_ring14_type(cx, b1, b2, b3, a1, a2, a3, a4, info.use_macrocycle14config),
        Type14::ShareRingBond | Type14::TwoInDiffRing => in_ring14_type(cx, b2, a1, a2, a3, a4, 0),
        Type14::TwoInSameRing => two_in_same_ring14_type(cx, b2, a1, a2, a3, a4, info.prefer_trans),
    };
    let nb = cx.nb;
    let tol12 = scale_tol_large_sp2(cx, a1, a2, a3);
    let tol23 = scale_tol_large_sp2(cx, a2, a3, a4);
    let (ba_l12, ba_u12) = angle_range(ba12, tol12);
    let (ba_l23, ba_u23) = angle_range(ba23, tol23);
    let extra = tv.extra_dist.unwrap_or(0.0);
    let (mut dl, mut du);
    match tv.kind {
        TorsionType::Cis => {
            dl = compute14_dist_cis(bl_l1, bl_l2, bl_l3, ba_l12, ba_l23) + extra;
            du = compute14_dist_cis(bl_u1, bl_u2, bl_u3, ba_u12, ba_u23) + extra;
            acc.cis_paths.insert(uid3(b1, b2, b3, nb));
        }
        TorsionType::Trans => {
            dl = compute14_dist_trans(bl_l1, bl_l2, bl_l3, ba_l12, ba_l23) + extra;
            du = compute14_dist_trans(bl_u1, bl_u2, bl_u3, ba_u12, ba_u23) + extra;
            acc.trans_paths.insert(uid3(b1, b2, b3, nb));
        }
        TorsionType::CisTrans | TorsionType::Flexible => {
            dl = compute14_dist_cis(bl_l1, bl_l2, bl_l3, ba_l12, ba_l23);
            du = compute14_dist_trans(bl_u1, bl_u2, bl_u3, ba_u12, ba_u23);
            if du < dl {
                std::mem::swap(&mut du, &mut dl);
            }
            if (du - dl).abs() < DIST12_DELTA {
                dl -= GEN_DIST_TOL;
                du += GEN_DIST_TOL;
            }
        }
        TorsionType::Custom => {
            let v = tv.value.unwrap_or(0.0);
            dl = compute14_dist_3d(bl_l1, bl_l2, bl_l3, ba_l12, ba_l23, v);
            du = compute14_dist_3d(bl_u1, bl_u2, bl_u3, ba_u12, ba_u23, v);
        }
    }
    if (du - dl).abs() < DIST12_DELTA {
        dl -= GEN_DIST_TOL;
        du += GEN_DIST_TOL;
    }
    acc.paths14.push(Path14 { bid1: b1, bid2: b2, bid3: b3, aid1: a1, aid2: a2, aid3: a3, aid4: a4, value: tv, type14: ty, ring_size: info.ring_size });
    acc.visited14[pid] = true;
    collected.entry(pid).or_default().push(Bounds14 { lower: dl, upper: du, aid1: a1, aid4: a4 });
}

fn record14_path(cx: &Ctx, b1: usize, b2: usize, b3: usize, acc: &mut ComputedData, rs: usize) {
    let a2 = acc.bond_adj.get(b1, b2) as usize;
    let a3 = acc.bond_adj.get(b2, b3) as usize;
    let a1 = cx.bond_other(b1, a2);
    // NB: RDKit takes the other atom of bid2 here
    let a4 = cx.bond_other(b2, a3);
    let value = if cx.hyb(a2) == Hybridization::SP2 && cx.hyb(a3) == Hybridization::SP2 {
        acc.cis_paths.insert(uid3(b1, b2, b3, cx.nb));
        TorsionValue::of(TorsionType::Cis)
    } else {
        TorsionValue::of(TorsionType::Flexible)
    };
    acc.paths14.push(Path14 { bid1: b1, bid2: b2, bid3: b3, aid1: a1, aid2: a2, aid3: a3, aid4: a4, value, type14: Type14::InRing, ring_size: rs });
}

fn set14_bounds(cx: &Ctx, mmat: &mut BoundsMatrix, acc: &mut ComputedData, dmat: &[f64], use_mc: bool, force_trans: bool) {
    let nb = cx.nb;
    let mut bond_rings = cx.ri.bond_rings.clone();
    bond_rings.sort_by_key(|r| std::cmp::Reverse(r.len()));
    let mut bid_is_macrocycle: HashSet<usize> = HashSet::new();
    let mut ring_bond_pairs = vec![false; nb * nb];
    let mut done_paths: HashSet<usize> = HashSet::new();
    let mut cis_ring_bond_pairs = vec![false; nb * nb];
    let mut collected: BTreeMap<usize, Vec<Bounds14>> = BTreeMap::new();
    for bring in &bond_rings {
        let rs = bring.len();
        if rs < 3 {
            continue;
        }
        let mut bid1 = bring[rs - 1];
        for i in 0..rs {
            let bid2 = bring[i];
            let bid3 = bring[(i + 1) % rs];
            let pid = uid2(bid1, bid2, nb);
            let id = uid3(bid1, bid2, bid3, nb);
            ring_bond_pairs[pid] = true;
            done_paths.insert(id);
            if rs > 5 {
                if use_mc && rs >= MIN_MACROCYCLE_RING_SIZE {
                    collect14(cx, bid1, bid2, bid3, Type14::MacrocycleAllInSameRing, acc, mmat, dmat, Info14 { use_macrocycle14config: use_mc, ..Default::default() }, &mut collected);
                    bid_is_macrocycle.insert(bid2);
                } else {
                    collect14(cx, bid1, bid2, bid3, Type14::InRing, acc, mmat, dmat, Info14 { ring_size: rs, ..Default::default() }, &mut collected);
                    cis_ring_bond_pairs[pid] = rs <= 8;
                }
            } else {
                record14_path(cx, bid1, bid2, bid3, acc, rs);
                cis_ring_bond_pairs[pid] = true;
            }
            bid1 = bid2;
        }
    }
    for (bid2, bond) in cx.mol.bonds().iter().enumerate() {
        let (aid2, aid3) = (bond.begin, bond.end);
        for &(_, bid1) in cx.mol.atom_bonds(aid2) {
            if bid1 == bid2 {
                continue;
            }
            for &(_, bid3) in cx.mol.atom_bonds(aid3) {
                if bid3 == bid2 {
                    continue;
                }
                let id = uid3(bid1, bid2, bid3, nb);
                if done_paths.contains(&id) {
                    continue;
                }
                let pid1 = uid2(bid1, bid2, nb);
                let pid2 = uid2(bid2, bid3, nb);
                let nr = |b: usize| cx.ri.num_bond_rings(b);
                if ring_bond_pairs[pid1] || ring_bond_pairs[pid2] {
                    if use_mc && bid_is_macrocycle.contains(&bid2) {
                        collect14(cx, bid1, bid2, bid3, Type14::MacrocycleTwoInSameRing, acc, mmat, dmat, Info14::default(), &mut collected);
                    } else {
                        let pt = cis_ring_bond_pairs[pid1] || cis_ring_bond_pairs[pid2];
                        collect14(cx, bid1, bid2, bid3, Type14::TwoInSameRing, acc, mmat, dmat, Info14 { prefer_trans: pt, ..Default::default() }, &mut collected);
                    }
                } else if (nr(bid1) > 0 && nr(bid2) > 0) || (nr(bid2) > 0 && nr(bid3) > 0) {
                    collect14(cx, bid1, bid2, bid3, Type14::TwoInDiffRing, acc, mmat, dmat, Info14::default(), &mut collected);
                } else if nr(bid2) > 0 {
                    collect14(cx, bid1, bid2, bid3, Type14::ShareRingBond, acc, mmat, dmat, Info14::default(), &mut collected);
                } else {
                    collect14(cx, bid1, bid2, bid3, Type14::InChain, acc, mmat, dmat, Info14 { force_trans_amides: force_trans, ..Default::default() }, &mut collected);
                }
            }
        }
    }
    for (_, b) in collected {
        let m = merge_bounds(b);
        check_and_set_bounds(m.aid1, m.aid4, m.lower, m.upper, mmat);
    }
}

fn d15_cis_cis(d1: f64, d2: f64, d3: f64, d4: f64, a12: f64, a23: f64, a34: f64) -> f64 {
    let dx14 = d2 - d3 * a23.cos() - d1 * a12.cos();
    let dy14 = d3 * a23.sin() - d1 * a12.sin();
    let d14 = (dx14 * dx14 + dy14 * dy14).sqrt();
    let cval = ((d3 - d2 * a23.cos() + d1 * (a12 + a23).cos()) / d14).clamp(-1.0, 1.0);
    let a143 = cval.acos();
    compute13_dist(d14, d4, a34 - a143)
}
fn d15_cis_trans(d1: f64, d2: f64, d3: f64, d4: f64, a12: f64, a23: f64, a34: f64) -> f64 {
    let dx14 = d2 - d3 * a23.cos() - d1 * a12.cos();
    let dy14 = d3 * a23.sin() - d1 * a12.sin();
    let d14 = (dx14 * dx14 + dy14 * dy14).sqrt();
    let cval = ((d3 - d2 * a23.cos() + d1 * (a12 + a23).cos()) / d14).clamp(-1.0, 1.0);
    let a143 = cval.acos();
    compute13_dist(d14, d4, a34 + a143)
}
fn d15_trans_trans(d1: f64, d2: f64, d3: f64, d4: f64, a12: f64, a23: f64, a34: f64) -> f64 {
    let dx14 = d2 - d3 * a23.cos() - d1 * a12.cos();
    let dy14 = d3 * a23.sin() + d1 * a12.sin();
    let d14 = (dx14 * dx14 + dy14 * dy14).sqrt();
    let cval = ((d3 - d2 * a23.cos() + d1 * (a12 - a23).cos()) / d14).clamp(-1.0, 1.0);
    let a143 = cval.acos();
    compute13_dist(d14, d4, a34 + a143)
}
fn d15_trans_cis(d1: f64, d2: f64, d3: f64, d4: f64, a12: f64, a23: f64, a34: f64) -> f64 {
    let dx14 = d2 - d3 * a23.cos() - d1 * a12.cos();
    let dy14 = d3 * a23.sin() + d1 * a12.sin();
    let d14 = (dx14 * dx14 + dy14 * dy14).sqrt();
    let cval = ((d3 - d2 * a23.cos() + d1 * (a12 - a23).cos()) / d14).clamp(-1.0, 1.0);
    let a143 = cval.acos();
    compute13_dist(d14, d4, a34 - a143)
}

#[allow(clippy::too_many_arguments)]
fn set15_helper(cx: &Ctx, bid1: usize, bid2: usize, bid3: usize, ty: TorsionType, acc: &mut ComputedData, mmat: &mut BoundsMatrix, dmat: &[f64]) {
    let nb = cx.nb;
    let na = cx.na;
    let aid2 = acc.bond_adj.get(bid1, bid2) as usize;
    let aid1 = cx.bond_other(bid1, aid2);
    let aid3 = acc.bond_adj.get(bid2, bid3) as usize;
    let aid4 = cx.bond_other(bid3, aid3);
    let d1 = acc.bond_lengths[bid1];
    let d2 = acc.bond_lengths[bid2];
    let d3 = acc.bond_lengths[bid3];
    let a12 = acc.bond_angles.get(bid1, bid2);
    let a23 = acc.bond_angles.get(bid2, bid3);
    for i in 0..nb {
        let mut du = -1.0;
        let mut dl;
        if acc.bond_adj.get(bid3, i) != aid4 as i64 {
            continue;
        }
        let aid5 = cx.bond_other(i, aid4);
        let pid = uid2(aid1, aid5, na);
        if acc.visited_bound(pid, DistType::D14) {
            return;
        }
        if dmat[aid1.max(aid5) * na + aid1.min(aid5)] < 3.9 {
            continue;
        }
        if aid1 == aid5 {
            continue;
        }
        if mmat.lower(aid1, aid5) < DIST12_DELTA || acc.set15_atoms[pid] {
            let d4 = acc.bond_lengths[i];
            let a34 = acc.bond_angles.get(bid3, i);
            let path_id = uid3(bid2, bid3, i, nb);
            let is_cis = acc.cis_paths.contains(&path_id);
            let is_trans = acc.trans_paths.contains(&path_id);
            match ty {
                TorsionType::Cis => {
                    if is_cis {
                        dl = d15_cis_cis(d1, d2, d3, d4, a12, a23, a34);
                        du = dl + DIST15_TOL;
                        dl -= DIST15_TOL;
                    } else if is_trans {
                        dl = d15_cis_trans(d1, d2, d3, d4, a12, a23, a34);
                        du = dl + DIST15_TOL;
                        dl -= DIST15_TOL;
                    } else {
                        dl = d15_cis_cis(d1, d2, d3, d4, a12, a23, a34) - DIST15_TOL;
                        du = d15_cis_trans(d1, d2, d3, d4, a12, a23, a34) + DIST15_TOL;
                    }
                }
                TorsionType::Trans => {
                    if is_cis {
                        dl = d15_trans_cis(d1, d2, d3, d4, a12, a23, a34);
                        du = dl + DIST15_TOL;
                        dl -= DIST15_TOL;
                    } else if is_trans {
                        dl = d15_trans_trans(d1, d2, d3, d4, a12, a23, a34);
                        du = dl + DIST15_TOL;
                        dl -= DIST15_TOL;
                    } else {
                        dl = d15_trans_cis(d1, d2, d3, d4, a12, a23, a34) - DIST15_TOL;
                        du = d15_trans_trans(d1, d2, d3, d4, a12, a23, a34) + DIST15_TOL;
                    }
                }
                _ => {
                    if is_cis {
                        dl = d15_cis_cis(d4, d3, d2, d1, a34, a23, a12) - DIST15_TOL;
                        du = d15_cis_trans(d4, d3, d2, d1, a34, a23, a12) + DIST15_TOL;
                    } else if is_trans {
                        dl = d15_trans_cis(d4, d3, d2, d1, a34, a23, a12) - DIST15_TOL;
                        du = d15_trans_trans(d4, d3, d2, d1, a34, a23, a12) + DIST15_TOL;
                    } else {
                        let vw1 = element(cx.anum(aid1)).r_vdw;
                        let vw5 = element(cx.anum(aid5)).r_vdw;
                        dl = VDW_SCALE_15 * (vw1 + vw5);
                    }
                }
            }
            if du < 0.0 {
                du = MAX_UPPER;
            }
            check_and_set_bounds(aid1, aid5, dl, du, mmat);
            acc.set15_atoms[pid] = true;
        }
    }
}

fn set15_bounds(cx: &Ctx, mmat: &mut BoundsMatrix, acc: &mut ComputedData, dmat: &[f64]) {
    let paths: Vec<(usize, usize, usize, TorsionType)> = acc.paths14.iter().map(|p| (p.bid1, p.bid2, p.bid3, p.value.kind)).collect();
    for (b1, b2, b3, ty) in paths {
        set15_helper(cx, b1, b2, b3, ty, acc, mmat, dmat);
        set15_helper(cx, b3, b2, b1, ty, acc, mmat, dmat);
    }
}

fn set_lower_bound_vdw(cx: &Ctx, mmat: &mut BoundsMatrix, dmat: &[f64]) {
    let npt = cx.na;
    let mut h_donor = vec![false; npt];
    let mut acceptor = vec![false; npt];
    for i in 1..npt {
        let ai = cx.anum(i);
        let vw1 = element(ai).r_vdw;
        if ai == 1 && cx.mol.neighbors(i).any(|n| matches!(cx.anum(n), 7 | 8)) {
            h_donor[i] = true;
        }
        if ai == 7 || ai == 8 {
            acceptor[i] = true;
        }
        for j in 0..i {
            let vw2 = element(cx.anum(j)).r_vdw;
            if mmat.lower(i, j) < DIST12_DELTA {
                if (h_donor[i] && acceptor[j]) || (acceptor[i] && h_donor[j]) {
                    mmat.set_lower(i, j, H_BOND_LENGTH);
                } else if dmat[i * npt + j] == 4.0 {
                    mmat.set_lower(i, j, VDW_SCALE_15 * (vw1 + vw2));
                } else if dmat[i * npt + j] == 5.0 {
                    mmat.set_lower(i, j, (VDW_SCALE_15 + 0.5 * (1.0 - VDW_SCALE_15)) * (vw1 + vw2));
                } else {
                    mmat.set_lower(i, j, vw1 + vw2);
                }
            }
        }
    }
}

/// Topological distance matrix as doubles (RDKit `MolOps::getDistanceMat`,
/// disconnected pairs get 1e8).
pub fn distance_matrix(mol: &Molecule) -> Vec<f64> {
    let n = mol.num_atoms();
    let td = mol.topological_distances();
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = if td[i][j] == usize::MAX { 1e8 } else { td[i][j] as f64 };
        }
    }
    out
}

fn ring_info(mol: &Molecule) -> std::borrow::Cow<'_, RingInfo> {
    match mol.ring_info() {
        Some(r) => std::borrow::Cow::Borrowed(r),
        None => std::borrow::Cow::Owned(RingInfo::compute(mol)),
    }
}

/// RDKit `setTopolBounds` (UFF 1-2 distances). Returns the recorded 1-4
/// paths (RDKit `paths14`).
pub fn set_topol_bounds(mol: &Molecule, mmat: &mut BoundsMatrix, opts: &TopolOptions) -> Vec<Path14> {
    let na = mol.num_atoms();
    let nb = mol.num_bonds();
    let ri = ring_info(mol);
    let cx = Ctx { mol, ri: &ri, na, nb };
    let mut acc = ComputedData::new(na, nb);
    let dmat = distance_matrix(mol);
    set12_bounds(&cx, mmat, &mut acc);
    if opts.set13bounds {
        set13_bounds(&cx, mmat, &mut acc);
        if opts.set14bounds {
            set14_bounds(&cx, mmat, &mut acc, &dmat, opts.use_macrocycle14config, opts.force_trans_amides);
            if opts.set15bounds {
                set15_bounds(&cx, mmat, &mut acc, &dmat);
            }
        }
    }
    let _ = opts.scale_vdw; // unused by RDKit as well
    set_lower_bound_vdw(&cx, mmat, &dmat);
    acc.paths14
}

/// RDKit `collectBondsAndAngles`: bonds and angles `[a1, a2(center), a3, flag]`
/// where flag = 1 for linear (triple bond or cumulated double bonds).
pub fn collect_bonds_and_angles(mol: &Molecule) -> (Vec<(usize, usize)>, Vec<[usize; 4]>) {
    let nb = mol.num_bonds();
    let mut bonds = Vec::with_capacity(nb);
    let mut angles = Vec::new();
    for i in 0..nb {
        let bi = mol.bond(i);
        bonds.push((bi.begin, bi.end));
        for j in i + 1..nb {
            let bj = mol.bond(j);
            let (a11, a12, a21, a22) = (bi.begin, bi.end, bj.begin, bj.end);
            if a11 != a21 && a11 != a22 && a12 != a21 && a12 != a22 {
                continue;
            }
            let mut t = [0usize; 4];
            if a12 == a21 {
                t[..3].copy_from_slice(&[a11, a12, a22]);
            } else if a12 == a22 {
                t[..3].copy_from_slice(&[a11, a12, a21]);
            } else if a11 == a21 {
                t[..3].copy_from_slice(&[a12, a11, a22]);
            } else if a11 == a22 {
                t[..3].copy_from_slice(&[a12, a11, a21]);
            }
            // triple bond, or consecutive double bonds (allene-like center)
            if bi.order == BondOrder::Triple
                || bj.order == BondOrder::Triple
                || (bi.order == BondOrder::Double && bj.order == BondOrder::Double && mol.degree(t[1]) == 2)
            {
                t[3] = 1;
            }
            angles.push(t);
        }
    }
    (bonds, angles)
}
