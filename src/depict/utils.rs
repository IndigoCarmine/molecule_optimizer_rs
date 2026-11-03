//! Helpers of RDKit's depictor (`Code/GraphMol/Depictor/DepictUtils.{h,cpp}`
//! and the bits of `RDGeneral/utils.cpp`, `GraphMol/Matrices.cpp` and
//! `GraphMol/Aromaticity.cpp` it relies on).

use super::geom::{Point2D, Transform2D};
use crate::core::{Hybridization, Molecule};
use crate::perception::rings::RingInfo;
use std::f64::consts::PI;

pub const COLLISION_THRES: f64 = 0.70;
pub const ANGLE_OPEN: f64 = 0.1222; // about 7 deg
pub const MAX_COLL_ITERS: u32 = 15;
pub const HETEROATOM_COLL_SCALE: f64 = 1.3;
pub const NUM_BONDS_FLIPS: u32 = 3;
pub const ANGLE_EXPANSION_INCREMENT: f64 = 0.0873; // 5 degrees per step
pub const MAX_ANGLE_EXPANSION: f64 = 3. * ANGLE_EXPANSION_INCREMENT;
pub const MAX_ANGLE_EXPANSION_ITERS: u32 = 20;
pub const NEIGH_RADIUS: f64 = 2.5;

/// Everything about the molecule the depictor needs.
pub struct Ctx<'a> {
    pub mol: &'a Molecule,
    /// rings as found by `symmetrizeSSSR(includeDativeBonds = true)`
    pub ri: RingInfo,
    /// RDKit `_CIPRank` atom property, if set
    pub cip: Option<Vec<u32>>,
    pub bond_len: f64,
}

impl Ctx<'_> {
    pub fn n(&self) -> usize {
        self.mol.num_atoms()
    }
    pub fn degree(&self, a: usize) -> usize {
        self.mol.degree(a)
    }
    pub fn anum(&self, a: usize) -> u8 {
        self.mol.atom(a).atomic_num
    }
    pub fn num_atom_rings(&self, a: usize) -> usize {
        self.ri.num_atom_rings(a)
    }
    pub fn num_bond_rings(&self, b: usize) -> usize {
        self.ri.num_bond_rings(b)
    }
}

/// `getAtomDepictRank`.
pub fn atom_depict_rank(mol: &Molecule, a: usize) -> i64 {
    const MAX_AT_NUM: i64 = 1000;
    const MAX_DEG: i64 = 100;
    let anum = mol.atom(a).atomic_num as i64;
    let anum = if anum == 1 { MAX_AT_NUM } else { anum };
    MAX_DEG * anum + mol.degree(a) as i64
}

/// `rankAtomsByRank(mol, atoms, ascending = true)`.
pub fn rank_atoms_by_rank(ctx: &Ctx, atoms: &[usize]) -> Vec<usize> {
    let n = ctx.n() as i64;
    let mut ra: Vec<(i64, usize)> = atoms
        .iter()
        .map(|&aid| {
            let rank = match &ctx.cip {
                Some(c) => c[aid] as i64,
                None => aid as i64 + n * atom_depict_rank(ctx.mol, aid),
            };
            (rank, aid)
        })
        .collect();
    ra.sort();
    ra.into_iter().map(|x| x.1).collect()
}

/// `embedRing`: regular polygon around the origin, first atom on +x.
pub fn embed_ring(ring: &[usize], bond_len: f64) -> Vec<(usize, Point2D)> {
    let na = ring.len();
    let ang = 2.0 * PI / na as f64;
    let al = bond_len / (2.0 * (1.0 - ang.cos())).sqrt();
    ring.iter()
        .enumerate()
        .map(|(i, &a)| {
            let x = al * (i as f64 * ang).cos();
            let y = al * (i as f64 * ang).sin();
            (a, Point2D::new(x, y))
        })
        .collect()
}

/// `computeBisectPoint`.
pub fn compute_bisect_point(rcr: Point2D, ang: f64, nb1: Point2D, nb2: Point2D) -> Point2D {
    let mut cloc = nb1;
    cloc += nb2;
    cloc *= 0.5;
    if ang > PI {
        cloc -= rcr;
        cloc *= -1.0;
        cloc += rcr;
    }
    cloc
}

/// `reflectPoint`: mirror `point` through the line `loc1`–`loc2`.
pub fn reflect_point(point: Point2D, loc1: Point2D, loc2: Point2D) -> Point2D {
    let org = Point2D::new(0.0, 0.0);
    let xaxis = Point2D::new(1.0, 0.0);
    let mut cent = loc1 + loc2;
    cent *= 0.5;
    let trans = Transform2D::align(org, xaxis, cent, loc1);
    let itrans = Transform2D::align(cent, loc1, org, xaxis);
    let mut res = trans.apply(point);
    res.y = -res.y;
    itrans.apply(res)
}

/// `setNbrOrder`.
pub fn set_nbr_order(ctx: &Ctx, aid: usize, nbrs: &[usize]) -> Vec<usize> {
    let mut reference: Option<usize> = None;
    for n in ctx.mol.neighbors(aid) {
        if !nbrs.contains(&n) {
            reference = Some(n);
        }
    }
    let mut thold = nbrs.to_vec();
    if let Some(r) = reference {
        thold.push(r);
    }
    debug_assert!(thold.len() > 3);
    let mut thold = rank_atoms_by_rank(ctx, &thold);
    let ln = thold.len();
    thold.swap(ln - 3, ln - 2);
    let mut res = Vec::with_capacity(thold.len());
    match reference.and_then(|r| thold.iter().position(|&x| x == r)) {
        Some(pos) => {
            res.extend_from_slice(&thold[pos + 1..]);
            res.extend_from_slice(&thold[..pos]);
        }
        None => res.extend_from_slice(&thold),
    }
    res
}

/// `pickFirstRingToEmbed`: fewest substituted atoms, then the largest ring.
pub fn pick_first_ring_to_embed(mol: &Molecule, fused: &[Vec<usize>]) -> usize {
    let mut res = 0;
    let mut max_size = 0;
    let mut minsubs = 100_000_000usize;
    for (cnt, ring) in fused.iter().enumerate() {
        let subs = ring.iter().filter(|&&a| mol.degree(a) > 2).count();
        if subs < minsubs {
            res = cnt;
            minsubs = subs;
            max_size = ring.len();
        } else if subs == minsubs && ring.len() > max_size {
            res = cnt;
            max_size = ring.len();
        }
    }
    res
}

/// `findNextRingToEmbed`: returns (next ring id, common atoms).
pub fn find_next_ring_to_embed(done_rings: &[usize], fused: &[Vec<usize>]) -> (usize, Vec<usize>) {
    let mut done_atoms = Vec::new();
    for (i, r) in fused.iter().enumerate() {
        if done_rings.contains(&i) {
            for &a in r {
                if !done_atoms.contains(&a) {
                    done_atoms.push(a);
                }
            }
        }
    }
    let mut max_common = 0;
    let mut next_id = 0;
    let mut res: Vec<usize> = Vec::new();
    for (rid, ring) in fused.iter().enumerate() {
        if done_rings.contains(&rid) {
            continue;
        }
        let common: Vec<usize> = ring.iter().copied().filter(|a| done_atoms.contains(a)).collect();
        if common.len() == 2 {
            return (rid, common);
        }
        if common.len() > max_common {
            max_common = common.len();
            next_id = rid;
            res = common;
        }
    }
    // make the common atoms a contiguous chain in ring order
    let n_cmn = res.len();
    let mut cmn_lst = 0;
    for i in 0..n_cmn {
        if res[i] == fused[next_id][i] {
            cmn_lst += 1;
        } else {
            break;
        }
    }
    if cmn_lst > 0 && cmn_lst < n_cmn {
        res.rotate_left(cmn_lst);
    }
    (next_id, res)
}

/// `getAllRotatableBonds`.
pub fn all_rotatable_bonds(ctx: &Ctx) -> Vec<usize> {
    (0..ctx.mol.num_bonds()).filter(|&b| !has_stereo(ctx.mol, b) && ctx.num_bond_rings(b) == 0).collect()
}

/// Bond stereo above `STEREOANY`.
pub fn has_stereo(mol: &Molecule, b: usize) -> bool {
    mol.bond(b).stereo != crate::core::BondStereo::None
}

/// `getRotatableBonds`: flippable bonds on the shortest path between two
/// atoms, excluding the two terminal bonds.
pub fn rotatable_bonds_between(ctx: &Ctx, aid1: usize, aid2: usize) -> Vec<usize> {
    let path = shortest_path(ctx.mol, aid1, aid2);
    let mut res = Vec::new();
    if path.len() >= 4 {
        let inner = &path[1..path.len() - 1];
        for w in inner.windows(2) {
            let b = ctx.mol.bond_between(w[0], w[1]).expect("path bond");
            if !has_stereo(ctx.mol, b) && ctx.num_bond_rings(b) == 0 {
                res.push(b);
            }
        }
    }
    res
}

/// `isSpiroCenter`.
pub fn is_spiro_center(ctx: &Ctx, aid: usize) -> bool {
    if ctx.degree(aid) != 4 || ctx.num_atom_rings(aid) != 2 {
        return false;
    }
    let rings: Vec<&Vec<usize>> = ctx.ri.atom_rings.iter().filter(|r| r.contains(&aid)).collect();
    if rings.len() != 2 {
        return false;
    }
    let shared = rings[0].iter().filter(|a| rings[1].contains(a)).count();
    if shared != 1 {
        return false;
    }
    let (mut n1, mut n2) = (0, 0);
    for nbr in ctx.mol.neighbors(aid) {
        let in1 = rings[0].contains(&nbr);
        let in2 = rings[1].contains(&nbr);
        if in1 && !in2 {
            n1 += 1;
        } else if in2 && !in1 {
            n2 += 1;
        } else {
            return false;
        }
    }
    n1 == 2 && n2 == 2
}

/// `findBondsPairsToPermuteDeg4`.
pub fn bond_pairs_to_permute_deg4(center: Point2D, nbr_bids: &[usize], nbr_locs: &[Point2D]) -> [(usize, usize); 2] {
    let pts: Vec<Point2D> = nbr_locs.iter().map(|&l| l - center).collect();
    let dp1 = pts[0].dot(pts[1]);
    if dp1.abs() < 1.0e-3 {
        let dp2 = pts[0].dot(pts[2]);
        if dp2.abs() < 1.0e-3 {
            [(nbr_bids[0], nbr_bids[1]), (nbr_bids[0], nbr_bids[2])]
        } else {
            [(nbr_bids[0], nbr_bids[1]), (nbr_bids[0], nbr_bids[3])]
        }
    } else {
        [(nbr_bids[0], nbr_bids[2]), (nbr_bids[0], nbr_bids[3])]
    }
}

/// `computeSubAngle`.
pub fn compute_sub_angle(degree: usize, h: Hybridization) -> f64 {
    match h {
        Hybridization::Unspecified | Hybridization::SP3 => {
            if degree == 4 {
                PI / 2.0
            } else {
                2.0 * PI / 3.0
            }
        }
        Hybridization::SP2 => 2.0 * PI / 3.0,
        _ => 2.0 * PI / degree as f64,
    }
}

/// `rotationDir`.
pub fn rotation_dir(center: Point2D, loc1: Point2D, loc2: Point2D, rem_angle: f64) -> i32 {
    let pt1 = loc1 - center;
    let pt2 = loc2 - center;
    let mut cross = pt1.x * pt2.y - pt1.y * pt2.x;
    let diff = PI - rem_angle;
    cross *= diff;
    if cross >= 0.0 { -1 } else { 1 }
}

/// `computeNormal`.
pub fn compute_normal(center: Point2D, other: Point2D) -> Point2D {
    let mut res = other - center;
    res.normalize();
    Point2D::new(-res.y, res.x)
}

/// `computeAngle`.
pub fn compute_angle(center: Point2D, loc1: Point2D, loc2: Point2D) -> f64 {
    (loc1 - center).angle_to(loc2 - center)
}

/// `MolOps::getShortestPath`: atoms from `a1` to `a2` inclusive (BFS in
/// neighbor order); empty if not connected.
pub fn shortest_path(mol: &Molecule, a1: usize, a2: usize) -> Vec<usize> {
    if a1 == a2 {
        return Vec::new();
    }
    const UNSEEN: isize = -1;
    const BEGIN: isize = -2;
    const END: isize = -3;
    let n = mol.num_atoms();
    let mut pred = vec![UNSEEN; n];
    pred[a1] = BEGIN;
    pred[a2] = END;
    let mut q = std::collections::VecDeque::new();
    q.push_back(a1);
    let mut done = false;
    while !done {
        let Some(&cur) = q.front() else { break };
        for nbr in mol.neighbors(cur) {
            match pred[nbr] {
                UNSEEN => {
                    pred[nbr] = cur as isize;
                    q.push_back(nbr);
                }
                END => {
                    pred[nbr] = cur as isize;
                    done = true;
                    break;
                }
                _ => {}
            }
        }
        q.pop_front();
    }
    let mut res = Vec::new();
    if done {
        let mut prev = a2;
        res.push(a2);
        loop {
            prev = pred[prev] as usize;
            if prev == a1 {
                break;
            }
            res.push(prev);
        }
        res.push(a1);
        res.reverse();
    }
    res
}

/// `MolOps::getDistanceMat` (topological, row major); disconnected pairs get
/// a large value.
pub fn distance_matrix(mol: &Molecule) -> Vec<f64> {
    let n = mol.num_atoms();
    let mut res = vec![0.0; n * n];
    for (i, row) in mol.topological_distances().into_iter().enumerate() {
        for (j, d) in row.into_iter().enumerate() {
            res[i * n + j] = if d == usize::MAX { 1.0e8 } else { d as f64 };
        }
    }
    res
}

/// `RingUtils::makeRingNeighborMap` (no size limits).
pub fn ring_neighbor_map(rings: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = rings.len();
    let mut map = vec![Vec::new(); n];
    for i in 0..n {
        for j in i + 1..n {
            if rings[i].iter().any(|a| rings[j].contains(a)) {
                map[i].push(j);
                map[j].push(i);
            }
        }
    }
    map
}

/// `RingUtils::pickFusedRings`: preorder DFS over the ring neighbor map.
pub fn pick_fused_rings(curr: usize, map: &[Vec<usize>], done: &mut [bool]) -> Vec<usize> {
    let mut res = vec![curr];
    done[curr] = true;
    let mut stack: Vec<(usize, usize)> = vec![(curr, 0)];
    while let Some(top) = stack.last_mut() {
        let (ring, next) = *top;
        if next == map[ring].len() {
            stack.pop();
            continue;
        }
        top.1 += 1;
        let neigh = map[ring][next];
        if done[neigh] {
            continue;
        }
        done[neigh] = true;
        res.push(neigh);
        stack.push((neigh, 0));
    }
    res
}

/// `boost::minstd_rand` (RDKit's `rng_type`).
#[derive(Debug, Clone)]
pub struct MinStd {
    x: u64,
}

impl MinStd {
    const M: u64 = 2_147_483_647;
    pub fn new(seed: u64) -> Self {
        let mut x = seed % Self::M;
        if x == 0 {
            x = 1;
        }
        MinStd { x }
    }
    pub fn next_u32(&mut self) -> u32 {
        self.x = self.x * 48271 % Self::M;
        self.x as u32
    }
    /// `boost::uniform_int<int>(lo, hi)`.
    pub fn uniform_int(&mut self, lo: u32, hi: u32) -> u32 {
        let range = (hi - lo) as u64;
        if range == 0 {
            return lo;
        }
        let bmin = 1u64;
        let brange = Self::M - 2; // max - min
        let bucket = (brange + 1) / (range + 1);
        loop {
            let r = (self.next_u32() as u64 - bmin) / bucket;
            if r <= range {
                return lo + r as u32;
            }
        }
    }
    /// `boost::uniform_real<double>(0, 1)`.
    pub fn uniform_01(&mut self) -> f64 {
        loop {
            let num = (self.next_u32() as u64 - 1) as f64;
            let div = (Self::M - 2) as f64 + 1.0;
            let r = num / div;
            if r < 1.0 {
                return r;
            }
        }
    }
}
