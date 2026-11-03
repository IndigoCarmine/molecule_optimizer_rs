//! Port of RDKit's `EmbeddedAtom` / `EmbeddedFrag`
//! (`Code/GraphMol/Depictor/EmbeddedFrag.{h,cpp}`): pieces of a molecule
//! with 2D coordinates that grow by adding atoms and merging with each other
//! until each connected component is one fragment, followed by the
//! collision-removal passes.

use super::geom::{Point2D, Transform2D};
use super::utils::*;
use crate::core::{BondOrder, BondStereo};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::f64::consts::PI;

/// `EmbeddedAtom`.
#[derive(Debug, Clone)]
pub struct EAtom {
    /// angle already used up around the atom (-1: none)
    pub angle: f64,
    pub nbr1: isize,
    pub nbr2: isize,
    pub cis_trans_nbr: isize,
    pub ccw: bool,
    pub rot_dir: i32,
    pub loc: Point2D,
    /// direction in which the next neighbor is added
    pub normal: Point2D,
    /// neighbors not embedded yet
    pub neighs: Vec<usize>,
    pub density: f64,
    /// coordinates given by the user
    pub fixed: bool,
}

impl Default for EAtom {
    fn default() -> Self {
        EAtom {
            angle: -1.0,
            nbr1: -1,
            nbr2: -1,
            cis_trans_nbr: -1,
            ccw: true,
            rot_dir: 0,
            loc: Point2D::default(),
            normal: Point2D::default(),
            neighs: Vec::new(),
            density: -1.0,
            fixed: false,
        }
    }
}

impl EAtom {
    fn at(loc: Point2D) -> Self {
        EAtom { loc, ..Default::default() }
    }
    fn transform(&mut self, t: &Transform2D) {
        let temp = t.apply(self.loc + self.normal);
        self.loc = t.apply(self.loc);
        self.normal = temp - self.loc;
    }
    fn reflect(&mut self, l1: Point2D, l2: Point2D) {
        let temp = reflect_point(self.loc + self.normal, l1, l2);
        self.loc = reflect_point(self.loc, l1, l2);
        self.normal = temp - self.loc;
        self.ccw = !self.ccw;
    }
}

/// Fragment list as RDKit's `std::list<EmbeddedFrag>`: removed entries
/// become `None`, so positions (and the relative order) never change.
pub type FragList = Vec<Option<EmbeddedFrag>>;

/// `EmbeddedFrag`.
#[derive(Debug, Clone, Default)]
pub struct EmbeddedFrag {
    pub done: bool,
    px: f64,
    nx: f64,
    py: f64,
    ny: f64,
    pub eatoms: BTreeMap<usize, EAtom>,
    attach: VecDeque<usize>,
}

/// `_recurseAtomOneSide`: atoms reachable from `end` without passing `beg`.
fn atoms_one_side(ctx: &Ctx, end: usize, beg: usize) -> Vec<usize> {
    let mut seen = vec![false; ctx.n()];
    let mut out = Vec::new();
    let mut stack = vec![end];
    seen[end] = true;
    while let Some(a) = stack.pop() {
        out.push(a);
        for nbr in ctx.mol.neighbors(a) {
            if nbr != beg && !seen[nbr] {
                seen[nbr] = true;
                stack.push(nbr);
            }
        }
    }
    out
}

fn cross_val(v1: Point2D, v2: Point2D) -> f64 {
    v1.x * v2.y - v2.x * v1.y
}

const SEG_EPS: f64 = 1e-6;

fn orientation(a: Point2D, b: Point2D, c: Point2D) -> f64 {
    cross_val(b - a, c - a)
}

fn point_on_segment(a: Point2D, b: Point2D, p: Point2D) -> bool {
    orientation(a, b, p).abs() <= SEG_EPS
        && p.x >= a.x.min(b.x) - SEG_EPS
        && p.x <= a.x.max(b.x) + SEG_EPS
        && p.y >= a.y.min(b.y) - SEG_EPS
        && p.y <= a.y.max(b.y) + SEG_EPS
}

fn segments_intersect(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> bool {
    if a.x.max(b.x) + SEG_EPS < c.x.min(d.x)
        || c.x.max(d.x) + SEG_EPS < a.x.min(b.x)
        || a.y.max(b.y) + SEG_EPS < c.y.min(d.y)
        || c.y.max(d.y) + SEG_EPS < a.y.min(b.y)
    {
        return false;
    }
    let o1 = orientation(a, b, c);
    let o2 = orientation(a, b, d);
    let o3 = orientation(c, d, a);
    let o4 = orientation(c, d, b);
    let first = (o1 > SEG_EPS && o2 < -SEG_EPS) || (o1 < -SEG_EPS && o2 > SEG_EPS);
    let second = (o3 > SEG_EPS && o4 < -SEG_EPS) || (o3 < -SEG_EPS && o4 > SEG_EPS);
    if first && second {
        return true;
    }
    point_on_segment(a, b, c) || point_on_segment(a, b, d) || point_on_segment(c, d, a) || point_on_segment(c, d, b)
}

/// `_findFarthestPair`.
fn farthest_pair(n: usize, beg1: usize, end1: usize, beg2: usize, end2: usize, dmat: &[f64]) -> (usize, usize) {
    let cands = [
        (dmat[beg1 * n + beg2], (beg1, beg2)),
        (dmat[beg1 * n + end2], (beg1, end2)),
        (dmat[end1 * n + beg2], (end1, beg2)),
        (dmat[end1 * n + end2], (end1, end2)),
    ];
    let mut best = cands[0];
    for c in &cands[1..] {
        // std::max keeps the first of equal elements
        if best.0 < c.0 {
            best = *c;
        }
    }
    best.1
}

/// `_findDeg1Neighbor`.
fn deg1_neighbor(ctx: &Ctx, aid: usize) -> usize {
    ctx.mol.neighbors(aid).next().expect("degree-one atom")
}

/// `_findClosestNeighbor`: neighbor of `aid2` topologically closest to `aid1`.
fn closest_neighbor(ctx: &Ctx, dmat: &[f64], aid1: usize, aid2: usize) -> usize {
    let mut res = 0;
    let mut mdist = 1.0e8;
    for nbr in ctx.mol.neighbors(aid2) {
        let d = dmat[aid1 * ctx.n() + nbr];
        if d < mdist {
            mdist = d;
            res = nbr;
        }
    }
    res
}

/// `_recurseDegTwoRingAtoms`.
fn deg_two_ring_atoms(ctx: &Ctx, aid: usize, path: &mut Vec<usize>, nbr_map: &mut HashMap<usize, Vec<usize>>) {
    let nbrs: Vec<usize> =
        ctx.mol.atom_bonds(aid).iter().filter(|&&(_, b)| ctx.num_bond_rings(b) > 0).map(|&(o, _)| o).collect();
    if nbrs.len() != 2 {
        return;
    }
    path.push(aid);
    nbr_map.insert(aid, nbrs.clone());
    for nbr in nbrs {
        if !path.contains(&nbr) {
            deg_two_ring_atoms(ctx, nbr, path, nbr_map);
        }
    }
}

/// `_anyNonRingBonds`.
fn non_ring_bonds_on_path(ctx: &Ctx, aid: usize, path: &[usize]) -> usize {
    let mut prev = aid;
    let mut n_open = 0;
    for &p in path {
        let b = ctx.mol.bond_between(prev, p).expect("path bond");
        if ctx.num_bond_rings(b) == 0 {
            n_open += 1;
        }
        prev = p;
    }
    n_open
}

/// `mirrorTransRingAtoms`: atoms of trans ring double bonds are mirrored
/// into the ring.
fn mirror_trans_ring_atoms(ctx: &Ctx, ring: &[usize], coords: &mut BTreeMap<usize, Point2D>) {
    let n = ring.len();
    for i in 0..n {
        let atom1 = ring[i];
        let atom2 = ring[(i + 1) % n];
        let Some(b) = ctx.mol.bond_between(atom1, atom2) else { continue };
        let bond = ctx.mol.bond(b);
        if bond.order != BondOrder::Double || bond.stereo == BondStereo::None {
            continue;
        }
        let Some((n0, n1)) = bond.stereo_atoms else { continue };
        let left_in = ring.contains(&n0);
        let right_in = ring.contains(&n1);
        let is_trans = if bond.stereo == BondStereo::Trans { left_in == right_in } else { left_in != right_in };
        if !is_trans {
            continue;
        }
        let left = ring[(i + n - 1) % n];
        let right = atom2;
        let last = coords[&left];
        let rf = coords[&right];
        let interest = coords[&atom1];
        let d = last - rf;
        let a = (d.x * d.x - d.y * d.y) / d.dot(d);
        let bb = 2.0 * d.x * d.y / d.dot(d);
        let x = a * (interest.x - rf.x) + bb * (interest.y - rf.y) + rf.x;
        let y = bb * (interest.x - rf.x) - a * (interest.y - rf.y) + rf.y;
        coords.insert(atom1, Point2D::new(x, y));
    }
}

impl EmbeddedFrag {
    // ------------------------------------------------------------------
    // constructors
    // ------------------------------------------------------------------

    /// A single atom at the origin.
    pub fn single(ctx: &Ctx, aid: usize) -> Self {
        let mut f = EmbeddedFrag::default();
        let e = EAtom { normal: Point2D::new(1.0, 0.0), ..Default::default() };
        f.eatoms.insert(aid, e);
        f.update_new_neighs(ctx, aid);
        f
    }

    /// User-specified coordinates.
    pub fn from_coord_map(ctx: &Ctx, map: &BTreeMap<usize, Point2D>) -> Self {
        let mut f = EmbeddedFrag::default();
        for (&aid, &p) in map {
            let mut e = EAtom::at(p);
            e.fixed = true;
            f.eatoms.insert(aid, e);
        }
        f.setup_new_neighs(ctx);
        f.setup_attachment_points(ctx);
        f
    }

    /// A fused ring system.
    pub fn fused_rings(ctx: &Ctx, rings: &[Vec<usize>]) -> Self {
        let mut f = EmbeddedFrag::default();
        f.embed_fused_rings(ctx, rings);
        f
    }

    /// The two atoms of a stereo double bond.
    pub fn cis_trans(ctx: &Ctx, bond: usize) -> Self {
        let b = ctx.mol.bond(bond);
        let (s0, s1) = b.stereo_atoms.expect("stereo atoms");
        let mut f = EmbeddedFrag::default();
        let beg = EAtom {
            loc: Point2D::new(0.0, 0.0),
            nbr1: b.end as isize,
            normal: Point2D::new(0.0, -1.0),
            ccw: false,
            cis_trans_nbr: s0 as isize,
            ..Default::default()
        };
        f.eatoms.insert(b.begin, beg);
        let cis = b.stereo == BondStereo::Cis;
        let end = EAtom {
            loc: Point2D::new(ctx.bond_len, 0.0),
            nbr1: b.begin as isize,
            cis_trans_nbr: s1 as isize,
            normal: if cis { Point2D::new(0.0, -1.0) } else { Point2D::new(0.0, 1.0) },
            ccw: cis,
            ..Default::default()
        };
        f.eatoms.insert(b.end, end);
        f
    }

    pub fn size(&self) -> usize {
        self.eatoms.len()
    }

    fn loc(&self, aid: isize) -> Point2D {
        self.eatoms.get(&(aid as usize)).map(|e| e.loc).unwrap_or_default()
    }

    // ------------------------------------------------------------------
    // neighbor bookkeeping
    // ------------------------------------------------------------------

    /// `findNumNeigh`.
    fn num_neigh(&self, pt: Point2D, radius: f64) -> usize {
        self.eatoms.values().filter(|e| (e.loc - pt).length() < radius).count()
    }

    /// `updateNewNeighs`.
    fn update_new_neighs(&mut self, ctx: &Ctx, aid: usize) {
        let mut heavy = Vec::new();
        let mut hs = Vec::new();
        for nbr in ctx.mol.neighbors(aid) {
            if !self.eatoms.contains_key(&nbr) {
                if ctx.anum(nbr) != 1 {
                    heavy.push(nbr);
                } else {
                    hs.push(nbr);
                }
            }
        }
        heavy.extend(hs);
        let mut neighs = heavy;
        let deg = ctx.degree(aid);
        if !neighs.is_empty() && (deg < 4 || neighs.len() < 3) {
            neighs = rank_atoms_by_rank(ctx, &neighs);
        } else if deg >= 4 && neighs.len() >= 3 {
            neighs = set_nbr_order(ctx, aid, &neighs);
        }
        let has = !neighs.is_empty();
        self.eatoms.get_mut(&aid).unwrap().neighs = neighs;
        if has && !self.attach.contains(&aid) {
            self.attach.push_back(aid);
        }
    }

    /// `setupNewNeighs`.
    pub fn setup_new_neighs(&mut self, ctx: &Ctx) {
        self.attach.clear();
        let keys: Vec<usize> = self.eatoms.keys().copied().collect();
        for k in keys {
            self.update_new_neighs(ctx, k);
        }
        let ranked = rank_atoms_by_rank(ctx, &self.attach.iter().copied().collect::<Vec<_>>());
        self.attach = ranked.into();
    }

    /// `setupAttachmentPoints`.
    fn setup_attachment_points(&mut self, ctx: &Ctx) {
        let attach: Vec<usize> = self.attach.iter().copied().collect();
        for dai in attach {
            let enbrs = &self.eatoms[&dai].neighs;
            let done: Vec<usize> = ctx.mol.neighbors(dai).filter(|n| !enbrs.contains(n)).collect();
            match done.len() {
                0 => {
                    let e = self.eatoms.get_mut(&dai).unwrap();
                    e.normal = Point2D::new(1.0, 0.0);
                    e.angle = -1.0;
                }
                1 => {
                    let nb = done[0];
                    let nloc = self.eatoms[&nb].loc;
                    let e = self.eatoms.get_mut(&dai).unwrap();
                    e.nbr1 = nb as isize;
                    e.normal = compute_normal(e.loc, nloc);
                }
                2 => {
                    let (l1, l2) = (self.eatoms[&done[0]].loc, self.eatoms[&done[1]].loc);
                    let e = self.eatoms.get_mut(&dai).unwrap();
                    e.nbr1 = done[0] as isize;
                    e.nbr2 = done[1] as isize;
                    e.angle = compute_angle(e.loc, l1, l2);
                }
                _ => self.compute_nbrs_and_ang(ctx, dai, &done),
            }
        }
    }

    /// `computeNbrsAndAng`: picks the gap new neighbors go into.
    fn compute_nbrs_and_ang(&mut self, ctx: &Ctx, aid: usize, done: &[usize]) {
        let c = self.eatoms[&aid].loc;
        let mut dirs: Vec<(f64, usize)> = done
            .iter()
            .map(|&n| {
                let d = self.eatoms[&n].loc - c;
                (d.y.atan2(d.x), n)
            })
            .collect();
        dirs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        let mut pairs: Vec<(f64, (usize, usize))> = Vec::with_capacity(dirs.len());
        for i in 0..dirs.len() {
            let s = dirs[i];
            let e = dirs[(i + 1) % dirs.len()];
            let mut angle = e.0 - s.0;
            if i + 1 == dirs.len() {
                angle += 2.0 * PI;
            }
            pairs.push((angle, (s.1, e.1)));
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        let mut winner = *pairs.last().unwrap();
        for pr in pairs.iter().rev() {
            if ctx.num_atom_rings(pr.1.0) <= 1 && ctx.num_atom_rings(pr.1.1) <= 1 {
                winner = *pr;
                break;
            }
        }
        let e = self.eatoms.get_mut(&aid).unwrap();
        e.rot_dir = 1;
        e.nbr1 = winner.1.1 as isize;
        e.nbr2 = winner.1.0 as isize;
        e.angle = 2.0 * PI - winner.0;
    }

    // ------------------------------------------------------------------
    // rings
    // ------------------------------------------------------------------

    /// `embedFusedRings` (without ring templates).
    fn embed_fused_rings(&mut self, ctx: &Ctx, rings: &[Vec<usize>]) {
        let coords: Vec<BTreeMap<usize, Point2D>> = rings
            .iter()
            .map(|ring| {
                let mut c: BTreeMap<usize, Point2D> = embed_ring(ring, ctx.bond_len).into_iter().collect();
                mirror_trans_ring_atoms(ctx, ring, &mut c);
                c
            })
            .collect();
        let first = pick_first_ring_to_embed(ctx.mol, rings);
        self.init_from_ring_coords(&rings[first], &coords[first]);
        let mut done_rings = vec![first];
        let mut union: HashSet<usize> = HashSet::new();
        for r in rings {
            union.extend(r.iter().copied());
        }
        while self.eatoms.len() < union.len() {
            let (next, common) = find_next_ring_to_embed(&done_rings, rings);
            if common.is_empty() {
                break;
            }
            let mut emb = EmbeddedFrag::default();
            emb.init_from_ring_coords(&rings[next], &coords[next]);
            let pins;
            if common.len() == 1 {
                let t = self.compute_one_atom_trans(common[0], &emb);
                emb.transform(&t);
                pins = vec![common[0]];
            } else {
                let aid1 = common[0];
                let aid2 = *common.last().unwrap();
                pins = vec![aid1, aid2];
                let t = self.compute_two_atom_trans(aid1, aid2, &coords[next]);
                emb.transform(&t);
                self.reflect_if_necessary_density(&mut emb, aid1, aid2);
            }
            self.merge_ring(&emb, common.len(), &pins);
            done_rings.push(next);
        }
    }

    /// `initFromRingCoords`.
    fn init_from_ring_coords(&mut self, ring: &[usize], map: &BTreeMap<usize, Point2D>) {
        let largest = PI * (1.0 - 2.0 / ring.len() as f64);
        let mut prev = *ring.last().unwrap();
        for (cnt, &ai) in ring.iter().enumerate() {
            let e = EAtom { loc: map[&ai], angle: largest, nbr1: prev as isize, ..Default::default() };
            if cnt > 0 {
                self.eatoms.get_mut(&prev).unwrap().nbr2 = ai as isize;
            }
            self.eatoms.insert(ai, e);
            prev = ai;
        }
        self.eatoms.get_mut(&prev).unwrap().nbr2 = ring[0] as isize;
    }

    /// `computeOneAtomTrans`.
    fn compute_one_atom_trans(&self, comm: usize, other: &EmbeddedFrag) -> Transform2D {
        let me = &self.eatoms[&comm];
        let rcr = me.loc;
        let oe = &other.eatoms[&comm];
        let ccr = oe.loc;
        let mut mid = other.loc(oe.nbr1);
        mid += other.loc(oe.nbr2);
        mid *= 0.5;
        let nbp1 = self.loc(me.nbr1);
        let nbp2 = self.loc(me.nbr2);
        let largest = 2.0 * PI - me.angle;
        let bpt = compute_bisect_point(rcr, largest, nbp1, nbp2);
        Transform2D::align(rcr, bpt, ccr, mid)
    }

    /// `computeTwoAtomTrans`.
    fn compute_two_atom_trans(&self, aid1: usize, aid2: usize, cor: &BTreeMap<usize, Point2D>) -> Transform2D {
        Transform2D::align(self.eatoms[&aid1].loc, self.eatoms[&aid2].loc, cor[&aid1], cor[&aid2])
    }

    /// `mergeRing`.
    fn merge_ring(&mut self, ring: &EmbeddedFrag, n_common: usize, pins: &[usize]) {
        for (&aid, o) in &ring.eatoms {
            match self.eatoms.get_mut(&aid) {
                None => {
                    self.eatoms.insert(aid, o.clone());
                }
                Some(e) => {
                    if n_common <= 2 && pins.contains(&aid) {
                        e.angle += o.angle;
                        if e.nbr1 == o.nbr1 {
                            e.nbr1 = o.nbr2;
                        } else if e.nbr1 == o.nbr2 {
                            e.nbr1 = o.nbr1;
                        } else if e.nbr2 == o.nbr1 {
                            e.nbr2 = o.nbr2;
                        } else if e.nbr2 == o.nbr2 {
                            e.nbr2 = o.nbr1;
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // reflections
    // ------------------------------------------------------------------

    pub fn transform(&mut self, t: &Transform2D) {
        for e in self.eatoms.values_mut() {
            e.transform(t);
        }
    }

    fn reflect(&mut self, l1: Point2D, l2: Point2D) {
        for e in self.eatoms.values_mut() {
            e.reflect(l1, l2);
        }
    }

    pub fn translate(&mut self, shift: Point2D) {
        for e in self.eatoms.values_mut() {
            e.loc += shift;
        }
    }

    /// `reflectIfNecessaryDensity`: put the new piece on the less crowded side.
    fn reflect_if_necessary_density(&self, emb: &mut EmbeddedFrag, aid1: usize, aid2: usize) {
        let pin1 = self.eatoms[&aid1].loc;
        let pin2 = self.eatoms[&aid2].loc;
        let mut dn = 0.0;
        let mut dr = 0.0;
        for (aid, o) in &emb.eatoms {
            if self.eatoms.contains_key(aid) {
                continue;
            }
            let loc1 = o.loc;
            let rloc1 = reflect_point(loc1, pin1, pin2);
            for t in self.eatoms.values() {
                let td = (t.loc - loc1).length();
                let rtd = (t.loc - rloc1).length();
                dn += if td > 1.0e-3 { 1.0 / td } else { 1000.0 };
                dr += if rtd > 1.0e-3 { 1.0 / rtd } else { 1000.0 };
            }
        }
        if dn - dr > 1.0e-4 {
            emb.reflect(pin1, pin2);
        }
    }

    /// `reflectIfNecessaryCisTrans`.
    fn reflect_if_necessary_cis_trans(&self, emb: &mut EmbeddedFrag, ct_case: u32, aid1: usize, aid2: usize) {
        let p1 = self.eatoms[&aid1].loc;
        let (p1norm, mut ratm);
        if ct_case == 1 {
            let oe = &emb.eatoms[&aid1];
            p1norm = oe.normal;
            let ring_atm = oe.cis_trans_nbr;
            match self.eatoms.get(&(ring_atm as usize)) {
                Some(r) if ring_atm >= 0 => ratm = r.loc,
                // RDKit: "stereochemistry around double bond may be incorrect"
                _ => return,
            }
        } else {
            let me = &self.eatoms[&aid1];
            p1norm = me.normal;
            ratm = emb.loc(me.cis_trans_nbr);
        }
        ratm -= p1;
        let dot = ratm.dot(p1norm);
        let p2 = self.eatoms[&aid2].loc;
        if dot < 0.0 {
            emb.reflect(p1, p2);
        }
    }

    /// `reflectIfNecessaryThirdPt`.
    fn reflect_if_necessary_third_pt(&self, emb: &mut EmbeddedFrag, aid1: usize, aid2: usize, aid3: usize) {
        let pt1 = self.eatoms[&aid1].loc;
        let pt2 = self.eatoms[&aid2].loc;
        let mut normal = pt2;
        normal -= pt1;
        normal.rotate90();
        let oth3 = emb.eatoms[&aid3].loc - pt1;
        let pt3 = self.eatoms[&aid3].loc - pt1;
        if normal.dot(pt3) * normal.dot(oth3) < 0.0 {
            emb.reflect(pt1, pt2);
        }
    }

    // ------------------------------------------------------------------
    // growing
    // ------------------------------------------------------------------

    /// `addNonRingAtom`.
    fn add_non_ring_atom(&mut self, ctx: &Ctx, aid: usize, to: usize) {
        debug_assert!(!self.eatoms.contains_key(&aid));
        if self.eatoms[&to].angle > 0.0 {
            self.add_atom_with_ang(aid, to);
        } else {
            self.add_atom_with_no_ang(ctx, aid, to);
        }
        let neighs = &mut self.eatoms.get_mut(&to).unwrap().neighs;
        if let Some(p) = neighs.iter().position(|&x| x == aid) {
            neighs.remove(p);
        }
        self.update_new_neighs(ctx, aid);
    }

    /// `addAtomToAtomWithAng`.
    fn add_atom_with_ang(&mut self, aid: usize, to: usize) {
        let rf = self.eatoms[&to].clone();
        let ref_loc = rf.loc;
        let nnbr = rf.neighs.len();
        let rem = 2.0 * PI - rf.angle;
        let mut curr = rem / (1 + nnbr) as f64;
        self.eatoms.get_mut(&to).unwrap().angle += curr;
        let nb1 = self.loc(rf.nbr1);
        let nb2 = self.loc(rf.nbr2);
        let can_choose = rf.rot_dir == 0;
        if can_choose {
            self.eatoms.get_mut(&to).unwrap().rot_dir = rotation_dir(ref_loc, nb1, nb2, rem);
        }
        curr *= self.eatoms[&to].rot_dir as f64;
        let mut curr_loc = Transform2D::rotation_about(ref_loc, curr).apply(nb2);
        if can_choose && (rem - PI).abs() < 1e-3 {
            let curr_loc2 = Transform2D::rotation_about(ref_loc, -curr).apply(nb2);
            if self.num_neigh(curr_loc, 0.5) > self.num_neigh(curr_loc2, 0.5) {
                curr_loc = curr_loc2;
                self.eatoms.get_mut(&to).unwrap().rot_dir *= -1;
            }
        }
        self.eatoms.get_mut(&to).unwrap().nbr2 = aid as isize;
        let tpt = curr_loc - ref_loc;
        let mut norm = Point2D::new(-tpt.y, tpt.x);
        let nccw = self.num_neigh(curr_loc + norm, NEIGH_RADIUS);
        let ncw = self.num_neigh(curr_loc - norm, NEIGH_RADIUS);
        norm.normalize();
        let (normal, ccw) = if nccw < ncw { (norm, false) } else { (-norm, true) };
        let e = EAtom { loc: curr_loc, nbr1: to as isize, angle: -1.0, normal, ccw, ..Default::default() };
        self.eatoms.insert(aid, e);
    }

    /// `addAtomToAtomWithNoAng`.
    fn add_atom_with_no_ang(&mut self, ctx: &Ctx, aid: usize, to: usize) {
        let rf = self.eatoms[&to].clone();
        let ref_loc = rf.loc;
        let mut ref_ccw = rf.ccw;
        let mut curr_loc = rf.normal;
        if rf.cis_trans_nbr >= 0 && rf.cis_trans_nbr as usize != aid {
            ref_ccw = !ref_ccw;
            curr_loc *= -1.0;
        }
        let deg = ctx.degree(to);
        let mut angle = compute_sub_angle(deg, ctx.mol.atom(to).hybridization);
        let origin = Point2D::new(0.0, 0.0);
        let mut flip_norm = false;
        {
            let e = self.eatoms.get_mut(&to).unwrap();
            if e.nbr1 >= 0 {
                e.angle = angle;
                e.nbr2 = aid as isize;
            } else {
                e.normal = Transform2D::rotation_about(origin, angle).apply(e.normal);
                e.nbr1 = aid as isize;
                flip_norm = true;
            }
        }
        angle -= PI / 2.0;
        if !ref_ccw {
            angle *= -1.0;
        }
        curr_loc = Transform2D::rotation_about(origin, angle).apply(curr_loc);
        curr_loc *= ctx.bond_len;
        curr_loc += ref_loc;
        let tpt = ref_loc - curr_loc;
        let mut norm = Point2D::new(-tpt.y, tpt.x);
        if ref_ccw ^ flip_norm {
            norm *= -1.0;
        }
        norm.normalize();
        let e = EAtom {
            loc: curr_loc,
            normal: norm,
            nbr1: to as isize,
            angle: -1.0,
            ccw: (!ref_ccw) ^ flip_norm,
            ..Default::default()
        };
        self.eatoms.insert(aid, e);
    }

    /// `findCommonAtoms`.
    fn common_atoms(&self, other: &EmbeddedFrag) -> Vec<usize> {
        self.eatoms.keys().copied().filter(|k| other.eatoms.contains_key(k)).collect()
    }

    /// `mergeNoCommon`.
    fn merge_no_common(&mut self, ctx: &Ctx, emb: &mut EmbeddedFrag, to: usize, nbr: usize) {
        self.add_non_ring_atom(ctx, nbr, to);
        emb.add_non_ring_atom(ctx, to, nbr);
        let mut comm = vec![to, nbr];
        self.merge_with_common(ctx, emb, &mut comm);
    }

    /// `mergeWithCommon`.
    fn merge_with_common(&mut self, ctx: &Ctx, emb: &mut EmbeddedFrag, comm: &mut Vec<usize>) {
        let mut ct_case = 0;
        if comm.len() == 1 {
            let c = comm[0];
            let other: isize;
            if self.eatoms[&c].cis_trans_nbr >= 0 {
                ct_case = 2;
                other = self.eatoms[&c].nbr1;
                emb.add_non_ring_atom(ctx, other as usize, c);
            } else if emb.eatoms[&c].cis_trans_nbr >= 0 {
                ct_case = 1;
                other = emb.eatoms[&c].nbr1;
                self.add_non_ring_atom(ctx, other as usize, c);
            } else {
                other = self.eatoms[&c].nbr1;
                if other >= 0 {
                    emb.add_non_ring_atom(ctx, other as usize, c);
                }
            }
            if other >= 0 {
                comm.push(other as usize);
            }
        }
        let t = if comm.len() == 1 {
            self.compute_one_atom_trans(comm[0], emb)
        } else {
            let (c1, c2) = (comm[0], comm[1]);
            Transform2D::align(self.eatoms[&c1].loc, self.eatoms[&c2].loc, emb.eatoms[&c1].loc, emb.eatoms[&c2].loc)
        };
        emb.transform(&t);
        if comm.len() >= 2 {
            if ct_case > 0 {
                self.reflect_if_necessary_cis_trans(emb, ct_case, comm[0], comm[1]);
            } else if comm.len() == 2 {
                self.reflect_if_necessary_density(emb, comm[0], comm[1]);
            } else {
                self.reflect_if_necessary_third_pt(emb, comm[0], comm[1], comm[2]);
            }
        }
        for (&aid, o) in &emb.eatoms {
            if !comm.contains(&aid) {
                self.eatoms.insert(aid, o.clone());
                if !o.neighs.is_empty() && !self.attach.contains(&aid) {
                    self.attach.push_back(aid);
                }
            } else {
                let e = self.eatoms.get_mut(&aid).unwrap();
                if o.cis_trans_nbr >= 0 {
                    e.cis_trans_nbr = o.cis_trans_nbr;
                    e.normal = o.normal;
                    e.ccw = o.ccw;
                }
                if o.angle > 0.0 {
                    e.angle = o.angle;
                    e.nbr1 = o.nbr1;
                    e.nbr2 = o.nbr2;
                }
            }
        }
        for &c in comm.iter() {
            self.update_new_neighs(ctx, c);
        }
    }

    fn remove_attach(&mut self, aid: usize) {
        if self.eatoms[&aid].neighs.is_empty()
            && let Some(p) = self.attach.iter().position(|&x| x == aid)
        {
            self.attach.remove(p);
        }
    }

    /// `mergeFragsWithComm`.
    fn merge_frags_with_comm(&mut self, ctx: &Ctx, efrags: &mut FragList) {
        loop {
            let mut found = None;
            for (i, f) in efrags.iter().enumerate() {
                if let Some(f) = f
                    && !f.done
                {
                    let comm = self.common_atoms(f);
                    if !comm.is_empty() {
                        found = Some((i, comm));
                        break;
                    }
                }
            }
            let Some((i, mut comm)) = found else { break };
            let mut other = efrags[i].take().unwrap();
            self.merge_with_common(ctx, &mut other, &mut comm);
            for c in comm {
                self.remove_attach(c);
            }
        }
    }

    /// `expandEfrag`.
    pub fn expand(&mut self, ctx: &Ctx, nratms: &mut Vec<usize>, efrags: &mut FragList) {
        self.merge_frags_with_comm(ctx, efrags);
        while let Some(&aid) = self.attach.front() {
            let nbrs = self.eatoms[&aid].neighs.clone();
            for nbri in nbrs {
                if let Some(p) = nratms.iter().position(|&x| x == nbri) {
                    self.add_non_ring_atom(ctx, nbri, aid);
                    nratms.remove(p);
                } else {
                    let idx = efrags
                        .iter()
                        .position(|f| f.as_ref().is_some_and(|f| !f.done && f.eatoms.contains_key(&nbri)));
                    if let Some(i) = idx {
                        let mut other = efrags[i].take().unwrap();
                        self.merge_no_common(ctx, &mut other, aid, nbri);
                        self.remove_attach(nbri);
                    }
                }
            }
            self.attach.pop_front();
            self.eatoms.get_mut(&aid).unwrap().neighs.clear();
            self.merge_frags_with_comm(ctx, efrags);
        }
    }

    // ------------------------------------------------------------------
    // orientation / layout
    // ------------------------------------------------------------------

    /// `computeBox`: (px, nx, py, ny) with nx/ny negated.
    pub fn compute_box(&mut self) {
        self.px = -1.0e8;
        self.nx = 1.0e8;
        self.py = -1.0e8;
        self.ny = 1.0e8;
        for e in self.eatoms.values() {
            self.px = self.px.max(e.loc.x);
            self.nx = self.nx.min(e.loc.x);
            self.py = self.py.max(e.loc.y);
            self.ny = self.ny.min(e.loc.y);
        }
        self.nx *= -1.0;
        self.ny *= -1.0;
    }

    pub fn bbox(&self) -> (f64, f64, f64, f64) {
        (self.px, self.nx, self.py, self.ny)
    }

    /// `canonicalizeOrientation`: principal axis along x.
    pub fn canonicalize_orientation(&mut self) {
        if self.eatoms.len() <= 1 {
            return;
        }
        let mut cent = Point2D::new(0.0, 0.0);
        for e in self.eatoms.values() {
            cent += e.loc;
        }
        cent *= 1.0 / self.eatoms.len() as f64;
        let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
        for e in self.eatoms.values_mut() {
            e.loc -= cent;
            xx += e.loc.x * e.loc.x;
            xy += e.loc.x * e.loc.y;
            yy += e.loc.y * e.loc.y;
        }
        let d = ((xx - yy) * (xx - yy) + 4.0 * xy * xy).sqrt();
        let mut eig1 = Point2D::new(2.0 * xy, (yy - xx) + d);
        if eig1.length() <= 1e-4 {
            return;
        }
        let eval1 = (xx + yy + d) / 2.0;
        eig1.normalize();
        let mut eig2 = Point2D::new(2.0 * xy, (yy - xx) - d);
        let eval2 = (xx + yy - d) / 2.0;
        if eig2.length() > 1e-4 {
            eig2.normalize();
            if eval2 > eval1 {
                std::mem::swap(&mut eig1, &mut eig2);
            }
        }
        let t = Transform2D { d: [eig1.x, eig1.y, 0.0, -eig1.y, eig1.x, 0.0, 0.0, 0.0, 1.0] };
        self.transform(&t);
    }

    // ------------------------------------------------------------------
    // flips
    // ------------------------------------------------------------------

    /// `flipAboutBond`.
    fn flip_about_bond(&mut self, ctx: &Ctx, bond: usize, flip_end: bool) {
        let b = ctx.mol.bond(bond);
        let (mut beg, mut end) = (b.begin, b.end);
        if !flip_end {
            std::mem::swap(&mut beg, &mut end);
        }
        let end_side = atoms_one_side(ctx, end, beg);
        let mut in_end = vec![false; ctx.n()];
        for &a in &end_side {
            in_end[a] = true;
        }
        let n_fixed = self.eatoms.values().filter(|e| e.fixed).count();
        if n_fixed > 0 && end_side.iter().any(|a| self.eatoms.get(a).is_some_and(|e| e.fixed)) {
            return;
        }
        let end_side_flip = (self.eatoms.len() - end_side.len()) >= end_side.len();
        let keys: Vec<usize> = self.eatoms.keys().copied().collect();
        for k in keys {
            if end_side_flip ^ !in_end[k] {
                // the bond atoms are read every time, as RDKit's references
                let (bl, el) = (self.eatoms[&beg].loc, self.eatoms[&end].loc);
                self.eatoms.get_mut(&k).unwrap().reflect(bl, el);
            }
        }
    }

    /// `flipAboutSpiroCenter`.
    fn flip_about_spiro_center(&mut self, ctx: &Ctx, spiro: usize) {
        let target = ctx.ri.atom_rings.iter().find(|r| r.contains(&spiro)).expect("spiro ring");
        let ring_nbrs: Vec<usize> = ctx.mol.neighbors(spiro).filter(|n| target.contains(n)).collect();
        if ring_nbrs.len() != 2 {
            return;
        }
        let to_flip = atoms_one_side(ctx, ring_nbrs[0], spiro);
        let spiro_loc = self.eatoms[&spiro].loc;
        let mid = (self.eatoms[&ring_nbrs[0]].loc + self.eatoms[&ring_nbrs[1]].loc) * 0.5;
        if to_flip.iter().any(|a| self.eatoms.get(a).is_none_or(|e| e.fixed)) {
            return;
        }
        for a in to_flip {
            self.eatoms.get_mut(&a).unwrap().reflect(spiro_loc, mid);
        }
    }

    /// `permuteBonds`.
    fn permute_bonds(&mut self, ctx: &Ctx, aid: usize, aid1: usize, aid2: usize) {
        let rl1 = self.eatoms[&aid].loc;
        let mut rl2 = self.eatoms[&aid1].loc + self.eatoms[&aid2].loc;
        rl2 *= 0.5;
        let fa = atoms_one_side(ctx, aid1, aid);
        let fb = atoms_one_side(ctx, aid2, aid);
        for a in fa.into_iter().chain(fb) {
            if let Some(e) = self.eatoms.get_mut(&a) {
                e.reflect(rl1, rl2);
            }
        }
    }

    // ------------------------------------------------------------------
    // collisions
    // ------------------------------------------------------------------

    /// `findCollisions`; also sets the per-atom densities.
    fn find_collisions(&mut self, ctx: &Ctx, dmat: &[f64], include_bonds: bool, include_bonded: bool) -> Vec<(usize, usize)> {
        let keys: Vec<usize> = self.eatoms.keys().copied().collect();
        let locs: Vec<Point2D> = self.eatoms.values().map(|e| e.loc).collect();
        let mut dens = vec![0.0f64; keys.len()];
        let mut res = Vec::new();
        let thr2 = COLLISION_THRES * COLLISION_THRES;
        for i in 0..keys.len() {
            let f1 = if ctx.anum(keys[i]) != 6 { HETEROATOM_COLL_SCALE } else { 1.0 };
            for j in 0..i {
                let f2 = if ctx.anum(keys[j]) != 6 { HETEROATOM_COLL_SCALE } else { 1.0 };
                let mut d2 = (locs[j] - locs[i]).length_sq();
                if d2 > 1.0e-3 {
                    dens[i] += 1.0 / d2;
                    dens[j] += 1.0 / d2;
                } else {
                    dens[i] += 1000.0;
                    dens[j] += 1000.0;
                }
                d2 /= f1 * f2;
                if d2 < thr2 && (include_bonded || ctx.mol.bond_between(keys[i], keys[j]).is_none()) {
                    res.push((keys[i], keys[j]));
                }
            }
        }
        for (e, d) in self.eatoms.values_mut().zip(dens) {
            e.density = d;
        }
        if include_bonds {
            let bonds = ctx.mol.bonds();
            for (b1, bd1) in bonds.iter().enumerate() {
                let (beg1, end1) = (bd1.begin, bd1.end);
                let (Some(l1), Some(l2)) = (self.eatoms.get(&beg1), self.eatoms.get(&end1)) else { continue };
                let (l1, l2) = (l1.loc, l2.loc);
                for bd2 in &bonds[b1 + 1..] {
                    let (beg2, end2) = (bd2.begin, bd2.end);
                    if beg1 == beg2 || beg1 == end2 || end1 == beg2 || end1 == end2 {
                        continue;
                    }
                    let (Some(l3), Some(l4)) = (self.eatoms.get(&beg2), self.eatoms.get(&end2)) else { continue };
                    if segments_intersect(l1, l2, l3.loc, l4.loc) {
                        res.push(farthest_pair(ctx.n(), beg1, end1, beg2, end2, dmat));
                    }
                }
            }
        }
        res
    }

    fn total_density(&self) -> f64 {
        self.eatoms.values().fold(0.0, |acc, e| e.density + acc)
    }

    /// `openAngles`.
    fn open_angles(&mut self, ctx: &Ctx, dmat: &[f64], aid1: usize, aid2: usize) {
        let deg1 = ctx.degree(aid1);
        let deg2 = ctx.degree(aid2);
        let fixed1 = self.eatoms[&aid1].fixed;
        let fixed2 = self.eatoms[&aid2].fixed;
        if (deg1 > 1 || fixed1) && (deg2 > 1 || fixed2) {
            return;
        }
        let (aid_a, aid_b, ty);
        if (deg1 == 1 && !fixed1) && (deg2 == 1 && !fixed2) {
            aid_a = deg1_neighbor(ctx, aid1);
            aid_b = deg1_neighbor(ctx, aid2);
            ty = 1;
        } else if (deg1 == 1 && !fixed1) && (deg2 > 1 || fixed2) {
            aid_a = deg1_neighbor(ctx, aid1);
            aid_b = closest_neighbor(ctx, dmat, aid_a, aid2);
            ty = 2;
        } else {
            aid_b = deg1_neighbor(ctx, aid2);
            aid_a = closest_neighbor(ctx, dmat, aid_b, aid1);
            ty = 3;
        }
        let la = self.eatoms[&aid_a].loc;
        let lb = self.eatoms[&aid_b].loc;
        let v2 = self.eatoms[&aid1].loc - la;
        let v1 = lb - la;
        let cross = v1.x * v2.y - v1.y * v2.x;
        match ty {
            1 => {
                let mut angle = ANGLE_OPEN;
                if cross < 0.0 {
                    angle *= -1.0;
                }
                let t1 = Transform2D::rotation_about(la, angle);
                let t2 = Transform2D::rotation_about(lb, -angle);
                t1.transform_point(&mut self.eatoms.get_mut(&aid1).unwrap().loc);
                t2.transform_point(&mut self.eatoms.get_mut(&aid2).unwrap().loc);
            }
            2 => {
                let mut angle = 2.0 * ANGLE_OPEN;
                if cross < 0.0 {
                    angle *= -1.0;
                }
                let t1 = Transform2D::rotation_about(la, angle);
                t1.transform_point(&mut self.eatoms.get_mut(&aid1).unwrap().loc);
            }
            _ => {
                let mut angle = -2.0 * ANGLE_OPEN;
                if cross < 0.0 {
                    angle *= -1.0;
                }
                let t2 = Transform2D::rotation_about(lb, angle);
                t2.transform_point(&mut self.eatoms.get_mut(&aid2).unwrap().loc);
            }
        }
    }

    fn try_bond_flip(
        &mut self,
        ctx: &Ctx,
        c: (usize, usize),
        ncols: usize,
        prev_density: f64,
        done_bonds: &mut HashMap<usize, u32>,
        dmat: &[f64],
    ) -> bool {
        for ri in rotatable_bonds_between(ctx, c.0, c.1) {
            let cnt = done_bonds.get(&ri).copied();
            if cnt.is_some_and(|n| n >= NUM_BONDS_FLIPS) {
                continue;
            }
            *done_bonds.entry(ri).or_insert(0) += 1;
            self.flip_about_bond(ctx, ri, true);
            let colls = self.find_collisions(ctx, dmat, true, true);
            let nd = self.total_density();
            if colls.len() < ncols {
                done_bonds.insert(ri, NUM_BONDS_FLIPS);
                return true;
            } else if colls.len() == ncols && nd < prev_density {
                return true;
            }
            self.flip_about_bond(ctx, ri, true);
            self.flip_about_bond(ctx, ri, false);
            let colls = self.find_collisions(ctx, dmat, true, true);
            let nd = self.total_density();
            if colls.len() < ncols {
                done_bonds.insert(ri, NUM_BONDS_FLIPS);
                return true;
            } else if colls.len() == ncols && nd < prev_density {
                return true;
            }
            self.flip_about_bond(ctx, ri, false);
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn try_spiro_flip(
        &mut self,
        ctx: &Ctx,
        c: (usize, usize),
        ncols: usize,
        prev_density: f64,
        done_spiros: &mut HashMap<usize, u32>,
        spiro: &[bool],
        dmat: &[f64],
    ) -> bool {
        let path = shortest_path(ctx.mol, c.0, c.1);
        for s in path.into_iter().filter(|&a| spiro[a]) {
            if done_spiros.get(&s).is_some_and(|&n| n >= NUM_BONDS_FLIPS) {
                continue;
            }
            self.flip_about_spiro_center(ctx, s);
            let colls = self.find_collisions(ctx, dmat, true, true);
            let nd = self.total_density();
            if colls.len() < ncols {
                done_spiros.insert(s, NUM_BONDS_FLIPS);
                return true;
            } else if colls.len() == ncols && nd < prev_density {
                *done_spiros.entry(s).or_insert(0) += 1;
                return true;
            }
            self.flip_about_spiro_center(ctx, s);
        }
        false
    }

    /// `removeCollisionsBondAndSpiroFlip`.
    pub fn remove_collisions_bond_and_spiro_flip(&mut self, ctx: &Ctx, dmat: &[f64]) {
        let spiro: Vec<bool> = (0..ctx.n()).map(|a| is_spiro_center(ctx, a)).collect();
        let mut colls = self.find_collisions(ctx, dmat, true, true);
        let mut done_bonds = HashMap::new();
        let mut done_spiros = HashMap::new();
        let mut iter = 0;
        while iter < MAX_COLL_ITERS && !colls.is_empty() {
            let ncols = colls.len();
            let c = colls[0];
            let prev = self.total_density();
            if !self.try_bond_flip(ctx, c, ncols, prev, &mut done_bonds, dmat) {
                self.try_spiro_flip(ctx, c, ncols, prev, &mut done_spiros, &spiro, dmat);
            }
            colls = self.find_collisions(ctx, dmat, true, true);
            iter += 1;
        }
    }

    /// `removeCollisionsOpenAngles`.
    pub fn remove_collisions_open_angles(&mut self, ctx: &Ctx, dmat: &[f64]) {
        for (a, b) in self.find_collisions(ctx, dmat, false, true) {
            self.open_angles(ctx, dmat, a, b);
        }
    }

    /// `removeCollisionsShortenBonds`.
    pub fn remove_collisions_shorten_bonds(&mut self, ctx: &Ctx, dmat: &[f64]) {
        let mut colls = self.find_collisions(ctx, dmat, false, true);
        let mut iter = 0;
        while !colls.is_empty() && iter < MAX_COLL_ITERS {
            let (mut aid1, mut aid2) = colls[0];
            let mut fixed1 = self.eatoms[&aid1].fixed;
            let mut fixed2 = self.eatoms[&aid2].fixed;
            if fixed1 && fixed2 {
                colls.remove(0);
                iter += 1;
                continue;
            }
            let mut deg1 = ctx.degree(aid1);
            let mut deg2 = ctx.degree(aid2);
            if fixed1 || (deg2 > deg1 && !fixed2) {
                std::mem::swap(&mut deg1, &mut deg2);
                std::mem::swap(&mut aid1, &mut aid2);
                std::mem::swap(&mut fixed1, &mut fixed2);
            }
            let path = shortest_path(ctx.mol, aid1, aid2);
            if path.is_empty() {
                colls.remove(0);
            } else {
                let path = &path[1..];
                if non_ring_bonds_on_path(ctx, aid1, path) > 0 {
                    for (aid, deg, ok) in [(aid1, deg1, true), (aid2, deg2, !fixed2)] {
                        if deg == 1 && ok {
                            let a = deg1_neighbor(ctx, aid);
                            let al = self.eatoms[&a].loc;
                            let mut loc = self.eatoms[&aid].loc;
                            loc -= al;
                            loc *= 0.9;
                            if loc.length() > 0.75 {
                                loc += al;
                                self.eatoms.get_mut(&aid).unwrap().loc = loc;
                            }
                        }
                    }
                } else {
                    // probably a bridged system: move the atoms of a chain of
                    // degree-two ring atoms inwards
                    let mut rpath = Vec::new();
                    let mut nbr_map = HashMap::new();
                    deg_two_ring_atoms(ctx, aid1, &mut rpath, &mut nbr_map);
                    if rpath.is_empty() {
                        deg_two_ring_atoms(ctx, aid2, &mut rpath, &mut nbr_map);
                    }
                    let mut moves: HashMap<usize, Point2D> = HashMap::new();
                    for &r in &rpath {
                        if self.eatoms[&r].fixed {
                            continue;
                        }
                        let nb = &nbr_map[&r];
                        let mut mv = self.eatoms[&nb[0]].loc;
                        mv += self.eatoms[&nb[1]].loc;
                        mv *= 0.5;
                        mv -= self.eatoms[&r].loc;
                        mv.normalize();
                        mv *= COLLISION_THRES;
                        moves.insert(r, mv);
                    }
                    for &r in &rpath {
                        let mv = moves.get(&r).copied().unwrap_or_default();
                        self.eatoms.get_mut(&r).unwrap().loc += mv;
                    }
                }
                colls = self.find_collisions(ctx, dmat, false, true);
            }
            iter += 1;
        }
    }

    /// `getAtomsOnSide`: BFS from `side_start` not crossing `center` or
    /// `exclude`.
    fn atoms_on_side(&self, ctx: &Ctx, center: usize, side_start: usize, exclude: usize) -> Vec<usize> {
        let mut res = Vec::new();
        let mut visited = vec![false; ctx.n()];
        visited[center] = true;
        visited[exclude] = true;
        let mut q = VecDeque::from([side_start]);
        while let Some(cur) = q.pop_front() {
            if visited[cur] {
                continue;
            }
            visited[cur] = true;
            if self.eatoms.contains_key(&cur) {
                res.push(cur);
            }
            for nbr in ctx.mol.neighbors(cur) {
                if !visited[nbr] {
                    q.push_back(nbr);
                }
            }
        }
        res
    }

    /// `openAngleByIncrement`.
    fn open_angle_by_increment(&mut self, ctx: &Ctx, prev: usize, center: usize, next: usize, inc: f64) -> bool {
        let cl = self.eatoms[&center].loc;
        let v1 = self.eatoms[&prev].loc - cl;
        let v2 = self.eatoms[&next].loc - cl;
        let side1 = self.atoms_on_side(ctx, center, prev, next);
        let side2 = self.atoms_on_side(ctx, center, next, prev);
        if side2.iter().any(|a| side1.contains(a)) {
            return false;
        }
        let fixed = |s: &[usize]| s.iter().any(|a| self.eatoms[a].fixed);
        let (f1, f2) = (fixed(&side1), fixed(&side2));
        if f1 && f2 {
            return false;
        }
        let rotate1 = f2 || (!f1 && side1.len() < side2.len());
        let angle_between = |a: Point2D, b: Point2D| {
            let den = a.length() * b.length();
            if den < 1e-8 {
                return -1.0;
            }
            ((a.x * b.x + a.y * b.y) / den).clamp(-1.0, 1.0).acos()
        };
        let orig = angle_between(v1, v2);
        if orig < 0.0 {
            return false;
        }
        let mut rot = 0.0;
        for cand in [inc, -inc] {
            let trial = Transform2D::rotation_about(cl, cand);
            let moved = trial.apply(self.eatoms[&if rotate1 { prev } else { next }].loc);
            let mv = moved - cl;
            let fixed_v = if rotate1 { v2 } else { v1 };
            if angle_between(mv, fixed_v) > orig + 1e-8 {
                rot = cand;
                break;
            }
        }
        if rot == 0.0 {
            return false;
        }
        let t = Transform2D::rotation_about(self.eatoms[&center].loc, rot);
        for a in if rotate1 { side1 } else { side2 } {
            t.transform_point(&mut self.eatoms.get_mut(&a).unwrap().loc);
        }
        true
    }

    /// `removeCollisionsPathAngleExpansion`.
    pub fn remove_collisions_path_angle_expansion(&mut self, ctx: &Ctx, dmat: &[f64]) {
        let mut atom_rings: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (ri, ring) in ctx.ri.atom_rings.iter().enumerate() {
            for &a in ring {
                atom_rings.entry(a).or_default().insert(ri);
            }
        }
        let canon = |c: (usize, usize)| (c.0.min(c.1), c.0.max(c.1));
        let find = |s: &mut Self| {
            let c = s.find_collisions(ctx, dmat, false, false);
            if c.is_empty() { s.find_collisions(ctx, dmat, true, false) } else { c }
        };
        let mut skipped: HashSet<(usize, usize)> = HashSet::new();
        let filter = |cs: &[(usize, usize)], skipped: &HashSet<(usize, usize)>| -> Vec<(usize, usize)> {
            cs.iter().copied().filter(|&c| !skipped.contains(&canon(c))).collect()
        };
        let mut all = find(self);
        let mut colls = filter(&all, &skipped);
        let mut totals: HashMap<(usize, usize), HashMap<usize, f64>> = HashMap::new();
        let mut iter = 0;
        while iter < MAX_ANGLE_EXPANSION_ITERS && !colls.is_empty() {
            let c = colls[0];
            let path = shortest_path(ctx.mol, c.0, c.1);
            if path.len() < 3 {
                skipped.insert(canon(c));
                colls.remove(0);
                iter += 1;
                continue;
            }
            let mut positions = Vec::new();
            for i in 1..path.len() - 1 {
                let (p, ce, nx) = (path[i - 1], path[i], path[i + 1]);
                let same_ring = match (atom_rings.get(&p), atom_rings.get(&ce), atom_rings.get(&nx)) {
                    (Some(pr), Some(cr), Some(nr)) => pr.iter().any(|r| cr.contains(r) && nr.contains(r)),
                    _ => false,
                };
                if !same_ring {
                    positions.push(i);
                }
            }
            if positions.is_empty() {
                skipped.insert(canon(c));
                colls.remove(0);
                iter += 1;
                continue;
            }
            let prev_count = all.len();
            let saved: Vec<(usize, Point2D)> = self.eatoms.iter().map(|(&k, e)| (k, e.loc)).collect();
            let mut any = false;
            for pos in positions {
                let center = path[pos];
                let tot = totals.entry(canon(c)).or_default().entry(center).or_insert(0.0);
                if *tot >= MAX_ANGLE_EXPANSION {
                    continue;
                }
                if self.open_angle_by_increment(ctx, path[pos - 1], center, path[pos + 1], ANGLE_EXPANSION_INCREMENT) {
                    *totals.get_mut(&canon(c)).unwrap().get_mut(&center).unwrap() += ANGLE_EXPANSION_INCREMENT;
                    any = true;
                }
            }
            if !any {
                skipped.insert(canon(c));
                colls.remove(0);
                iter += 1;
                continue;
            }
            let new_colls = find(self);
            if new_colls.len() > prev_count {
                for (k, l) in saved {
                    self.eatoms.get_mut(&k).unwrap().loc = l;
                }
                totals.remove(&canon(c));
                skipped.insert(canon(c));
                all = find(self);
            } else {
                all = new_colls;
            }
            colls = filter(&all, &skipped);
            iter += 1;
        }
    }

    // ------------------------------------------------------------------
    // random sampling
    // ------------------------------------------------------------------

    /// `computeDistMat` + `mimicDistMatAndDensityCostFunc` without a
    /// reference distance matrix (density term only).
    fn density_cost(&self, ctx: &Ctx) -> f64 {
        let na = ctx.n();
        if na < 2 {
            return 0.0;
        }
        let mut d2d = vec![0.0f64; na * (na - 1) / 2];
        let items: Vec<(usize, Point2D)> = self.eatoms.iter().map(|(&k, e)| (k, e.loc)).collect();
        for i in 0..items.len() {
            for j in 0..i {
                let (mut ai, mut aj) = (items[i].0, items[j].0);
                let d = (items[j].1 - items[i].1).length();
                if ai < aj {
                    std::mem::swap(&mut ai, &mut aj);
                }
                d2d[ai * (ai - 1) / 2 + aj] = d;
            }
        }
        let mut res1 = 0.0;
        for d in d2d {
            let d2 = d * d;
            res1 += if d2 > 1.0e-3 { 1.0 / d2 } else { 1000.0 };
        }
        res1
    }

    /// `randomSampleFlipsAndPermutations` (density cost only).
    pub fn random_sample_flips_and_permutations(
        &mut self,
        ctx: &Ctx,
        n_bonds_per_sample: u32,
        n_samples: u32,
        seed: i32,
        permute_deg4: bool,
        rng: &mut MinStd,
    ) {
        let rot_bonds: Vec<usize> = all_rotatable_bonds(ctx)
            .into_iter()
            .filter(|&b| {
                let bd = ctx.mol.bond(b);
                self.eatoms.contains_key(&bd.begin) && self.eatoms.contains_key(&bd.end)
            })
            .collect();
        let nb = rot_bonds.len();
        let mut deg4: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();
        if permute_deg4 {
            for a in 0..ctx.n() {
                if ctx.degree(a) == 4 && ctx.num_atom_rings(a) == 0 {
                    let (aids, bids): (Vec<usize>, Vec<usize>) = ctx.mol.atom_bonds(a).iter().copied().unzip();
                    if aids.iter().all(|x| self.eatoms.get(x).is_some_and(|e| !e.fixed)) {
                        deg4.push((a, bids, aids));
                    }
                }
            }
        }
        let nt = nb + deg4.len();
        let per_sample = (nt as u32).min(n_bonds_per_sample);
        if nt == 0 {
            return;
        }
        if seed > 0 {
            *rng = MinStd::new(seed as u64);
        }
        let mut best = self.density_cost(ctx);
        let mut best_crd: Vec<(usize, Point2D)> = self.eatoms.iter().map(|(&k, e)| (k, e.loc)).collect();
        for _ in 0..n_samples {
            for _ in 0..per_sample {
                let ri = rng.uniform_int(0, nt as u32 - 1) as usize;
                if ri < nb {
                    self.flip_about_bond(ctx, rot_bonds[ri], true);
                } else {
                    let (ai, bids, aids) = deg4[ri - nb].clone();
                    let locs: Vec<Point2D> = aids.iter().map(|x| self.eatoms[x].loc).collect();
                    let pairs = bond_pairs_to_permute_deg4(self.eatoms[&ai].loc, &bids, &locs);
                    let fbi = if rng.uniform_01() > 0.5 { 1 } else { 0 };
                    let a1 = ctx.mol.bond(pairs[fbi].0).other(ai);
                    let a2 = ctx.mol.bond(pairs[fbi].1).other(ai);
                    self.permute_bonds(ctx, ai, a1, a2);
                }
            }
            let d = self.density_cost(ctx);
            if best - d > 1e-4 {
                best = d;
                best_crd = self.eatoms.iter().map(|(&k, e)| (k, e.loc)).collect();
            }
        }
        for (k, l) in best_crd {
            self.eatoms.get_mut(&k).unwrap().loc = l;
        }
    }
}
