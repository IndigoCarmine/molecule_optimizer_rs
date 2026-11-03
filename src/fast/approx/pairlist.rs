//! Buffered cluster-pair (Verlet) list.
//!
//! Construction:
//!
//! 1. In periodic systems every atom is translated into the home cell by a
//!    lattice vector `T_i` (kept until the next rebuild, so the kernels use
//!    `x_i + T_i`; the force field itself always sees the caller's
//!    coordinates). Isolated systems use their bounding box.
//! 2. The cell (or bounding box) is divided into columns along the first
//!    two lattice directions; the column size is chosen so that four atoms
//!    occupy a roughly cubic volume at the system's average density.
//!    Atoms of a column are sorted by their height along the third lattice
//!    direction and grouped into clusters of four (the last cluster of a
//!    column is padded with dummy atoms).
//! 3. For every cluster `I`, neighbor columns within reach are scanned for
//!    all periodic images (shifts `n_a a + n_b b + n_c c`, `n ∈ {-1,0,1}`);
//!    clusters are z-sorted, so the candidate range in a column is found by
//!    binary search on the cluster heights. A cluster pair is kept if the
//!    distance between the bounding boxes is below `rlist = cutoff +
//!    buffer`.
//! 4. Each kept pair stores a 16-bit interaction mask (bit `4a + b` for
//!    atom `a` of `I` and `b` of `J`): dummies, self pairs, excluded and
//!    interfragment pairs, and atom pairs farther than `rlist` are cleared.
//!
//! The list is *symmetric* (both `(I, J)` and `(J, I)` are stored), so each
//! I-cluster computes the gradient of its own atoms only: there are no
//! write conflicts and the result does not depend on the number of threads.
//!
//! Validity: if no atom moved more than `buffer / 2` since the build, every
//! pair closer than `cutoff` was closer than `rlist` at build time and is
//! therefore in the list.

use super::params::NbParams;
use crate::core::UnitCell;
use rayon::prelude::*;
use wide::f64x4;

/// Entry of the j-cluster list of an i-cluster.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CjEntry {
    pub j: u32,
    pub mask: u16,
    /// Index into [`PairList::shifts`].
    pub shift: u8,
}

/// Dummy atom marker.
pub(crate) const DUMMY: u32 = u32::MAX;
/// Index of the zero shift.
pub(crate) const ZERO_SHIFT: u8 = 13;

/// A built cluster-pair list.
#[derive(Debug, Clone)]
pub(crate) struct PairList {
    pub nclusters: usize,
    /// Atoms of every cluster (`DUMMY` for padding).
    pub atoms: Vec<[u32; 4]>,
    /// `cluster * 4 + lane` of every atom.
    pub slot: Vec<u32>,
    /// Lattice translation applied to each atom (zero for isolated systems).
    pub tshift: Vec<[f64; 3]>,
    /// Positions at build time (displacement reference).
    pub ref_pos: Vec<f64>,
    /// CSR offsets into `cj` (length `nclusters + 1`).
    pub ci: Vec<u32>,
    pub cj: Vec<CjEntry>,
    /// Shift vectors `n_a a + n_b b + n_c c`, index `9(n_a+1) + 3(n_b+1) + (n_c+1)`.
    pub shifts: [[f64; 3]; 27],
    /// Charges times `sqrt(C)` per cluster lane (0 for dummies).
    pub q: Vec<f64x4>,
    /// vdW types per cluster lane (0 for dummies).
    pub types: Vec<[u32; 4]>,
    /// `true` if every pair is in the list regardless of distance (never
    /// needs a rebuild).
    pub all_pairs: bool,
}

#[derive(Clone, Copy)]
struct Cluster {
    lo: [f64; 3],
    hi: [f64; 3],
    hmin: f64,
    hmax: f64,
    cx: u32,
    cy: u32,
}

/// Builds the list for positions `pos`.
pub(crate) fn build(p: &NbParams, pos: &[f64], cell: Option<&UnitCell>, rlist: f64, all_pairs: bool) -> PairList {
    let n = p.n;
    // --- geometry: wrapped positions, fractional column coordinates, heights
    let mut tshift = vec![[0.0; 3]; n];
    let mut wpos = vec![[0.0; 3]; n];
    let mut frac = vec![[0.0; 2]; n];
    let mut height = vec![0.0; n];
    let mut shifts = [[0.0; 3]; 27];
    let (wa, wb, volume) = match cell {
        Some(c) => {
            let [va, vb, vc] = c.vectors();
            let w = c.perpendicular_widths();
            for (k, s) in shifts.iter_mut().enumerate() {
                let (na, nb, nc) = ((k / 9) as f64 - 1.0, ((k / 3) % 3) as f64 - 1.0, (k % 3) as f64 - 1.0);
                for d in 0..3 {
                    s[d] = na * va[d] + nb * vb[d] + nc * vc[d];
                }
            }
            tshift
                .par_iter_mut()
                .zip(wpos.par_iter_mut())
                .zip(frac.par_iter_mut().zip(height.par_iter_mut()))
                .enumerate()
                .for_each(|(i, ((t, wp), (fr, h)))| {
                    let r = [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];
                    let s = c.to_fractional(r);
                    let f = s.map(f64::floor);
                    let tt = c.to_cartesian(f.map(|x| -x));
                    *t = tt;
                    *wp = [r[0] + tt[0], r[1] + tt[1], r[2] + tt[2]];
                    *fr = [s[0] - f[0], s[1] - f[1]];
                    *h = (s[2] - f[2]) * w[2];
                });
            (w[0], w[1], c.volume())
        }
        None => {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for i in 0..n {
                for d in 0..3 {
                    lo[d] = lo[d].min(pos[3 * i + d]);
                    hi[d] = hi[d].max(pos[3 * i + d]);
                }
            }
            let len = [0, 1, 2].map(|d| (hi[d] - lo[d]).max(1e-6));
            for i in 0..n {
                let r = [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];
                wpos[i] = r;
                frac[i] = [(r[0] - lo[0]) / len[0], (r[1] - lo[1]) / len[1]];
                height[i] = r[2] - lo[2];
            }
            (len[0], len[1], len.iter().map(|&l| l.max(2.0)).product::<f64>())
        }
    };
    let periodic = cell.is_some();
    // --- columns
    let side = (4.0 * volume / n as f64).cbrt();
    let ncol = |w: f64| ((w / side + 0.5).floor() as usize).clamp(1, 1024);
    let (nx, ny) = (ncol(wa), ncol(wb));
    let col_of = |i: usize| -> (u32, u32) {
        let cx = ((frac[i][0] * nx as f64) as usize).min(nx - 1);
        let cy = ((frac[i][1] * ny as f64) as usize).min(ny - 1);
        (cx as u32, cy as u32)
    };
    let mut order: Vec<(u32, f64, u32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (cx, cy) = col_of(i);
            (cx * ny as u32 + cy, height[i], i as u32)
        })
        .collect();
    order.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)));
    // --- clusters
    let ncols = nx * ny;
    let mut col_start = vec![0u32; ncols + 1];
    let mut atoms: Vec<[u32; 4]> = Vec::with_capacity(n / 4 + ncols);
    let mut clusters: Vec<Cluster> = Vec::with_capacity(n / 4 + ncols);
    let mut k = 0;
    for col in 0..ncols {
        col_start[col] = atoms.len() as u32;
        let start = k;
        while k < n && order[k].0 as usize == col {
            k += 1;
        }
        for chunk in order[start..k].chunks(4) {
            let mut a = [DUMMY; 4];
            let mut cl = Cluster {
                lo: [f64::INFINITY; 3],
                hi: [f64::NEG_INFINITY; 3],
                hmin: f64::INFINITY,
                hmax: f64::NEG_INFINITY,
                cx: (col / ny) as u32,
                cy: (col % ny) as u32,
            };
            for (l, &(_, h, i)) in chunk.iter().enumerate() {
                a[l] = i;
                let w = wpos[i as usize];
                for d in 0..3 {
                    cl.lo[d] = cl.lo[d].min(w[d]);
                    cl.hi[d] = cl.hi[d].max(w[d]);
                }
                cl.hmin = cl.hmin.min(h);
                cl.hmax = cl.hmax.max(h);
            }
            atoms.push(a);
            clusters.push(cl);
        }
    }
    col_start[ncols] = atoms.len() as u32;
    let nclusters = atoms.len();
    let mut slot = vec![0u32; n];
    let mut q = Vec::with_capacity(nclusters);
    let mut types = Vec::with_capacity(nclusters);
    let sq = p.coulomb.sqrt();
    for (c, a) in atoms.iter().enumerate() {
        let mut qq = [0.0; 4];
        let mut tt = [0u32; 4];
        for l in 0..4 {
            if a[l] != DUMMY {
                let i = a[l] as usize;
                slot[i] = (4 * c + l) as u32;
                qq[l] = p.q[i] * sq;
                tt[l] = p.atom_type[i];
            }
        }
        q.push(f64x4::new(qq));
        types.push(tt);
    }
    // --- cluster-pair search
    let rl2 = rlist * rlist;
    let wc = cell.map(|c| c.perpendicular_widths()[2]).unwrap_or(0.0);
    let reach = |w: f64, nc: usize| -> i64 {
        if all_pairs { nc as i64 } else { ((rlist * nc as f64 / w).floor() as i64 + 1).min(nc as i64) }
    };
    let (rx, ry) = (reach(wa, nx), reach(wb, ny));
    let sc_range: &[i64] = if periodic { &[-1, 0, 1] } else { &[0] };
    let mask_of = |ci: usize, cj: usize, shift: usize| -> u16 {
        let s = shifts[shift];
        let mut m = 0u16;
        for a in 0..4 {
            let ia = atoms[ci][a];
            if ia == DUMMY {
                continue;
            }
            let wi = wpos[ia as usize];
            for b in 0..4 {
                let jb = atoms[cj][b];
                if jb == DUMMY || (ia == jb && shift == ZERO_SHIFT as usize) {
                    continue;
                }
                if p.is_masked(ia as usize, jb as usize) {
                    continue;
                }
                if !all_pairs {
                    let wj = wpos[jb as usize];
                    let d = [wi[0] - wj[0] - s[0], wi[1] - wj[1] - s[1], wi[2] - wj[2] - s[2]];
                    if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] >= rl2 {
                        continue;
                    }
                }
                m |= 1 << (4 * a + b);
            }
        }
        m
    };
    const BLOCK: usize = 64;
    let blocks: Vec<(Vec<u32>, Vec<CjEntry>)> = (0..nclusters.div_ceil(BLOCK))
        .into_par_iter()
        .map(|blk| {
            let mut counts = Vec::with_capacity(BLOCK);
            let mut out = Vec::new();
            for ci in blk * BLOCK..((blk + 1) * BLOCK).min(nclusters) {
                let before = out.len();
                if all_pairs {
                    for cj in 0..nclusters {
                        let m = mask_of(ci, cj, ZERO_SHIFT as usize);
                        if m != 0 {
                            out.push(CjEntry { j: cj as u32, mask: m, shift: ZERO_SHIFT });
                        }
                    }
                } else {
                    let ic = &clusters[ci];
                    for dx in -rx..=rx {
                        let mut cx = ic.cx as i64 + dx;
                        let mut sa = 0i64;
                        if periodic {
                            sa = cx.div_euclid(nx as i64);
                            cx = cx.rem_euclid(nx as i64);
                        } else if cx < 0 || cx >= nx as i64 {
                            continue;
                        }
                        for dy in -ry..=ry {
                            let mut cy = ic.cy as i64 + dy;
                            let mut sb = 0i64;
                            if periodic {
                                sb = cy.div_euclid(ny as i64);
                                cy = cy.rem_euclid(ny as i64);
                            } else if cy < 0 || cy >= ny as i64 {
                                continue;
                            }
                            let col = cx as usize * ny + cy as usize;
                            let (c0, c1) = (col_start[col] as usize, col_start[col + 1] as usize);
                            if c0 == c1 {
                                continue;
                            }
                            for &sc in sc_range {
                                let shift = ((sa + 1) * 9 + (sb + 1) * 3 + (sc + 1)) as usize;
                                let s = shifts[shift];
                                let sh = sc as f64 * wc;
                                let lo_h = ic.hmin - rlist - sh;
                                let hi_h = ic.hmax + rlist - sh;
                                let first = c0 + clusters[c0..c1].partition_point(|c| c.hmax < lo_h);
                                for cj in first..c1 {
                                    let jc = &clusters[cj];
                                    if jc.hmin > hi_h {
                                        break;
                                    }
                                    let mut d2 = 0.0;
                                    for d in 0..3 {
                                        let g = (jc.lo[d] + s[d] - ic.hi[d]).max(ic.lo[d] - jc.hi[d] - s[d]).max(0.0);
                                        d2 += g * g;
                                    }
                                    if d2 >= rl2 {
                                        continue;
                                    }
                                    let m = mask_of(ci, cj, shift);
                                    if m != 0 {
                                        out.push(CjEntry { j: cj as u32, mask: m, shift: shift as u8 });
                                    }
                                }
                            }
                        }
                    }
                }
                counts.push((out.len() - before) as u32);
            }
            (counts, out)
        })
        .collect();
    let total: usize = blocks.iter().map(|b| b.1.len()).sum();
    let mut ci = Vec::with_capacity(nclusters + 1);
    let mut cj = Vec::with_capacity(total);
    ci.push(0u32);
    for (counts, out) in blocks {
        for c in counts {
            ci.push(ci.last().unwrap() + c);
        }
        cj.extend(out);
    }
    PairList { nclusters, atoms, slot, tshift, ref_pos: pos[..3 * n].to_vec(), ci, cj, shifts, q, types, all_pairs }
}

impl PairList {
    /// Cluster-ordered coordinates `x_i + T_i` (dummies at the origin).
    pub(crate) fn gather(&self, pos: &[f64]) -> super::kernel::ClusterCoords {
        let nc = self.nclusters;
        let mut x = vec![f64x4::splat(0.0); nc];
        let mut y = vec![f64x4::splat(0.0); nc];
        let mut z = vec![f64x4::splat(0.0); nc];
        x.par_iter_mut().zip(y.par_iter_mut()).zip(z.par_iter_mut()).enumerate().with_min_len(256).for_each(
            |(c, ((x, y), z))| {
                let mut a = [[0.0; 4]; 3];
                for l in 0..4 {
                    let i = self.atoms[c][l];
                    if i != DUMMY {
                        let i = i as usize;
                        let t = self.tshift[i];
                        for d in 0..3 {
                            a[d][l] = pos[3 * i + d] + t[d];
                        }
                    }
                }
                *x = f64x4::new(a[0]);
                *y = f64x4::new(a[1]);
                *z = f64x4::new(a[2]);
            },
        );
        super::kernel::ClusterCoords { x, y, z }
    }

    /// Largest squared displacement of any atom since the build (SIMD,
    /// parallel; the maximum is independent of the evaluation order).
    pub(crate) fn max_displacement2(&self, pos: &[f64]) -> f64 {
        const CH: usize = 3 * 4 * 1024;
        let n3 = self.ref_pos.len();
        pos[..n3]
            .par_chunks(CH)
            .zip(self.ref_pos.par_chunks(CH))
            .map(|(p, r)| {
                let mut m = f64x4::splat(0.0);
                let full = p.len() / 12;
                for k in 0..full {
                    let o = 12 * k;
                    let lane = |d: usize| {
                        f64x4::new([
                            p[o + d] - r[o + d],
                            p[o + 3 + d] - r[o + 3 + d],
                            p[o + 6 + d] - r[o + 6 + d],
                            p[o + 9 + d] - r[o + 9 + d],
                        ])
                    };
                    let (dx, dy, dz) = (lane(0), lane(1), lane(2));
                    m = m.max(dx * dx + dy * dy + dz * dz);
                }
                let mut mm = m.to_array().iter().fold(0.0f64, |a, &b| a.max(b));
                for i in (12 * full / 3)..(p.len() / 3) {
                    let d = [p[3 * i] - r[3 * i], p[3 * i + 1] - r[3 * i + 1], p[3 * i + 2] - r[3 * i + 2]];
                    mm = mm.max(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                }
                mm
            })
            .reduce(|| 0.0, f64::max)
    }

    /// Number of i-j cluster pairs.
    pub(crate) fn num_cluster_pairs(&self) -> usize {
        self.cj.len()
    }
    /// Number of atom pairs with a set mask bit (each counted twice).
    pub(crate) fn num_masked_pairs(&self) -> usize {
        self.cj.iter().map(|e| e.mask.count_ones() as usize).sum()
    }
}
