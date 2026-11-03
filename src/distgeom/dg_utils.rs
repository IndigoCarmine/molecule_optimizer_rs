//! Metric-matrix embedding utilities: random distance matrices, initial
//! coordinates from the metric matrix (power eigen solver), random
//! coordinates.
//!
//! Port of RDKit `Code/DistGeom/DistGeomUtils.cpp` (pickRandomDistMat,
//! computeInitialCoords, computeRandomCoords),
//! `Code/Numerics/EigenSolvers/PowerEigenSolver.cpp`, and the needed parts of
//! `Code/Numerics/{SymmMatrix,Vector}.h`.
//! RDKit: Copyright (C) 2004-2026 Greg Landrum and other RDKit contributors,
//! BSD-3-Clause.

use super::bounds::BoundsMatrix;
use super::rng::{DoubleSource, MinstdRand};

const EIGVAL_TOL: f64 = 0.001;

/// Packed lower-triangular symmetric matrix (RDKit `SymmMatrix<double>`).
#[derive(Debug, Clone)]
pub struct SymmMatrix {
    n: usize,
    data: Vec<f64>,
}

impl SymmMatrix {
    pub fn new(n: usize, init: f64) -> Self {
        SymmMatrix { n, data: vec![init; n * (n + 1) / 2] }
    }
    pub fn num_rows(&self) -> usize {
        self.n
    }
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if j <= i { self.data[i * (i + 1) / 2 + j] } else { self.data[j * (j + 1) / 2 + i] }
    }
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        if j <= i {
            self.data[i * (i + 1) / 2 + j] = v;
        } else {
            self.data[j * (j + 1) / 2 + i] = v;
        }
    }
    pub fn data(&self) -> &[f64] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// RDKit `multiply(SymmMatrix, Vector, Vector)`.
    fn mul_vec(&self, x: &[f64], y: &mut [f64]) {
        let n = self.n;
        let a = &self.data;
        for i in 0..n {
            let mut acc = 0.0;
            let mut id_a = i * (i + 1) / 2;
            for &xj in x.iter().take(i + 1) {
                acc += a[id_a] * xj;
                id_a += 1;
            }
            id_a -= 1;
            for (j, &xj) in x.iter().enumerate().take(n).skip(i + 1) {
                id_a += j;
                acc += a[id_a] * xj;
            }
            y[i] = acc;
        }
    }
}

/// RDKit `DistGeom::pickRandomDistMat`. Returns the largest distance.
pub fn pick_random_dist_mat(mmat: &BoundsMatrix, dist_mat: &mut SymmMatrix, rng: &mut DoubleSource) -> f64 {
    let npt = mmat.num_rows();
    assert_eq!(npt, dist_mat.num_rows());
    let mut largest = -1.0;
    let d = dist_mat.data_mut();
    for i in 1..npt {
        let id = i * (i + 1) / 2;
        for j in 0..i {
            let ub = mmat.upper(i, j);
            let lb = mmat.lower(i, j);
            let rval = rng.next_f64();
            let v = lb + rval * (ub - lb);
            d[id + j] = v;
            if v > largest {
                largest = v;
            }
        }
    }
    largest
}

fn norm_l2(v: &[f64]) -> f64 {
    let mut s = 0.0;
    for x in v {
        s += x * x;
    }
    s.sqrt()
}

fn normalize(v: &mut [f64]) {
    let l = norm_l2(v);
    if l < 1e-16 {
        return;
    }
    for x in v.iter_mut() {
        *x /= l;
    }
}

/// RDKit `Vector::setToRandom(seed)`.
fn set_to_random(v: &mut [f64], seed: u32) {
    let mut g = DoubleSource::new(if seed > 0 { seed } else { 1 });
    for x in v.iter_mut() {
        *x = g.next_f64();
    }
    normalize(v);
}

/// RDKit `RDNumeric::EigenSolvers::powerEigenSolver`. `mat` is deflated in
/// place. `eig_vecs` is `num_eig × N` row-major. Returns `converged`.
pub fn power_eigen_solver(
    num_eig: usize,
    mat: &mut SymmMatrix,
    eig_vals: &mut [f64],
    eig_vecs: &mut [f64],
    mut seed: i32,
) -> bool {
    const MAX_ITERATIONS: usize = 1000;
    const TOLERANCE: f64 = 0.001;
    const HUGE_EIGVAL: f64 = 1.0e10;
    const TINY_EIGVAL: f64 = 1.0e-10;
    let n = mat.num_rows();
    let mut v = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut converged = false;
    if seed <= 0 {
        // RDKit uses clock() here; we stay deterministic.
        seed = 1;
    }
    for ei in 0..num_eig {
        let mut eig_val = -HUGE_EIGVAL;
        seed = seed.wrapping_add(ei as i32);
        set_to_random(&mut v, seed as u32);
        converged = false;
        for _ in 0..MAX_ITERATIONS {
            mat.mul_vec(&v, &mut z);
            let prev_val = eig_val;
            // largestAbsValId
            let mut res = -1.0;
            let mut id = n;
            for (i, zi) in z.iter().enumerate() {
                if zi.abs() > res {
                    res = zi.abs();
                    id = i;
                }
            }
            eig_val = z[id];
            if eig_val.abs() < TINY_EIGVAL {
                break;
            }
            for i in 0..n {
                v[i] = z[i] / eig_val;
            }
            if (eig_val - prev_val).abs() < TOLERANCE {
                converged = true;
                break;
            }
        }
        if !converged {
            break;
        }
        normalize(&mut v);
        eig_vecs[ei * n..(ei + 1) * n].copy_from_slice(&v);
        eig_vals[ei] = eig_val;
        let md = mat.data_mut();
        for i in 0..n {
            let id = i * (i + 1) / 2;
            for j in 0..=i {
                md[id + j] -= eig_val * v[i] * v[j];
            }
        }
    }
    converged
}

/// RDKit `DistGeom::computeInitialCoords`. `positions` is a flat array with
/// dimension `dim` (3 or 4) per point.
pub fn compute_initial_coords(
    dist_mat: &SymmMatrix,
    positions: &mut [f64],
    dim: usize,
    rng: &mut DoubleSource,
    rand_neg_eig: bool,
    num_zero_fail: u32,
) -> bool {
    let n = dist_mat.num_rows();
    assert_eq!(positions.len(), n * dim);
    let data = dist_mat.data();
    let mut sq = SymmMatrix::new(n, 0.0);
    let mut t = SymmMatrix::new(n, 0.0);
    let mut eig_vecs = vec![0.0; dim * n];
    let mut eig_vals = vec![0.0; dim];

    let mut sum_sq_d2 = 0.0;
    {
        let sqd = sq.data_mut();
        for (i, d) in data.iter().enumerate() {
            sqd[i] = d * d;
            sum_sq_d2 += sqd[i];
        }
    }
    sum_sq_d2 /= (n * n) as f64;

    let mut sq_d0i = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            sq_d0i[i] += sq.get(i, j);
        }
        sq_d0i[i] /= n as f64;
        sq_d0i[i] -= sum_sq_d2;
        if sq_d0i[i] < EIGVAL_TOL && n > 3 {
            return false;
        }
    }
    for i in 0..n {
        for j in 0..=i {
            let val = 0.5 * (sq_d0i[i] + sq_d0i[j] - sq.get(i, j));
            t.set(i, j, val);
        }
    }
    let n_eigs = dim.min(n);
    power_eigen_solver(n_eigs, &mut t, &mut eig_vals, &mut eig_vecs, (sum_sq_d2 * n as f64) as i32);

    let mut found_neg = false;
    let mut zero_eigs = 0u32;
    for e in eig_vals.iter_mut().take(dim) {
        if *e > EIGVAL_TOL {
            *e = e.sqrt();
        } else if e.abs() < EIGVAL_TOL {
            *e = 0.0;
            zero_eigs += 1;
        } else {
            found_neg = true;
        }
    }
    if found_neg && !rand_neg_eig {
        return false;
    }
    if zero_eigs >= num_zero_fail && n > 3 {
        return false;
    }
    for i in 0..n {
        for j in 0..dim {
            positions[i * dim + j] = if eig_vals[j] >= 0.0 {
                eig_vals[j] * eig_vecs[j * n + i]
            } else {
                1.0 - 2.0 * rng.next_f64()
            };
        }
    }
    true
}

/// RDKit `DistGeom::computeRandomCoords`.
pub fn compute_random_coords(positions: &mut [f64], box_size: f64, rng: &mut DoubleSource) -> bool {
    for p in positions.iter_mut() {
        *p = box_size * (rng.next_f64() - 0.5);
    }
    true
}

#[allow(dead_code)]
fn _unused(_: MinstdRand) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_solver_dense() {
        // [[2,1],[1,2]]: eigenvalues 3 and 1. (RDKit's solver uses the largest
        // component of M·v as eigenvalue estimate, so it is only approximate.)
        let mut m = SymmMatrix::new(2, 0.0);
        m.set(0, 0, 2.0);
        m.set(1, 0, 1.0);
        m.set(1, 1, 2.0);
        let orig = m.clone();
        let mut vals = vec![0.0; 2];
        let mut vecs = vec![0.0; 4];
        assert!(power_eigen_solver(2, &mut m, &mut vals, &mut vecs, 7));
        assert!((vals[0] - 3.0).abs() < 1e-2, "{vals:?}");
        assert!((vals[1] - 1.0).abs() < 2e-2, "{vals:?}");
        let mut mv = vec![0.0; 2];
        orig.mul_vec(&vecs[0..2], &mut mv);
        for k in 0..2 {
            assert!((mv[k] - vals[0] * vecs[k]).abs() < 1e-2);
        }
    }

    #[test]
    fn initial_coords_reproduce_square() {
        // planar 5-point set; the 3rd coordinate comes out ~0
        let pts = [[0.0, 0.0], [1.2, 0.0], [1.0, 1.1], [0.0, 1.0], [0.3, 0.6]];
        let n = pts.len();
        let mut d = SymmMatrix::new(n, 0.0);
        for i in 0..n {
            for j in 0..i {
                let dx: f64 = pts[i][0] - pts[j][0];
                let dy: f64 = pts[i][1] - pts[j][1];
                d.set(i, j, (dx * dx + dy * dy).sqrt());
            }
        }
        let mut pos = vec![0.0; n * 3];
        let mut rng = DoubleSource::new(1);
        assert!(compute_initial_coords(&d, &mut pos, 3, &mut rng, true, 2));
        let p = |i: usize| [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];
        let dd = |a: [f64; 3], b: [f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        assert!((dd(p(0), p(2)) - (1.0f64 + 1.21).sqrt()).abs() < 1e-2);
    }
}
