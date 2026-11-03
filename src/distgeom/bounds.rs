//! Distance bounds matrix and triangle smoothing.
//!
//! Port of RDKit `Code/DistGeom/BoundsMatrix.h` and
//! `Code/DistGeom/TriangleSmooth.cpp`.
//! RDKit: Copyright (C) 2004-2026 Greg Landrum and other RDKit contributors,
//! BSD-3-Clause.

/// N×N matrix holding upper bounds in the upper triangle (`i < j`) and lower
/// bounds in the lower triangle, exactly like RDKit's `BoundsMatrix`.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundsMatrix {
    n: usize,
    data: Vec<f64>,
}

impl BoundsMatrix {
    /// All entries zero.
    pub fn new(n: usize) -> Self {
        BoundsMatrix { n, data: vec![0.0; n * n] }
    }

    /// RDKit `initBoundsMat` defaults: lower 0, upper 1000.
    pub fn new_initialized(n: usize) -> Self {
        let mut m = Self::new(n);
        m.init(0.0, 1000.0);
        m
    }

    /// RDKit `initBoundsMat(mmat, defaultMin, defaultMax)`.
    pub fn init(&mut self, default_min: f64, default_max: f64) {
        for i in 1..self.n {
            for j in 0..i {
                self.set_upper(i, j, default_max);
                self.set_lower(i, j, default_min);
            }
        }
    }

    pub fn num_rows(&self) -> usize {
        self.n
    }

    /// Raw element (row-major), as `getValUnchecked`.
    #[inline]
    pub fn val(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }
    #[inline]
    pub fn set_val(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] = v;
    }

    #[inline]
    pub fn upper(&self, i: usize, j: usize) -> f64 {
        if i < j { self.val(i, j) } else { self.val(j, i) }
    }
    #[inline]
    pub fn lower(&self, i: usize, j: usize) -> f64 {
        if i < j { self.val(j, i) } else { self.val(i, j) }
    }
    #[inline]
    pub fn set_upper(&mut self, i: usize, j: usize, v: f64) {
        debug_assert!(v >= 0.0, "negative upper bound");
        if i < j { self.set_val(i, j, v) } else { self.set_val(j, i, v) }
    }
    #[inline]
    pub fn set_lower(&mut self, i: usize, j: usize, v: f64) {
        debug_assert!(v >= 0.0, "negative lower bound");
        if i < j { self.set_val(j, i, v) } else { self.set_val(i, j, v) }
    }

    /// Sets the upper bound only if smaller than the current one and larger
    /// than the lower bound.
    pub fn set_upper_if_better(&mut self, i: usize, j: usize, v: f64) {
        if v < self.upper(i, j) && v > self.lower(i, j) {
            self.set_upper(i, j, v);
        }
    }
    pub fn set_lower_if_better(&mut self, i: usize, j: usize, v: f64) {
        if v > self.lower(i, j) && v < self.upper(i, j) {
            self.set_lower(i, j, v);
        }
    }

    /// All lower bounds ≤ upper bounds.
    pub fn check_valid(&self) -> bool {
        for i in 1..self.n {
            for j in 0..i {
                if self.upper(i, j) < self.lower(i, j) {
                    return false;
                }
            }
        }
        true
    }

    /// Upper-bound matrix as nested rows (diagonal 0).
    pub fn upper_matrix(&self) -> Vec<Vec<f64>> {
        (0..self.n).map(|i| (0..self.n).map(|j| if i == j { 0.0 } else { self.upper(i, j) }).collect()).collect()
    }
    /// Lower-bound matrix as nested rows (diagonal 0).
    pub fn lower_matrix(&self) -> Vec<Vec<f64>> {
        (0..self.n).map(|i| (0..self.n).map(|j| if i == j { 0.0 } else { self.lower(i, j) }).collect()).collect()
    }

    /// Raw row-major data (upper bounds above the diagonal, lower below),
    /// same layout as RDKit's `GetMoleculeBoundsMatrix`.
    pub fn raw(&self) -> &[f64] {
        &self.data
    }

    /// Triangle-bounds smoothing (RDKit `triangleSmoothBounds`). Returns
    /// `false` if the bounds are inconsistent.
    pub fn triangle_smooth(&mut self, tol: f64) -> bool {
        triangle_smooth_bounds(self, tol)
    }
}

/// RDKit `DistGeom::triangleSmoothBounds`.
pub fn triangle_smooth_bounds(bm: &mut BoundsMatrix, tol: f64) -> bool {
    let npt = bm.n;
    if npt == 0 {
        return true;
    }
    for k in 0..npt {
        for i in 0..npt - 1 {
            if i == k {
                continue;
            }
            let (ii, ik) = if i > k { (k, i) } else { (i, k) };
            let uik = bm.val(ii, ik);
            let lik = bm.val(ik, ii);
            for j in i + 1..npt {
                if j == k {
                    continue;
                }
                let (jj, jk) = if j > k { (k, j) } else { (j, k) };
                let ukj = bm.val(jj, jk);
                let sum_uik_ukj = uik + ukj;
                if bm.val(i, j) > sum_uik_ukj {
                    bm.set_val(i, j, sum_uik_ukj);
                }
                let diff_lik_ujk = lik - ukj;
                let diff_ljk_uik = bm.val(jk, jj) - uik;
                if bm.val(j, i) < diff_lik_ujk {
                    bm.set_val(j, i, diff_lik_ujk);
                } else if bm.val(j, i) < diff_ljk_uik {
                    bm.set_val(j, i, diff_ljk_uik);
                }
                let l_bound = bm.val(j, i);
                let u_bound = bm.val(i, j);
                if tol > 0.0 && (l_bound - u_bound) / l_bound > 0.0 && (l_bound - u_bound) / l_bound < tol {
                    bm.set_val(i, j, l_bound);
                } else if l_bound - u_bound > 0.0 {
                    return false;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothing_tightens_upper_bounds() {
        let mut m = BoundsMatrix::new_initialized(3);
        m.set_upper(0, 1, 1.5);
        m.set_lower(0, 1, 1.4);
        m.set_upper(1, 2, 1.5);
        m.set_lower(1, 2, 1.4);
        assert!(m.triangle_smooth(0.0));
        assert!((m.upper(0, 2) - 3.0).abs() < 1e-12);
        assert!(m.check_valid());
    }

    #[test]
    fn smoothing_detects_inconsistency() {
        let mut m = BoundsMatrix::new_initialized(3);
        m.set_upper(0, 1, 1.0);
        m.set_lower(0, 1, 0.9);
        m.set_upper(1, 2, 1.0);
        m.set_lower(1, 2, 0.9);
        m.set_lower(0, 2, 5.0);
        m.set_upper(0, 2, 6.0);
        assert!(!m.triangle_smooth(0.0));
    }
}
