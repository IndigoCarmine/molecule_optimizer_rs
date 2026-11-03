//! Smooth particle-mesh Ewald: reciprocal-space part.
//!
//! Implementation of U. Essmann, L. Perera, M. L. Berkowitz, T. Darden,
//! H. Lee, L. G. Pedersen, "A smooth particle mesh Ewald method",
//! J. Chem. Phys. 103, 8577 (1995):
//!
//! * charges are spread on a `K1 × K2 × K3` grid with cardinal B-splines
//!   of order `p` in the scaled fractional coordinates `u_d = K_d s_d`;
//! * `E_rec = 1/2 Σ_m D(m) |F(Q)(m)|²`,
//!   `D(m) = exp(-π²|m|²/β²) B(m) / (π V |m|²)`, with the B-spline moduli
//!   `B(m) = Π_d |Σ_{j=0}^{p-2} M_p(j+1) exp(2πi m_d j/K_d)|^-2`;
//! * the gradient is obtained by gathering the convolution
//!   `F⁻¹[D · F(Q)]` with the spline derivatives.
//!
//! Determinism: the charge grid is filled plane by plane (each plane is
//! owned by one task, which visits the atoms touching it in a fixed
//! order), the 3D FFT is done as independent 1D line transforms, and the
//! energy is reduced over fixed chunks in a fixed order, so results do not
//! depend on the number of threads.

use crate::core::UnitCell;
use crate::error::{Error, Result};
use rayon::prelude::*;
use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

/// Smallest `n' ≥ n` whose prime factors are 2, 3, 5 and 7.
pub fn fft_friendly_size(n: usize) -> usize {
    let mut m = n.max(1);
    loop {
        let mut r = m;
        for f in [2, 3, 5, 7] {
            while r.is_multiple_of(f) {
                r /= f;
            }
        }
        if r == 1 {
            return m;
        }
        m += 1;
    }
}

/// Values `M_p(w + j)` and derivatives `M_p'(w + j)`, `j = 0..p`, of the
/// cardinal B-spline of order `p ≥ 2` for `w ∈ [0, 1)`.
pub fn bspline(w: f64, p: usize, th: &mut [f64], dth: &mut [f64]) {
    debug_assert!(p >= 2 && th.len() >= p && dth.len() >= p);
    th[0] = w;
    th[1] = 1.0 - w;
    for v in th[2..p].iter_mut() {
        *v = 0.0;
    }
    if p == 2 {
        dth[0] = 1.0;
        dth[1] = -1.0;
        return;
    }
    for n in 3..=p {
        if n == p {
            // derivative from order p-1: M_p'(x) = M_{p-1}(x) - M_{p-1}(x-1)
            dth[0] = th[0];
            for j in 1..p {
                dth[j] = th[j] - th[j - 1];
            }
        }
        let inv = 1.0 / (n - 1) as f64;
        // descending j so that th[j-1] is still the order n-1 value
        for j in (0..n).rev() {
            let x = w + j as f64;
            let cur = if j < n - 1 { th[j] } else { 0.0 };
            let prev = if j > 0 { th[j - 1] } else { 0.0 };
            th[j] = (x * cur + (n as f64 - x) * prev) * inv;
        }
    }
}

/// Reciprocal-space SPME evaluator for a fixed cell.
pub(crate) struct Pme {
    pub order: usize,
    pub k: [usize; 3],
    pub beta: f64,
    pub(crate) recip: [[f64; 3]; 3],
    /// Influence function `D(m)` in the transposed (y, z, x) layout.
    pub(crate) infl: Vec<f64>,
    fwd: [Arc<dyn Fft<f64>>; 3],
    inv: [Arc<dyn Fft<f64>>; 3],
    /// Charged atoms and their charges (times `sqrt(C)`).
    pub(crate) atoms: Vec<u32>,
    pub(crate) q: Vec<f64>,
    work: Mutex<Option<(Vec<Complex64>, Vec<Complex64>)>>,
}

const LINES_PER_TASK: usize = 16;

impl Pme {
    pub fn new(cell: &UnitCell, beta: f64, grid_spacing: f64, order: usize, q_scaled: &[f64]) -> Result<Pme> {
        if !(3..=16).contains(&order) {
            return Err(Error::InvalidArgument(format!("PME spline order must be in 3..=16 (got {order})")));
        }
        if grid_spacing.is_nan() || grid_spacing <= 0.0 {
            return Err(Error::InvalidArgument(format!("PME grid spacing must be positive (got {grid_spacing})")));
        }
        let len = cell.lengths();
        let k = len.map(|l| fft_friendly_size(((l / grid_spacing).ceil() as usize).max(order + 1)));
        if k.iter().product::<usize>() > 1 << 28 {
            return Err(Error::InvalidArgument("PME grid too large".into()));
        }
        let recip = cell.reciprocal();
        let volume = cell.volume();
        // B-spline moduli
        let mut th = vec![0.0; order];
        let mut dth = vec![0.0; order];
        bspline(0.0, order, &mut th, &mut dth);
        let bmod = |kk: usize| -> Vec<f64> {
            let mut b: Vec<f64> = (0..kk)
                .map(|m| {
                    let (mut re, mut im) = (0.0, 0.0);
                    for j in 0..order - 1 {
                        let arg = 2.0 * PI * (m * j) as f64 / kk as f64;
                        re += th[j + 1] * arg.cos();
                        im += th[j + 1] * arg.sin();
                    }
                    re * re + im * im
                })
                .collect();
            for m in 0..kk {
                if b[m] < 1e-7 {
                    b[m] = 0.5 * (b[(m + kk - 1) % kk] + b[(m + 1) % kk]);
                }
            }
            b
        };
        let bm = [bmod(k[0]), bmod(k[1]), bmod(k[2])];
        let [k1, k2, k3] = k;
        let fac = PI * PI / (beta * beta);
        let mut infl = vec![0.0; k1 * k2 * k3];
        infl.par_chunks_mut(k3 * k1).enumerate().for_each(|(y, ch)| {
            let my = if y <= k2 / 2 { y as f64 } else { y as f64 - k2 as f64 };
            for z in 0..k3 {
                let mz = if z <= k3 / 2 { z as f64 } else { z as f64 - k3 as f64 };
                for x in 0..k1 {
                    let mx = if x <= k1 / 2 { x as f64 } else { x as f64 - k1 as f64 };
                    if x == 0 && y == 0 && z == 0 {
                        ch[z * k1 + x] = 0.0;
                        continue;
                    }
                    let mut m = [0.0; 3];
                    for d in 0..3 {
                        m[d] = mx * recip[0][d] + my * recip[1][d] + mz * recip[2][d];
                    }
                    let m2 = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
                    ch[z * k1 + x] = (-fac * m2).exp() / (PI * volume * m2) / (bm[0][x] * bm[1][y] * bm[2][z]);
                }
            }
        });
        let mut planner = FftPlanner::<f64>::new();
        let fwd = [planner.plan_fft_forward(k1), planner.plan_fft_forward(k2), planner.plan_fft_forward(k3)];
        let inv = [planner.plan_fft_inverse(k1), planner.plan_fft_inverse(k2), planner.plan_fft_inverse(k3)];
        let mut atoms = Vec::new();
        let mut q = Vec::new();
        for (i, &qi) in q_scaled.iter().enumerate() {
            if qi != 0.0 {
                atoms.push(i as u32);
                q.push(qi);
            }
        }
        Ok(Pme { order, k, beta, recip, infl, fwd, inv, atoms, q, work: Mutex::new(None) })
    }

    /// Reciprocal-space energy; adds the gradient to `grad` if given.
    pub fn energy_grad(&self, pos: &[f64], grad: Option<&mut [f64]>) -> f64 {
        let na = self.atoms.len();
        if na == 0 {
            return 0.0;
        }
        let p = self.order;
        let [k1, k2, k3] = self.k;
        let kk = [k1, k2, k3];
        // splines
        let mut base = vec![[0u32; 3]; na];
        let mut th = vec![0.0; na * 3 * p];
        let mut dth = vec![0.0; na * 3 * p];
        base.par_iter_mut()
            .zip(th.par_chunks_mut(3 * p).zip(dth.par_chunks_mut(3 * p)))
            .enumerate()
            .with_min_len(64)
            .for_each(|(a, (b, (t, dt)))| {
                let i = self.atoms[a] as usize;
                let r = [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];
                for d in 0..3 {
                    let s = self.recip[d][0] * r[0] + self.recip[d][1] * r[1] + self.recip[d][2] * r[2];
                    let u = s * kk[d] as f64;
                    let fl = u.floor();
                    let mut w = u - fl;
                    if w >= 1.0 {
                        w = 0.0;
                    }
                    b[d] = (fl as i64).rem_euclid(kk[d] as i64) as u32;
                    bspline(w, p, &mut t[d * p..(d + 1) * p], &mut dt[d * p..(d + 1) * p]);
                }
            });
        // atoms bucketed by their first grid index (fixed order: atom index)
        let mut start = vec![0usize; k1 + 1];
        for b in &base {
            start[b[0] as usize + 1] += 1;
        }
        for x in 0..k1 {
            start[x + 1] += start[x];
        }
        let mut fill = start.clone();
        let mut bucket = vec![0u32; na];
        for (a, b) in base.iter().enumerate() {
            bucket[fill[b[0] as usize]] = a as u32;
            fill[b[0] as usize] += 1;
        }
        let n = k1 * k2 * k3;
        let mut work = self.work.lock().ok().and_then(|mut g| g.take()).unwrap_or_else(|| {
            (vec![Complex64::new(0.0, 0.0); n], vec![Complex64::new(0.0, 0.0); n])
        });
        let (ga, gb) = (&mut work.0, &mut work.1);
        // spread
        ga.par_chunks_mut(k2 * k3).enumerate().for_each(|(x, plane)| {
            plane.iter_mut().for_each(|c| *c = Complex64::new(0.0, 0.0));
            for j1 in 0..p {
                let b = (x + j1) % k1;
                for &a in &bucket[start[b]..start[b + 1]] {
                    let a = a as usize;
                    let t = &th[a * 3 * p..(a + 1) * 3 * p];
                    let w1 = self.q[a] * t[j1];
                    let (b2, b3) = (base[a][1] as usize, base[a][2] as usize);
                    for j2 in 0..p {
                        let y = (b2 + k2 * p - j2) % k2;
                        let w12 = w1 * t[p + j2];
                        let row = &mut plane[y * k3..(y + 1) * k3];
                        for j3 in 0..p {
                            let z = (b3 + k3 * p - j3) % k3;
                            row[z].re += w12 * t[2 * p + j3];
                        }
                    }
                }
            }
        });
        self.forward(ga, gb);
        // energy and D·F(Q)
        let do_grad = grad.is_some();
        let parts: Vec<f64> = gb
            .par_chunks_mut(k3 * k1)
            .zip(self.infl.par_chunks(k3 * k1))
            .map(|(g, d)| {
                let mut e = 0.0;
                for (c, &dv) in g.iter_mut().zip(d) {
                    e += dv * c.norm_sqr();
                    if do_grad {
                        *c *= dv;
                    }
                }
                e
            })
            .collect();
        let energy = 0.5 * parts.iter().sum::<f64>();
        if let Some(grad) = grad {
            self.inverse(gb, ga);
            let conv: &[Complex64] = ga;
            let gr: Vec<[f64; 3]> = (0..na)
                .into_par_iter()
                .with_min_len(64)
                .map(|a| {
                    let t = &th[a * 3 * p..(a + 1) * 3 * p];
                    let dt = &dth[a * 3 * p..(a + 1) * 3 * p];
                    let (b1, b2, b3) = (base[a][0] as usize, base[a][1] as usize, base[a][2] as usize);
                    let (mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0);
                    for j1 in 0..p {
                        let x = (b1 + k1 * p - j1) % k1;
                        for j2 in 0..p {
                            let y = (b2 + k2 * p - j2) % k2;
                            let row = &conv[(x * k2 + y) * k3..(x * k2 + y + 1) * k3];
                            let (mut c0, mut c1) = (0.0, 0.0);
                            for j3 in 0..p {
                                let z = (b3 + k3 * p - j3) % k3;
                                let v = row[z].re;
                                c0 += v * t[2 * p + j3];
                                c1 += v * dt[2 * p + j3];
                            }
                            s1 += dt[j1] * t[p + j2] * c0;
                            s2 += t[j1] * dt[p + j2] * c0;
                            s3 += t[j1] * t[p + j2] * c1;
                        }
                    }
                    let q = self.q[a];
                    let s = [s1 * k1 as f64 * q, s2 * k2 as f64 * q, s3 * k3 as f64 * q];
                    let mut g = [0.0; 3];
                    for d in 0..3 {
                        g[d] = s[0] * self.recip[0][d] + s[1] * self.recip[1][d] + s[2] * self.recip[2][d];
                    }
                    g
                })
                .collect();
            for (a, g) in gr.iter().enumerate() {
                let i = self.atoms[a] as usize;
                for d in 0..3 {
                    grad[3 * i + d] += g[d];
                }
            }
        }
        if let Ok(mut g) = self.work.lock() {
            *g = Some(work);
        }
        energy
    }

    /// Forward 3D FFT of `a` (layout x, y, z) into `b` (layout y, z, x).
    fn forward(&self, a: &mut [Complex64], b: &mut [Complex64]) {
        let [k1, k2, k3] = self.k;
        let [fx, fy, fz] = &self.fwd;
        a.par_chunks_mut(k3 * LINES_PER_TASK)
            .for_each_init(|| scratch(fz), |s, ch| fz.process_with_scratch(ch, s));
        a.par_chunks_mut(k2 * k3).for_each_init(
            || (vec![Complex64::new(0.0, 0.0); k2 * k3], scratch(fy)),
            |(tmp, s), plane| {
                transpose(plane, tmp, k2, k3);
                fy.process_with_scratch(tmp, s);
                transpose(tmp, plane, k3, k2);
            },
        );
        let a: &[Complex64] = a;
        b.par_chunks_mut(k3 * k1).enumerate().for_each_init(
            || scratch(fx),
            |s, (y, ch)| {
                for z in 0..k3 {
                    for x in 0..k1 {
                        ch[z * k1 + x] = a[(x * k2 + y) * k3 + z];
                    }
                }
                fx.process_with_scratch(ch, s);
            },
        );
    }

    /// Inverse (unnormalized) 3D FFT of `b` (layout y, z, x) into `a`
    /// (layout x, y, z).
    fn inverse(&self, b: &mut [Complex64], a: &mut [Complex64]) {
        let [k1, k2, k3] = self.k;
        let [ix, iy, iz] = &self.inv;
        b.par_chunks_mut(k1 * LINES_PER_TASK)
            .for_each_init(|| scratch(ix), |s, ch| ix.process_with_scratch(ch, s));
        let bb: &[Complex64] = b;
        a.par_chunks_mut(k2 * k3).enumerate().for_each(|(x, plane)| {
            for y in 0..k2 {
                for z in 0..k3 {
                    plane[y * k3 + z] = bb[(y * k3 + z) * k1 + x];
                }
            }
        });
        a.par_chunks_mut(k2 * k3).for_each_init(
            || (vec![Complex64::new(0.0, 0.0); k2 * k3], scratch(iy)),
            |(tmp, s), plane| {
                transpose(plane, tmp, k2, k3);
                iy.process_with_scratch(tmp, s);
                transpose(tmp, plane, k3, k2);
            },
        );
        a.par_chunks_mut(k3 * LINES_PER_TASK)
            .for_each_init(|| scratch(iz), |s, ch| iz.process_with_scratch(ch, s));
    }
}

fn scratch(f: &Arc<dyn Fft<f64>>) -> Vec<Complex64> {
    vec![Complex64::new(0.0, 0.0); f.get_inplace_scratch_len()]
}

/// `dst[c * rows + r] = src[r * cols + c]`.
fn transpose(src: &[Complex64], dst: &mut [Complex64], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn friendly_sizes() {
        assert_eq!(fft_friendly_size(11), 12);
        assert_eq!(fft_friendly_size(13), 14);
        assert_eq!(fft_friendly_size(17), 18);
        assert_eq!(fft_friendly_size(97), 98);
        assert_eq!(fft_friendly_size(64), 64);
    }

    #[test]
    fn bspline_partition_of_unity_and_derivative() {
        for p in 3..=8 {
            let mut th = vec![0.0; p];
            let mut dth = vec![0.0; p];
            for &w in &[0.0, 0.1, 0.37, 0.5, 0.99] {
                bspline(w, p, &mut th, &mut dth);
                let s: f64 = th.iter().sum();
                assert!((s - 1.0).abs() < 1e-14, "p={p} w={w} sum={s}");
                let ds: f64 = dth.iter().sum();
                assert!(ds.abs() < 1e-14);
                // finite-difference derivative
                let h = 1e-6;
                let mut tp = vec![0.0; p];
                let mut tm = vec![0.0; p];
                let mut scratch = vec![0.0; p];
                if w > h && w < 1.0 - h {
                    bspline(w + h, p, &mut tp, &mut scratch);
                    bspline(w - h, p, &mut tm, &mut scratch);
                    for j in 0..p {
                        let fd = (tp[j] - tm[j]) / (2.0 * h);
                        assert!((fd - dth[j]).abs() < 1e-7, "p={p} j={j}: {fd} vs {}", dth[j]);
                    }
                }
            }
        }
        // order 4 at w = 0: M4(0..4) = 0, 1/6, 2/3, 1/6
        let mut th = [0.0; 4];
        let mut dth = [0.0; 4];
        bspline(0.0, 4, &mut th, &mut dth);
        assert!((th[1] - 1.0 / 6.0).abs() < 1e-15 && (th[2] - 2.0 / 3.0).abs() < 1e-15);
    }
}
