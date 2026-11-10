//! Energy terms of the GAFF functional form (eq. 1 of Wang et al. 2004)
//! that the other force fields of this crate do not provide: the harmonic
//! angle, the cosine torsion (also used for impropers) and the plain
//! Coulomb term. Bonds and the 12-6 Lennard-Jones term reuse
//! [`crate::ff::uff::BondStretchContrib`] and [`crate::ff::uff::VdwContrib`].

use crate::core::geometry::{dot, norm, sub};
use crate::ff::Contrib;
use crate::ff::utils::{Torsion, p3};

/// Harmonic angle `k (θ - θ0)²` (θ in radians).
#[derive(Debug, Clone)]
pub struct HarmonicAngleContrib {
    /// `i`, centre `j`, `k`.
    pub idx: [usize; 3],
    pub force_constant: f64,
    /// Equilibrium angle in radians.
    pub theta0: f64,
}

impl Contrib for HarmonicAngleContrib {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        let [i, j, k] = self.idx;
        let (r1, r2) = (sub(p3(pos, i), p3(pos, j)), sub(p3(pos, k), p3(pos, j)));
        let (l1, l2) = (norm(r1), norm(r2));
        if l1 <= 0.0 || l2 <= 0.0 {
            return 0.0;
        }
        let d = (dot(r1, r2) / (l1 * l2)).clamp(-1.0, 1.0).acos() - self.theta0;
        self.force_constant * d * d
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let [i, j, k] = self.idx;
        let (r1, r2) = (sub(p3(pos, i), p3(pos, j)), sub(p3(pos, k), p3(pos, j)));
        let (l1, l2) = (norm(r1), norm(r2));
        if l1 <= 0.0 || l2 <= 0.0 {
            return;
        }
        let cos = (dot(r1, r2) / (l1 * l2)).clamp(-1.0, 1.0);
        let sin = (1.0 - cos * cos).sqrt();
        // (θ - θ0) / sin θ; at a straight angle with θ0 = π the limit is -1
        let ratio = if sin > 1e-8 {
            (cos.acos() - self.theta0) / sin
        } else if cos < 0.0 && (self.theta0 - std::f64::consts::PI).abs() < 1e-6 {
            -1.0
        } else {
            0.0
        };
        // dE/dx = 2k (θ - θ0) · (-1 / sin θ) · dcos/dx
        let pre = -2.0 * self.force_constant * ratio;
        for d in 0..3 {
            let (u1, u2) = (r1[d] / l1, r2[d] / l2);
            let g1 = pre * (u2 - cos * u1) / l1;
            let g3 = pre * (u1 - cos * u2) / l2;
            grad[3 * i + d] += g1;
            grad[3 * k + d] += g3;
            grad[3 * j + d] -= g1 + g3;
        }
    }
    fn label(&self) -> &'static str {
        "AngleBend"
    }
}

/// Cosine torsion `k (1 + cos(n φ - γ))` with `γ` = 0° or 180°
/// (`cos_phase` = `cos γ` = ±1), for proper torsions and impropers.
#[derive(Debug, Clone)]
pub struct FourierTorsionContrib {
    pub idx: [usize; 4],
    pub force_constant: f64,
    pub periodicity: u32,
    /// `cos γ`: `1.0` for phase 0°, `-1.0` for phase 180°.
    pub cos_phase: f64,
    /// Reported as `"Improper"` instead of `"Torsion"`.
    pub improper: bool,
}

impl FourierTorsionContrib {
    /// `(cos(n φ), d cos(n φ) / d cos φ)` from `c = cos φ` (Chebyshev
    /// polynomials `T_n(c)` and `n U_{n-1}(c)`).
    #[inline]
    fn cos_n(&self, c: f64) -> (f64, f64) {
        let n = self.periodicity;
        if n == 0 {
            return (1.0, 0.0);
        }
        let (mut t_prev, mut t) = (1.0, c);
        let (mut u_prev, mut u) = (0.0, 1.0);
        for _ in 1..n {
            let t_next = 2.0 * c * t - t_prev;
            let u_next = 2.0 * c * u - u_prev;
            (t_prev, t) = (t, t_next);
            (u_prev, u) = (u, u_next);
        }
        (t, n as f64 * u)
    }
}

impl Contrib for FourierTorsionContrib {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        let c = Torsion::at(pos, self.idx).cos_phi;
        self.force_constant * (1.0 + self.cos_phase * self.cos_n(c).0)
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let t = Torsion::at(pos, self.idx);
        t.add_grad(grad, self.idx, self.force_constant * self.cos_phase * self.cos_n(t.cos_phi).1);
    }
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> f64 {
        let t = Torsion::at(pos, self.idx);
        let (cn, dcn) = self.cos_n(t.cos_phi);
        t.add_grad(grad, self.idx, self.force_constant * self.cos_phase * dcn);
        self.force_constant * (1.0 + self.cos_phase * cn)
    }
    fn label(&self) -> &'static str {
        if self.improper { "Improper" } else { "Torsion" }
    }
}

/// Coulomb term `qq / r`, with `qq = C q_i q_j · scale / ε` in kcal·Å/mol.
#[derive(Debug, Clone)]
pub struct CoulombContrib {
    pub idx1: usize,
    pub idx2: usize,
    pub qq: f64,
}

impl Contrib for CoulombContrib {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        let r = norm(sub(p3(pos, self.idx1), p3(pos, self.idx2)));
        if r <= 0.0 { 0.0 } else { self.qq / r }
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let d = sub(p3(pos, self.idx1), p3(pos, self.idx2));
        let r = norm(d);
        if r <= 0.0 {
            return;
        }
        let pre = -self.qq / (r * r * r);
        for (x, dx) in d.iter().enumerate() {
            grad[3 * self.idx1 + x] += pre * dx;
            grad[3 * self.idx2 + x] -= pre * dx;
        }
    }
    fn label(&self) -> &'static str {
        "Electrostatic"
    }
}
