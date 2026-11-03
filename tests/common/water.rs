//! Generated water boxes and a simple bonded water model (shared by the
//! approximate-mode tests and benchmarks).
#![allow(dead_code)]

use molecule_optimizer_rs::core::{Atom, BondOrder, Molecule, UnitCell};
use molecule_optimizer_rs::fast::backend::NonbondedBackend;
use molecule_optimizer_rs::ff::ForceField;
use molecule_optimizer_rs::perception::sanitize;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub const R_OH: f64 = 0.9572;
pub const THETA_HOH: f64 = 104.52;

fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let [w, x, y, z] = q;
    let m = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ];
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// `nx × ny × nz` randomly oriented waters on a lattice with the given
/// spacing (3.1 Å ≈ liquid density) in a periodic orthorhombic box.
/// Atoms are ordered O, H, H per water.
pub fn water_box(n: [usize; 3], spacing: f64, seed: u64) -> Molecule {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let half = (THETA_HOH / 2.0).to_radians();
    let h1 = [R_OH * half.sin(), R_OH * half.cos(), 0.0];
    let h2 = [-R_OH * half.sin(), R_OH * half.cos(), 0.0];
    let mut m = Molecule::new();
    for ix in 0..n[0] {
        for iy in 0..n[1] {
            for iz in 0..n[2] {
                let c = [ix, iy, iz].map(|i| (i as f64 + 0.5) * spacing);
                let c = [0, 1, 2].map(|d| c[d] + rng.gen_range(-0.15..0.15));
                // uniform random rotation (normalized Gaussian quaternion)
                let mut q = [0.0f64; 4];
                loop {
                    for v in &mut q {
                        *v = rng.gen_range(-1.0..1.0);
                    }
                    let l2: f64 = q.iter().map(|v| v * v).sum();
                    if l2 > 1e-3 && l2 <= 1.0 {
                        let l = l2.sqrt();
                        q.iter_mut().for_each(|v| *v /= l);
                        break;
                    }
                }
                let a = rotate(q, h1);
                let b = rotate(q, h2);
                let o = m.add_atom(Atom::new(8), c);
                let p1 = m.add_atom(Atom::new(1), [c[0] + a[0], c[1] + a[1], c[2] + a[2]]);
                let p2 = m.add_atom(Atom::new(1), [c[0] + b[0], c[1] + b[1], c[2] + b[2]]);
                m.add_bond(o, p1, BondOrder::Single);
                m.add_bond(o, p2, BondOrder::Single);
            }
        }
    }
    sanitize(&mut m).expect("sanitize water box");
    m.cell = Some(UnitCell::orthorhombic(n[0] as f64 * spacing, n[1] as f64 * spacing, n[2] as f64 * spacing).unwrap());
    m
}

/// Number of waters `k³` closest to `target` atoms.
pub fn cube_side_for_atoms(target: usize) -> usize {
    ((target as f64 / 3.0).cbrt()).round() as usize
}

/// Harmonic bonded water model (O-H stretch + H-O-H bend), atoms ordered
/// O, H, H.
pub struct BondedWater {
    pub nwat: usize,
    pub kb: f64,
    pub ka: f64,
}

impl BondedWater {
    pub fn new(nwat: usize) -> Self {
        BondedWater { nwat, kb: 1000.0, ka: 100.0 }
    }
    pub fn energy_grad(&self, pos: &[f64], mut grad: Option<&mut [f64]>) -> f64 {
        let th0 = THETA_HOH.to_radians();
        let mut e = 0.0;
        for w in 0..self.nwat {
            let (o, a, b) = (3 * w, 3 * w + 1, 3 * w + 2);
            let p = |i: usize| [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];
            let (po, pa, pb) = (p(o), p(a), p(b));
            let va = [pa[0] - po[0], pa[1] - po[1], pa[2] - po[2]];
            let vb = [pb[0] - po[0], pb[1] - po[1], pb[2] - po[2]];
            let la = (va[0] * va[0] + va[1] * va[1] + va[2] * va[2]).sqrt();
            let lb = (vb[0] * vb[0] + vb[1] * vb[1] + vb[2] * vb[2]).sqrt();
            e += 0.5 * self.kb * ((la - R_OH).powi(2) + (lb - R_OH).powi(2));
            let c = ((va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]) / (la * lb)).clamp(-1.0, 1.0);
            let th = c.acos();
            e += 0.5 * self.ka * (th - th0).powi(2);
            if let Some(g) = grad.as_deref_mut() {
                let ga = self.kb * (la - R_OH) / la;
                let gb = self.kb * (lb - R_OH) / lb;
                let s = (1.0 - c * c).max(1e-12).sqrt();
                let dth = -self.ka * (th - th0) / s; // dE/dcos
                for d in 0..3 {
                    let dca = vb[d] / (la * lb) - c * va[d] / (la * la);
                    let dcb = va[d] / (la * lb) - c * vb[d] / (lb * lb);
                    let fa = ga * va[d] + dth * dca;
                    let fb = gb * vb[d] + dth * dcb;
                    g[3 * a + d] += fa;
                    g[3 * b + d] += fb;
                    g[3 * o + d] -= fa + fb;
                }
            }
        }
        e
    }
}

/// Bonded water model + nonbonded backend as a [`ForceField`].
pub struct WaterFf<B: NonbondedBackend> {
    pub bonded: BondedWater,
    pub nb: B,
}

impl<B: NonbondedBackend> ForceField for WaterFf<B> {
    fn num_points(&self) -> usize {
        3 * self.bonded.nwat
    }
    fn energy(&self, pos: &[f64]) -> f64 {
        self.bonded.energy_grad(pos, None) + self.nb.energy(pos).total()
    }
    fn energy_grad(&self, pos: &[f64], grad: &mut [f64]) -> f64 {
        grad.iter_mut().for_each(|g| *g = 0.0);
        let e = self.bonded.energy_grad(pos, Some(grad));
        e + self.nb.energy_grad(pos, grad).total()
    }
    fn on_accepted_step(&self, pos: &[f64]) {
        self.nb.on_accepted_step(pos);
    }
}
