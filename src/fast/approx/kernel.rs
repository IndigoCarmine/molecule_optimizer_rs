//! Pair interaction functions: scalar per-pair reference implementations
//! and 4×4 cluster-pair kernels (`wide::f64x4`, four j-atoms per vector).
//!
//! Energy of a pair at distance `r` (no exclusion, same fragment if
//! required):
//!
//! * van der Waals (only for `r < rc`):
//!   - MMFF buffered 14-7
//!     `eps (1.07 R/(r + 0.07 R))^7 (1.12 R^7/(r^7 + 0.12 R^7) - 2)`;
//!   - UFF Lennard-Jones `D ((x/r)^12 - 2 (x/r)^6)`;
//!   - modifier: potential shift `E(r) - E(rc)`, or a fifth-order switch
//!     `S(t) = 1 - 10t³ + 15t⁴ - 6t⁵`, `t = (r - r_sw)/(rc - r_sw)`
//!     (`r_sw ≥ rc` = plain truncation).
//! * electrostatics `C qi qj f(r)` with `C = 332.0716 / D`:
//!   - full: `f = 1/ρ^n` for all pairs (`ρ = r + δ`, `δ` = MMFF's 0.05 Å
//!     buffer, `n = 2` for the distance-dependent dielectric);
//!   - reaction field (`r < rc`): `f = 1/ρ^n + k r² - c` with
//!     `k = κ n / (rc ρc^(n+1))`, `κ = (eps_rf - D)/(2 eps_rf + D)` and
//!     `c = 1/ρc^n + k rc²` (for `eps_rf = ∞`, `κ = 1/2`: energy and force
//!     vanish at the cutoff; for `δ = 0`, `n = 1` this is the textbook
//!     reaction field);
//!   - Ewald real space (`r < rc`): `f = erfc(βr)/r` (no buffer).

use super::erfc::{ERFC_K, erfc_exp, erfc_exp_inv};
use super::pairlist::PairList;
use super::params::{NbParams, VdwKind};
use wide::{CmpLt, f64x4};

/// `2/√π`
pub(crate) const TWO_OVER_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Modifier {
    /// Plain truncation at the cutoff.
    None,
    /// `E(r) - E(rc)`.
    Shift,
    /// Switch between `rsw` and `rc` (`inv_w = 1/(rc - rsw)`).
    Switch { rsw: f64, inv_w: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ElecKind {
    None,
    Full,
    ReactionField,
    Ewald,
}

/// Kernel settings.
#[derive(Debug, Clone)]
pub(crate) struct KernelParams {
    pub vdw: VdwKind,
    pub rc2: f64,
    pub modifier: Modifier,
    pub elec: ElecKind,
    /// Exponent `n = 2` (distance-dependent dielectric).
    pub ele_n2: bool,
    /// Electrostatic buffer `δ`.
    pub delta: f64,
    pub k_rf: f64,
    pub c_rf: f64,
    pub beta: f64,
    /// Per type pair `E_vdw(rc)` (potential shift).
    pub shift: Vec<f64>,
}

impl KernelParams {
    pub fn new(p: &NbParams, rc: f64, modifier: Modifier, elec: ElecKind, eps_rf: f64, beta: f64) -> Self {
        let ele_n2 = p.distance_dielectric;
        let delta = if elec == ElecKind::Ewald { 0.0 } else { p.ele_buffer };
        let (mut k_rf, mut c_rf) = (0.0, 0.0);
        if elec == ElecKind::ReactionField {
            let d = p.coulomb_dielectric();
            let kappa = if eps_rf.is_infinite() { 0.5 } else { (eps_rf - d) / (2.0 * eps_rf + d) };
            let rho = rc + delta;
            let n = if ele_n2 { 2.0 } else { 1.0 };
            k_rf = kappa * n / (rc * rho.powf(n + 1.0));
            c_rf = 1.0 / rho.powf(n) + k_rf * rc * rc;
        }
        let nt = p.ntypes;
        let mut shift = vec![0.0; nt * nt];
        if modifier == Modifier::Shift {
            for k in 0..nt * nt {
                shift[k] = vdw_raw(p.kind, p.p1[k], p.p2[k], p.p3[k], rc).0;
            }
        }
        KernelParams { vdw: p.kind, rc2: rc * rc, modifier, elec, ele_n2, delta, k_rf, c_rf, beta, shift }
    }
}

impl NbParams {
    /// The dielectric constant `D` (`332.0716 / coulomb`).
    pub(crate) fn coulomb_dielectric(&self) -> f64 {
        if self.coulomb > 0.0 { super::params::MMFF_COULOMB / self.coulomb } else { 1.0 }
    }
}

// ---------------------------------------------------------------------------
// scalar reference
// ---------------------------------------------------------------------------

/// Unmodified vdW energy and `dE/dr`.
#[inline]
pub(crate) fn vdw_raw(kind: VdwKind, p1: f64, p2: f64, p3: f64, r: f64) -> (f64, f64) {
    match kind {
        VdwKind::Mmff => {
            let (rs, eps, rs7) = (p1, p2, p3);
            let d1 = 1.0 / (r + 0.07 * rs);
            let a = 1.07 * rs * d1;
            let a2 = a * a;
            let a7 = a2 * a2 * a2 * a;
            let r2 = r * r;
            let r6 = r2 * r2 * r2;
            let den = 1.0 / (r6 * r + 0.12 * rs7);
            let b = 1.12 * rs7 * den - 2.0;
            (eps * a7 * b, -7.0 * eps * a7 * (b * d1 + 1.12 * rs7 * r6 * den * den))
        }
        VdwKind::Uff => {
            let (x2, d) = (p1, p2);
            let s = x2 / (r * r);
            let s3 = s * s * s;
            let s6 = s3 * s3;
            (d * (s6 - 2.0 * s3), -12.0 * d * (s6 - s3) / r)
        }
    }
}

/// vdW energy and `dE/dr` with the modifier (caller ensures `r < rc`).
#[inline]
pub(crate) fn vdw_modified(kp: &KernelParams, p: &NbParams, idx: usize, r: f64) -> (f64, f64) {
    let (e, de) = vdw_raw(kp.vdw, p.p1[idx], p.p2[idx], p.p3[idx], r);
    match kp.modifier {
        Modifier::None => (e, de),
        Modifier::Shift => (e - kp.shift[idx], de),
        Modifier::Switch { rsw, inv_w } => {
            if r <= rsw {
                (e, de)
            } else {
                let t = (r - rsw) * inv_w;
                let s = 1.0 - t * t * t * (10.0 - 15.0 * t + 6.0 * t * t);
                let ds = -30.0 * t * t * (1.0 - t) * (1.0 - t) * inv_w;
                (e * s, de * s + e * ds)
            }
        }
    }
}

/// Electrostatic shape `f(r)` and `df/dr` of the kernel mode (caller
/// applies the cutoff for reaction field / Ewald).
#[inline]
pub(crate) fn elec_shape(kp: &KernelParams, r: f64) -> (f64, f64) {
    match kp.elec {
        ElecKind::None => (0.0, 0.0),
        ElecKind::Full | ElecKind::ReactionField => {
            let inv = 1.0 / (r + kp.delta);
            let (mut f, mut df) = if kp.ele_n2 {
                let f = inv * inv;
                (f, -2.0 * f * inv)
            } else {
                (inv, -inv * inv)
            };
            if kp.elec == ElecKind::ReactionField {
                f += kp.k_rf * r * r - kp.c_rf;
                df += 2.0 * kp.k_rf * r;
            }
            (f, df)
        }
        ElecKind::Ewald => {
            let (ec, e) = erfc_exp(kp.beta * r);
            let f = ec / r;
            (f, -(f + kp.beta * TWO_OVER_SQRT_PI * e) / r)
        }
    }
}

/// Reference energy of one pair: returns `(E_vdw, E_elec, (dE/dr)/r)`.
/// `qq` already contains the Coulomb prefactor (and any scaling).
#[inline]
pub(crate) fn pair_ref(kp: &KernelParams, p: &NbParams, idx: usize, qq: f64, r2: f64) -> (f64, f64, f64) {
    let in_rc = r2 < kp.rc2;
    let elec_on = kp.elec != ElecKind::None && (kp.elec == ElecKind::Full || in_rc);
    if !in_rc && !elec_on {
        return (0.0, 0.0, 0.0);
    }
    let r = r2.sqrt();
    let (ev, dev) = if in_rc { vdw_modified(kp, p, idx, r) } else { (0.0, 0.0) };
    let (ee, dee) = if elec_on {
        let (f, df) = elec_shape(kp, r);
        (qq * f, qq * df)
    } else {
        (0.0, 0.0)
    };
    (ev, ee, (dev + dee) / r)
}

// ---------------------------------------------------------------------------
// cluster kernels
// ---------------------------------------------------------------------------

const fn lane(b: bool) -> f64 {
    if b { f64::from_bits(u64::MAX) } else { 0.0 }
}
const fn mask4(bits: usize) -> f64x4 {
    f64x4::new([lane(bits & 1 != 0), lane(bits & 2 != 0), lane(bits & 4 != 0), lane(bits & 8 != 0)])
}
const MASKS: [f64x4; 16] = {
    let mut m = [f64x4::new([0.0; 4]); 16];
    let mut i = 0;
    while i < 16 {
        m[i] = mask4(i);
        i += 1;
    }
    m
};

/// Cluster-ordered coordinates (structure of arrays) for one evaluation.
pub(crate) struct ClusterCoords {
    pub x: Vec<f64x4>,
    pub y: Vec<f64x4>,
    pub z: Vec<f64x4>,
}

/// Evaluation context shared by all I-clusters.
pub(crate) struct Ctx<'a> {
    pub kp: &'a KernelParams,
    pub p: &'a NbParams,
    pub list: &'a PairList,
    pub c: &'a ClusterCoords,
}

const VDW_MMFF: u8 = 0;
const VDW_UFF: u8 = 1;
const EL_NONE: u8 = 0;
const EL_FULL: u8 = 1;
const EL_RF: u8 = 2;
const EL_EWALD: u8 = 3;
const MOD_NONE: u8 = 0;
const MOD_SHIFT: u8 = 1;
const MOD_SWITCH: u8 = 2;

#[inline(always)]
fn gather(tab: &[f64], base: usize, t: &[u32; 4]) -> f64x4 {
    f64x4::new([tab[base + t[0] as usize], tab[base + t[1] as usize], tab[base + t[2] as usize], tab[base + t[3] as usize]])
}

/// SIMD kernel for I-cluster `ci`: returns `(E_vdw, E_elec)` (already
/// halved: the list is symmetric) and writes the gradient of its four atoms
/// to `g` (`[x0..x3, y0..y3, z0..z3]`) if `GRAD`.
#[inline(always)]
fn simd_i<const VDW: u8, const EL: u8, const MODI: u8, const GRAD: bool>(ctx: &Ctx, ci: usize, g: &mut [f64; 12]) -> (f64, f64) {
    let kp = ctx.kp;
    let p = ctx.p;
    let l = ctx.list;
    let nt = p.ntypes;
    let one = f64x4::splat(1.0);
    let zero = f64x4::splat(0.0);
    let rc2 = f64x4::splat(kp.rc2);
    let xi = ctx.c.x[ci].to_array();
    let yi = ctx.c.y[ci].to_array();
    let zi = ctx.c.z[ci].to_array();
    let qi = l.q[ci].to_array();
    let ti = l.types[ci];
    let mut ev_acc = zero;
    let mut ee_acc = zero;
    let mut gx = [zero; 4];
    let mut gy = [zero; 4];
    let mut gz = [zero; 4];
    let (k_rf, c_rf, beta, delta) = (f64x4::splat(kp.k_rf), f64x4::splat(kp.c_rf), f64x4::splat(kp.beta), f64x4::splat(kp.delta));
    let (rsw, inv_w) = match kp.modifier {
        Modifier::Switch { rsw, inv_w } => (f64x4::splat(rsw), f64x4::splat(inv_w)),
        _ => (zero, zero),
    };
    for e in &l.cj[l.ci[ci] as usize..l.ci[ci + 1] as usize] {
        let cj = e.j as usize;
        let s = l.shifts[e.shift as usize];
        let xj = ctx.c.x[cj] + f64x4::splat(s[0]);
        let yj = ctx.c.y[cj] + f64x4::splat(s[1]);
        let zj = ctx.c.z[cj] + f64x4::splat(s[2]);
        let qj = l.q[cj];
        let tj = &l.types[cj];
        for a in 0..4 {
            let bits = ((e.mask >> (4 * a)) & 0xF) as usize;
            if bits == 0 {
                continue;
            }
            let pm = MASKS[bits];
            let dx = f64x4::splat(xi[a]) - xj;
            let dy = f64x4::splat(yi[a]) - yj;
            let dz = f64x4::splat(zi[a]) - zj;
            let r2 = dx * dx + dy * dy + dz * dz;
            let vm = pm & r2.cmp_lt(rc2);
            let em = if EL == EL_FULL { pm } else { vm };
            if em.none() && vm.none() {
                continue;
            }
            let r2s = (em | vm).blend(r2, one);
            let r = r2s.sqrt();
            let base = ti[a] as usize * nt;
            // All reciprocals from one division: denominators r, A, B, C
            // (constant 1 when unused, folded by the compiler).
            let (rs, eps, rs7) = if VDW == VDW_MMFF {
                (gather(&p.p1, base, tj), gather(&p.p2, base, tj), gather(&p.p3, base, tj))
            } else {
                (gather(&p.p1, base, tj), gather(&p.p2, base, tj), zero)
            };
            let r6 = r2s * r2s * r2s;
            let (da, db) = if VDW == VDW_MMFF {
                (r + f64x4::splat(0.07) * rs, r6 * r + f64x4::splat(0.12) * rs7)
            } else {
                (one, one)
            };
            let dc = match EL {
                EL_FULL | EL_RF => r + delta,
                EL_EWALD => beta * r + f64x4::splat(ERFC_K),
                _ => one,
            };
            let p1 = r * da;
            let p2 = p1 * db;
            let inv = one / (p2 * dc);
            let inv_c = p2 * inv;
            let t = dc * inv;
            let inv_b = p1 * t;
            let t2 = db * t;
            let inv_a = r * t2;
            let rinv = da * t2;
            // van der Waals
            let (ev, dev) = if VDW == VDW_MMFF {
                let d1 = inv_a;
                let at = f64x4::splat(1.07) * rs * d1;
                let a2 = at * at;
                let a7 = a2 * a2 * a2 * at;
                let den = inv_b;
                let b = f64x4::splat(1.12) * rs7 * den - f64x4::splat(2.0);
                (eps * a7 * b, f64x4::splat(-7.0) * eps * a7 * (b * d1 + f64x4::splat(1.12) * rs7 * r6 * den * den))
            } else {
                let (x2, d) = (rs, eps);
                let s = x2 * rinv * rinv;
                let s3 = s * s * s;
                let s6 = s3 * s3;
                (d * (s6 - f64x4::splat(2.0) * s3), f64x4::splat(-12.0) * d * (s6 - s3) * rinv)
            };
            let (ev, dev) = match MODI {
                MOD_SHIFT => (ev - gather(&kp.shift, base, tj), dev),
                MOD_SWITCH => {
                    let t = ((r - rsw) * inv_w).max(zero);
                    let t2 = t * t;
                    let sw = one - t2 * t * (f64x4::splat(10.0) - f64x4::splat(15.0) * t + f64x4::splat(6.0) * t2);
                    let omt = one - t;
                    let dsw = f64x4::splat(-30.0) * t2 * omt * omt * inv_w;
                    (ev * sw, dev * sw + ev * dsw)
                }
                _ => (ev, dev),
            };
            let ev = vm.blend(ev, zero);
            let dev = vm.blend(dev, zero);
            ev_acc += ev;
            let mut de = dev;
            // electrostatics
            if EL != EL_NONE {
                let qq = f64x4::splat(qi[a]) * qj;
                let (f, df) = if EL == EL_EWALD {
                    let (ec, ex) = erfc_exp_inv(beta * r, inv_c);
                    let f = ec * rinv;
                    (f, -(f + beta * f64x4::splat(TWO_OVER_SQRT_PI) * ex) * rinv)
                } else {
                    let inv = inv_c;
                    let (mut f, mut df) = if kp.ele_n2 {
                        let f = inv * inv;
                        (f, f64x4::splat(-2.0) * f * inv)
                    } else {
                        (inv, -(inv * inv))
                    };
                    if EL == EL_RF {
                        f += k_rf * r2s - c_rf;
                        df += f64x4::splat(2.0) * k_rf * r;
                    }
                    (f, df)
                };
                ee_acc += em.blend(qq * f, zero);
                de += em.blend(qq * df, zero);
            }
            if GRAD {
                let gs = de * rinv;
                gx[a] += gs * dx;
                gy[a] += gs * dy;
                gz[a] += gs * dz;
            }
        }
    }
    if GRAD {
        for a in 0..4 {
            g[a] = gx[a].reduce_add();
            g[4 + a] = gy[a].reduce_add();
            g[8 + a] = gz[a].reduce_add();
        }
    }
    (0.5 * ev_acc.reduce_add(), 0.5 * ee_acc.reduce_add())
}

/// Scalar cluster kernel (same traversal, per-pair reference functions).
fn scalar_i<const GRAD: bool>(ctx: &Ctx, ci: usize, g: &mut [f64; 12]) -> (f64, f64) {
    let (kp, p, l) = (ctx.kp, ctx.p, ctx.list);
    let nt = p.ntypes;
    let xi = ctx.c.x[ci].to_array();
    let yi = ctx.c.y[ci].to_array();
    let zi = ctx.c.z[ci].to_array();
    let qi = l.q[ci].to_array();
    let ti = l.types[ci];
    let (mut ev, mut ee) = (0.0, 0.0);
    let mut gl = [0.0; 12];
    for e in &l.cj[l.ci[ci] as usize..l.ci[ci + 1] as usize] {
        let cj = e.j as usize;
        let s = l.shifts[e.shift as usize];
        let xj = ctx.c.x[cj].to_array();
        let yj = ctx.c.y[cj].to_array();
        let zj = ctx.c.z[cj].to_array();
        let qj = l.q[cj].to_array();
        let tj = l.types[cj];
        for a in 0..4 {
            for b in 0..4 {
                if e.mask & (1 << (4 * a + b)) == 0 {
                    continue;
                }
                let dx = xi[a] - (xj[b] + s[0]);
                let dy = yi[a] - (yj[b] + s[1]);
                let dz = zi[a] - (zj[b] + s[2]);
                let r2 = dx * dx + dy * dy + dz * dz;
                let idx = ti[a] as usize * nt + tj[b] as usize;
                let (v, el, gs) = pair_ref(kp, p, idx, qi[a] * qj[b], r2);
                ev += v;
                ee += el;
                if GRAD {
                    gl[a] += gs * dx;
                    gl[4 + a] += gs * dy;
                    gl[8 + a] += gs * dz;
                }
            }
        }
    }
    if GRAD {
        *g = gl;
    }
    (0.5 * ev, 0.5 * ee)
}

/// Kernel entry point selected once per evaluation.
pub(crate) type ClusterKernel = fn(&Ctx, usize, &mut [f64; 12]) -> (f64, f64);

macro_rules! pick_simd {
    ($v:expr, $e:expr, $m:expr, $g:expr; $($vv:ident),*; $($ee:ident),*; $($mm:ident),*) => {{
        pick_simd!(@v $v, $e, $m, $g; [$($vv),*]; [$($ee),*]; [$($mm),*])
    }};
    (@v $v:expr, $e:expr, $m:expr, $g:expr; [$($vv:ident),*]; $es:tt; $ms:tt) => {
        match $v {
            $( $vv => pick_simd!(@e $vv, $e, $m, $g; $es; $ms), )*
            _ => unreachable!(),
        }
    };
    (@e $vv:ident, $e:expr, $m:expr, $g:expr; [$($ee:ident),*]; $ms:tt) => {
        match $e {
            $( $ee => pick_simd!(@m $vv, $ee, $m, $g; $ms), )*
            _ => unreachable!(),
        }
    };
    (@m $vv:ident, $ee:ident, $m:expr, $g:expr; [$($mm:ident),*]) => {
        match $m {
            $( $mm => if $g { simd_i::<$vv, $ee, $mm, true> as ClusterKernel } else { simd_i::<$vv, $ee, $mm, false> as ClusterKernel }, )*
            _ => unreachable!(),
        }
    };
}

/// Selects the kernel for the given settings.
pub(crate) fn select_kernel(kp: &KernelParams, simd: bool, grad: bool) -> ClusterKernel {
    if !simd {
        return if grad { scalar_i::<true> } else { scalar_i::<false> };
    }
    let v = match kp.vdw {
        VdwKind::Mmff => VDW_MMFF,
        VdwKind::Uff => VDW_UFF,
    };
    let e = match kp.elec {
        ElecKind::None => EL_NONE,
        ElecKind::Full => EL_FULL,
        ElecKind::ReactionField => EL_RF,
        ElecKind::Ewald => EL_EWALD,
    };
    let m = match kp.modifier {
        Modifier::None => MOD_NONE,
        Modifier::Shift => MOD_SHIFT,
        Modifier::Switch { .. } => MOD_SWITCH,
    };
    pick_simd!(v, e, m, grad; VDW_MMFF, VDW_UFF; EL_NONE, EL_FULL, EL_RF, EL_EWALD; MOD_NONE, MOD_SHIFT, MOD_SWITCH)
}
