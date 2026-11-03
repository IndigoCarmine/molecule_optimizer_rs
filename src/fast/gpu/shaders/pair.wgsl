// Pair interaction functions (f32), mirroring `fast::approx::kernel`.
//
// Generated constants (header): VDW (0 = MMFF 14-7, 1 = UFF LJ),
// ELEC (0 none, 1 full, 2 reaction field, 3 Ewald real space),
// MODI (0 none, 1 potential shift, 2 switch), ELE_N2 (distance-dependent
// dielectric), SPLIT_C (reaction field without the constant shift),
// ERFC_A0..ERFC_A20 (erfc polynomial).

struct NbParams {
    rc2: f32,
    k_rf: f32,
    c_rf: f32,
    beta: f32,
    delta: f32,
    rsw: f32,
    inv_w: f32,
    scale14: f32,
    nclusters: u32,
    natoms: u32,
    pad0: u32,
    pad1: u32,
}

const TWO_OVER_SQRT_PI: f32 = 1.1283791670955126;

// 1/x refined by one Newton step (GPU divisions may be approximate).
fn recip(x: f32) -> f32 {
    let y = 1.0 / x;
    return fma(fma(-x, y, 1.0), y, y);
}

// (erfc(x), exp(-x^2)) for x >= 0: erfc = exp(-x^2) h(u) / (x + 2) with the
// polynomial h of `fast::approx::erfc` (u = 1.25 t + 0.25, t = (x-2)/(x+2)).
fn erfc_exp(x: f32) -> vec2<f32> {
    let e = exp(-(x * x));
    let inv = 1.0 / (x + 2.0);
    let t = 1.0 - 4.0 * inv;
    let u = min(t * 1.25 + 0.25, 1.0);
    let u2 = u * u;
    let u4 = u2 * u2;
    let u8 = u4 * u4;
    let u16 = u8 * u8;
    let b0 = fma(ERFC_A1, u, ERFC_A0);
    let b1 = fma(ERFC_A3, u, ERFC_A2);
    let b2 = fma(ERFC_A5, u, ERFC_A4);
    let b3 = fma(ERFC_A7, u, ERFC_A6);
    let b4 = fma(ERFC_A9, u, ERFC_A8);
    let b5 = fma(ERFC_A11, u, ERFC_A10);
    let b6 = fma(ERFC_A13, u, ERFC_A12);
    let b7 = fma(ERFC_A15, u, ERFC_A14);
    let b8 = fma(ERFC_A17, u, ERFC_A16);
    let b9 = fma(ERFC_A19, u, ERFC_A18);
    let c0 = fma(b1, u2, b0);
    let c1 = fma(b3, u2, b2);
    let c2 = fma(b5, u2, b4);
    let c3 = fma(b7, u2, b6);
    let c4 = fma(b9, u2, b8);
    let d0 = fma(c1, u4, c0);
    let d1 = fma(c3, u4, c2);
    let d2 = fma(ERFC_A20, u4, c4);
    let e0 = fma(d1, u8, d0);
    let h = fma(d2, u16, e0);
    return vec2<f32>(e * h * inv, e);
}

// Unmodified van der Waals energy and dE/dr; `p` = (p1, p2, p3, shift).
fn vdw_raw(p: vec4<f32>, r: f32, r2: f32) -> vec2<f32> {
    if (VDW == 0u) {
        let rs = p.x;
        let eps = p.y;
        let rs7 = p.z;
        let d1 = 1.0 / (r + 0.07 * rs);
        let a = 1.07 * rs * d1;
        let a2 = a * a;
        let a7 = a2 * a2 * a2 * a;
        let r6 = r2 * r2 * r2;
        let den = 1.0 / (r6 * r + 0.12 * rs7);
        let b = 1.12 * rs7 * den - 2.0;
        return vec2<f32>(eps * a7 * b, -7.0 * eps * a7 * (b * d1 + 1.12 * rs7 * r6 * den * den));
    } else {
        let s = p.x / r2;
        let s3 = s * s * s;
        let s6 = s3 * s3;
        return vec2<f32>(p.y * (s6 - 2.0 * s3), -12.0 * p.y * (s6 - s3) / r);
    }
}

// van der Waals energy and dE/dr with the modifier (caller: r < rc).
fn vdw_mod(p: vec4<f32>, r: f32, r2: f32) -> vec2<f32> {
    let v = vdw_raw(p, r, r2);
    if (MODI == 1u) {
        return vec2<f32>(v.x - p.w, v.y);
    } else if (MODI == 2u) {
        if (r <= P.rsw) {
            return v;
        }
        let t = (r - P.rsw) * P.inv_w;
        let t2 = t * t;
        let s = 1.0 - t2 * t * (10.0 - 15.0 * t + 6.0 * t2);
        let omt = 1.0 - t;
        let ds = -30.0 * t2 * omt * omt * P.inv_w;
        return vec2<f32>(v.x * s, v.y * s + v.x * ds);
    }
    return v;
}

// Electrostatic shape f(r) and df/dr (caller applies the cutoff).
fn elec_shape(r: f32, r2: f32) -> vec2<f32> {
    if (ELEC == 3u) {
        let ee = erfc_exp(P.beta * r);
        let inv = 1.0 / r;
        let f = ee.x * inv;
        return vec2<f32>(f, -(f + P.beta * TWO_OVER_SQRT_PI * ee.y) * inv);
    }
    let inv = recip(r + P.delta);
    var f: f32;
    var df: f32;
    if (ELE_N2) {
        f = inv * inv;
        df = -2.0 * f * inv;
    } else {
        f = inv;
        df = -inv * inv;
    }
    if (ELEC == 2u) {
        // SPLIT_C: the constant -c is applied by the caller (in f64 on the CPU)
        f += P.k_rf * r2;
        if (!SPLIT_C) {
            f -= P.c_rf;
        }
        df += 2.0 * P.k_rf * r;
    }
    return vec2<f32>(f, df);
}
