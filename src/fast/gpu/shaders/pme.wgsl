// Smooth PME on the GPU (reciprocal space), see `fast::approx::pme`.
//
// * spread: one invocation per charged atom computes its B-spline weights
//   (from the fractional offsets computed in f64 on the CPU) and adds the
//   ORDER^3 charge contributions to the grid as 64-bit fixed-point numbers
//   (two u32 words with carry, 2^-40 resolution). Integer addition is
//   associative, so the grid is bitwise independent of the execution order.
// * convert: fixed point -> complex f32 grid (and clears the fixed-point grid).
// * influence: E = 1/2 sum D |F(Q)|^2 (per-workgroup partial sums, fixed
//   tree) and F(Q) <- D F(Q).
// * gather: gradient of every charged atom from the convolved grid.
//
// Generated constants: ORDER, GRAD.

struct PmeParams {
    k1: u32,
    k2: u32,
    k3: u32,
    nq: u32,
    ntot: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    recip0: vec4<f32>,
    recip1: vec4<f32>,
    recip2: vec4<f32>,
}

@group(0) @binding(0) var<uniform> PP: PmeParams;
// per charged atom: (grid base index x, y, z, atom index), (w x, y, z, charge * sqrt(C))
@group(0) @binding(1) var<storage, read> atoms_b: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> atoms_w: array<vec4<f32>>;
// per charged atom: theta (3 * ORDER), dtheta (3 * ORDER)
@group(0) @binding(3) var<storage, read_write> theta: array<f32>;
// fixed-point charge grid: (low word, high word) per grid point
@group(0) @binding(4) var<storage, read_write> qgrid: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> ga: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> fx: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read> infl: array<f32>;
@group(0) @binding(8) var<storage, read_write> epart: array<f32>;
@group(0) @binding(9) var<storage, read> fy: array<vec2<f32>>;
@group(0) @binding(10) var<storage, read_write> pgrad: array<vec4<f32>>;

struct Spline {
    th: array<f32, ORDER>,
    dth: array<f32, ORDER>,
}

// Values M_p(w + j) and derivatives, j = 0..p (same recursion as the CPU).
fn bspline(w: f32) -> Spline {
    var s: Spline;
    for (var j = 0u; j < ORDER; j++) {
        s.th[j] = 0.0;
        s.dth[j] = 0.0;
    }
    s.th[0] = w;
    s.th[1] = 1.0 - w;
    for (var n = 3u; n <= ORDER; n++) {
        if (n == ORDER) {
            s.dth[0] = s.th[0];
            for (var j = 1u; j < ORDER; j++) {
                s.dth[j] = s.th[j] - s.th[j - 1u];
            }
        }
        let inv = 1.0 / f32(n - 1u);
        for (var jj = 0u; jj < n; jj++) {
            let j = n - 1u - jj;
            let x = w + f32(j);
            var cur = 0.0;
            if (j < n - 1u) {
                cur = s.th[j];
            }
            var prev = 0.0;
            if (j > 0u) {
                prev = s.th[j - 1u];
            }
            s.th[j] = (x * cur + (f32(n) - x) * prev) * inv;
        }
    }
    return s;
}

fn linear_id(lid: u32, wid: vec3<u32>, nwg: vec3<u32>, wg: u32) -> u32 {
    return (wid.x + wid.y * nwg.x) * wg + lid;
}

fn add_fixed(idx: u32, v: f32) {
    // v * 2^40 = hi * 2^32 + lo
    let s = v * 256.0;
    var hf = floor(s);
    var fr = s - hf;
    if (fr >= 1.0) {
        hf += 1.0;
        fr = 0.0;
    }
    let lo = u32(fr * 4294967296.0);
    let hi = bitcast<u32>(i32(hf));
    let old = atomicAdd(&qgrid[2u * idx], lo);
    let carry = select(0u, 1u, old + lo < old);
    atomicAdd(&qgrid[2u * idx + 1u], hi + carry);
}

@compute @workgroup_size(64)
fn spread(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let a = linear_id(lid, wid, nwg, 64u);
    if (a >= PP.nq) {
        return;
    }
    let b = atoms_b[a];
    let w = atoms_w[a];
    let s0 = bspline(w.x);
    let s1 = bspline(w.y);
    let s2 = bspline(w.z);
    let o = a * 6u * ORDER;
    for (var j = 0u; j < ORDER; j++) {
        theta[o + j] = s0.th[j];
        theta[o + ORDER + j] = s1.th[j];
        theta[o + 2u * ORDER + j] = s2.th[j];
        theta[o + 3u * ORDER + j] = s0.dth[j];
        theta[o + 4u * ORDER + j] = s1.dth[j];
        theta[o + 5u * ORDER + j] = s2.dth[j];
    }
    for (var j1 = 0u; j1 < ORDER; j1++) {
        let x = (b.x + PP.k1 * ORDER - j1) % PP.k1;
        let w1 = w.w * s0.th[j1];
        for (var j2 = 0u; j2 < ORDER; j2++) {
            let y = (b.y + PP.k2 * ORDER - j2) % PP.k2;
            let w12 = w1 * s1.th[j2];
            let row = (x * PP.k2 + y) * PP.k3;
            for (var j3 = 0u; j3 < ORDER; j3++) {
                let z = (b.z + PP.k3 * ORDER - j3) % PP.k3;
                add_fixed(row + z, w12 * s2.th[j3]);
            }
        }
    }
}

@compute @workgroup_size(256)
fn convert(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = linear_id(lid, wid, nwg, 256u);
    if (g >= PP.ntot) {
        return;
    }
    let lo = atomicLoad(&qgrid[2u * g]);
    let hi = atomicLoad(&qgrid[2u * g + 1u]);
    atomicStore(&qgrid[2u * g], 0u);
    atomicStore(&qgrid[2u * g + 1u], 0u);
    ga[g] = vec2<f32>(f32(bitcast<i32>(hi)) * (1.0 / 256.0) + f32(lo) * 9.094947017729282e-13, 0.0);
}

var<workgroup> ered: array<f32, 256>;

@compute @workgroup_size(256)
fn influence(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let group = wid.x + wid.y * nwg.x;
    let g = group * 256u + lid;
    var e = 0.0;
    if (g < PP.ntot) {
        let c = fx[g];
        let d = infl[g];
        e = d * dot(c, c);
        if (GRAD) {
            fx[g] = c * d;
        }
    }
    ered[lid] = e;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            ered[lid] += ered[lid + s];
        }
        workgroupBarrier();
    }
    if (lid == 0u && group * 256u < PP.ntot) {
        epart[group] = ered[0];
    }
}

@compute @workgroup_size(64)
fn gather(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let a = linear_id(lid, wid, nwg, 64u);
    if (a >= PP.nq) {
        return;
    }
    let b = atoms_b[a];
    let q = atoms_w[a].w;
    let o = a * 6u * ORDER;
    var s1 = 0.0;
    var s2 = 0.0;
    var s3 = 0.0;
    for (var j1 = 0u; j1 < ORDER; j1++) {
        let x = (b.x + PP.k1 * ORDER - j1) % PP.k1;
        let t1 = theta[o + j1];
        let dt1 = theta[o + 3u * ORDER + j1];
        for (var j2 = 0u; j2 < ORDER; j2++) {
            let y = (b.y + PP.k2 * ORDER - j2) % PP.k2;
            let t2 = theta[o + ORDER + j2];
            let dt2 = theta[o + 4u * ORDER + j2];
            let row = (x * PP.k2 + y) * PP.k3;
            var c0 = 0.0;
            var c1 = 0.0;
            for (var j3 = 0u; j3 < ORDER; j3++) {
                let z = (b.z + PP.k3 * ORDER - j3) % PP.k3;
                let v = fy[row + z].x;
                c0 += v * theta[o + 2u * ORDER + j3];
                c1 += v * theta[o + 5u * ORDER + j3];
            }
            s1 += dt1 * t2 * c0;
            s2 += t1 * dt2 * c0;
            s3 += t1 * t2 * c1;
        }
    }
    let sv = vec3<f32>(s1 * f32(PP.k1) * q, s2 * f32(PP.k2) * q, s3 * f32(PP.k3) * q);
    let g = sv.x * PP.recip0.xyz + sv.y * PP.recip1.xyz + sv.z * PP.recip2.xyz;
    pgrad[b.w] = vec4<f32>(g, 0.0);
}
