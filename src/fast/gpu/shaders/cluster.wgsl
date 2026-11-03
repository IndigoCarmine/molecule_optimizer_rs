// Cluster-pair kernel: TPC invocations per i-cluster (WG / TPC clusters per
// workgroup). Each invocation walks every TPC-th j-cluster entry of its
// i-cluster, sums the (up to) 16 pair interactions of an entry in registers
// (the 4x4 body is generated unrolled at `//!BODY`) and adds the entry
// totals to Kahan-compensated accumulators. The TPC partial sums are
// combined with a fixed tree in workgroup memory, and the first invocation
// writes the gradient of the four i-atoms and the cluster energies. The
// list is symmetric, so no other cluster writes these values: no atomics,
// results are bitwise reproducible.
//
// Generated constants: WG, TPC, GRAD (+ those of pair.wgsl).

@group(0) @binding(0) var<uniform> P: NbParams;
// per slot (4 per cluster): coordinates relative to the cluster origin, charge * sqrt(C)
@group(0) @binding(1) var<storage, read> xyzq: array<vec4<f32>>;
// per slot: type-pair table row (toff + type * ntypes) and the type
@group(0) @binding(2) var<storage, read> stype: array<vec2<u32>>;
// type-pair table (p1, p2, p3, shift)
@group(0) @binding(3) var<storage, read> tab: array<vec4<f32>>;
// per cluster: first entry, end entry
@group(0) @binding(4) var<storage, read> cinfo: array<vec2<u32>>;
// per entry: (j-cluster, interaction mask) and the origin difference O_J + S - O_I
@group(0) @binding(5) var<storage, read> cj: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> cjd: array<vec4<f32>>;
// outputs: gradient per slot; (E_vdw, E_elec, sum of qi qj of the
// reaction-field pairs, -) per cluster (not halved)
@group(0) @binding(7) var<storage, read_write> cgrad: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> cener: array<vec4<f32>>;

const NACC: u32 = 15u;
var<workgroup> red: array<f32, WG * NACC>;

// One pair: (E_vdw, E_elec, qq of a reaction-field pair, (dE/dr)/r).
fn interact(d: vec3<f32>, qq: f32, tidx: u32) -> vec4<f32> {
    let r2 = dot(d, d);
    let in_rc = r2 < P.rc2;
    if (!in_rc && ELEC != 1u) {
        return vec4<f32>(0.0);
    }
    let r = sqrt(r2);
    var o = vec4<f32>(0.0);
    var de = 0.0;
    if (in_rc) {
        let v = vdw_mod(tab[tidx], r, r2);
        o.x = v.x;
        de = v.y;
    }
    if (ELEC != 0u) {
        let f = elec_shape(r, r2);
        o.y = qq * f.x;
        de += qq * f.y;
        if (ELEC == 2u) {
            o.z = qq;
        }
    }
    o.w = de / r;
    return o;
}

fn kahan3(s: ptr<function, vec3<f32>>, c: ptr<function, vec3<f32>>, x: vec3<f32>) {
    let y = x - *c;
    let t = *s + y;
    *c = (t - *s) - y;
    *s = t;
}

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let group = wid.x + wid.y * nwg.x;
    let ci = group * (WG / TPC) + lid / TPC;
    let t = lid % TPC;
    var s0 = vec3<f32>(0.0);
    var s1 = vec3<f32>(0.0);
    var s2 = vec3<f32>(0.0);
    var s3 = vec3<f32>(0.0);
    var se = vec3<f32>(0.0);
    var c0 = vec3<f32>(0.0);
    var c1 = vec3<f32>(0.0);
    var c2 = vec3<f32>(0.0);
    var c3 = vec3<f32>(0.0);
    var ce = vec3<f32>(0.0);
    if (ci < P.nclusters) {
        let pi0 = xyzq[4u * ci];
        let pi1 = xyzq[4u * ci + 1u];
        let pi2 = xyzq[4u * ci + 2u];
        let pi3 = xyzq[4u * ci + 3u];
        let ti0 = stype[4u * ci].x;
        let ti1 = stype[4u * ci + 1u].x;
        let ti2 = stype[4u * ci + 2u].x;
        let ti3 = stype[4u * ci + 3u].x;
        let info = cinfo[ci];
        for (var e = info.x + t; e < info.y; e += TPC) {
            let jm = cj[e];
            let cj4 = 4u * jm.x;
            let mask = jm.y;
            let sh = cjd[e].xyz;
            let pj0 = xyzq[cj4];
            let pj1 = xyzq[cj4 + 1u];
            let pj2 = xyzq[cj4 + 2u];
            let pj3 = xyzq[cj4 + 3u];
            let xj0 = pj0.xyz + sh;
            let xj1 = pj1.xyz + sh;
            let xj2 = pj2.xyz + sh;
            let xj3 = pj3.xyz + sh;
            let tj0 = stype[cj4].y;
            let tj1 = stype[cj4 + 1u].y;
            let tj2 = stype[cj4 + 2u].y;
            let tj3 = stype[cj4 + 3u].y;
            var en = vec3<f32>(0.0);
            var g0 = vec3<f32>(0.0);
            var g1 = vec3<f32>(0.0);
            var g2 = vec3<f32>(0.0);
            var g3 = vec3<f32>(0.0);
//!BODY
            kahan3(&se, &ce, en);
            if (GRAD) {
                kahan3(&s0, &c0, g0);
                kahan3(&s1, &c1, g1);
                kahan3(&s2, &c2, g2);
                kahan3(&s3, &c3, g3);
            }
        }
    }
    let a0 = s0 - c0;
    let a1 = s1 - c1;
    let a2 = s2 - c2;
    let a3 = s3 - c3;
    let ae = se - ce;
    if (TPC == 1u) {
        if (ci < P.nclusters) {
            if (GRAD) {
                cgrad[4u * ci] = vec4<f32>(a0, 0.0);
                cgrad[4u * ci + 1u] = vec4<f32>(a1, 0.0);
                cgrad[4u * ci + 2u] = vec4<f32>(a2, 0.0);
                cgrad[4u * ci + 3u] = vec4<f32>(a3, 0.0);
            }
            cener[ci] = vec4<f32>(ae, 0.0);
        }
        return;
    }
    let acc = array<f32, NACC>(a0.x, a0.y, a0.z, a1.x, a1.y, a1.z, a2.x, a2.y, a2.z, a3.x, a3.y, a3.z, ae.x, ae.y, ae.z);
    for (var k = 0u; k < NACC; k++) {
        red[k * WG + lid] = acc[k];
    }
    workgroupBarrier();
    for (var s = TPC / 2u; s > 0u; s >>= 1u) {
        if (t < s) {
            for (var k = 0u; k < NACC; k++) {
                red[k * WG + lid] += red[k * WG + lid + s];
            }
        }
        workgroupBarrier();
    }
    if (t == 0u && ci < P.nclusters) {
        if (GRAD) {
            for (var a = 0u; a < 4u; a++) {
                cgrad[4u * ci + a] = vec4<f32>(red[(3u * a) * WG + lid], red[(3u * a + 1u) * WG + lid], red[(3u * a + 2u) * WG + lid], 0.0);
            }
        }
        cener[ci] = vec4<f32>(red[12u * WG + lid], red[13u * WG + lid], red[14u * WG + lid], 0.0);
    }
}
