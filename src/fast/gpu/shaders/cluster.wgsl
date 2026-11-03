// Cluster-pair kernel: TPC invocations per i-cluster (WG / TPC clusters per
// workgroup). Each invocation walks every TPC-th j-cluster entry of its
// i-cluster, sums the 16 pair interactions of an entry in registers and adds
// the entry totals to Kahan-compensated accumulators. The TPC partial sums
// are combined with a fixed tree in workgroup memory, and the first
// invocation writes the gradient of the four i-atoms and the cluster energy.
// The list is symmetric, so no other cluster writes these values: no
// atomics, results are bitwise reproducible.
//
// Generated constants: WG, TPC, GRAD (+ those of pair.wgsl).

@group(0) @binding(0) var<uniform> P: NbParams;
// per slot (4 per cluster): coordinates relative to the cluster origin, charge * sqrt(C)
@group(0) @binding(1) var<storage, read> xyzq: array<vec4<f32>>;
// per slot: index into the type-pair table row (toff + type * ntypes) and the type
@group(0) @binding(2) var<storage, read> stype: array<vec2<u32>>;
// type-pair table (p1, p2, p3, shift)
@group(0) @binding(3) var<storage, read> tab: array<vec4<f32>>;
// per cluster: first entry, end entry
@group(0) @binding(4) var<storage, read> cinfo: array<vec2<u32>>;
// per entry (4 words): j-cluster, mask, then (shift x, y, z, unused) in cjd
@group(0) @binding(5) var<storage, read> cj: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> cjd: array<vec4<f32>>;
// outputs: gradient per slot, (E_vdw, E_elec) per cluster (not halved)
@group(0) @binding(7) var<storage, read_write> cgrad: array<vec4<f32>>;
// (E_vdw, E_elec, sum of qi qj of the reaction-field pairs, -) per cluster
@group(0) @binding(8) var<storage, read_write> cener: array<vec4<f32>>;

const NACC: u32 = 15u;
var<workgroup> red: array<f32, WG * NACC>;

fn kahan(s: ptr<function, array<f32, NACC>>, c: ptr<function, array<f32, NACC>>, k: u32, x: f32) {
    let y = x - (*c)[k];
    let t = (*s)[k] + y;
    (*c)[k] = (t - (*s)[k]) - y;
    (*s)[k] = t;
}

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let group = wid.x + wid.y * nwg.x;
    let ci = group * (WG / TPC) + lid / TPC;
    let t = lid % TPC;
    var acc: array<f32, NACC>;
    var cmp: array<f32, NACC>;
    for (var k = 0u; k < NACC; k++) {
        acc[k] = 0.0;
        cmp[k] = 0.0;
    }
    if (ci < P.nclusters) {
        var xi: array<vec3<f32>, 4>;
        var qi: array<f32, 4>;
        var ti: array<u32, 4>;
        for (var a = 0u; a < 4u; a++) {
            let v = xyzq[4u * ci + a];
            xi[a] = v.xyz;
            qi[a] = v.w;
            ti[a] = stype[4u * ci + a].x;
        }
        let info = cinfo[ci];
        for (var e = info.x + t; e < info.y; e += TPC) {
            let jm = cj[e];
            let cj4 = 4u * jm.x;
            let mask = jm.y;
            let sh = cjd[e].xyz;
            var xj: array<vec3<f32>, 4>;
            var qj: array<f32, 4>;
            var tj: array<u32, 4>;
            for (var b = 0u; b < 4u; b++) {
                let v = xyzq[cj4 + b];
                xj[b] = v.xyz + sh;
                qj[b] = v.w;
                tj[b] = stype[cj4 + b].y;
            }
            var g: array<vec3<f32>, 4>;
            var ev = 0.0;
            var ee = 0.0;
            var qs = 0.0;
            for (var a = 0u; a < 4u; a++) {
                g[a] = vec3<f32>(0.0);
                let bits = (mask >> (4u * a)) & 15u;
                if (bits == 0u) {
                    continue;
                }
                for (var b = 0u; b < 4u; b++) {
                    if ((bits & (1u << b)) == 0u) {
                        continue;
                    }
                    let d = xi[a] - xj[b];
                    let r2 = dot(d, d);
                    let in_rc = r2 < P.rc2;
                    if (!in_rc && ELEC != 1u) {
                        continue;
                    }
                    let r = sqrt(r2);
                    var de = 0.0;
                    if (in_rc) {
                        let v = vdw_mod(tab[ti[a] + tj[b]], r, r2);
                        ev += v.x;
                        de = v.y;
                    }
                    if (ELEC != 0u) {
                        let qq = qi[a] * qj[b];
                        let f = elec_shape(r, r2);
                        ee += qq * f.x;
                        if (ELEC == 2u) {
                            qs += qq;
                        }
                        de += qq * f.y;
                    }
                    if (GRAD) {
                        g[a] += (de / r) * d;
                    }
                }
            }
            kahan(&acc, &cmp, 12u, ev);
            kahan(&acc, &cmp, 13u, ee);
            kahan(&acc, &cmp, 14u, qs);
            if (GRAD) {
                for (var a = 0u; a < 4u; a++) {
                    kahan(&acc, &cmp, 3u * a, g[a].x);
                    kahan(&acc, &cmp, 3u * a + 1u, g[a].y);
                    kahan(&acc, &cmp, 3u * a + 2u, g[a].z);
                }
            }
        }
    }
    if (TPC == 1u) {
        if (ci < P.nclusters) {
            if (GRAD) {
                for (var a = 0u; a < 4u; a++) {
                    cgrad[4u * ci + a] = vec4<f32>(acc[3u * a] - cmp[3u * a], acc[3u * a + 1u] - cmp[3u * a + 1u], acc[3u * a + 2u] - cmp[3u * a + 2u], 0.0);
                }
            }
            cener[ci] = vec4<f32>(acc[12u] - cmp[12u], acc[13u] - cmp[13u], acc[14u] - cmp[14u], 0.0);
        }
        return;
    }
    for (var k = 0u; k < NACC; k++) {
        red[k * WG + lid] = acc[k] - cmp[k];
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
