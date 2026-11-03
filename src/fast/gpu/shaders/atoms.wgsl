// Per-atom kernel: explicit pairs (MMFF 1-4 pairs, Ewald exclusion
// corrections) and assembly of the per-atom gradient
// (cluster gradient + explicit pairs + reciprocal-space PME).
//
// Generated constants: WG, GRAD, HAS_PME (+ those of pair.wgsl).

@group(0) @binding(0) var<uniform> P: NbParams;
@group(0) @binding(1) var<storage, read> xyzq: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tab: array<vec4<f32>>;
// per atom: slot, first explicit entry, end entry
@group(0) @binding(3) var<storage, read> ainfo: array<vec4<u32>>;
// two words per explicit entry: (partner slot, is14, table index, -) and the
// constant part of the displacement (origin and lattice offsets, f32 bits)
@group(0) @binding(4) var<storage, read> ex: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> cgrad: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> pgrad: array<vec4<f32>>;
// outputs: gradient (3 per atom), (E_vdw, E_elec) per atom (not halved)
@group(0) @binding(7) var<storage, read_write> grad: array<f32>;
@group(0) @binding(8) var<storage, read_write> aener: array<vec2<f32>>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = (wid.x + wid.y * nwg.x) * WG + lid;
    if (i >= P.natoms) {
        return;
    }
    let info = ainfo[i];
    let pi = xyzq[info.x];
    var g = vec3<f32>(0.0);
    var ev = 0.0;
    var ee = 0.0;
    for (var k = info.y; k < info.z; k++) {
        let e = ex[2u * k];
        let pj = xyzq[e.x];
        let d = pi.xyz - pj.xyz + bitcast<vec3<f32>>(ex[2u * k + 1u].xyz);
        let r2 = dot(d, d);
        let r = sqrt(r2);
        let qq = pi.w * pj.w;
        var de = 0.0;
        if (e.y != 0u) {
            if (r2 < P.rc2) {
                let v = vdw_mod(tab[e.z], r, r2);
                ev += v.x;
                de = v.y;
            }
            let s = P.scale14;
            if (ELEC == 1u || (ELEC == 2u && r2 < P.rc2)) {
                let f = elec_shape(r, r2);
                ee += s * qq * f.x;
                de += s * qq * f.y;
            } else if (ELEC == 3u) {
                // s/r - erf(br)/r = (s - 1)/r + erfc(br)/r
                let f = elec_shape(r, r2);
                let inv = 1.0 / r;
                ee += qq * ((s - 1.0) * inv + f.x);
                de += qq * (-(s - 1.0) * inv * inv + f.y);
            }
        } else if (ELEC == 3u) {
            // excluded pair: -erf(br)/r = erfc(br)/r - 1/r
            let f = elec_shape(r, r2);
            let inv = 1.0 / r;
            ee += qq * (f.x - inv);
            de += qq * (f.y + inv * inv);
        }
        g += (de / r) * d;
    }
    aener[i] = vec2<f32>(ev, ee);
    if (GRAD) {
        g += cgrad[info.x].xyz;
        if (HAS_PME) {
            g += pgrad[i].xyz;
        }
        grad[3u * i] = g.x;
        grad[3u * i + 1u] = g.y;
        grad[3u * i + 2u] = g.z;
    }
}
