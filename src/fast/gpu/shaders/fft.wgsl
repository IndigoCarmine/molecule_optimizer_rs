// One radix-R pass of an out-of-place Stockham autosort FFT applied to all
// lines of a 3D grid along one axis (complex f32, unnormalized).
//
// Line element k of line L lives at base(L) + k * stride with
// base(L) = (L / stride) * n * stride + L % stride. For the pass with
// sub-transform size ns (product of the previous radices), invocation j
// (0 <= j < n / R) reads v[r] = src[j + r n / R], applies the twiddles
// w^(r (j mod ns) n / (ns R)), w = exp(-2 pi i / n) (conjugated for the
// inverse), does a radix-R DFT and writes v[q] to
// dst[(j / ns) ns R + j mod ns + q ns].
//
// Generated constants: R, CR / SR (cos / sin of 2 pi m / R, m < R).

struct FftParams {
    n: u32,
    ns: u32,
    stride: u32,
    lines: u32,
    tw_off: u32,
    inverse: u32,
    jfast: u32,
    total: u32,
}

@group(0) @binding(0) var<uniform> F: FftParams;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;
// exp(-2 pi i m / n), m < n, for every axis length (concatenated)
@group(0) @binding(3) var<storage, read> tw: array<vec2<f32>>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = (wid.x + wid.y * nwg.x) * 64u + lid;
    if (g >= F.total) {
        return;
    }
    let m = F.n / R;
    var j: u32;
    var l: u32;
    if (F.jfast != 0u) {
        j = g % m;
        l = g / m;
    } else {
        l = g % F.lines;
        j = g / F.lines;
    }
    let st = F.stride;
    let base = (l / st) * F.n * st + l % st;
    let k = j % F.ns;
    let tstep = F.n / (F.ns * R);
    let sgn = select(1.0, -1.0, F.inverse != 0u);
    var v: array<vec2<f32>, R>;
    for (var r = 0u; r < R; r++) {
        var x = src[base + (j + r * m) * st];
        if (r > 0u) {
            var t = tw[F.tw_off + (k * r * tstep) % F.n];
            t.y *= sgn;
            x = cmul(x, t);
        }
        v[r] = x;
    }
    let od = (j / F.ns) * F.ns * R + k;
    for (var q = 0u; q < R; q++) {
        var acc = v[0];
        for (var r = 1u; r < R; r++) {
            let mm = (r * q) % R;
            // exp(-+ 2 pi i mm / R)
            let w = vec2<f32>(CR[mm], -sgn * SR[mm]);
            acc += cmul(v[r], w);
        }
        dst[base + (od + q * F.ns) * st] = acc;
    }
}
