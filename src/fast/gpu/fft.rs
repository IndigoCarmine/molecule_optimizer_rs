//! 3D complex FFT in WGSL compute shaders (f32, unnormalized).
//!
//! Each axis is transformed by a sequence of out-of-place Stockham
//! autosort passes (radix 4, 2, 3, 5, 7 — every length produced by
//! [`crate::fast::approx::fft_friendly_size`]); a pass processes all lines
//! of the grid along its axis in one dispatch and ping-pongs between two
//! buffers. Twiddle factors are computed in `f64` on the CPU and stored as
//! `f32`. The results are deterministic (no atomics, fixed operation order).

use super::GpuContext;
use super::util::{self, Bind, Const};
use crate::error::{Error, Result};
use std::collections::HashMap;

const FFT_WGSL: &str = include_str!("shaders/fft.wgsl");
const RADICES: [usize; 5] = [4, 2, 3, 5, 7];

/// Radix sequence of `n` (radix 4 first), or `None` if `n` has other prime
/// factors.
pub(crate) fn factorize(n: usize) -> Option<Vec<usize>> {
    let mut r = n;
    let mut out = Vec::new();
    while r.is_multiple_of(4) {
        out.push(4);
        r /= 4;
    }
    for f in [2, 3, 5, 7] {
        while r.is_multiple_of(f) {
            out.push(f);
            r /= f;
        }
    }
    (r == 1).then_some(out)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
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

struct Pass {
    radix: usize,
    bind: wgpu::BindGroup,
    groups: usize,
    _uniform: wgpu::Buffer,
}

/// Compiled radix pipelines (shared by all plans of a context).
pub(crate) struct FftKernels {
    layout: wgpu::BindGroupLayout,
    pipes: HashMap<usize, wgpu::ComputePipeline>,
}

impl FftKernels {
    pub fn new(ctx: &GpuContext) -> FftKernels {
        let layout = util::seq_layout(ctx, "fft", &[Bind::Uniform, Bind::Read, Bind::ReadWrite, Bind::Read]);
        let mut pipes = HashMap::new();
        for r in RADICES {
            let mut h = util::header(&[("R", Const::U(r as u32))]);
            let cs: Vec<String> = (0..r).map(|m| util::f32_literal((2.0 * std::f64::consts::PI * m as f64 / r as f64).cos())).collect();
            let sn: Vec<String> = (0..r).map(|m| util::f32_literal((2.0 * std::f64::consts::PI * m as f64 / r as f64).sin())).collect();
            h.push_str(&format!("var<private> CR: array<f32, {r}> = array<f32, {r}>({});\n", cs.join(", ")));
            h.push_str(&format!("var<private> SR: array<f32, {r}> = array<f32, {r}>({});\n", sn.join(", ")));
            let m = util::module(ctx, "fft", &[&h, FFT_WGSL]);
            pipes.insert(r, util::pipeline(ctx, "fft", &m, "main", &layout));
        }
        FftKernels { layout, pipes }
    }
}

/// Forward and inverse transform plan on two fixed grid buffers `a`, `b`.
///
/// The forward transform reads `a`; its result ends in
/// [`FftPlan::forward_output`]; the inverse transform starts from that
/// buffer and ends in [`FftPlan::inverse_output`].
pub(crate) struct FftPlan {
    pub dims: [usize; 3],
    fwd: Vec<Pass>,
    inv: Vec<Pass>,
    fwd_out_b: bool,
    inv_out_b: bool,
    _tw: wgpu::Buffer,
}

impl FftPlan {
    pub fn new(ctx: &GpuContext, k: &FftKernels, dims: [usize; 3], a: &wgpu::Buffer, b: &wgpu::Buffer) -> Result<FftPlan> {
        let total: usize = dims.iter().product();
        let mut factors = Vec::new();
        for &n in &dims {
            factors.push(factorize(n).ok_or_else(|| Error::InvalidArgument(format!("GPU FFT: unsupported length {n} (factors 2, 3, 5, 7 only)")))?);
        }
        // twiddle tables exp(-2 pi i m / n)
        let mut tw: Vec<[f32; 2]> = Vec::new();
        let mut tw_off = [0usize; 3];
        for (d, &n) in dims.iter().enumerate() {
            tw_off[d] = tw.len();
            for m in 0..n {
                let ang = -2.0 * std::f64::consts::PI * m as f64 / n as f64;
                tw.push([ang.cos() as f32, ang.sin() as f32]);
            }
        }
        let twb = util::storage(ctx, "fft twiddles", tw.len() * 8);
        ctx.queue().write_buffer(&twb, 0, bytemuck::cast_slice(&tw));
        let strides = [dims[1] * dims[2], dims[2], 1];
        let mut in_b = false;
        let make = |inverse: bool, in_b: &mut bool| -> Vec<Pass> {
            let mut passes = Vec::new();
            // axis order z, y, x
            for d in [2usize, 1, 0] {
                let n = dims[d];
                let mut ns = 1;
                for &r in &factors[d] {
                    let p = FftParams {
                        n: n as u32,
                        ns: ns as u32,
                        stride: strides[d] as u32,
                        lines: (total / n) as u32,
                        tw_off: tw_off[d] as u32,
                        inverse: inverse as u32,
                        jfast: (strides[d] == 1) as u32,
                        total: (total / r) as u32,
                    };
                    let u = util::uniform(ctx, "fft pass", std::mem::size_of::<FftParams>());
                    ctx.queue().write_buffer(&u, 0, bytemuck::bytes_of(&p));
                    let (src, dst) = if *in_b { (b, a) } else { (a, b) };
                    let bind = util::seq_bind_group(ctx, "fft pass", &k.layout, &[&u, src, dst, &twb]);
                    passes.push(Pass { radix: r, bind, groups: (total / r).div_ceil(64), _uniform: u });
                    *in_b = !*in_b;
                    ns *= r;
                }
            }
            passes
        };
        let fwd = make(false, &mut in_b);
        let fwd_out_b = in_b;
        let inv = make(true, &mut in_b);
        let inv_out_b = in_b;
        Ok(FftPlan { dims, fwd, inv, fwd_out_b, inv_out_b, _tw: twb })
    }

    fn encode(k: &FftKernels, passes: &[Pass], pass: &mut wgpu::ComputePass) {
        for p in passes {
            pass.set_pipeline(&k.pipes[&p.radix]);
            pass.set_bind_group(0, &p.bind, &[]);
            util::dispatch(pass, p.groups);
        }
    }
    pub fn encode_forward(&self, k: &FftKernels, pass: &mut wgpu::ComputePass) {
        Self::encode(k, &self.fwd, pass);
    }
    pub fn encode_inverse(&self, k: &FftKernels, pass: &mut wgpu::ComputePass) {
        Self::encode(k, &self.inv, pass);
    }
    /// `true` if the forward result is in `b` (else `a`).
    pub fn forward_output(&self) -> bool {
        self.fwd_out_b
    }
    /// `true` if the inverse result is in `b` (else `a`).
    pub fn inverse_output(&self) -> bool {
        self.inv_out_b
    }
    /// Number of compute passes (dispatches) per direction.
    pub fn num_passes(&self) -> usize {
        self.fwd.len()
    }
}

/// Standalone 3D FFT on the GPU (host data in and out), mainly for testing
/// the PME transform against a CPU FFT.
pub struct GpuFft3d {
    ctx: GpuContext,
    kernels: FftKernels,
    plan: FftPlan,
    a: wgpu::Buffer,
    b: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl GpuFft3d {
    /// Plan for a `dims[0] × dims[1] × dims[2]` grid (row-major, last index
    /// fastest); every length must factor into 2, 3, 5 and 7.
    pub fn new(ctx: &GpuContext, dims: [usize; 3]) -> Result<GpuFft3d> {
        let total: usize = dims.iter().product();
        if total == 0 {
            return Err(Error::InvalidArgument("empty FFT grid".into()));
        }
        let kernels = FftKernels::new(ctx);
        let a = util::storage(ctx, "fft a", total * 8);
        let b = util::storage(ctx, "fft b", total * 8);
        let plan = FftPlan::new(ctx, &kernels, dims, &a, &b)?;
        let staging = util::readback(ctx, "fft staging", total * 8);
        Ok(GpuFft3d { ctx: ctx.clone(), kernels, plan, a, b, staging })
    }

    /// Number of radix passes per transform.
    pub fn num_passes(&self) -> usize {
        self.plan.num_passes()
    }

    /// In-place transform of `data` (`[re, im]`): forward
    /// `X_k = Σ x_j exp(-2πi j·k/N)` or unnormalized inverse (`+` sign).
    pub fn transform(&self, data: &mut [[f32; 2]], inverse: bool) {
        let total: usize = self.plan.dims.iter().product();
        assert_eq!(data.len(), total, "FFT data length");
        let input = if inverse && self.plan.forward_output() { &self.b } else { &self.a };
        self.ctx.queue().write_buffer(input, 0, bytemuck::cast_slice(data));
        let mut enc = self.ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fft") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("fft"), timestamp_writes: None });
            if inverse {
                self.plan.encode_inverse(&self.kernels, &mut pass);
            } else {
                self.plan.encode_forward(&self.kernels, &mut pass);
            }
        }
        let out_b = if inverse { self.plan.inverse_output() } else { self.plan.forward_output() };
        enc.copy_buffer_to_buffer(if out_b { &self.b } else { &self.a }, 0, &self.staging, 0, (total * 8) as u64);
        self.ctx.queue().submit([enc.finish()]);
        util::map_blocking(&self.ctx, &self.staging, total * 8);
        util::read_mapped(&self.staging, 0, data);
        self.staging.unmap();
    }
}
