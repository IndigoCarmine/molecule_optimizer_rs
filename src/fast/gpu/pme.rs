//! Reciprocal-space smooth PME on the GPU.
//!
//! Per evaluation the CPU computes, in `f64`, the grid base index and the
//! fractional offset `w ∈ [0, 1)` of every charged atom (the only place where
//! absolute coordinates enter, so no `f32` precision is lost on large boxes)
//! and uploads them; everything else runs on the GPU: B-splines, charge
//! spreading (64-bit fixed-point integer atomics, deterministic), the 3D FFT
//! ([`super::fft`]), the influence function / energy, the inverse FFT and
//! the force gather.

use super::GpuContext;
use super::fft::{FftKernels, FftPlan};
use super::util::{self, Bind, Const};
use crate::error::{Error, Result};
use crate::fast::approx::pme::Pme;
use rayon::prelude::*;

const PME_WGSL: &str = include_str!("shaders/pme.wgsl");
/// Largest spline order supported on the GPU.
pub(crate) const MAX_GPU_ORDER: usize = 12;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PmeParams {
    k1: u32,
    k2: u32,
    k3: u32,
    nq: u32,
    ntot: u32,
    pad: [u32; 3],
    recip0: [f32; 4],
    recip1: [f32; 4],
    recip2: [f32; 4],
}

pub(crate) struct PmeGpu {
    k: [usize; 3],
    recip: [[f64; 3]; 3],
    atoms: Vec<u32>,
    q: Vec<f64>,
    ntot: usize,
    npart: usize,
    atoms_b: wgpu::Buffer,
    atoms_w: wgpu::Buffer,
    pub epart: wgpu::Buffer,
    pub pgrad: wgpu::Buffer,
    kernels: FftKernels,
    plan: FftPlan,
    spread: wgpu::ComputePipeline,
    convert: wgpu::ComputePipeline,
    influence: [wgpu::ComputePipeline; 2],
    gather: wgpu::ComputePipeline,
    bg_spread: wgpu::BindGroup,
    bg_convert: wgpu::BindGroup,
    bg_influence: wgpu::BindGroup,
    bg_gather: wgpu::BindGroup,
    h_b: Vec<[u32; 4]>,
    h_w: Vec<[f32; 4]>,
    _keep: Vec<wgpu::Buffer>,
}

impl PmeGpu {
    pub fn new(ctx: &GpuContext, pme: &Pme, natoms: usize) -> Result<PmeGpu> {
        let order = pme.order;
        if order > MAX_GPU_ORDER {
            return Err(Error::InvalidArgument(format!("GPU PME supports spline orders up to {MAX_GPU_ORDER} (got {order})")));
        }
        let k = pme.k;
        let [k1, k2, k3] = k;
        let ntot = k1 * k2 * k3;
        let nq = pme.atoms.len();
        let npart = ntot.div_ceil(256);
        let params = PmeParams {
            k1: k1 as u32,
            k2: k2 as u32,
            k3: k3 as u32,
            nq: nq as u32,
            ntot: ntot as u32,
            pad: [0; 3],
            recip0: [pme.recip[0][0] as f32, pme.recip[0][1] as f32, pme.recip[0][2] as f32, 0.0],
            recip1: [pme.recip[1][0] as f32, pme.recip[1][1] as f32, pme.recip[1][2] as f32, 0.0],
            recip2: [pme.recip[2][0] as f32, pme.recip[2][1] as f32, pme.recip[2][2] as f32, 0.0],
        };
        let pbuf = util::uniform(ctx, "pme params", std::mem::size_of::<PmeParams>());
        ctx.queue().write_buffer(&pbuf, 0, bytemuck::bytes_of(&params));
        let atoms_b = util::storage(ctx, "pme atoms b", nq * 16);
        let atoms_w = util::storage(ctx, "pme atoms w", nq * 16);
        let theta = util::storage(ctx, "pme theta", nq * 6 * order * 4);
        let qgrid = util::storage(ctx, "pme fixed-point grid", ntot * 8);
        let ga = util::storage(ctx, "pme grid a", ntot * 8);
        let gb = util::storage(ctx, "pme grid b", ntot * 8);
        // influence function: CPU layout (y, z, x) -> (x, y, z)
        let mut infl = vec![0f32; ntot];
        infl.par_chunks_mut(k2 * k3).enumerate().for_each(|(x, plane)| {
            for y in 0..k2 {
                for z in 0..k3 {
                    plane[y * k3 + z] = pme.infl[(y * k3 + z) * k1 + x] as f32;
                }
            }
        });
        let ibuf = util::storage(ctx, "pme influence", ntot * 4);
        ctx.queue().write_buffer(&ibuf, 0, bytemuck::cast_slice(&infl));
        let epart = util::storage(ctx, "pme energy partials", npart * 4);
        let pgrad = util::storage(ctx, "pme gradient", natoms * 16);
        let kernels = FftKernels::new(ctx);
        let plan = FftPlan::new(ctx, &kernels, k, &ga, &gb)?;
        let fx = if plan.forward_output() { &gb } else { &ga };
        let fy = if plan.inverse_output() { &gb } else { &ga };
        let module = |grad: bool| {
            let h = util::header(&[("ORDER", Const::U(order as u32)), ("GRAD", Const::B(grad))]);
            util::module(ctx, "pme", &[&h, PME_WGSL])
        };
        let (m0, m1) = (module(false), module(true));
        use Bind::*;
        let l_spread = util::layout(ctx, "pme spread", &[(0, Uniform), (1, Read), (2, Read), (3, ReadWrite), (4, ReadWrite)]);
        let l_convert = util::layout(ctx, "pme convert", &[(0, Uniform), (4, ReadWrite), (5, ReadWrite)]);
        let l_infl = util::layout(ctx, "pme influence", &[(0, Uniform), (6, ReadWrite), (7, Read), (8, ReadWrite)]);
        let l_gather = util::layout(ctx, "pme gather", &[(0, Uniform), (1, Read), (2, Read), (3, ReadWrite), (9, Read), (10, ReadWrite)]);
        let spread = util::pipeline(ctx, "pme spread", &m1, "spread", &l_spread);
        let convert = util::pipeline(ctx, "pme convert", &m1, "convert", &l_convert);
        let influence = [util::pipeline(ctx, "pme influence", &m0, "influence", &l_infl), util::pipeline(ctx, "pme influence", &m1, "influence", &l_infl)];
        let gather = util::pipeline(ctx, "pme gather", &m1, "gather", &l_gather);
        let bg_spread = util::bind_group(ctx, "pme spread", &l_spread, &[(0, &pbuf), (1, &atoms_b), (2, &atoms_w), (3, &theta), (4, &qgrid)]);
        let bg_convert = util::bind_group(ctx, "pme convert", &l_convert, &[(0, &pbuf), (4, &qgrid), (5, &ga)]);
        let bg_influence = util::bind_group(ctx, "pme influence", &l_infl, &[(0, &pbuf), (6, fx), (7, &ibuf), (8, &epart)]);
        let bg_gather = util::bind_group(ctx, "pme gather", &l_gather, &[(0, &pbuf), (1, &atoms_b), (2, &atoms_w), (3, &theta), (9, fy), (10, &pgrad)]);
        Ok(PmeGpu {
            k,
            recip: pme.recip,
            atoms: pme.atoms.clone(),
            q: pme.q.clone(),
            ntot,
            npart,
            atoms_b,
            atoms_w,
            epart,
            pgrad,
            kernels,
            plan,
            spread,
            convert,
            influence,
            gather,
            bg_spread,
            bg_convert,
            bg_influence,
            bg_gather,
            h_b: vec![[0; 4]; nq],
            h_w: vec![[0.0; 4]; nq],
            _keep: vec![pbuf, theta, qgrid, ga, gb, ibuf],
        })
    }

    /// Number of energy partial sums written per evaluation.
    pub fn num_partials(&self) -> usize {
        self.npart
    }
    pub fn num_charged(&self) -> usize {
        self.atoms.len()
    }

    /// Computes the grid base indices and fractional offsets (f64) and
    /// uploads them.
    pub fn prepare(&mut self, ctx: &GpuContext, pos: &[f64]) {
        let (k, recip, atoms, q) = (self.k, &self.recip, &self.atoms, &self.q);
        self.h_b.par_iter_mut().zip(self.h_w.par_iter_mut()).enumerate().with_min_len(1024).for_each(|(a, (b, w))| {
            let i = atoms[a] as usize;
            let r = [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];
            for d in 0..3 {
                let s = recip[d][0] * r[0] + recip[d][1] * r[1] + recip[d][2] * r[2];
                let u = s * k[d] as f64;
                let fl = u.floor();
                let mut ww = u - fl;
                if ww >= 1.0 {
                    ww = 0.0;
                }
                b[d] = (fl as i64).rem_euclid(k[d] as i64) as u32;
                w[d] = ww as f32;
            }
            b[3] = i as u32;
            w[3] = q[a] as f32;
        });
        if !self.h_b.is_empty() {
            ctx.queue().write_buffer(&self.atoms_b, 0, bytemuck::cast_slice(&self.h_b));
            ctx.queue().write_buffer(&self.atoms_w, 0, bytemuck::cast_slice(&self.h_w));
        }
    }

    pub fn encode(&self, pass: &mut wgpu::ComputePass, grad: bool) {
        let nq = self.atoms.len();
        if nq == 0 {
            return;
        }
        pass.set_pipeline(&self.spread);
        pass.set_bind_group(0, &self.bg_spread, &[]);
        util::dispatch(pass, nq.div_ceil(64));
        pass.set_pipeline(&self.convert);
        pass.set_bind_group(0, &self.bg_convert, &[]);
        util::dispatch(pass, self.ntot.div_ceil(256));
        self.plan.encode_forward(&self.kernels, pass);
        pass.set_pipeline(&self.influence[grad as usize]);
        pass.set_bind_group(0, &self.bg_influence, &[]);
        util::dispatch(pass, self.npart);
        if grad {
            self.plan.encode_inverse(&self.kernels, pass);
            pass.set_pipeline(&self.gather);
            pass.set_bind_group(0, &self.bg_gather, &[]);
            util::dispatch(pass, nq.div_ceil(64));
        }
    }
}
