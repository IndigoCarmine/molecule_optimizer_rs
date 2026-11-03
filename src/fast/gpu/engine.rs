//! GPU evaluation engine shared by [`super::WgpuNonbonded`] (one large
//! system, optional PME) and [`super::GpuBatch`] (many systems in one
//! dispatch).
//!
//! # Data layout
//!
//! The cluster-pair lists of all systems are concatenated (global cluster,
//! slot and atom indices; per-system type-pair tables at a table offset).
//! Per system and list build the CPU computes, in `f64`:
//!
//! * a cluster origin `O_c` (mean of the build-time positions `x_i + T_i`
//!   of its atoms);
//! * per cluster-pair entry `(I, J, shift)` the origin difference
//!   `O_J + S - O_I` (stored as `f32`);
//! * per explicit pair `(i, j)` the constant `(O_ci - T_i) - (O_cj - T_j) + L_ij`
//!   (`L_ij` = minimum-image lattice vector at build time).
//!
//! Per evaluation only the cluster-relative coordinates `x_i + T_i - O_c`
//! (`f32`, a few Å, so `f32` keeps ~1e-7 Å resolution on any box size) and
//! the charges are uploaded (plus the PME grid offsets). Pair displacements
//! are `(x_a - O_I) - (x_b - O_J) - (O_J + S - O_I)` in `f32`.
//!
//! Outputs read back per evaluation: cluster energies, per-atom explicit-pair
//! energies, PME energy partial sums (all summed in `f64` on the CPU) and the
//! per-atom gradient.

use super::GpuContext;
use super::pme::PmeGpu;
use super::util::{self, Bind, Const, Grow};
use crate::error::{Error, Result};
use crate::fast::approx::ApproxNonbonded;
use crate::fast::approx::kernel::{ElecKind, KernelParams, Modifier};
use crate::fast::approx::pairlist::{DUMMY, PairList};
use crate::fast::approx::params::VdwKind;
use crate::fast::backend::NonbondedEnergy;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

const PAIR_WGSL: &str = include_str!("shaders/pair.wgsl");
const CLUSTER_WGSL: &str = include_str!("shaders/cluster.wgsl");
const ATOMS_WGSL: &str = include_str!("shaders/atoms.wgsl");
const WG: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NbParamsGpu {
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
    pad: [u32; 2],
}

/// One system of an evaluation.
pub(crate) struct SysRef<'a> {
    /// Template backend (parameters, exclusions, explicit pairs).
    pub nb: &'a ApproxNonbonded,
    /// Index of the template (type-pair table).
    pub tmpl: usize,
    pub list: Arc<PairList>,
    pub pos: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
struct SysLayout {
    coff: usize,
    nclusters: usize,
    aoff: usize,
    natoms: usize,
}

/// Unrolled 4×4 body of the cluster kernel (bit `4a + b` of the mask).
fn cluster_body() -> String {
    let mut s = String::new();
    for a in 0..4 {
        s.push_str(&format!("            if (((mask >> {}u) & 15u) != 0u) {{\n", 4 * a));
        for b in 0..4 {
            s.push_str(&format!(
                "                if ((mask & {}u) != 0u) {{\n                    let d = pi{a}.xyz - xj{b};\n                    let o = interact(d, pi{a}.w * pj{b}.w, ti{a} + tj{b});\n                    en += o.xyz;\n                    g{a} += o.w * d;\n                }}\n",
                1u32 << (4 * a + b)
            ));
        }
        s.push_str("            }\n");
    }
    s
}

fn same_kernel_settings(a: &KernelParams, b: &KernelParams) -> bool {
    a.vdw == b.vdw
        && a.rc2 == b.rc2
        && a.modifier == b.modifier
        && a.elec == b.elec
        && a.ele_n2 == b.ele_n2
        && a.delta == b.delta
        && a.k_rf == b.k_rf
        && a.c_rf == b.c_rf
        && a.beta == b.beta
}

pub(crate) struct Engine {
    ctx: GpuContext,
    header: String,
    cluster_layout: wgpu::BindGroupLayout,
    atoms_layout: wgpu::BindGroupLayout,
    cluster_pipes: HashMap<(u32, bool), wgpu::ComputePipeline>,
    atoms_pipes: Vec<wgpu::ComputePipeline>,
    params: wgpu::Buffer,
    base_params: NbParamsGpu,
    /// Reaction-field constant `c` (f64; applied on the CPU).
    c_rf: f64,
    tab: wgpu::Buffer,
    tab_off: Vec<usize>,
    xyzq: Grow,
    stype: Grow,
    cinfo: Grow,
    cj: Grow,
    cjd: Grow,
    cgrad: Grow,
    cener: Grow,
    ainfo: Grow,
    ex: Grow,
    grad: Grow,
    aener: Grow,
    dummy: wgpu::Buffer,
    bg_cluster: Option<wgpu::BindGroup>,
    bg_atoms: Option<wgpu::BindGroup>,
    pme: Option<PmeGpu>,
    staging: [Option<(wgpu::Buffer, usize)>; 2],
    next_staging: usize,
    // host side
    sys: Vec<SysLayout>,
    list_ids: Vec<usize>,
    origins: Vec<[f64; 3]>,
    nclusters: usize,
    natoms: usize,
    nentries: usize,
    tpc: u32,
    uploads: usize,
    h_xyzq: Vec<[f32; 4]>,
    h_stype: Vec<[u32; 2]>,
    h_cinfo: Vec<[u32; 2]>,
    h_cj: Vec<[u32; 2]>,
    h_cjd: Vec<[f32; 4]>,
    h_ainfo: Vec<[u32; 4]>,
    h_ex: Vec<[u32; 4]>,
    // results of the last evaluation
    r_cener: Vec<[f32; 4]>,
    r_aener: Vec<[f32; 2]>,
    r_epart: Vec<f32>,
    pub(crate) r_grad: Vec<f32>,
    pub(crate) energies: Vec<NonbondedEnergy>,
}

impl Engine {
    /// `templates`: distinct parameter sets (all with the same kernel
    /// settings). PME is set up from the first template if it has one
    /// (single-system mode only).
    pub fn new(ctx: &GpuContext, templates: &[&ApproxNonbonded], with_pme: bool) -> Result<Engine> {
        let t0 = templates.first().ok_or_else(|| Error::InvalidArgument("no systems".into()))?;
        let kp = &t0.kp;
        for t in templates {
            if !same_kernel_settings(kp, &t.kp) || t.p.kind != t0.p.kind || t.p.scale14 != t0.p.scale14 {
                return Err(Error::InvalidArgument("all systems of a GPU batch need the same force field and options".into()));
            }
        }
        let erfc_consts: Vec<(String, Const)> =
            crate::fast::approx::erfc::A.iter().enumerate().map(|(k, &a)| (format!("ERFC_A{k}"), Const::F(a))).collect();
        let mut consts: Vec<(&str, Const)> = vec![
            ("VDW", Const::U(if kp.vdw == VdwKind::Mmff { 0 } else { 1 })),
            (
                "ELEC",
                Const::U(match kp.elec {
                    ElecKind::None => 0,
                    ElecKind::Full => 1,
                    ElecKind::ReactionField => 2,
                    ElecKind::Ewald => 3,
                }),
            ),
            (
                "MODI",
                Const::U(match kp.modifier {
                    Modifier::None => 0,
                    Modifier::Shift => 1,
                    Modifier::Switch { .. } => 2,
                }),
            ),
            ("ELE_N2", Const::B(kp.ele_n2)),
        ];
        for (n, c) in &erfc_consts {
            consts.push((n.as_str(), *c));
        }
        let header = util::header(&consts);
        let (rsw, inv_w) = match kp.modifier {
            Modifier::Switch { rsw, inv_w } => (rsw, inv_w),
            _ => (0.0, 0.0),
        };
        let base_params = NbParamsGpu {
            rc2: kp.rc2 as f32,
            k_rf: kp.k_rf as f32,
            c_rf: kp.c_rf as f32,
            beta: kp.beta as f32,
            delta: kp.delta as f32,
            rsw: rsw as f32,
            inv_w: inv_w as f32,
            scale14: t0.p.scale14 as f32,
            nclusters: 0,
            natoms: 0,
            pad: [0; 2],
        };
        // type-pair tables
        let mut tab: Vec<[f32; 4]> = Vec::new();
        let mut tab_off = Vec::new();
        for t in templates {
            tab_off.push(tab.len());
            for k in 0..t.p.ntypes * t.p.ntypes {
                tab.push([t.p.p1[k] as f32, t.p.p2[k] as f32, t.p.p3[k] as f32, t.kp.shift[k] as f32]);
            }
        }
        if tab.len() >= u32::MAX as usize / 2 {
            return Err(Error::InvalidArgument("GPU type table too large".into()));
        }
        let tabb = util::storage(ctx, "type-pair table", tab.len() * 16);
        ctx.queue().write_buffer(&tabb, 0, bytemuck::cast_slice(&tab));
        use Bind::*;
        let cluster_layout = util::seq_layout(ctx, "cluster", &[Uniform, Read, Read, Read, Read, Read, Read, ReadWrite, ReadWrite]);
        let atoms_layout = util::seq_layout(ctx, "atoms", &[Uniform, Read, Read, Read, Read, Read, Read, ReadWrite, ReadWrite]);
        let pme = match (&t0.pme, with_pme) {
            (Some(p), true) => Some(PmeGpu::new(ctx, p, t0.p.n)?),
            _ => None,
        };
        let has_pme = pme.as_ref().is_some_and(|p| p.num_charged() > 0);
        let atoms_pipes = [false, true]
            .iter()
            .map(|&g| {
                let h = util::header(&[("WG", Const::U(WG)), ("GRAD", Const::B(g)), ("HAS_PME", Const::B(has_pme)), ("SPLIT_C", Const::B(false))]);
                let m = util::module(ctx, "atoms", &[&header, &h, PAIR_WGSL, ATOMS_WGSL]);
                util::pipeline(ctx, "atoms", &m, "main", &atoms_layout)
            })
            .collect();
        Ok(Engine {
            ctx: ctx.clone(),
            header,
            cluster_layout,
            atoms_layout,
            cluster_pipes: HashMap::new(),
            atoms_pipes,
            params: util::uniform(ctx, "nonbonded params", std::mem::size_of::<NbParamsGpu>()),
            base_params,
            c_rf: if kp.elec == ElecKind::ReactionField { kp.c_rf } else { 0.0 },
            tab: tabb,
            tab_off,
            xyzq: Grow::new(ctx, "slot coordinates", 16),
            stype: Grow::new(ctx, "slot types", 16),
            cinfo: Grow::new(ctx, "cluster info", 16),
            cj: Grow::new(ctx, "cluster pairs", 16),
            cjd: Grow::new(ctx, "cluster pair shifts", 16),
            cgrad: Grow::new(ctx, "cluster gradient", 16),
            cener: Grow::new(ctx, "cluster energies", 16),
            ainfo: Grow::new(ctx, "atom info", 16),
            ex: Grow::new(ctx, "explicit pairs", 16),
            grad: Grow::new(ctx, "gradient", 16),
            aener: Grow::new(ctx, "atom energies", 16),
            dummy: util::storage(ctx, "dummy", 16),
            bg_cluster: None,
            bg_atoms: None,
            pme,
            staging: [None, None],
            next_staging: 0,
            sys: Vec::new(),
            list_ids: Vec::new(),
            origins: Vec::new(),
            nclusters: 0,
            natoms: 0,
            nentries: 0,
            tpc: 1,
            uploads: 0,
            h_xyzq: Vec::new(),
            h_stype: Vec::new(),
            h_cinfo: Vec::new(),
            h_cj: Vec::new(),
            h_cjd: Vec::new(),
            h_ainfo: Vec::new(),
            h_ex: Vec::new(),
            r_cener: Vec::new(),
            r_aener: Vec::new(),
            r_epart: Vec::new(),
            r_grad: Vec::new(),
            energies: Vec::new(),
        })
    }

    /// Number of pair-list uploads so far.
    pub fn num_uploads(&self) -> usize {
        self.uploads
    }
    /// Invocations per i-cluster of the current cluster kernel.
    pub fn threads_per_cluster(&self) -> u32 {
        self.tpc
    }
    /// Atom offset of system `s` in [`Engine::r_grad`] (times 3).
    pub fn atom_offset(&self, s: usize) -> usize {
        self.sys[s].aoff
    }

    fn cluster_pipe(&mut self, tpc: u32, grad: bool) -> &wgpu::ComputePipeline {
        let (ctx, header, layout) = (&self.ctx, &self.header, &self.cluster_layout);
        self.cluster_pipes.entry((tpc, grad)).or_insert_with(|| {
            let h = util::header(&[("WG", Const::U(WG)), ("TPC", Const::U(tpc)), ("GRAD", Const::B(grad)), ("SPLIT_C", Const::B(true))]);
            let src = CLUSTER_WGSL.replace("//!BODY\n", &cluster_body()).replace("//!BODY\r\n", &cluster_body());
            let m = util::module(ctx, "cluster", &[header, &h, PAIR_WGSL, &src]);
            util::pipeline(ctx, "cluster", &m, "main", layout)
        })
    }

    fn upload_topology(&mut self, sys: &[SysRef]) {
        self.sys.clear();
        self.list_ids.clear();
        self.origins.clear();
        self.h_stype.clear();
        self.h_cinfo.clear();
        self.h_cj.clear();
        self.h_cjd.clear();
        self.h_ainfo.clear();
        self.h_ex.clear();
        let (mut coff, mut aoff) = (0usize, 0usize);
        for s in sys {
            let (l, p) = (&*s.list, &s.nb.p);
            let toff = self.tab_off[s.tmpl];
            let nt = p.ntypes;
            self.sys.push(SysLayout { coff, nclusters: l.nclusters, aoff, natoms: p.n });
            self.list_ids.push(Arc::as_ptr(&s.list) as usize);
            let wpos = |i: usize| -> [f64; 3] {
                let t = l.tshift[i];
                [l.ref_pos[3 * i] + t[0], l.ref_pos[3 * i + 1] + t[1], l.ref_pos[3 * i + 2] + t[2]]
            };
            for a in &l.atoms {
                let mut o = [0.0; 3];
                let mut cnt = 0.0;
                for &i in a.iter().filter(|&&i| i != DUMMY) {
                    let w = wpos(i as usize);
                    for d in 0..3 {
                        o[d] += w[d];
                    }
                    cnt += 1.0;
                }
                if cnt > 0.0 {
                    o = o.map(|v| v / cnt);
                }
                self.origins.push(o);
                for &i in a {
                    if i == DUMMY {
                        self.h_stype.push([toff as u32, 0]);
                    } else {
                        let ti = p.atom_type[i as usize] as usize;
                        self.h_stype.push([(toff + ti * nt) as u32, ti as u32]);
                    }
                }
            }
            for c in 0..l.nclusters {
                let (b, e) = (l.ci[c] as usize, l.ci[c + 1] as usize);
                let start = self.h_cj.len();
                let oi = self.origins[coff + c];
                for ent in &l.cj[b..e] {
                    let oj = self.origins[coff + ent.j as usize];
                    let sh = l.shifts[ent.shift as usize];
                    self.h_cj.push([(coff + ent.j as usize) as u32, ent.mask as u32]);
                    self.h_cjd.push([(oj[0] + sh[0] - oi[0]) as f32, (oj[1] + sh[1] - oi[1]) as f32, (oj[2] + sh[2] - oi[2]) as f32, 0.0]);
                }
                self.h_cinfo.push([start as u32, self.h_cj.len() as u32]);
            }
            let ex = &s.nb.explicit;
            for i in 0..p.n {
                let si = l.slot[i] as usize;
                let start = self.h_ex.len() / 2;
                let oi = self.origins[coff + si / 4];
                let ti = l.tshift[i];
                for k in ex.start[i] as usize..ex.start[i + 1] as usize {
                    let j = ex.partner[k] as usize;
                    let sj = l.slot[j] as usize;
                    let oj = self.origins[coff + sj / 4];
                    let tj = l.tshift[j];
                    let d0 = [0, 1, 2].map(|d| l.ref_pos[3 * i + d] - l.ref_pos[3 * j + d]);
                    let dm = match &s.nb.cell {
                        Some(c) => c.min_image(d0),
                        None => d0,
                    };
                    let cst = [0, 1, 2].map(|d| ((oi[d] - ti[d]) - (oj[d] - tj[d]) + (dm[d] - d0[d])) as f32);
                    let tidx = toff + p.atom_type[i] as usize * nt + p.atom_type[j] as usize;
                    self.h_ex.push([(4 * coff + sj) as u32, ex.is14[k] as u32, tidx as u32, 0]);
                    self.h_ex.push([cst[0].to_bits(), cst[1].to_bits(), cst[2].to_bits(), 0]);
                }
                self.h_ainfo.push([(4 * coff + si) as u32, start as u32, (self.h_ex.len() / 2) as u32, 0]);
            }
            coff += l.nclusters;
            aoff += p.n;
        }
        self.nclusters = coff;
        self.natoms = aoff;
        self.nentries = self.h_cj.len();
        let ctx = self.ctx.clone();
        let mut grown = false;
        grown |= self.stype.upload(&ctx, &self.h_stype);
        grown |= self.cinfo.upload(&ctx, &self.h_cinfo);
        grown |= self.cj.upload(&ctx, &self.h_cj);
        grown |= self.cjd.upload(&ctx, &self.h_cjd);
        grown |= self.ainfo.upload(&ctx, &self.h_ainfo);
        grown |= self.ex.upload(&ctx, &self.h_ex);
        grown |= self.xyzq.ensure(&ctx, 4 * self.nclusters * 16);
        grown |= self.cgrad.ensure(&ctx, 4 * self.nclusters * 16);
        grown |= self.cener.ensure(&ctx, self.nclusters * 16);
        grown |= self.grad.ensure(&ctx, self.natoms * 12);
        grown |= self.aener.ensure(&ctx, self.natoms * 8);
        let mut prm = self.base_params;
        prm.nclusters = self.nclusters as u32;
        prm.natoms = self.natoms as u32;
        ctx.queue().write_buffer(&self.params, 0, bytemuck::bytes_of(&prm));
        if grown || self.bg_cluster.is_none() {
            self.bg_cluster = Some(util::seq_bind_group(
                &ctx,
                "cluster",
                &self.cluster_layout,
                &[&self.params, &self.xyzq.buf, &self.stype.buf, &self.tab, &self.cinfo.buf, &self.cj.buf, &self.cjd.buf, &self.cgrad.buf, &self.cener.buf],
            ));
            let pg = self.pme.as_ref().map(|p| &p.pgrad).unwrap_or(&self.dummy);
            self.bg_atoms = Some(util::seq_bind_group(
                &ctx,
                "atoms",
                &self.atoms_layout,
                &[&self.params, &self.xyzq.buf, &self.tab, &self.ainfo.buf, &self.ex.buf, &self.cgrad.buf, pg, &self.grad.buf, &self.aener.buf],
            ));
        }
        // invocations per i-cluster from the average list length
        let avg = self.nentries as f64 / self.nclusters.max(1) as f64;
        let mut tpc = 1u32;
        while tpc < 64 && (2 * tpc) as f64 * 32.0 <= avg {
            tpc *= 2;
        }
        if let Some(v) = std::env::var("MOLOPT_GPU_TPC").ok().and_then(|s| s.parse::<u32>().ok())
            && v.is_power_of_two()
            && v <= 64
        {
            tpc = v;
        }
        self.tpc = tpc;
        self.h_xyzq.resize(4 * self.nclusters, [0.0; 4]);
        self.uploads += 1;
    }

    /// Evaluates all systems; results in [`Engine::energies`] and (if `grad`)
    /// [`Engine::r_grad`] (3 per atom, systems concatenated).
    pub fn evaluate(&mut self, sys: &[SysRef], grad: bool) {
        let changed = self.list_ids.len() != sys.len() || sys.iter().zip(&self.list_ids).any(|(s, &id)| Arc::as_ptr(&s.list) as usize != id);
        if changed {
            self.upload_topology(sys);
        }
        let ctx = self.ctx.clone();
        // cluster-relative coordinates
        for (s, lay) in sys.iter().zip(&self.sys) {
            let (l, pos, qs) = (&*s.list, s.pos, &s.nb.qs);
            assert!(pos.len() >= 3 * lay.natoms, "position vector too short");
            let origins = &self.origins[lay.coff..lay.coff + lay.nclusters];
            self.h_xyzq[4 * lay.coff..4 * (lay.coff + lay.nclusters)].par_chunks_mut(4).zip(origins.par_iter()).zip(l.atoms.par_iter()).with_min_len(256).for_each(
                |((out, o), a)| {
                    for k in 0..4 {
                        let i = a[k];
                        out[k] = if i == DUMMY {
                            [0.0; 4]
                        } else {
                            let i = i as usize;
                            let t = l.tshift[i];
                            [
                                (pos[3 * i] + t[0] - o[0]) as f32,
                                (pos[3 * i + 1] + t[1] - o[1]) as f32,
                                (pos[3 * i + 2] + t[2] - o[2]) as f32,
                                qs[i] as f32,
                            ]
                        };
                    }
                },
            );
        }
        self.xyzq.upload(&ctx, &self.h_xyzq);
        if let Some(p) = self.pme.as_mut() {
            p.prepare(&ctx, sys[0].pos);
        }
        // sizes of the readback parts (bytes)
        let (nc, na) = (self.nclusters, self.natoms);
        let npart = self.pme.as_ref().map(|p| p.num_partials()).unwrap_or(0);
        let off_aener = 16 * nc;
        let off_epart = off_aener + 8 * na;
        let off_grad = off_epart + 4 * npart;
        let total = if grad { off_grad + 12 * na } else { off_grad };
        let si = self.next_staging;
        self.next_staging ^= 1;
        if self.staging[si].as_ref().is_none_or(|(_, cap)| *cap < total) {
            let cap = (total + total / 4).max(64).div_ceil(16) * 16;
            self.staging[si] = Some((util::readback(&ctx, "staging", cap), cap));
        }
        let tpc = self.tpc;
        let groups_c = nc.div_ceil((WG / tpc) as usize);
        let groups_a = na.div_ceil(WG as usize);
        self.cluster_pipe(tpc, grad);
        let cpipe = &self.cluster_pipes[&(tpc, grad)];
        let mut enc = ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nonbonded") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("nonbonded"), timestamp_writes: None });
            if let Some(p) = &self.pme {
                p.encode(&mut pass, grad);
            }
            pass.set_pipeline(cpipe);
            pass.set_bind_group(0, self.bg_cluster.as_ref().unwrap(), &[]);
            util::dispatch(&mut pass, groups_c);
            pass.set_pipeline(&self.atoms_pipes[grad as usize]);
            pass.set_bind_group(0, self.bg_atoms.as_ref().unwrap(), &[]);
            util::dispatch(&mut pass, groups_a);
        }
        let staging = &self.staging[si].as_ref().unwrap().0;
        if nc > 0 {
            enc.copy_buffer_to_buffer(&self.cener.buf, 0, staging, 0, (16 * nc) as u64);
        }
        if na > 0 {
            enc.copy_buffer_to_buffer(&self.aener.buf, 0, staging, off_aener as u64, (8 * na) as u64);
        }
        if let Some(p) = &self.pme
            && npart > 0
        {
            enc.copy_buffer_to_buffer(&p.epart, 0, staging, off_epart as u64, (4 * npart) as u64);
        }
        if grad && na > 0 {
            enc.copy_buffer_to_buffer(&self.grad.buf, 0, staging, off_grad as u64, (12 * na) as u64);
        }
        ctx.queue().submit([enc.finish()]);
        util::map_blocking(&ctx, staging, total);
        {
            let view = staging.get_mapped_range(0..(total.max(4).div_ceil(4) * 4) as u64).expect("mapped readback");
            self.r_cener.resize(nc, [0.0; 4]);
            self.r_aener.resize(na, [0.0; 2]);
            self.r_epart.resize(npart, 0.0);
            bytemuck::cast_slice_mut::<_, u8>(&mut self.r_cener).copy_from_slice(&view[..off_aener]);
            bytemuck::cast_slice_mut::<_, u8>(&mut self.r_aener).copy_from_slice(&view[off_aener..off_epart]);
            bytemuck::cast_slice_mut::<_, u8>(&mut self.r_epart).copy_from_slice(&view[off_epart..off_grad]);
            if grad {
                self.r_grad.resize(3 * na, 0.0);
                bytemuck::cast_slice_mut::<_, u8>(&mut self.r_grad).copy_from_slice(&view[off_grad..off_grad + 12 * na]);
            }
        }
        staging.unmap();
        // energies (f64)
        self.energies.clear();
        for (k, lay) in self.sys.iter().enumerate() {
            let (mut v, mut e, mut qq) = (0.0f64, 0.0f64, 0.0f64);
            for c in &self.r_cener[lay.coff..lay.coff + lay.nclusters] {
                v += c[0] as f64;
                e += c[1] as f64;
                qq += c[2] as f64;
            }
            // reaction-field constant shift of the cluster pairs (f64)
            e -= self.c_rf * qq;
            for a in &self.r_aener[lay.aoff..lay.aoff + lay.natoms] {
                v += a[0] as f64;
                e += a[1] as f64;
            }
            v *= 0.5;
            e *= 0.5;
            if k == 0 && self.pme.is_some() {
                e += 0.5 * self.r_epart.iter().map(|&x| x as f64).sum::<f64>() + sys[0].nb.ewald_const;
            }
            self.energies.push(NonbondedEnergy { vdw: v, electrostatic: e });
        }
    }
}
