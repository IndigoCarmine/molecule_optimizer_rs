//! Small wgpu helpers: shader assembly, buffers, layouts, dispatch.

use super::GpuContext;
use std::fmt::Write;

/// Compile-time constant of a generated WGSL header.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Const {
    U(u32),
    B(bool),
    F(f64),
}

/// `const NAME: type = value;` lines.
pub(crate) fn header(consts: &[(&str, Const)]) -> String {
    let mut s = String::new();
    for (name, c) in consts {
        match c {
            Const::U(v) => writeln!(s, "const {name}: u32 = {v}u;"),
            Const::B(v) => writeln!(s, "const {name}: bool = {v};"),
            Const::F(v) => writeln!(s, "const {name}: f32 = {};", f32_literal(*v)),
        }
        .unwrap();
    }
    s
}

/// WGSL abstract-float literal of `v` (round trip through `f64`).
pub(crate) fn f32_literal(v: f64) -> String {
    let s = format!("{v:?}");
    if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") { s } else { format!("{s}.0") }
}

pub(crate) fn module(ctx: &GpuContext, label: &str, parts: &[&str]) -> wgpu::ShaderModule {
    let src: String = parts.concat();
    ctx.device().create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) })
}

/// Binding kinds of a bind-group layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bind {
    Uniform,
    Read,
    ReadWrite,
}

pub(crate) fn layout(ctx: &GpuContext, label: &str, entries: &[(u32, Bind)]) -> wgpu::BindGroupLayout {
    let e: Vec<wgpu::BindGroupLayoutEntry> = entries
        .iter()
        .map(|&(binding, b)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: match b {
                    Bind::Uniform => wgpu::BufferBindingType::Uniform,
                    Bind::Read => wgpu::BufferBindingType::Storage { read_only: true },
                    Bind::ReadWrite => wgpu::BufferBindingType::Storage { read_only: false },
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    ctx.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries: &e })
}

/// Sequential layout `0..n`.
pub(crate) fn seq_layout(ctx: &GpuContext, label: &str, kinds: &[Bind]) -> wgpu::BindGroupLayout {
    let e: Vec<(u32, Bind)> = kinds.iter().enumerate().map(|(i, &b)| (i as u32, b)).collect();
    layout(ctx, label, &e)
}

pub(crate) fn pipeline(ctx: &GpuContext, label: &str, module: &wgpu::ShaderModule, entry: &str, layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let pl = ctx.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    ctx.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

pub(crate) fn bind_group(ctx: &GpuContext, label: &str, layout: &wgpu::BindGroupLayout, entries: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
    let e: Vec<wgpu::BindGroupEntry> =
        entries.iter().map(|&(binding, b)| wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }).collect();
    ctx.device().create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &e })
}

pub(crate) fn seq_bind_group(ctx: &GpuContext, label: &str, layout: &wgpu::BindGroupLayout, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let e: Vec<(u32, &wgpu::Buffer)> = buffers.iter().enumerate().map(|(i, &b)| (i as u32, b)).collect();
    bind_group(ctx, label, layout, &e)
}

/// Storage buffer of at least `bytes` bytes (rounded up to 16, minimum 16).
pub(crate) fn storage(ctx: &GpuContext, label: &str, bytes: usize) -> wgpu::Buffer {
    let size = (bytes.max(16).div_ceil(16) * 16) as u64;
    ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

pub(crate) fn uniform(ctx: &GpuContext, label: &str, bytes: usize) -> wgpu::Buffer {
    let size = (bytes.max(16).div_ceil(16) * 16) as u64;
    ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub(crate) fn readback(ctx: &GpuContext, label: &str, bytes: usize) -> wgpu::Buffer {
    let size = (bytes.max(16).div_ceil(16) * 16) as u64;
    ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A storage buffer that grows (by 1.5x) when more capacity is needed.
pub(crate) struct Grow {
    pub buf: wgpu::Buffer,
    pub cap: usize,
    label: &'static str,
}

impl Grow {
    pub fn new(ctx: &GpuContext, label: &'static str, bytes: usize) -> Grow {
        let cap = bytes.max(16).div_ceil(16) * 16;
        Grow { buf: storage(ctx, label, cap), cap, label }
    }
    /// Ensures a capacity of `bytes`; returns `true` if the buffer was replaced.
    pub fn ensure(&mut self, ctx: &GpuContext, bytes: usize) -> bool {
        if bytes <= self.cap {
            return false;
        }
        let cap = (bytes + bytes / 2).div_ceil(16) * 16;
        self.buf = storage(ctx, self.label, cap);
        self.cap = cap;
        true
    }
    /// Writes `data` at offset 0 (growing the buffer first if necessary);
    /// returns `true` if the buffer was replaced.
    pub fn upload<T: bytemuck::Pod>(&mut self, ctx: &GpuContext, data: &[T]) -> bool {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let grown = self.ensure(ctx, bytes.len());
        if !bytes.is_empty() {
            ctx.queue().write_buffer(&self.buf, 0, bytes);
        }
        grown
    }
}

/// Maximum workgroups per dispatch dimension used by [`dispatch`].
pub(crate) const MAX_GROUPS_X: u32 = 32768;

/// Dispatches `groups` workgroups as a 2D grid (shaders compute the linear
/// workgroup index as `wid.x + wid.y * num_workgroups.x`).
pub(crate) fn dispatch(pass: &mut wgpu::ComputePass, groups: usize) {
    if groups == 0 {
        return;
    }
    let g = groups as u32;
    let x = g.min(MAX_GROUPS_X);
    pass.dispatch_workgroups(x, g.div_ceil(x), 1);
}

/// Copies a mapped readback range into `out`.
pub(crate) fn read_mapped<T: bytemuck::Pod>(buf: &wgpu::Buffer, offset: usize, out: &mut [T]) {
    let bytes = std::mem::size_of_val(out);
    if bytes == 0 {
        return;
    }
    let view = buf.get_mapped_range(offset as u64..(offset + bytes) as u64).expect("mapped readback range");
    bytemuck::cast_slice_mut::<T, u8>(out).copy_from_slice(&view);
}

/// Maps `buf` for reading and blocks until the mapping is available.
pub(crate) fn map_blocking(ctx: &GpuContext, buf: &wgpu::Buffer, bytes: usize) {
    let size = (bytes.max(4).div_ceil(4) * 4) as u64;
    buf.map_async(wgpu::MapMode::Read, 0..size, |r| r.expect("GPU readback mapping failed"));
    ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("GPU poll failed");
}
