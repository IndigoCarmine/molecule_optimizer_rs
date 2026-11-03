//! Adapter / device selection.

use crate::error::{Error, Result};
use std::sync::Arc;

/// Options of [`GpuContext::new_blocking_with`].
#[derive(Debug, Clone)]
pub struct GpuContextOptions {
    /// Backends to consider (default: Vulkan, Metal, DX12; overridable with
    /// the `WGPU_BACKEND` environment variable, e.g. `vulkan`, `dx12`, `metal`).
    pub backends: wgpu::Backends,
    /// Adapter preference (default: high performance, overridable with
    /// `WGPU_POWER_PREF=low|high|none`).
    pub power_preference: wgpu::PowerPreference,
    /// Allow a software adapter (WARP, lavapipe, SwiftShader) if no hardware
    /// adapter is available (default `true`).
    pub allow_software: bool,
    /// Use a software adapter only (for testing; default from the
    /// environment variable `MOLOPT_GPU_SOFTWARE=1`).
    pub force_software: bool,
}

impl Default for GpuContextOptions {
    fn default() -> Self {
        GpuContextOptions {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
            power_preference: wgpu::PowerPreference::from_env().unwrap_or(wgpu::PowerPreference::HighPerformance),
            allow_software: true,
            force_software: std::env::var_os("MOLOPT_GPU_SOFTWARE").is_some_and(|v| v == "1"),
        }
    }
}

struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
}

/// A wgpu device and queue shared by GPU backends (cheap to clone).
#[derive(Clone)]
pub struct GpuContext {
    inner: Arc<Inner>,
}

/// Two contexts are equal if they share the same device.
impl PartialEq for GpuContext {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext").field("adapter", &self.inner.info.name).field("backend", &self.inner.info.backend).finish()
    }
}

fn instance(backends: wgpu::Backends) -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    wgpu::Instance::new(desc)
}

/// Lists the adapters visible to wgpu with the default backends.
pub fn enumerate_adapters() -> Vec<wgpu::AdapterInfo> {
    let backends = GpuContextOptions::default().backends;
    pollster::block_on(instance(backends).enumerate_adapters(backends)).iter().map(|a| a.get_info()).collect()
}

fn is_software(info: &wgpu::AdapterInfo) -> bool {
    info.device_type == wgpu::DeviceType::Cpu
}

impl GpuContext {
    /// Selects an adapter with the default options and creates a device
    /// (blocking). Returns an error if no usable adapter exists.
    pub fn new_blocking() -> Result<GpuContext> {
        Self::new_blocking_with(&GpuContextOptions::default())
    }

    /// [`GpuContext::new_blocking`] with explicit options.
    ///
    /// Adapters are ranked discrete > integrated > virtual > other > software;
    /// among equal device types Vulkan / Metal are preferred over DX12.
    pub fn new_blocking_with(opts: &GpuContextOptions) -> Result<GpuContext> {
        let inst = instance(opts.backends);
        let adapters = pollster::block_on(inst.enumerate_adapters(opts.backends));
        let rank = |info: &wgpu::AdapterInfo| -> i32 {
            let mut r = match info.device_type {
                wgpu::DeviceType::DiscreteGpu => 40,
                wgpu::DeviceType::IntegratedGpu => 30,
                wgpu::DeviceType::VirtualGpu => 20,
                wgpu::DeviceType::Other => 10,
                wgpu::DeviceType::Cpu => 0,
            };
            if opts.power_preference == wgpu::PowerPreference::LowPower && info.device_type == wgpu::DeviceType::IntegratedGpu {
                r += 15;
            }
            r + match info.backend {
                wgpu::Backend::Vulkan | wgpu::Backend::Metal => 2,
                wgpu::Backend::Dx12 => 1,
                _ => 0,
            }
        };
        let mut best: Option<(i32, wgpu::Adapter)> = None;
        for a in adapters {
            if !a.get_downlevel_capabilities().flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) {
                continue;
            }
            let info = a.get_info();
            let sw = is_software(&info);
            if (opts.force_software && !sw) || (sw && !opts.allow_software && !opts.force_software) {
                continue;
            }
            let r = rank(&info);
            if best.as_ref().is_none_or(|(br, _)| r > *br) {
                best = Some((r, a));
            }
        }
        let Some((_, adapter)) = best else {
            return Err(Error::InvalidArgument("no usable GPU adapter with compute shaders found".into()));
        };
        let info = adapter.get_info();
        let alim = adapter.limits();
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = alim.max_storage_buffers_per_shader_stage.min(16);
        limits.max_storage_buffer_binding_size = alim.max_storage_buffer_binding_size;
        limits.max_buffer_size = alim.max_buffer_size;
        limits.max_compute_workgroup_storage_size = alim.max_compute_workgroup_storage_size;
        limits.max_compute_invocations_per_workgroup = alim.max_compute_invocations_per_workgroup;
        limits.max_compute_workgroup_size_x = alim.max_compute_workgroup_size_x;
        limits.max_compute_workgroups_per_dimension = alim.max_compute_workgroups_per_dimension;
        limits.max_uniform_buffer_binding_size = alim.max_uniform_buffer_binding_size;
        limits.max_bind_groups = alim.max_bind_groups.min(4);
        if limits.max_storage_buffers_per_shader_stage < 8 || limits.max_compute_invocations_per_workgroup < 256 {
            return Err(Error::InvalidArgument(format!("GPU adapter {} does not meet the compute limits", info.name)));
        }
        let desc = wgpu::DeviceDescriptor {
            label: Some("molecule_optimizer_rs"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&desc))
            .map_err(|e| Error::InvalidArgument(format!("cannot create GPU device on {}: {e}", info.name)))?;
        Ok(GpuContext { inner: Arc::new(Inner { device, queue, info, limits }) })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.inner.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }
    /// Adapter description (name, backend, driver).
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.inner.info
    }
    /// `"<name> (<backend>, <device type>, driver ...)"`.
    pub fn description(&self) -> String {
        let i = &self.inner.info;
        format!("{} ({:?}, {:?}, driver {} {})", i.name, i.backend, i.device_type, i.driver, i.driver_info)
    }
    /// `true` for CPU (software) adapters.
    pub fn is_software(&self) -> bool {
        is_software(&self.inner.info)
    }
    /// Device limits.
    pub fn limits(&self) -> &wgpu::Limits {
        &self.inner.limits
    }
}
