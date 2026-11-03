//! GPU backend (wgpu + WGSL; Vulkan / Metal / DX12) for the approximate
//! nonbonded mode (feature `gpu`).
//!
//! * [`GpuContext`]: adapter / device selection ([`GpuContext::new_blocking`]
//!   prefers discrete, then integrated GPUs, then software adapters).
//! * [`WgpuNonbonded`] ([`build_gpu_nonbonded`]): [`crate::fast::NonbondedBackend`]
//!   for one system with the same model and options as
//!   [`crate::fast::approx::ApproxNonbonded`] — MMFF buffered 14-7 / UFF
//!   Lennard-Jones with potential shift or switch, Coulomb (constant or
//!   distance-dependent dielectric), reaction field, Ewald real space and
//!   smooth PME (spreading, 3D FFT, influence function and gather on the
//!   GPU), MMFF 1-4 pairs and Ewald exclusion corrections. The pair list
//!   is built on the CPU (shared with the wrapped CPU backend, rebuilt
//!   lazily) and uploaded only after a rebuild; per evaluation only the
//!   coordinates go up and energies / gradient come back.
//!   [`crate::fast::FastOptions::gpu`] selects it in [`crate::fast::FastBuilder`].
//! * [`GpuBatch`]: many independent small systems (conformers, different
//!   molecules) per dispatch.
//! * [`GpuFft3d`]: the WGSL Stockham FFT used by PME (radix 2, 3, 4, 5, 7,
//!   8, 9), standalone.
//!
//! # Precision
//!
//! All GPU arithmetic is `f32`. Coordinates are uploaded relative to the
//! origin of their cluster (differences computed in `f64` on the CPU,
//! periodic shifts folded into per-pair origin offsets), so pair distances
//! keep `f32` resolution independently of the box size; each i-cluster sums
//! its interactions with Kahan compensation; energies are reduced in `f64`
//! on the CPU (the reaction-field constant is applied there as well); PME
//! grid offsets are computed in `f64` on the CPU. Typical deviations from
//! the `f64` CPU backend: energy ~1e-6 relative, gradient ~1e-6 relative RMS.
//!
//! # Determinism
//!
//! The cluster-pair list is symmetric and every i-cluster writes only its
//! own atoms (one group of invocations per cluster, fixed reduction trees);
//! PME charge spreading uses 64-bit fixed-point integer atomics, which are
//! order independent. Results are bitwise reproducible run-to-run.

mod batch;
mod context;
mod engine;
mod fft;
mod nonbonded;
mod pme;
mod util;

pub use batch::GpuBatch;
pub use context::{GpuContext, GpuContextOptions, enumerate_adapters};
pub use fft::GpuFft3d;
pub use nonbonded::{WgpuNonbonded, build_gpu_nonbonded};
