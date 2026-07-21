//! Backend abstractions for CPU and GPU execution.
//!
//! This module defines the [`Backend`] trait and implementations:
//! - [`Cpu`]: CPU backend with SIMD acceleration
//! - `Cuda`: CUDA backend (optional, requires `cuda` feature)

mod contract_plan;
mod cpu;
mod traits;

pub use cpu::Cpu;
pub use traits::{Backend, BackendScalar, Storage};

#[cfg(any(feature = "cuda", feature = "cuda-tropical"))]
mod cuda;

#[cfg(any(feature = "cuda", feature = "cuda-tropical"))]
pub use cuda::{Cuda, CudaComplex, CudaError, CudaStorage};
