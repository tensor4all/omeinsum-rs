//! # OMEinsum-rs
//!
//! High-performance Einstein summation with support for both tropical and standard algebras.
//!
//! ## Features
//!
//! - **Algebra-agnostic**: Works with standard arithmetic `(+, ×)` and tropical semirings `(max, +)`, `(min, +)`
//! - **Optimized contraction**: Integration with [omeco](https://github.com/GiggleLiu/omeco) for contraction order optimization
//! - **Backpropagation**: Gradient computation for both tropical and standard operations
//! - **Zero-copy views**: Stride-based tensor with efficient permute/reshape
//! - **CPU + CUDA**: Support for both backends (CUDA optional)
//!
//! ## Quick Start
//!
//! ```rust
//! use omeinsum::{Tensor, Einsum, einsum, Cpu};
//! use omeinsum::algebra::Standard;
//!
//! // Standard matrix multiplication
//! let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
//! let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
//!
//! // C[i,k] = Σ_j A[i,j] × B[j,k]
//! let c = einsum::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
//! ```
//!
//! With the `tropical` feature enabled:
//!
//! ```rust,ignore
//! use omeinsum::{einsum, Tensor, Cpu, MaxPlus};
//!
//! // Tropical (max-plus) matrix multiplication
//! // C[i,k] = max_j (A[i,j] + B[j,k])
//! let c_tropical = einsum::<MaxPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                         User API                            │
//! │   einsum(tensors, ixs, iy) → Tensor                        │
//! │   Einsum::new(ixs, iy).optimize().execute(tensors)         │
//! └─────────────────────────────────────────────────────────────┘
//!                               │
//!                               ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      Einsum Engine                          │
//! │   omeco::optimize_code() → NestedEinsum (contraction tree) │
//! │   Execute tree via binary contractions                      │
//! └─────────────────────────────────────────────────────────────┘
//!                               │
//!                               ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                 Algebra<T> dispatch                         │
//! │   Standard<T>: (+, ×) → BLAS/loops                         │
//! │   MaxPlus<T>:  (max, +) → tropical-gemm                    │
//! │   MinPlus<T>:  (min, +) → tropical-gemm                    │
//! └─────────────────────────────────────────────────────────────┘
//! ```

pub mod algebra;
pub mod backend;
pub mod einsum;
pub mod tensor;

// Re-exports
pub use algebra::{Algebra, CloneSemiring, Complex32, Complex64, Semiring, Standard};
pub use backend::{Backend, BackendScalar, Cpu, Storage};
pub use einsum::{cost_and_gradient, einsum, einsum_with_grad, EinBuilder, Einsum};
pub use tensor::{DenseTensor, Tensor, TensorView};

#[cfg(feature = "tropical")]
pub use algebra::{MaxMul, MaxPlus, MinPlus};

#[cfg(feature = "cuda")]
pub use backend::Cuda;
