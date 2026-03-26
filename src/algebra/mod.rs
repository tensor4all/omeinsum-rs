//! Algebraic structures for tensor operations.
//!
//! This module defines the [`Semiring`] trait and implementations for:
//! - [`Standard<T>`]: Standard arithmetic `(+, ×)` (always available)
//! - `MaxPlus<T>`: Tropical max-plus `(max, +)` (requires `tropical` feature)
//! - `MinPlus<T>`: Tropical min-plus `(min, +)` (requires `tropical` feature)
//! - `MaxMul<T>`: Tropical max-mul `(max, ×)` (requires `tropical` feature)

mod semiring;
mod standard;

#[cfg(feature = "tropical")]
mod tropical;

pub use semiring::{Algebra, CloneSemiring, Semiring};
pub use standard::Standard;

#[cfg(feature = "tropical")]
pub use tropical::{MaxMul, MaxPlus, MinPlus};

// Re-export complex types for convenience
pub use num_complex::{Complex32, Complex64};

/// Marker trait for scalar types that can be used in tensors.
pub trait Scalar:
    Copy
    + Clone
    + Send
    + Sync
    + Default
    + std::fmt::Debug
    + 'static
    + bytemuck::Pod
    + std::ops::AddAssign
{
}

impl Scalar for f32 {}
impl Scalar for f64 {}
impl Scalar for i32 {}
impl Scalar for i64 {}
impl Scalar for u32 {}
impl Scalar for u64 {}
impl Scalar for Complex32 {}
impl Scalar for Complex64 {}
