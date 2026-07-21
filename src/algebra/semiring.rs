//! Core algebraic traits for tensor operations.

use super::Scalar;

/// Base semiring trait for tensor contraction. Allows heap-allocated types.
///
/// # Semiring Laws
///
/// For a semiring (S, ⊕, ⊗, 0, 1):
/// - (S, ⊕, 0) is a commutative monoid
/// - (S, ⊗, 1) is a monoid
/// - ⊗ distributes over ⊕
/// - 0 annihilates: a ⊗ 0 = 0 ⊗ a = 0
///
/// # Examples
///
/// | Semiring | ⊕ | ⊗ | 0 | 1 |
/// |----------|---|---|---|---|
/// | Standard | + | × | 0 | 1 |
/// | MaxPlus  | max | + | -∞ | 0 |
/// | MinPlus  | min | + | +∞ | 0 |
/// | MaxMul   | max | × | 0 | 1 |
pub trait GenericSemiring: Clone + Send + Sync + 'static {
    /// Additive identity (zero element for ⊕)
    fn zero() -> Self;
    /// Multiplicative identity (one element for ⊗)
    fn one() -> Self;
    /// Addition operation (⊕)
    fn add(self, rhs: Self) -> Self;
    /// Multiplication operation (⊗)
    fn mul(self, rhs: Self) -> Self;
    /// Check if this is the zero element
    fn is_zero(&self) -> bool;
}

/// Copy semiring with scalar conversion. Enables GEMM optimization.
///
/// Extends [`GenericSemiring`] with `Copy` and scalar bridging for BLAS backends.
pub trait Semiring: GenericSemiring + Copy {
    /// The underlying scalar type
    type Scalar: Scalar;
    /// Create from scalar value
    fn from_scalar(s: Self::Scalar) -> Self;
    /// Extract scalar value
    fn to_scalar(self) -> Self::Scalar;
}

/// Extended semiring operations for automatic differentiation.
///
/// This trait adds argmax tracking needed for tropical backpropagation.
pub trait Algebra: Semiring {
    /// Index type for argmax tracking
    type Index: Copy + Clone + Send + Sync + Default + std::fmt::Debug + 'static;

    /// Addition with argmax tracking.
    ///
    /// Returns (result, winner_index) where winner_index indicates
    /// which operand "won" the addition (relevant for tropical max/min).
    fn add_with_argmax(
        self,
        self_idx: Self::Index,
        rhs: Self,
        rhs_idx: Self::Index,
    ) -> (Self, Self::Index);

    /// Backward pass for addition.
    ///
    /// Given output gradient `grad_out`, compute gradients for inputs.
    /// For standard arithmetic: both inputs get `grad_out`.
    /// For tropical: only the winner gets `grad_out`.
    fn add_backward(
        self,
        rhs: Self,
        grad_out: Self::Scalar,
        winner_idx: Option<Self::Index>,
    ) -> (Self::Scalar, Self::Scalar);

    /// Backward pass for multiplication.
    ///
    /// Given output gradient `grad_out`, compute gradients for inputs.
    /// Standard: grad_a = grad_out × b, grad_b = grad_out × a
    /// Tropical (add): grad_a = grad_out, grad_b = grad_out
    fn mul_backward(self, rhs: Self, grad_out: Self::Scalar) -> (Self::Scalar, Self::Scalar);

    /// Whether this algebra requires argmax tracking for backprop.
    fn needs_argmax() -> bool {
        false
    }

    /// Check if `self` is "better" than `other` for tropical selection.
    ///
    /// For MaxPlus: returns true if self > other
    /// For MinPlus: returns true if self < other
    /// For Standard: not meaningful (always false)
    fn is_better(&self, other: &Self) -> bool;
}
