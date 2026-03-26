//! Tropical semiring implementations.
//!
//! - [`MaxPlus<T>`]: `(max, +)` semiring for longest path, Viterbi
//! - [`MinPlus<T>`]: `(min, +)` semiring for shortest path
//! - [`MaxMul<T>`]: `(max, ×)` semiring for max probability

use super::semiring::{Algebra, CloneSemiring, Semiring};
use super::Scalar;
use num_traits::{Bounded, One, Zero};

// ============================================================================
// MaxPlus: (max, +) semiring
// ============================================================================

/// Tropical max-plus semiring `(max, +)`.
///
/// Operations:
/// - Addition (⊕): `max(a, b)`
/// - Multiplication (⊗): `a + b`
/// - Zero: `-∞`
/// - One: `0`
///
/// Used for: longest path, Viterbi algorithm, max-probability (log space)
///
/// # Example
///
/// ```rust
/// use omeinsum::algebra::{MaxPlus, CloneSemiring, Semiring};
///
/// let a = MaxPlus(2.0f32);
/// let b = MaxPlus(3.0f32);
///
/// assert_eq!(a.add(b).to_scalar(), 3.0);  // max(2, 3) = 3
/// assert_eq!(a.mul(b).to_scalar(), 5.0);  // 2 + 3 = 5
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct MaxPlus<T: Scalar>(pub T);

impl<T> CloneSemiring for MaxPlus<T>
where
    T: Scalar + Bounded + Zero + PartialOrd + std::ops::Add<Output = T>,
{
    #[inline]
    fn zero() -> Self {
        MaxPlus(T::min_value()) // -∞
    }

    #[inline]
    fn one() -> Self {
        MaxPlus(T::zero()) // 0
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        if self.0 >= rhs.0 {
            self
        } else {
            rhs
        }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        MaxPlus(self.0 + rhs.0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == T::min_value()
    }
}

impl<T> Semiring for MaxPlus<T>
where
    T: Scalar + Bounded + Zero + PartialOrd + std::ops::Add<Output = T>,
{
    type Scalar = T;

    #[inline]
    fn from_scalar(s: T) -> Self {
        MaxPlus(s)
    }

    #[inline]
    fn to_scalar(self) -> T {
        self.0
    }
}

impl<T> Algebra for MaxPlus<T>
where
    T: Scalar + Bounded + Zero + PartialOrd + std::ops::Add<Output = T>,
{
    type Index = u32;

    #[inline]
    fn add_with_argmax(
        self,
        self_idx: Self::Index,
        rhs: Self,
        rhs_idx: Self::Index,
    ) -> (Self, Self::Index) {
        if self.0 >= rhs.0 {
            (self, self_idx)
        } else {
            (rhs, rhs_idx)
        }
    }

    #[inline]
    fn add_backward(
        self,
        rhs: Self,
        grad_out: Self::Scalar,
        winner_idx: Option<Self::Index>,
    ) -> (Self::Scalar, Self::Scalar) {
        // Tropical max: only the winner gets the gradient
        match winner_idx {
            Some(0) => (grad_out, T::zero()),
            Some(_) => (T::zero(), grad_out),
            None => {
                // Recompute winner
                if self.0 >= rhs.0 {
                    (grad_out, T::zero())
                } else {
                    (T::zero(), grad_out)
                }
            }
        }
    }

    #[inline]
    fn mul_backward(self, _rhs: Self, grad_out: Self::Scalar) -> (Self::Scalar, Self::Scalar) {
        // Tropical multiplication is addition: both get gradient
        (grad_out, grad_out)
    }

    #[inline]
    fn needs_argmax() -> bool {
        true
    }

    #[inline]
    fn is_better(&self, other: &Self) -> bool {
        // MaxPlus: larger is better
        self.0 > other.0
    }
}

// ============================================================================
// MinPlus: (min, +) semiring
// ============================================================================

/// Tropical min-plus semiring `(min, +)`.
///
/// Operations:
/// - Addition (⊕): `min(a, b)`
/// - Multiplication (⊗): `a + b`
/// - Zero: `+∞`
/// - One: `0`
///
/// Used for: shortest path (Dijkstra), min-cost problems
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct MinPlus<T: Scalar>(pub T);

impl<T> CloneSemiring for MinPlus<T>
where
    T: Scalar + Bounded + Zero + PartialOrd + std::ops::Add<Output = T>,
{
    #[inline]
    fn zero() -> Self {
        MinPlus(T::max_value()) // +∞
    }

    #[inline]
    fn one() -> Self {
        MinPlus(T::zero()) // 0
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        if self.0 <= rhs.0 {
            self
        } else {
            rhs
        }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        MinPlus(self.0 + rhs.0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == T::max_value()
    }
}

impl<T> Semiring for MinPlus<T>
where
    T: Scalar + Bounded + Zero + PartialOrd + std::ops::Add<Output = T>,
{
    type Scalar = T;

    #[inline]
    fn from_scalar(s: T) -> Self {
        MinPlus(s)
    }

    #[inline]
    fn to_scalar(self) -> T {
        self.0
    }
}

impl<T> Algebra for MinPlus<T>
where
    T: Scalar + Bounded + Zero + PartialOrd + std::ops::Add<Output = T>,
{
    type Index = u32;

    #[inline]
    fn add_with_argmax(
        self,
        self_idx: Self::Index,
        rhs: Self,
        rhs_idx: Self::Index,
    ) -> (Self, Self::Index) {
        if self.0 <= rhs.0 {
            (self, self_idx)
        } else {
            (rhs, rhs_idx)
        }
    }

    #[inline]
    fn add_backward(
        self,
        rhs: Self,
        grad_out: Self::Scalar,
        winner_idx: Option<Self::Index>,
    ) -> (Self::Scalar, Self::Scalar) {
        match winner_idx {
            Some(0) => (grad_out, T::zero()),
            Some(_) => (T::zero(), grad_out),
            None => {
                if self.0 <= rhs.0 {
                    (grad_out, T::zero())
                } else {
                    (T::zero(), grad_out)
                }
            }
        }
    }

    #[inline]
    fn mul_backward(self, _rhs: Self, grad_out: Self::Scalar) -> (Self::Scalar, Self::Scalar) {
        (grad_out, grad_out)
    }

    #[inline]
    fn needs_argmax() -> bool {
        true
    }

    #[inline]
    fn is_better(&self, other: &Self) -> bool {
        // MinPlus: smaller is better
        self.0 < other.0
    }
}

// ============================================================================
// MaxMul: (max, ×) semiring
// ============================================================================

/// Tropical max-mul semiring `(max, ×)`.
///
/// Operations:
/// - Addition (⊕): `max(a, b)`
/// - Multiplication (⊗): `a × b`
/// - Zero: `0`
/// - One: `1`
///
/// Used for: max probability (non-log space), fuzzy logic
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct MaxMul<T: Scalar>(pub T);

impl<T> CloneSemiring for MaxMul<T>
where
    T: Scalar + Zero + One + PartialOrd + std::ops::Mul<Output = T>,
{
    #[inline]
    fn zero() -> Self {
        MaxMul(T::zero())
    }

    #[inline]
    fn one() -> Self {
        MaxMul(T::one())
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        if self.0 >= rhs.0 {
            self
        } else {
            rhs
        }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        MaxMul(self.0 * rhs.0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == T::zero()
    }
}

impl<T> Semiring for MaxMul<T>
where
    T: Scalar + Zero + One + PartialOrd + std::ops::Mul<Output = T>,
{
    type Scalar = T;

    #[inline]
    fn from_scalar(s: T) -> Self {
        MaxMul(s)
    }

    #[inline]
    fn to_scalar(self) -> T {
        self.0
    }
}

impl<T> Algebra for MaxMul<T>
where
    T: Scalar + Zero + One + PartialOrd + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
{
    type Index = u32;

    #[inline]
    fn add_with_argmax(
        self,
        self_idx: Self::Index,
        rhs: Self,
        rhs_idx: Self::Index,
    ) -> (Self, Self::Index) {
        if self.0 >= rhs.0 {
            (self, self_idx)
        } else {
            (rhs, rhs_idx)
        }
    }

    #[inline]
    fn add_backward(
        self,
        rhs: Self,
        grad_out: Self::Scalar,
        winner_idx: Option<Self::Index>,
    ) -> (Self::Scalar, Self::Scalar) {
        match winner_idx {
            Some(0) => (grad_out, T::zero()),
            Some(_) => (T::zero(), grad_out),
            None => {
                if self.0 >= rhs.0 {
                    (grad_out, T::zero())
                } else {
                    (T::zero(), grad_out)
                }
            }
        }
    }

    #[inline]
    fn mul_backward(self, rhs: Self, grad_out: Self::Scalar) -> (Self::Scalar, Self::Scalar) {
        // Standard chain rule for multiplication
        (grad_out * rhs.0, grad_out * self.0)
    }

    #[inline]
    fn needs_argmax() -> bool {
        true
    }

    #[inline]
    fn is_better(&self, other: &Self) -> bool {
        // MaxMul: larger is better (same as MaxPlus)
        self.0 > other.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxplus_f32() {
        let a = MaxPlus(2.0f32);
        let b = MaxPlus(3.0f32);

        assert_eq!(a.add(b).to_scalar(), 3.0); // max(2, 3) = 3
        assert_eq!(a.mul(b).to_scalar(), 5.0); // 2 + 3 = 5
        assert_eq!(MaxPlus::<f32>::zero().to_scalar(), f32::MIN);
        assert_eq!(MaxPlus::<f32>::one().to_scalar(), 0.0);
    }

    #[test]
    fn test_maxplus_argmax() {
        let a = MaxPlus(2.0f32);
        let b = MaxPlus(3.0f32);

        let (result, idx) = a.add_with_argmax(0, b, 1);
        assert_eq!(result.to_scalar(), 3.0);
        assert_eq!(idx, 1); // b won

        let (result, idx) = b.add_with_argmax(0, a, 1);
        assert_eq!(result.to_scalar(), 3.0);
        assert_eq!(idx, 0); // b (now first) won
    }

    #[test]
    fn test_minplus_f32() {
        let a = MinPlus(2.0f32);
        let b = MinPlus(3.0f32);

        assert_eq!(a.add(b).to_scalar(), 2.0); // min(2, 3) = 2
        assert_eq!(a.mul(b).to_scalar(), 5.0); // 2 + 3 = 5
        assert_eq!(MinPlus::<f32>::zero().to_scalar(), f32::MAX);
        assert_eq!(MinPlus::<f32>::one().to_scalar(), 0.0);
    }

    #[test]
    fn test_maxmul_f32() {
        let a = MaxMul(2.0f32);
        let b = MaxMul(3.0f32);

        assert_eq!(a.add(b).to_scalar(), 3.0); // max(2, 3) = 3
        assert_eq!(a.mul(b).to_scalar(), 6.0); // 2 × 3 = 6
        assert_eq!(MaxMul::<f32>::zero().to_scalar(), 0.0);
        assert_eq!(MaxMul::<f32>::one().to_scalar(), 1.0);
    }

    #[test]
    fn test_tropical_backward() {
        let a = MaxPlus(2.0f32);
        let b = MaxPlus(3.0f32);

        // b wins, so only b gets gradient
        let (ga, gb) = a.add_backward(b, 1.0, Some(1));
        assert_eq!(ga, 0.0);
        assert_eq!(gb, 1.0);

        // Tropical mul backward: both get gradient
        let (ga, gb) = a.mul_backward(b, 1.0);
        assert_eq!(ga, 1.0);
        assert_eq!(gb, 1.0);
    }
}
