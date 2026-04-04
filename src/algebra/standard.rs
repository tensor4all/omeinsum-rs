//! Standard arithmetic semiring `(+, ×)`.

use super::semiring::{Algebra, GenericSemiring, Semiring};
use super::Scalar;
use num_traits::{One, Zero};

/// Standard arithmetic semiring with addition and multiplication.
///
/// This represents the usual `(+, ×)` operations used in linear algebra.
///
/// # Example
///
/// ```rust
/// use omeinsum::algebra::{Standard, GenericSemiring, Semiring};
///
/// let a = Standard(2.0f32);
/// let b = Standard(3.0f32);
///
/// assert_eq!(a.add(b).to_scalar(), 5.0);  // 2 + 3 = 5
/// assert_eq!(a.mul(b).to_scalar(), 6.0);  // 2 × 3 = 6
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Standard<T: Scalar>(pub T);

impl<
        T: Scalar + Zero + One + PartialEq + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    > GenericSemiring for Standard<T>
{
    #[inline]
    fn zero() -> Self {
        Standard(T::zero())
    }

    #[inline]
    fn one() -> Self {
        Standard(T::one())
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Standard(self.0 + rhs.0)
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Standard(self.0 * rhs.0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == T::zero()
    }
}

impl<
        T: Scalar + Zero + One + PartialEq + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    > Semiring for Standard<T>
{
    type Scalar = T;

    #[inline]
    fn from_scalar(s: T) -> Self {
        Standard(s)
    }

    #[inline]
    fn to_scalar(self) -> T {
        self.0
    }
}

impl<
        T: Scalar + Zero + One + PartialEq + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    > Algebra for Standard<T>
{
    type Index = u32;

    #[inline]
    fn add_with_argmax(
        self,
        _self_idx: Self::Index,
        rhs: Self,
        _rhs_idx: Self::Index,
    ) -> (Self, Self::Index) {
        // Standard addition doesn't track argmax
        (self.add(rhs), 0)
    }

    #[inline]
    fn add_backward(
        self,
        _rhs: Self,
        grad_out: Self::Scalar,
        _winner_idx: Option<Self::Index>,
    ) -> (Self::Scalar, Self::Scalar) {
        // Standard addition: both inputs get the full gradient
        (grad_out, grad_out)
    }

    #[inline]
    fn mul_backward(self, rhs: Self, grad_out: Self::Scalar) -> (Self::Scalar, Self::Scalar) {
        // Standard multiplication: chain rule
        // d/da (a × b) = b, d/db (a × b) = a
        (
            Standard(grad_out).mul(rhs).to_scalar(),
            Standard(grad_out).mul(self).to_scalar(),
        )
    }

    #[inline]
    fn needs_argmax() -> bool {
        false
    }

    #[inline]
    fn is_better(&self, _other: &Self) -> bool {
        // Standard algebra accumulates all values, no "better" comparison
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_f32() {
        let a = Standard(2.0f32);
        let b = Standard(3.0f32);

        assert_eq!(a.add(b).to_scalar(), 5.0);
        assert_eq!(a.mul(b).to_scalar(), 6.0);
        assert_eq!(Standard::<f32>::zero().to_scalar(), 0.0);
        assert_eq!(Standard::<f32>::one().to_scalar(), 1.0);
    }

    #[test]
    fn test_standard_backward() {
        let a = Standard(2.0f32);
        let b = Standard(3.0f32);

        // Add backward
        let (ga, gb) = a.add_backward(b, 1.0, None);
        assert_eq!(ga, 1.0);
        assert_eq!(gb, 1.0);

        // Mul backward: d/da(a*b) = b, d/db(a*b) = a
        let (ga, gb) = a.mul_backward(b, 1.0);
        assert_eq!(ga, 3.0); // grad_out * b
        assert_eq!(gb, 2.0); // grad_out * a
    }
}
