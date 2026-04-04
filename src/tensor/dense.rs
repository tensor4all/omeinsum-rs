//! Simple owned tensor for CloneSemiring types.
//!
//! [`DenseTensor`] stores elements in a flat `Vec<S>` with column-major layout.
//! Designed for generic loop contraction — no strides, no Arc, no backend.

use crate::algebra::CloneSemiring;

/// A dense tensor for generic (non-GEMM) contraction.
///
/// Elements are stored in column-major order. Element at multi-index
/// `[i0, i1, ..., in]` is at position `i0 + i1*shape[0] + i2*shape[0]*shape[1] + ...`.
#[derive(Clone, Debug)]
pub struct DenseTensor<S> {
    data: Vec<S>,
    shape: Vec<usize>,
}

impl<S: CloneSemiring> DenseTensor<S> {
    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product::<usize>().max(1);
        let data = (0..len).map(|_| S::zero()).collect();
        Self { data, shape }
    }

    /// Create a tensor from data and shape.
    ///
    /// Data must be in column-major order. Panics if `data.len() != shape.iter().product()`.
    pub fn from_data(data: Vec<S>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product::<usize>().max(1);
        assert_eq!(
            data.len(),
            expected,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected,
        );
        Self { data, shape }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Access element by flat (column-major) index.
    pub fn get(&self, index: usize) -> &S {
        &self.data[index]
    }

    /// Mutable access by flat (column-major) index.
    pub fn get_mut(&mut self, index: usize) -> &mut S {
        &mut self.data[index]
    }

    /// Consume into raw data and shape.
    pub fn into_data(self) -> (Vec<S>, Vec<usize>) {
        (self.data, self.shape)
    }

    /// Borrow data as slice.
    pub fn data(&self) -> &[S] {
        &self.data
    }

    /// Mutable borrow of data.
    pub fn data_mut(&mut self) -> &mut [S] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct TestSemiring(f64);

    impl CloneSemiring for TestSemiring {
        fn zero() -> Self { TestSemiring(0.0) }
        fn one() -> Self { TestSemiring(1.0) }
        fn add(self, rhs: Self) -> Self { TestSemiring(self.0 + rhs.0) }
        fn mul(self, rhs: Self) -> Self { TestSemiring(self.0 * rhs.0) }
        fn is_zero(&self) -> bool { self.0 == 0.0 }
    }

    #[test]
    fn test_zeros() {
        let t: DenseTensor<TestSemiring> = DenseTensor::zeros(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.len(), 6);
        for i in 0..6 {
            assert!(t.get(i).is_zero());
        }
    }

    #[test]
    fn test_from_data() {
        let data = vec![TestSemiring(1.0), TestSemiring(2.0), TestSemiring(3.0)];
        let t = DenseTensor::from_data(data, vec![3]);
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.get(0), &TestSemiring(1.0));
        assert_eq!(t.get(2), &TestSemiring(3.0));
    }

    #[test]
    #[should_panic]
    fn test_from_data_shape_mismatch() {
        let data = vec![TestSemiring(1.0), TestSemiring(2.0)];
        DenseTensor::from_data(data, vec![3]);
    }

    #[test]
    fn test_into_data_roundtrip() {
        let data = vec![TestSemiring(1.0), TestSemiring(2.0)];
        let t = DenseTensor::from_data(data.clone(), vec![2]);
        let (recovered_data, recovered_shape) = t.into_data();
        assert_eq!(recovered_data, data);
        assert_eq!(recovered_shape, vec![2]);
    }

    #[test]
    fn test_get_mut() {
        let mut t = DenseTensor::zeros(vec![2, 2]);
        *t.get_mut(1) = TestSemiring(42.0);
        assert_eq!(t.get(1), &TestSemiring(42.0));
    }
}
