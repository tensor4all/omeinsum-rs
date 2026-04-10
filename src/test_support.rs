#![cfg(test)]

use crate::algebra::{Algebra, Scalar};
use crate::backend::{Backend, BackendScalar};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TestBackend {
    pub(crate) id: usize,
}

impl TestBackend {
    pub(crate) fn new(id: usize) -> Self {
        Self { id }
    }
}

impl Default for TestBackend {
    fn default() -> Self {
        Self { id: 0 }
    }
}

impl Backend for TestBackend {
    type Storage<T: Scalar> = Vec<T>;

    fn name() -> &'static str {
        "test"
    }

    fn synchronize(&self) {}

    fn alloc<T: Scalar>(&self, len: usize) -> Self::Storage<T> {
        vec![T::default(); len]
    }

    fn from_slice<T: Scalar>(&self, data: &[T]) -> Self::Storage<T> {
        data.to_vec()
    }

    fn copy_strided<T: Scalar>(
        &self,
        src: &Self::Storage<T>,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Self::Storage<T> {
        let numel: usize = shape.iter().product();
        let mut dst = Vec::with_capacity(numel);

        for linear_idx in 0..numel {
            let mut remaining = linear_idx;
            let mut src_offset = offset;
            for dim in 0..shape.len() {
                let coord = remaining % shape[dim];
                remaining /= shape[dim];
                src_offset += coord * strides[dim];
            }
            dst.push(src[src_offset]);
        }

        dst
    }

    fn contract<A: Algebra>(
        &self,
        _a: &Self::Storage<A::Scalar>,
        _shape_a: &[usize],
        _strides_a: &[usize],
        _modes_a: &[i32],
        _b: &Self::Storage<A::Scalar>,
        _shape_b: &[usize],
        _strides_b: &[usize],
        _modes_b: &[i32],
        _shape_c: &[usize],
        _modes_c: &[i32],
    ) -> Self::Storage<A::Scalar>
    where
        A::Scalar: BackendScalar<Self>,
    {
        unimplemented!("TestBackend only supports unary code paths in tests")
    }

    fn contract_with_argmax<A: Algebra<Index = u32>>(
        &self,
        _a: &Self::Storage<A::Scalar>,
        _shape_a: &[usize],
        _strides_a: &[usize],
        _modes_a: &[i32],
        _b: &Self::Storage<A::Scalar>,
        _shape_b: &[usize],
        _strides_b: &[usize],
        _modes_b: &[i32],
        _shape_c: &[usize],
        _modes_c: &[i32],
    ) -> (Self::Storage<A::Scalar>, Self::Storage<u32>)
    where
        A::Scalar: BackendScalar<Self>,
    {
        unimplemented!("TestBackend only supports unary code paths in tests")
    }
}

impl<T: Scalar> BackendScalar<TestBackend> for T {}
