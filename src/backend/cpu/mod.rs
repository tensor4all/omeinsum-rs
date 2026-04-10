//! CPU backend implementation.

mod contract;

use super::traits::{Backend, BackendScalar, Storage};
use crate::algebra::{Algebra, Scalar, Standard};
use std::any::TypeId;

/// CPU backend using Vec storage.
#[derive(Clone, Debug, Default)]
pub struct Cpu;

impl Cpu {
    /// General matrix multiplication (internal implementation).
    ///
    /// Computes C = A ⊗ B where ⊗ is the semiring multiplication
    /// and the reduction uses semiring addition.
    ///
    /// This is an internal implementation detail used by the contract method.
    /// Users should use `einsum()` or `contract_binary()` instead.
    pub(crate) fn gemm_internal<A: Algebra>(
        &self,
        a: &[A::Scalar],
        m: usize,
        k: usize,
        b: &[A::Scalar],
        n: usize,
    ) -> Vec<A::Scalar> {
        // Fast path: faer for Standard f32/f64
        if TypeId::of::<A>() == TypeId::of::<Standard<f32>>() {
            // SAFETY: A::Scalar is f32 when A is Standard<f32>
            let a_f32: &[f32] = unsafe { std::mem::transmute(a) };
            let b_f32: &[f32] = unsafe { std::mem::transmute(b) };
            let result = faer_gemm_f32(a_f32, m, k, b_f32, n);
            return unsafe { std::mem::transmute::<Vec<f32>, Vec<A::Scalar>>(result) };
        }
        if TypeId::of::<A>() == TypeId::of::<Standard<f64>>() {
            let a_f64: &[f64] = unsafe { std::mem::transmute(a) };
            let b_f64: &[f64] = unsafe { std::mem::transmute(b) };
            let result = faer_gemm_f64(a_f64, m, k, b_f64, n);
            return unsafe { std::mem::transmute::<Vec<f64>, Vec<A::Scalar>>(result) };
        }

        // Try to use optimized tropical-gemm if available
        #[cfg(feature = "tropical-kernels")]
        {
            if let Some(result) = try_tropical_gemm::<A>(a, m, k, b, n) {
                return result;
            }
        }

        // Fallback to generic loop implementation
        generic_gemm::<A>(a, m, k, b, n)
    }

    /// GEMM with argmax tracking (internal implementation).
    ///
    /// Returns (result, argmax) where argmax[i, j] is the k index
    /// that "won" the reduction for element [i, j].
    pub(crate) fn gemm_with_argmax_internal<A: Algebra<Index = u32>>(
        &self,
        a: &[A::Scalar],
        m: usize,
        k: usize,
        b: &[A::Scalar],
        n: usize,
    ) -> (Vec<A::Scalar>, Vec<u32>) {
        // Try to use optimized tropical-gemm if available
        #[cfg(feature = "tropical-kernels")]
        {
            if let Some(result) = try_tropical_gemm_with_argmax::<A>(a, m, k, b, n) {
                return result;
            }
        }

        // Fallback to generic loop implementation
        generic_gemm_with_argmax::<A>(a, m, k, b, n)
    }

    /// Backward pass for GEMM w.r.t. A (internal implementation).
    /// Used primarily for testing CPU-specific backward implementations.
    #[allow(dead_code)]
    pub(crate) fn gemm_backward_a_internal<A: Algebra>(
        &self,
        grad_c: &[A::Scalar],
        argmax: &[u32],
        _b: &[A::Scalar],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<A::Scalar> {
        let mut grad_a = vec![A::Scalar::default(); m * k];

        // For tropical: grad_a[i, argmax[i,j]] += grad_c[i,j]
        // For standard: grad_a = grad_c @ b.T
        // Column-major: element (i, j) is at index j * nrows + i
        if A::needs_argmax() {
            for j in 0..n {
                for i in 0..m {
                    let idx = argmax[j * m + i] as usize; // argmax[i, j] in column-major
                                                          // grad_a[i, idx] += grad_c[i, j]
                    grad_a[idx * m + i] += grad_c[j * m + i];
                }
            }
        }

        grad_a
    }

    /// Backward pass for GEMM w.r.t. B (internal implementation).
    /// Used primarily for testing CPU-specific backward implementations.
    #[allow(dead_code)]
    pub(crate) fn gemm_backward_b_internal<A: Algebra>(
        &self,
        grad_c: &[A::Scalar],
        argmax: &[u32],
        _a: &[A::Scalar],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<A::Scalar> {
        let mut grad_b = vec![A::Scalar::default(); k * n];

        // Column-major: element (i, j) is at index j * nrows + i
        if A::needs_argmax() {
            for j in 0..n {
                for i in 0..m {
                    let idx = argmax[j * m + i] as usize; // argmax[i, j] in column-major
                                                          // grad_b[idx, j] += grad_c[i, j]
                    grad_b[j * k + idx] += grad_c[j * m + i];
                }
            }
        }

        grad_b
    }

    /// Batched GEMM (internal implementation).
    pub(crate) fn gemm_batched_internal<A: Algebra>(
        &self,
        a: &[A::Scalar],
        batch_size: usize,
        m: usize,
        k: usize,
        b: &[A::Scalar],
        n: usize,
    ) -> Vec<A::Scalar> {
        let a_batch_stride = m * k;
        let b_batch_stride = k * n;
        let c_batch_stride = m * n;

        let mut c = vec![A::zero().to_scalar(); batch_size * m * n];

        for batch in 0..batch_size {
            let a_offset = batch * a_batch_stride;
            let b_offset = batch * b_batch_stride;
            let c_offset = batch * c_batch_stride;

            let a_slice = &a[a_offset..a_offset + a_batch_stride];
            let b_slice = &b[b_offset..b_offset + b_batch_stride];

            let c_batch = generic_gemm::<A>(a_slice, m, k, b_slice, n);
            c[c_offset..c_offset + c_batch_stride].copy_from_slice(&c_batch);
        }

        c
    }

    /// Batched GEMM with argmax tracking (internal implementation).
    pub(crate) fn gemm_batched_with_argmax_internal<A: Algebra<Index = u32>>(
        &self,
        a: &[A::Scalar],
        batch_size: usize,
        m: usize,
        k: usize,
        b: &[A::Scalar],
        n: usize,
    ) -> (Vec<A::Scalar>, Vec<u32>) {
        let a_batch_stride = m * k;
        let b_batch_stride = k * n;
        let c_batch_stride = m * n;

        let mut c = vec![A::zero().to_scalar(); batch_size * m * n];
        let mut argmax = vec![0u32; batch_size * m * n];

        for batch in 0..batch_size {
            let a_offset = batch * a_batch_stride;
            let b_offset = batch * b_batch_stride;
            let c_offset = batch * c_batch_stride;

            let a_slice = &a[a_offset..a_offset + a_batch_stride];
            let b_slice = &b[b_offset..b_offset + b_batch_stride];

            let (c_batch, argmax_batch) = generic_gemm_with_argmax::<A>(a_slice, m, k, b_slice, n);
            c[c_offset..c_offset + c_batch_stride].copy_from_slice(&c_batch);
            argmax[c_offset..c_offset + c_batch_stride].copy_from_slice(&argmax_batch);
        }

        (c, argmax)
    }
}

impl<T: Scalar> Storage<T> for Vec<T> {
    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn get(&self, index: usize) -> T {
        self[index]
    }

    #[inline]
    fn set(&mut self, index: usize, value: T) {
        self[index] = value;
    }

    #[inline]
    fn to_vec(&self) -> Vec<T> {
        self.clone()
    }

    #[inline]
    fn from_slice(data: &[T]) -> Self {
        data.to_vec()
    }

    #[inline]
    fn zeros(len: usize) -> Self {
        vec![T::default(); len]
    }
}

impl Backend for Cpu {
    type Storage<T: Scalar> = Vec<T>;

    fn name() -> &'static str {
        "cpu"
    }

    fn synchronize(&self) {
        // No-op for CPU
    }

    fn alloc<T: Scalar>(&self, len: usize) -> Vec<T> {
        vec![T::default(); len]
    }

    fn from_slice<T: Scalar>(&self, data: &[T]) -> Vec<T> {
        data.to_vec()
    }

    fn contract<A: Algebra>(
        &self,
        a: &Self::Storage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &Self::Storage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> Self::Storage<A::Scalar>
    where
        A::Scalar: BackendScalar<Self>,
    {
        contract::contract::<A>(
            self, a, shape_a, strides_a, modes_a, b, shape_b, strides_b, modes_b, shape_c, modes_c,
        )
    }

    fn contract_with_argmax<A: Algebra<Index = u32>>(
        &self,
        a: &Self::Storage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &Self::Storage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> (Self::Storage<A::Scalar>, Self::Storage<u32>)
    where
        A::Scalar: BackendScalar<Self>,
    {
        contract::contract_with_argmax::<A>(
            self, a, shape_a, strides_a, modes_a, b, shape_b, strides_b, modes_b, shape_c, modes_c,
        )
    }

    fn copy_strided<T: Scalar>(
        &self,
        src: &Vec<T>,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Vec<T> {
        let numel: usize = shape.iter().product();
        let mut dst = vec![T::default(); numel];

        // Iterate over all indices and copy
        let mut indices = vec![0usize; shape.len()];
        for dst_elem in dst.iter_mut() {
            // Compute source offset using strides
            let src_offset: usize = offset
                + indices
                    .iter()
                    .zip(strides.iter())
                    .map(|(i, s)| i * s)
                    .sum::<usize>();

            *dst_elem = src[src_offset];

            // Increment indices (column-major order: first dimension first)
            for dim in 0..shape.len() {
                indices[dim] += 1;
                if indices[dim] < shape[dim] {
                    break;
                }
                indices[dim] = 0;
            }
        }

        dst
    }
}

/// GEMM using faer for f32 (column-major layout).
///
/// Computes C = A @ B where A is m×k, B is k×n, C is m×n.
fn faer_gemm_f32(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    use faer::Mat;

    // Create matrices from column-major data
    // Column-major: element (i, j) is at index j * nrows + i
    let a_mat = Mat::from_fn(m, k, |i, j| a[j * m + i]);
    let b_mat = Mat::from_fn(k, n, |i, j| b[j * k + i]);

    // Multiply
    let c_mat = &a_mat * &b_mat;

    // Convert back to column-major Vec
    let mut c = vec![0.0f32; m * n];
    for j in 0..n {
        for i in 0..m {
            c[j * m + i] = c_mat[(i, j)];
        }
    }
    c
}

/// GEMM using faer for f64 (column-major layout).
fn faer_gemm_f64(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    use faer::Mat;

    let a_mat = Mat::from_fn(m, k, |i, j| a[j * m + i]);
    let b_mat = Mat::from_fn(k, n, |i, j| b[j * k + i]);

    let c_mat = &a_mat * &b_mat;

    let mut c = vec![0.0f64; m * n];
    for j in 0..n {
        for i in 0..m {
            c[j * m + i] = c_mat[(i, j)];
        }
    }
    c
}

/// Generic GEMM using semiring operations (column-major layout).
fn generic_gemm<A: Algebra>(
    a: &[A::Scalar],
    m: usize,
    k: usize,
    b: &[A::Scalar],
    n: usize,
) -> Vec<A::Scalar> {
    let mut c = vec![A::zero().to_scalar(); m * n];

    // Column-major: element (i, j) is at index j * nrows + i
    for j in 0..n {
        for i in 0..m {
            let mut acc = A::zero();
            for kk in 0..k {
                let a_val = A::from_scalar(a[kk * m + i]); // A[i, kk] in column-major
                let b_val = A::from_scalar(b[j * k + kk]); // B[kk, j] in column-major
                let prod = a_val.mul(b_val);
                acc = acc.add(prod);
            }
            c[j * m + i] = acc.to_scalar();
        }
    }

    c
}

/// Generic GEMM with argmax tracking (column-major layout).
fn generic_gemm_with_argmax<A: Algebra<Index = u32>>(
    a: &[A::Scalar],
    m: usize,
    k: usize,
    b: &[A::Scalar],
    n: usize,
) -> (Vec<A::Scalar>, Vec<u32>) {
    let mut c = vec![A::zero().to_scalar(); m * n];
    let mut argmax = vec![0u32; m * n];

    // Column-major: element (i, j) is at index j * nrows + i
    for j in 0..n {
        for i in 0..m {
            let mut acc = A::zero();
            let mut best_k = 0u32;

            for kk in 0..k {
                let a_val = A::from_scalar(a[kk * m + i]); // A[i, kk] in column-major
                let b_val = A::from_scalar(b[j * k + kk]); // B[kk, j] in column-major
                let prod = a_val.mul(b_val);
                let (new_acc, winner) = acc.add_with_argmax(best_k, prod, kk as u32);
                acc = new_acc;
                best_k = winner;
            }

            c[j * m + i] = acc.to_scalar();
            argmax[j * m + i] = best_k;
        }
    }

    (c, argmax)
}

// Optional: Use tropical-gemm for optimized kernels
#[cfg(feature = "tropical-kernels")]
fn try_tropical_gemm<A: Algebra>(
    a: &[A::Scalar],
    m: usize,
    k: usize,
    b: &[A::Scalar],
    n: usize,
) -> Option<Vec<A::Scalar>> {
    use crate::algebra::{MaxMul, MaxPlus, MinPlus};
    use std::any::TypeId;
    use tropical_gemm::{
        tropical_matmul, TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus, TropicalSemiring,
    };

    // Dispatch based on algebra type using TypeId
    // The tropical-gemm types have identical repr(transparent) layout to our types,
    // and both wrap the scalar directly, so we can safely transmute the output.

    if TypeId::of::<A>() == TypeId::of::<MaxPlus<f32>>() {
        // SAFETY: A::Scalar is f32, and MaxPlus<f32> has repr(transparent) over f32
        let a_f32: &[f32] = unsafe { std::mem::transmute(a) };
        let b_f32: &[f32] = unsafe { std::mem::transmute(b) };

        let result: Vec<TropicalMaxPlus<f32>> =
            tropical_matmul::<TropicalMaxPlus<f32>>(a_f32, m, k, b_f32, n);

        // Convert TropicalMaxPlus<f32> -> f32, both are repr(transparent) over f32
        let scalars: Vec<f32> = result.into_iter().map(|x| x.value()).collect();

        // SAFETY: A::Scalar is f32
        Some(unsafe { std::mem::transmute(scalars) })
    } else if TypeId::of::<A>() == TypeId::of::<MaxPlus<f64>>() {
        let a_f64: &[f64] = unsafe { std::mem::transmute(a) };
        let b_f64: &[f64] = unsafe { std::mem::transmute(b) };

        let result: Vec<TropicalMaxPlus<f64>> =
            tropical_matmul::<TropicalMaxPlus<f64>>(a_f64, m, k, b_f64, n);
        let scalars: Vec<f64> = result.into_iter().map(|x| x.value()).collect();

        Some(unsafe { std::mem::transmute(scalars) })
    } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f32>>() {
        let a_f32: &[f32] = unsafe { std::mem::transmute(a) };
        let b_f32: &[f32] = unsafe { std::mem::transmute(b) };

        let result: Vec<TropicalMinPlus<f32>> =
            tropical_matmul::<TropicalMinPlus<f32>>(a_f32, m, k, b_f32, n);
        let scalars: Vec<f32> = result.into_iter().map(|x| x.value()).collect();

        Some(unsafe { std::mem::transmute(scalars) })
    } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f64>>() {
        let a_f64: &[f64] = unsafe { std::mem::transmute(a) };
        let b_f64: &[f64] = unsafe { std::mem::transmute(b) };

        let result: Vec<TropicalMinPlus<f64>> =
            tropical_matmul::<TropicalMinPlus<f64>>(a_f64, m, k, b_f64, n);
        let scalars: Vec<f64> = result.into_iter().map(|x| x.value()).collect();

        Some(unsafe { std::mem::transmute(scalars) })
    } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f32>>() {
        let a_f32: &[f32] = unsafe { std::mem::transmute(a) };
        let b_f32: &[f32] = unsafe { std::mem::transmute(b) };

        let result: Vec<TropicalMaxMul<f32>> =
            tropical_matmul::<TropicalMaxMul<f32>>(a_f32, m, k, b_f32, n);
        let scalars: Vec<f32> = result.into_iter().map(|x| x.value()).collect();

        Some(unsafe { std::mem::transmute(scalars) })
    } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f64>>() {
        let a_f64: &[f64] = unsafe { std::mem::transmute(a) };
        let b_f64: &[f64] = unsafe { std::mem::transmute(b) };

        let result: Vec<TropicalMaxMul<f64>> =
            tropical_matmul::<TropicalMaxMul<f64>>(a_f64, m, k, b_f64, n);
        let scalars: Vec<f64> = result.into_iter().map(|x| x.value()).collect();

        Some(unsafe { std::mem::transmute(scalars) })
    } else {
        // Unsupported type, fall back to generic implementation
        None
    }
}

#[cfg(feature = "tropical-kernels")]
fn try_tropical_gemm_with_argmax<A: Algebra<Index = u32>>(
    a: &[A::Scalar],
    m: usize,
    k: usize,
    b: &[A::Scalar],
    n: usize,
) -> Option<(Vec<A::Scalar>, Vec<u32>)> {
    use crate::algebra::{MaxMul, MaxPlus, MinPlus};
    use std::any::TypeId;
    use tropical_gemm::{
        tropical_matmul_with_argmax, TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus,
        TropicalSemiring,
    };

    // Dispatch based on algebra type using TypeId
    if TypeId::of::<A>() == TypeId::of::<MaxPlus<f32>>() {
        let a_f32: &[f32] = unsafe { std::mem::transmute(a) };
        let b_f32: &[f32] = unsafe { std::mem::transmute(b) };

        let result = tropical_matmul_with_argmax::<TropicalMaxPlus<f32>>(a_f32, m, k, b_f32, n);

        // Convert to column-major storage
        // Note: tropical-gemm's accessor functions use (col, row) order internally
        let mut scalars = Vec::with_capacity(m * n);
        let mut argmax = Vec::with_capacity(m * n);
        for j in 0..n {
            for i in 0..m {
                scalars.push(result.get(j, i).value());
                argmax.push(result.get_argmax(j, i));
            }
        }

        Some((unsafe { std::mem::transmute(scalars) }, argmax))
    } else if TypeId::of::<A>() == TypeId::of::<MaxPlus<f64>>() {
        let a_f64: &[f64] = unsafe { std::mem::transmute(a) };
        let b_f64: &[f64] = unsafe { std::mem::transmute(b) };

        let result = tropical_matmul_with_argmax::<TropicalMaxPlus<f64>>(a_f64, m, k, b_f64, n);

        // Convert to column-major storage
        let mut scalars = Vec::with_capacity(m * n);
        let mut argmax = Vec::with_capacity(m * n);
        for j in 0..n {
            for i in 0..m {
                scalars.push(result.get(j, i).value());
                argmax.push(result.get_argmax(j, i));
            }
        }

        Some((unsafe { std::mem::transmute(scalars) }, argmax))
    } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f32>>() {
        let a_f32: &[f32] = unsafe { std::mem::transmute(a) };
        let b_f32: &[f32] = unsafe { std::mem::transmute(b) };

        let result = tropical_matmul_with_argmax::<TropicalMinPlus<f32>>(a_f32, m, k, b_f32, n);

        // Convert to column-major storage
        let mut scalars = Vec::with_capacity(m * n);
        let mut argmax = Vec::with_capacity(m * n);
        for j in 0..n {
            for i in 0..m {
                scalars.push(result.get(j, i).value());
                argmax.push(result.get_argmax(j, i));
            }
        }

        Some((unsafe { std::mem::transmute(scalars) }, argmax))
    } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f64>>() {
        let a_f64: &[f64] = unsafe { std::mem::transmute(a) };
        let b_f64: &[f64] = unsafe { std::mem::transmute(b) };

        let result = tropical_matmul_with_argmax::<TropicalMinPlus<f64>>(a_f64, m, k, b_f64, n);

        // Convert to column-major storage
        let mut scalars = Vec::with_capacity(m * n);
        let mut argmax = Vec::with_capacity(m * n);
        for j in 0..n {
            for i in 0..m {
                scalars.push(result.get(j, i).value());
                argmax.push(result.get_argmax(j, i));
            }
        }

        Some((unsafe { std::mem::transmute(scalars) }, argmax))
    } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f32>>() {
        let a_f32: &[f32] = unsafe { std::mem::transmute(a) };
        let b_f32: &[f32] = unsafe { std::mem::transmute(b) };

        let result = tropical_matmul_with_argmax::<TropicalMaxMul<f32>>(a_f32, m, k, b_f32, n);

        // Convert to column-major storage
        let mut scalars = Vec::with_capacity(m * n);
        let mut argmax = Vec::with_capacity(m * n);
        for j in 0..n {
            for i in 0..m {
                scalars.push(result.get(j, i).value());
                argmax.push(result.get_argmax(j, i));
            }
        }

        Some((unsafe { std::mem::transmute(scalars) }, argmax))
    } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f64>>() {
        let a_f64: &[f64] = unsafe { std::mem::transmute(a) };
        let b_f64: &[f64] = unsafe { std::mem::transmute(b) };

        let result = tropical_matmul_with_argmax::<TropicalMaxMul<f64>>(a_f64, m, k, b_f64, n);

        // Convert to column-major storage
        let mut scalars = Vec::with_capacity(m * n);
        let mut argmax = Vec::with_capacity(m * n);
        for j in 0..n {
            for i in 0..m {
                scalars.push(result.get(j, i).value());
                argmax.push(result.get_argmax(j, i));
            }
        }

        Some((unsafe { std::mem::transmute(scalars) }, argmax))
    } else {
        // Unsupported type, fall back to generic implementation
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Standard;

    #[cfg(feature = "tropical")]
    use crate::algebra::MaxPlus;

    #[test]
    fn test_cpu_gemm_standard() {
        let cpu = Cpu;
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2

        let c = cpu.gemm_internal::<Standard<f32>>(&a, 2, 2, &b, 2);

        // [1 2] × [1 2] = [1*1+2*3  1*2+2*4] = [7  10]
        // [3 4]   [3 4]   [3*1+4*3  3*2+4*4]   [15 22]
        assert_eq!(c, vec![7.0, 10.0, 15.0, 22.0]);
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_cpu_gemm_maxplus() {
        let cpu = Cpu;
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2

        let c = cpu.gemm_internal::<MaxPlus<f32>>(&a, 2, 2, &b, 2);

        // MaxPlus: C[i,j] = max_k(A[i,k] + B[k,j])
        // C[0,0] = max(1+1, 2+3) = max(2, 5) = 5
        // C[0,1] = max(1+2, 2+4) = max(3, 6) = 6
        // C[1,0] = max(3+1, 4+3) = max(4, 7) = 7
        // C[1,1] = max(3+2, 4+4) = max(5, 8) = 8
        assert_eq!(c, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_cpu_gemm_with_argmax() {
        let cpu = Cpu;
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];

        let (c, argmax) = cpu.gemm_with_argmax_internal::<MaxPlus<f32>>(&a, 2, 2, &b, 2);

        assert_eq!(c, vec![5.0, 6.0, 7.0, 8.0]);
        // All winners should be k=1 (second column of A, second row of B)
        assert_eq!(argmax, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_copy_strided() {
        let cpu = Cpu;
        // Column-major: data [1,2,3,4,5,6] for shape [2,3] represents:
        // [[1, 3, 5],
        //  [2, 4, 6]]
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        // Transpose: shape [3, 2], strides [2, 1] (original col-major strides permuted)
        // This reads the original matrix as transposed
        let dst = cpu.copy_strided(&src, &[3, 2], &[2, 1], 0);

        // Transposed matrix in column-major:
        // [[1, 2],
        //  [3, 4],
        //  [5, 6]] -> column-major data: [1, 3, 5, 2, 4, 6]
        assert_eq!(dst, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    /// Test that optimized tropical-gemm kernels produce same results as generic implementation.
    #[cfg(feature = "tropical-kernels")]
    #[test]
    fn test_tropical_gemm_optimized_maxplus() {
        use crate::algebra::MaxPlus;

        let cpu = Cpu;
        let m = 64;
        let k = 64;
        let n = 64;

        let a: Vec<f32> = (0..m * k).map(|i| (i % 100) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 100) as f32).collect();

        // Test MaxPlus<f32>
        let c_opt = cpu.gemm_internal::<MaxPlus<f32>>(&a, m, k, &b, n);
        let c_generic = generic_gemm::<MaxPlus<f32>>(&a, m, k, &b, n);

        for (i, (opt, gen)) in c_opt.iter().zip(c_generic.iter()).enumerate() {
            assert!(
                (opt - gen).abs() < 1e-6,
                "MaxPlus mismatch at index {}: opt={}, gen={}",
                i,
                opt,
                gen
            );
        }
    }

    #[cfg(feature = "tropical-kernels")]
    #[test]
    fn test_tropical_gemm_optimized_minplus() {
        use crate::algebra::MinPlus;

        let cpu = Cpu;
        let m = 32;
        let k = 32;
        let n = 32;

        let a: Vec<f32> = (0..m * k).map(|i| (i % 50) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 50) as f32).collect();

        // Test MinPlus<f32>
        let c_opt = cpu.gemm_internal::<MinPlus<f32>>(&a, m, k, &b, n);
        let c_generic = generic_gemm::<MinPlus<f32>>(&a, m, k, &b, n);

        for (i, (opt, gen)) in c_opt.iter().zip(c_generic.iter()).enumerate() {
            assert!(
                (opt - gen).abs() < 1e-6,
                "MinPlus mismatch at index {}: opt={}, gen={}",
                i,
                opt,
                gen
            );
        }
    }

    #[cfg(feature = "tropical-kernels")]
    #[test]
    fn test_tropical_gemm_optimized_maxmul() {
        use crate::algebra::MaxMul;

        let cpu = Cpu;
        let m = 16;
        let k = 16;
        let n = 16;

        // Use small values to avoid overflow in multiplication
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 10) as f32) * 0.1 + 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 10) as f32) * 0.1 + 0.1).collect();

        // Test MaxMul<f32>
        let c_opt = cpu.gemm_internal::<MaxMul<f32>>(&a, m, k, &b, n);
        let c_generic = generic_gemm::<MaxMul<f32>>(&a, m, k, &b, n);

        for (i, (opt, gen)) in c_opt.iter().zip(c_generic.iter()).enumerate() {
            assert!(
                (opt - gen).abs() < 1e-5,
                "MaxMul mismatch at index {}: opt={}, gen={}",
                i,
                opt,
                gen
            );
        }
    }

    #[cfg(feature = "tropical-kernels")]
    #[test]
    fn test_tropical_gemm_with_argmax_optimized() {
        use crate::algebra::MaxPlus;

        let cpu = Cpu;
        let m = 32;
        let k = 32;
        let n = 32;

        let a: Vec<f32> = (0..m * k).map(|i| (i % 100) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 100) as f32).collect();

        // Test MaxPlus<f32> with argmax
        let (c_opt, argmax_opt) = cpu.gemm_with_argmax_internal::<MaxPlus<f32>>(&a, m, k, &b, n);
        let (c_generic, argmax_generic) = generic_gemm_with_argmax::<MaxPlus<f32>>(&a, m, k, &b, n);

        for (i, (opt, gen)) in c_opt.iter().zip(c_generic.iter()).enumerate() {
            assert!(
                (opt - gen).abs() < 1e-6,
                "MaxPlus with argmax: value mismatch at index {}: opt={}, gen={}",
                i,
                opt,
                gen
            );
        }

        for (i, (opt, gen)) in argmax_opt.iter().zip(argmax_generic.iter()).enumerate() {
            assert_eq!(
                opt, gen,
                "MaxPlus with argmax: argmax mismatch at index {}: opt={}, gen={}",
                i, opt, gen
            );
        }
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_gemm_backward() {
        let cpu = Cpu;
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2

        let (_c, argmax) = cpu.gemm_with_argmax_internal::<MaxPlus<f32>>(&a, 2, 3, &b, 2);

        let grad_c = vec![1.0f32; 4];
        let grad_a = cpu.gemm_backward_a_internal::<MaxPlus<f32>>(&grad_c, &argmax, &b, 2, 3, 2);
        let grad_b = cpu.gemm_backward_b_internal::<MaxPlus<f32>>(&grad_c, &argmax, &a, 2, 3, 2);

        assert_eq!(grad_a.len(), 6);
        assert_eq!(grad_b.len(), 6);

        // Verify that gradients accumulated correctly (no unsafe transmute issues)
        // The sum of all gradients should equal the sum of all grad_c elements
        // since each grad_c element contributes exactly once to grad_a and grad_b
        let grad_a_sum: f32 = grad_a.iter().sum();
        let grad_b_sum: f32 = grad_b.iter().sum();
        let grad_c_sum: f32 = grad_c.iter().sum();

        assert_eq!(grad_a_sum, grad_c_sum, "grad_a sum should equal grad_c sum");
        assert_eq!(grad_b_sum, grad_c_sum, "grad_b sum should equal grad_c sum");
    }
}
