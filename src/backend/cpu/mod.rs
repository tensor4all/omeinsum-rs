//! CPU backend implementation.

mod buffer_pool;
mod contract;

use super::traits::{Backend, BackendScalar, Storage};
use crate::algebra::{Algebra, Scalar, Standard};
use num_complex::{Complex32, Complex64};
use std::any::TypeId;

/// CPU backend using Vec storage.
#[derive(Clone, Debug, Default)]
pub struct Cpu;

#[derive(Clone, Copy)]
pub(crate) struct MatrixLayout<'a, T> {
    pub data: &'a [T],
    pub rows: usize,
    pub cols: usize,
    pub row_stride: isize,
    pub col_stride: isize,
}

impl<'a, T> MatrixLayout<'a, T> {
    pub(crate) fn column_major(data: &'a [T], rows: usize, cols: usize) -> Self {
        Self {
            data,
            rows,
            cols,
            row_stride: 1,
            col_stride: rows as isize,
        }
    }

    #[cfg(test)]
    pub(crate) fn column_major_transposed(data: &'a [T], rows: usize, cols: usize) -> Self {
        Self {
            data,
            rows,
            cols,
            row_stride: cols as isize,
            col_stride: 1,
        }
    }
}

fn layout_offset_bounds<T>(layout: &MatrixLayout<'_, T>) -> (isize, isize) {
    let row_extent = if layout.rows == 0 {
        0
    } else {
        (layout.rows as isize - 1) * layout.row_stride
    };
    let col_extent = if layout.cols == 0 {
        0
    } else {
        (layout.cols as isize - 1) * layout.col_stride
    };
    let offsets = [0, row_extent, col_extent, row_extent + col_extent];
    (
        *offsets.iter().min().expect("offset bounds must exist"),
        *offsets.iter().max().expect("offset bounds must exist"),
    )
}

fn faer_mat_ref<'a, T>(layout: MatrixLayout<'a, T>) -> faer::MatRef<'a, T> {
    let (min_offset, max_offset) = layout_offset_bounds(&layout);
    if layout.rows > 0 && layout.cols > 0 {
        assert!(
            !layout.data.is_empty(),
            "matrix layout requires backing storage"
        );
        assert!(
            max_offset >= min_offset,
            "matrix layout offsets must be ordered"
        );
        assert!(
            ((max_offset - min_offset) as usize) < layout.data.len(),
            "matrix layout exceeds backing storage"
        );
    }

    let ptr = unsafe { layout.data.as_ptr().offset(-min_offset) };
    unsafe {
        faer::MatRef::from_raw_parts(
            ptr,
            layout.rows,
            layout.cols,
            layout.row_stride,
            layout.col_stride,
        )
    }
}

fn matrix_layout_batch_view<'a, T>(
    layout: MatrixLayout<'a, T>,
    batch: usize,
) -> MatrixLayout<'a, T> {
    assert!(batch < layout.data.len(), "batch offset must be in bounds");
    MatrixLayout {
        data: &layout.data[batch..],
        ..layout
    }
}

const STANDARD_BATCHED_FAER_MIN_OPS_PER_BATCH: usize = 1024;

fn should_use_standard_batched_gemm(batch_size: usize, m: usize, k: usize, n: usize) -> bool {
    batch_size > 0
        && m > 0
        && k > 0
        && n > 0
        && m.saturating_mul(k).saturating_mul(n) >= STANDARD_BATCHED_FAER_MIN_OPS_PER_BATCH
}

#[cfg(test)]
mod allocation_counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    pub(crate) struct CountingAllocator;

    thread_local! {
        static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
        static COUNT_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    }

    fn record_allocation() {
        COUNT_ALLOCATIONS.with(|active| {
            if active.get() {
                ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
            }
        });
    }

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record_allocation();
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record_allocation();
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            record_allocation();
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    pub(crate) fn with_allocation_counting<T>(f: impl FnOnce() -> T) -> (T, usize) {
        ALLOCATION_COUNT.with(|count| count.set(0));
        COUNT_ALLOCATIONS.with(|active| active.set(true));
        let result = f();
        COUNT_ALLOCATIONS.with(|active| active.set(false));
        let allocations = ALLOCATION_COUNT.with(Cell::get);
        (result, allocations)
    }

    #[cfg(test)]
    mod tests {
        use super::with_allocation_counting;
        use std::hint::spin_loop;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::thread;

        #[test]
        fn allocation_counts_are_isolated_between_threads() {
            let phase = Arc::new(AtomicUsize::new(0));
            let worker_phase = Arc::clone(&phase);
            let worker = thread::spawn(move || {
                let (buffer, allocations) = with_allocation_counting(|| {
                    let buffer = Vec::<u8>::with_capacity(64);
                    worker_phase.store(1, Ordering::Release);
                    while worker_phase.load(Ordering::Acquire) != 2 {
                        spin_loop();
                    }
                    buffer
                });
                drop(buffer);
                allocations
            });

            while phase.load(Ordering::Acquire) != 1 {
                spin_loop();
            }
            let ((), allocations) = with_allocation_counting(|| {
                phase.store(2, Ordering::Release);
            });

            assert_eq!(allocations, 0);
            assert_eq!(worker.join().unwrap(), 1);
        }
    }
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: allocation_counting::CountingAllocator =
    allocation_counting::CountingAllocator;

impl Cpu {
    pub(crate) fn gemm_standard_layout_internal<A: Algebra>(
        &self,
        a: MatrixLayout<'_, A::Scalar>,
        b: MatrixLayout<'_, A::Scalar>,
    ) -> Option<Vec<A::Scalar>> {
        if TypeId::of::<A>() == TypeId::of::<Standard<f32>>() {
            let a_f32 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[f32]>(a.data) },
                rows: a.rows,
                cols: a.cols,
                row_stride: a.row_stride,
                col_stride: a.col_stride,
            };
            let b_f32 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[f32]>(b.data) },
                rows: b.rows,
                cols: b.cols,
                row_stride: b.row_stride,
                col_stride: b.col_stride,
            };
            let result = faer_gemm_layout(a_f32, b_f32);
            return Some(unsafe { std::mem::transmute::<Vec<f32>, Vec<A::Scalar>>(result) });
        }
        if TypeId::of::<A>() == TypeId::of::<Standard<f64>>() {
            let a_f64 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[f64]>(a.data) },
                rows: a.rows,
                cols: a.cols,
                row_stride: a.row_stride,
                col_stride: a.col_stride,
            };
            let b_f64 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[f64]>(b.data) },
                rows: b.rows,
                cols: b.cols,
                row_stride: b.row_stride,
                col_stride: b.col_stride,
            };
            let result = faer_gemm_layout(a_f64, b_f64);
            return Some(unsafe { std::mem::transmute::<Vec<f64>, Vec<A::Scalar>>(result) });
        }
        if TypeId::of::<A>() == TypeId::of::<Standard<Complex32>>() {
            // SAFETY: TypeId proves A::Scalar is Complex32 for this branch.
            let a_c32 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[Complex32]>(a.data) },
                rows: a.rows,
                cols: a.cols,
                row_stride: a.row_stride,
                col_stride: a.col_stride,
            };
            let b_c32 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[Complex32]>(b.data) },
                rows: b.rows,
                cols: b.cols,
                row_stride: b.row_stride,
                col_stride: b.col_stride,
            };
            let result = faer_gemm_layout(a_c32, b_c32);
            return Some(unsafe { std::mem::transmute::<Vec<Complex32>, Vec<A::Scalar>>(result) });
        }
        if TypeId::of::<A>() == TypeId::of::<Standard<Complex64>>() {
            // SAFETY: TypeId proves A::Scalar is Complex64 for this branch.
            let a_c64 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[Complex64]>(a.data) },
                rows: a.rows,
                cols: a.cols,
                row_stride: a.row_stride,
                col_stride: a.col_stride,
            };
            let b_c64 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[Complex64]>(b.data) },
                rows: b.rows,
                cols: b.cols,
                row_stride: b.row_stride,
                col_stride: b.col_stride,
            };
            let result = faer_gemm_layout(a_c64, b_c64);
            return Some(unsafe { std::mem::transmute::<Vec<Complex64>, Vec<A::Scalar>>(result) });
        }

        None
    }

    pub(crate) fn gemm_batched_standard_layout_internal<A: Algebra>(
        &self,
        batch_size: usize,
        a: MatrixLayout<'_, A::Scalar>,
        b: MatrixLayout<'_, A::Scalar>,
    ) -> Option<Vec<A::Scalar>> {
        if TypeId::of::<A>() == TypeId::of::<Standard<f32>>() {
            let a_f32 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[f32]>(a.data) },
                rows: a.rows,
                cols: a.cols,
                row_stride: a.row_stride,
                col_stride: a.col_stride,
            };
            let b_f32 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[f32]>(b.data) },
                rows: b.rows,
                cols: b.cols,
                row_stride: b.row_stride,
                col_stride: b.col_stride,
            };
            let result = faer_batched_gemm_layout(batch_size, a_f32, b_f32);
            return Some(unsafe { std::mem::transmute::<Vec<f32>, Vec<A::Scalar>>(result) });
        }
        if TypeId::of::<A>() == TypeId::of::<Standard<f64>>() {
            let a_f64 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[f64]>(a.data) },
                rows: a.rows,
                cols: a.cols,
                row_stride: a.row_stride,
                col_stride: a.col_stride,
            };
            let b_f64 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[f64]>(b.data) },
                rows: b.rows,
                cols: b.cols,
                row_stride: b.row_stride,
                col_stride: b.col_stride,
            };
            let result = faer_batched_gemm_layout(batch_size, a_f64, b_f64);
            return Some(unsafe { std::mem::transmute::<Vec<f64>, Vec<A::Scalar>>(result) });
        }
        if TypeId::of::<A>() == TypeId::of::<Standard<Complex32>>() {
            // SAFETY: TypeId proves A::Scalar is Complex32 for this branch.
            let a_c32 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[Complex32]>(a.data) },
                rows: a.rows,
                cols: a.cols,
                row_stride: a.row_stride,
                col_stride: a.col_stride,
            };
            let b_c32 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[Complex32]>(b.data) },
                rows: b.rows,
                cols: b.cols,
                row_stride: b.row_stride,
                col_stride: b.col_stride,
            };
            let result = faer_batched_gemm_layout(batch_size, a_c32, b_c32);
            return Some(unsafe { std::mem::transmute::<Vec<Complex32>, Vec<A::Scalar>>(result) });
        }
        if TypeId::of::<A>() == TypeId::of::<Standard<Complex64>>() {
            // SAFETY: TypeId proves A::Scalar is Complex64 for this branch.
            let a_c64 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[Complex64]>(a.data) },
                rows: a.rows,
                cols: a.cols,
                row_stride: a.row_stride,
                col_stride: a.col_stride,
            };
            let b_c64 = MatrixLayout {
                data: unsafe { std::mem::transmute::<&[A::Scalar], &[Complex64]>(b.data) },
                rows: b.rows,
                cols: b.cols,
                row_stride: b.row_stride,
                col_stride: b.col_stride,
            };
            let result = faer_batched_gemm_layout(batch_size, a_c64, b_c64);
            return Some(unsafe { std::mem::transmute::<Vec<Complex64>, Vec<A::Scalar>>(result) });
        }

        None
    }

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
        // Fast path: faer for native real and complex Standard scalars.
        if TypeId::of::<A>() == TypeId::of::<Standard<f32>>() {
            // SAFETY: A::Scalar is f32 when A is Standard<f32>
            let a_f32: &[f32] = unsafe { std::mem::transmute(a) };
            let b_f32: &[f32] = unsafe { std::mem::transmute(b) };
            let result = faer_gemm(a_f32, m, k, b_f32, n);
            return unsafe { std::mem::transmute::<Vec<f32>, Vec<A::Scalar>>(result) };
        }
        if TypeId::of::<A>() == TypeId::of::<Standard<f64>>() {
            let a_f64: &[f64] = unsafe { std::mem::transmute(a) };
            let b_f64: &[f64] = unsafe { std::mem::transmute(b) };
            let result = faer_gemm(a_f64, m, k, b_f64, n);
            return unsafe { std::mem::transmute::<Vec<f64>, Vec<A::Scalar>>(result) };
        }
        if TypeId::of::<A>() == TypeId::of::<Standard<Complex32>>() {
            // SAFETY: A::Scalar is Complex32 when A is Standard<Complex32>.
            let a_c32: &[Complex32] = unsafe { std::mem::transmute(a) };
            let b_c32: &[Complex32] = unsafe { std::mem::transmute(b) };
            let result = faer_gemm(a_c32, m, k, b_c32, n);
            return unsafe { std::mem::transmute::<Vec<Complex32>, Vec<A::Scalar>>(result) };
        }
        if TypeId::of::<A>() == TypeId::of::<Standard<Complex64>>() {
            // SAFETY: A::Scalar is Complex64 when A is Standard<Complex64>.
            let a_c64: &[Complex64] = unsafe { std::mem::transmute(a) };
            let b_c64: &[Complex64] = unsafe { std::mem::transmute(b) };
            let result = faer_gemm(a_c64, m, k, b_c64, n);
            return unsafe { std::mem::transmute::<Vec<Complex64>, Vec<A::Scalar>>(result) };
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
        if TypeId::of::<A>() == TypeId::of::<Standard<f32>>() {
            let a_f32: &[f32] = unsafe { std::mem::transmute(a) };
            let b_f32: &[f32] = unsafe { std::mem::transmute(b) };
            let result = standard_batched_gemm(a_f32, batch_size, m, k, b_f32, n);
            return unsafe { std::mem::transmute::<Vec<f32>, Vec<A::Scalar>>(result) };
        }
        if TypeId::of::<A>() == TypeId::of::<Standard<f64>>() {
            let a_f64: &[f64] = unsafe { std::mem::transmute(a) };
            let b_f64: &[f64] = unsafe { std::mem::transmute(b) };
            let result = standard_batched_gemm(a_f64, batch_size, m, k, b_f64, n);
            return unsafe { std::mem::transmute::<Vec<f64>, Vec<A::Scalar>>(result) };
        }
        if TypeId::of::<A>() == TypeId::of::<Standard<Complex32>>() {
            // SAFETY: A::Scalar is Complex32 when A is Standard<Complex32>.
            let a_c32: &[Complex32] = unsafe { std::mem::transmute(a) };
            let b_c32: &[Complex32] = unsafe { std::mem::transmute(b) };
            let result = standard_batched_gemm(a_c32, batch_size, m, k, b_c32, n);
            return unsafe { std::mem::transmute::<Vec<Complex32>, Vec<A::Scalar>>(result) };
        }
        if TypeId::of::<A>() == TypeId::of::<Standard<Complex64>>() {
            // SAFETY: A::Scalar is Complex64 when A is Standard<Complex64>.
            let a_c64: &[Complex64] = unsafe { std::mem::transmute(a) };
            let b_c64: &[Complex64] = unsafe { std::mem::transmute(b) };
            let result = standard_batched_gemm(a_c64, batch_size, m, k, b_c64, n);
            return unsafe { std::mem::transmute::<Vec<Complex64>, Vec<A::Scalar>>(result) };
        }

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

            // Try the SIMD tropical path per batch; fall back to the generic
            // semiring loop if the algebra isn't one tropical-gemm supports.
            // `try_tropical_gemm` already takes care of the column-major /
            // row-major swap (see the comment in its body).
            #[cfg(feature = "tropical-kernels")]
            {
                if let Some(c_batch) = try_tropical_gemm::<A>(a_slice, m, k, b_slice, n) {
                    c[c_offset..c_offset + c_batch_stride].copy_from_slice(&c_batch);
                    continue;
                }
            }

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

/// GEMM using faer's native real and complex kernels.
///
/// Inputs and output are column-major. Inputs are borrowed; only the output is
/// allocated.
fn faer_gemm<T>(a: &[T], m: usize, k: usize, b: &[T], n: usize) -> Vec<T>
where
    T: faer::traits::ComplexField + Copy,
{
    faer_gemm_layout(
        MatrixLayout {
            data: a,
            rows: m,
            cols: k,
            row_stride: 1,
            col_stride: m as isize,
        },
        MatrixLayout {
            data: b,
            rows: k,
            cols: n,
            row_stride: 1,
            col_stride: k as isize,
        },
    )
}

fn faer_gemm_layout<T>(a: MatrixLayout<'_, T>, b: MatrixLayout<'_, T>) -> Vec<T>
where
    T: faer::traits::ComplexField + Copy,
{
    let mut c = vec![faer::traits::math_utils::zero::<T>(); a.rows * b.cols];
    faer_gemm_layout_into(a, b, &mut c);
    c
}

fn faer_gemm_layout_into<T>(a: MatrixLayout<'_, T>, b: MatrixLayout<'_, T>, c: &mut [T])
where
    T: faer::traits::ComplexField + Copy,
{
    use faer::{linalg::matmul::matmul, Accum, MatMut, Par};

    assert_eq!(c.len(), a.rows * b.cols);
    let a_mat = faer_mat_ref(a);
    let b_mat = faer_mat_ref(b);
    let mut c_mat =
        unsafe { MatMut::from_raw_parts_mut(c.as_mut_ptr(), a.rows, b.cols, 1, a.rows as isize) };
    matmul(
        c_mat.as_mut(),
        Accum::Replace,
        a_mat,
        b_mat,
        faer::traits::math_utils::one::<T>(),
        Par::Seq,
    );
}

fn standard_batched_gemm<T>(
    a: &[T],
    batch_size: usize,
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> Vec<T>
where
    T: faer::traits::ComplexField + Copy + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    if should_use_standard_batched_gemm(batch_size, m, k, n) {
        return faer_batched_gemm(a, batch_size, m, k, b, n);
    }

    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let c_batch_stride = m * n;
    let mut c = vec![faer::traits::math_utils::zero::<T>(); batch_size * c_batch_stride];

    for batch in 0..batch_size {
        let a_offset = batch * a_batch_stride;
        let b_offset = batch * b_batch_stride;
        let c_offset = batch * c_batch_stride;

        for j in 0..n {
            for i in 0..m {
                let mut acc = faer::traits::math_utils::zero::<T>();
                for kk in 0..k {
                    acc += a[a_offset + kk * m + i] * b[b_offset + j * k + kk];
                }
                c[c_offset + j * m + i] = acc;
            }
        }
    }

    c
}

fn faer_batched_gemm<T>(a: &[T], batch_size: usize, m: usize, k: usize, b: &[T], n: usize) -> Vec<T>
where
    T: faer::traits::ComplexField + Copy,
{
    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let c_batch_stride = m * n;
    let mut c = vec![faer::traits::math_utils::zero::<T>(); batch_size * c_batch_stride];

    for batch in 0..batch_size {
        let a_offset = batch * a_batch_stride;
        let b_offset = batch * b_batch_stride;
        let c_offset = batch * c_batch_stride;
        faer_gemm_layout_into(
            MatrixLayout::column_major(&a[a_offset..a_offset + a_batch_stride], m, k),
            MatrixLayout::column_major(&b[b_offset..b_offset + b_batch_stride], k, n),
            &mut c[c_offset..c_offset + c_batch_stride],
        );
    }

    c
}

fn faer_batched_gemm_layout<T>(
    batch_size: usize,
    a: MatrixLayout<'_, T>,
    b: MatrixLayout<'_, T>,
) -> Vec<T>
where
    T: faer::traits::ComplexField + Copy,
{
    assert_eq!(a.cols, b.rows, "GEMM inner dimensions must match");
    let c_batch_stride = a.rows * b.cols;
    let mut c = vec![faer::traits::math_utils::zero::<T>(); batch_size * c_batch_stride];
    if batch_size == 0 || a.rows == 0 || a.cols == 0 || b.cols == 0 {
        return c;
    }

    for batch in 0..batch_size {
        let c_offset = batch * c_batch_stride;
        faer_gemm_layout_into(
            matrix_layout_batch_view(a, batch),
            matrix_layout_batch_view(b, batch),
            &mut c[c_offset..c_offset + c_batch_stride],
        );
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

    // Dispatch based on algebra type using TypeId. The tropical-gemm types have
    // identical repr(transparent) layout to our types, so we can safely transmute
    // the input slices and the output Vec.
    //
    // Layout note (don't "fix" the arg order — the swap is intentional):
    // omeinsum stores tensors in column-major order (see `generic_gemm` docstring
    // and the module-wide "column-major: idx j*nrows+i" convention). `tropical-gemm`'s
    // `tropical_matmul` API documents its inputs/outputs as row-major. Passing our
    // column-major bytes to a row-major callee is equivalent to passing A^T and B^T,
    // which would compute (A^T × B^T). Using the identity
    //     (A × B) = ((B^T × A^T))^T
    // combined with the observation that a column-major m×k matrix's raw bytes are
    // byte-identical to the row-major k×m representation of its transpose, we can
    // get the intended column-major A×B out of a row-major matmul by calling
    //     tropical_matmul(b, n, k, a, m)
    // i.e. swap `a` ↔ `b` and swap `m` ↔ `n`. The returned row-major n×m Vec is
    // byte-identical to the column-major m×n storage of A×B. Zero data transposes.

    if TypeId::of::<A>() == TypeId::of::<MaxPlus<f32>>() {
        // SAFETY: A::Scalar is f32, and MaxPlus<f32> has repr(transparent) over f32
        let a_f32: &[f32] = unsafe { std::mem::transmute(a) };
        let b_f32: &[f32] = unsafe { std::mem::transmute(b) };

        let result: Vec<TropicalMaxPlus<f32>> =
            tropical_matmul::<TropicalMaxPlus<f32>>(b_f32, n, k, a_f32, m);

        // Convert TropicalMaxPlus<f32> -> f32, both are repr(transparent) over f32
        let scalars: Vec<f32> = result.into_iter().map(|x| x.value()).collect();

        // SAFETY: A::Scalar is f32
        Some(unsafe { std::mem::transmute(scalars) })
    } else if TypeId::of::<A>() == TypeId::of::<MaxPlus<f64>>() {
        let a_f64: &[f64] = unsafe { std::mem::transmute(a) };
        let b_f64: &[f64] = unsafe { std::mem::transmute(b) };

        let result: Vec<TropicalMaxPlus<f64>> =
            tropical_matmul::<TropicalMaxPlus<f64>>(b_f64, n, k, a_f64, m);
        let scalars: Vec<f64> = result.into_iter().map(|x| x.value()).collect();

        Some(unsafe { std::mem::transmute(scalars) })
    } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f32>>() {
        let a_f32: &[f32] = unsafe { std::mem::transmute(a) };
        let b_f32: &[f32] = unsafe { std::mem::transmute(b) };

        let result: Vec<TropicalMinPlus<f32>> =
            tropical_matmul::<TropicalMinPlus<f32>>(b_f32, n, k, a_f32, m);
        let scalars: Vec<f32> = result.into_iter().map(|x| x.value()).collect();

        Some(unsafe { std::mem::transmute(scalars) })
    } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f64>>() {
        let a_f64: &[f64] = unsafe { std::mem::transmute(a) };
        let b_f64: &[f64] = unsafe { std::mem::transmute(b) };

        let result: Vec<TropicalMinPlus<f64>> =
            tropical_matmul::<TropicalMinPlus<f64>>(b_f64, n, k, a_f64, m);
        let scalars: Vec<f64> = result.into_iter().map(|x| x.value()).collect();

        Some(unsafe { std::mem::transmute(scalars) })
    } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f32>>() {
        let a_f32: &[f32] = unsafe { std::mem::transmute(a) };
        let b_f32: &[f32] = unsafe { std::mem::transmute(b) };

        let result: Vec<TropicalMaxMul<f32>> =
            tropical_matmul::<TropicalMaxMul<f32>>(b_f32, n, k, a_f32, m);
        let scalars: Vec<f32> = result.into_iter().map(|x| x.value()).collect();

        Some(unsafe { std::mem::transmute(scalars) })
    } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f64>>() {
        let a_f64: &[f64] = unsafe { std::mem::transmute(a) };
        let b_f64: &[f64] = unsafe { std::mem::transmute(b) };

        let result: Vec<TropicalMaxMul<f64>> =
            tropical_matmul::<TropicalMaxMul<f64>>(b_f64, n, k, a_f64, m);
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

    fn generic_batched_gemm_for_test<A: Algebra>(
        a: &[A::Scalar],
        batch_size: usize,
        m: usize,
        k: usize,
        b: &[A::Scalar],
        n: usize,
    ) -> Vec<A::Scalar> {
        let mut result = Vec::with_capacity(batch_size * m * n);
        for batch in 0..batch_size {
            result.extend(generic_gemm::<A>(
                &a[batch * m * k..(batch + 1) * m * k],
                m,
                k,
                &b[batch * k * n..(batch + 1) * k * n],
                n,
            ));
        }
        result
    }

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

    #[test]
    fn test_cpu_gemm_standard_complex_hand_checked() {
        let cpu = Cpu;
        let a32 = vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(4.0, -1.0),
        ];
        let b32 = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(-1.0, 0.0),
        ];
        let expected32 = vec![
            Complex32::new(5.0, -1.0),
            Complex32::new(10.0, -6.0),
            Complex32::new(-3.0, 1.0),
            Complex32::new(-4.0, 4.0),
        ];
        assert_eq!(
            cpu.gemm_internal::<Standard<Complex32>>(&a32, 2, 2, &b32, 2),
            expected32
        );

        let a64: Vec<Complex64> = a32
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let b64: Vec<Complex64> = b32
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let expected64: Vec<Complex64> = expected32
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        assert_eq!(
            cpu.gemm_internal::<Standard<Complex64>>(&a64, 2, 2, &b64, 2),
            expected64
        );
    }

    #[test]
    fn test_cpu_gemm_standard_complex_rectangular_matches_generic() {
        let cpu = Cpu;
        let (m, k, n) = (3usize, 2usize, 4usize);
        let a32: Vec<Complex32> = (0..m * k)
            .map(|index| Complex32::new(index as f32 * 0.25 - 0.5, index as f32 * -0.125 + 0.25))
            .collect();
        let b32: Vec<Complex32> = (0..k * n)
            .map(|index| Complex32::new(index as f32 * -0.2 + 0.75, index as f32 * 0.15 - 0.3))
            .collect();
        let actual32 = cpu.gemm_internal::<Standard<Complex32>>(&a32, m, k, &b32, n);
        let expected32 = generic_gemm::<Standard<Complex32>>(&a32, m, k, &b32, n);
        for (actual, expected) in actual32.iter().zip(&expected32) {
            assert!((*actual - *expected).norm() <= 1e-5);
        }

        let a64: Vec<Complex64> = a32
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let b64: Vec<Complex64> = b32
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let actual64 = cpu.gemm_internal::<Standard<Complex64>>(&a64, m, k, &b64, n);
        let expected64 = generic_gemm::<Standard<Complex64>>(&a64, m, k, &b64, n);
        for (actual, expected) in actual64.iter().zip(&expected64) {
            assert!((*actual - *expected).norm() <= 1e-12);
        }
    }

    #[test]
    fn test_cpu_gemm_standard_complex_degenerate_dimensions() {
        let cpu = Cpu;
        let empty32 = cpu.gemm_internal::<Standard<Complex32>>(&[], 0, 3, &[], 0);
        assert!(empty32.is_empty());

        let zeros64 = cpu.gemm_internal::<Standard<Complex64>>(&[], 2, 0, &[], 3);
        assert_eq!(zeros64, vec![Complex64::new(0.0, 0.0); 6]);
    }

    #[test]
    fn test_complex_layout_gemm_accepts_contiguous_and_transposed_views() {
        let cpu = Cpu;
        let a32 = vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(4.0, -1.0),
        ];
        let b32 = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(-1.0, 0.0),
        ];
        let actual32 = cpu
            .gemm_standard_layout_internal::<Standard<Complex32>>(
                MatrixLayout::column_major(&a32, 2, 2),
                MatrixLayout::column_major(&b32, 2, 2),
            )
            .expect("Complex32 layouts should use faer");
        let expected32 = generic_gemm::<Standard<Complex32>>(&a32, 2, 2, &b32, 2);
        assert_eq!(actual32, expected32);

        let a64: Vec<Complex64> = a32
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let b64: Vec<Complex64> = b32
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let actual64 = cpu
            .gemm_standard_layout_internal::<Standard<Complex64>>(
                MatrixLayout::column_major(&a64, 2, 2),
                MatrixLayout::column_major_transposed(&b64, 2, 2),
            )
            .expect("Complex64 transpose layouts should use faer");
        let b64_transposed = vec![b64[0], b64[2], b64[1], b64[3]];
        let expected64 = generic_gemm::<Standard<Complex64>>(&a64, 2, 2, &b64_transposed, 2);
        assert_eq!(actual64, expected64);
    }

    #[test]
    fn test_complex_layout_gemm_accepts_nonunit_and_negative_strides() {
        let cpu = Cpu;
        let a32 = vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, -1.0),
            Complex32::new(-1.0, 2.0),
            Complex32::new(0.5, 0.0),
            Complex32::new(4.0, 1.0),
        ];
        let b32 = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 1.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(0.0, 0.5),
            Complex32::new(3.0, 0.0),
            Complex32::new(-2.0, 1.0),
        ];
        let expected32 = generic_gemm::<Standard<Complex32>>(&a32, 2, 3, &b32, 2);

        let a_negative = vec![a32[1], a32[0], a32[3], a32[2], a32[5], a32[4]];
        let b_negative = vec![b32[3], b32[4], b32[5], b32[0], b32[1], b32[2]];
        let actual32 = cpu
            .gemm_standard_layout_internal::<Standard<Complex32>>(
                MatrixLayout {
                    data: &a_negative,
                    rows: 2,
                    cols: 3,
                    row_stride: -1,
                    col_stride: 2,
                },
                MatrixLayout {
                    data: &b_negative,
                    rows: 3,
                    cols: 2,
                    row_stride: 1,
                    col_stride: -3,
                },
            )
            .expect("negative Complex32 strides should use faer");
        for (actual, expected) in actual32.iter().zip(&expected32) {
            assert!((*actual - *expected).norm() <= 1e-5);
        }

        let a64: Vec<Complex64> = a32
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let b64: Vec<Complex64> = b32
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let expected64 = generic_gemm::<Standard<Complex64>>(&a64, 2, 3, &b64, 2);
        let mut a_positive = vec![Complex64::new(99.0, 99.0); 13];
        for (index, offset) in [0usize, 2, 5, 7, 10, 12].into_iter().enumerate() {
            a_positive[offset] = a64[index];
        }
        let mut b_positive = vec![Complex64::new(99.0, 99.0); 12];
        for (index, offset) in [0usize, 2, 4, 7, 9, 11].into_iter().enumerate() {
            b_positive[offset] = b64[index];
        }
        let actual64 = cpu
            .gemm_standard_layout_internal::<Standard<Complex64>>(
                MatrixLayout {
                    data: &a_positive,
                    rows: 2,
                    cols: 3,
                    row_stride: 2,
                    col_stride: 5,
                },
                MatrixLayout {
                    data: &b_positive,
                    rows: 3,
                    cols: 2,
                    row_stride: 2,
                    col_stride: 7,
                },
            )
            .expect("non-unit Complex64 strides should use faer");
        for (actual, expected) in actual64.iter().zip(&expected64) {
            assert!((*actual - *expected).norm() <= 1e-12);
        }
    }

    #[test]
    fn test_complex_layout_gemm_into_does_not_copy_inputs() {
        let a32 = vec![Complex32::new(1.0, 1.0); 4];
        let b32 = vec![Complex32::new(2.0, -1.0); 4];
        let mut c32 = vec![Complex32::new(0.0, 0.0); 4];
        let a64 = vec![Complex64::new(1.0, 1.0); 4];
        let b64 = vec![Complex64::new(2.0, -1.0); 4];
        let mut c64 = vec![Complex64::new(0.0, 0.0); 4];

        faer_gemm_layout_into(
            MatrixLayout::column_major(&a32, 2, 2),
            MatrixLayout::column_major(&b32, 2, 2),
            &mut c32,
        );
        faer_gemm_layout_into(
            MatrixLayout::column_major(&a64, 2, 2),
            MatrixLayout::column_major(&b64, 2, 2),
            &mut c64,
        );

        let ((), allocations) = allocation_counting::with_allocation_counting(|| {
            faer_gemm_layout_into(
                MatrixLayout::column_major(&a32, 2, 2),
                MatrixLayout::column_major(&b32, 2, 2),
                &mut c32,
            );
            faer_gemm_layout_into(
                MatrixLayout::column_major(&a64, 2, 2),
                MatrixLayout::column_major(&b64, 2, 2),
                &mut c64,
            );
        });

        assert_eq!(allocations, 0, "borrowed complex GEMM must not copy inputs");
    }

    #[test]
    fn test_faer_layout_gemm_accepts_rhs_transpose_view() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];

        let c = faer_gemm_layout(
            MatrixLayout::column_major(&a, 2, 2),
            MatrixLayout::column_major_transposed(&b, 2, 2),
        );

        let expected = faer_gemm(&a, 2, 2, &[1.0, 3.0, 2.0, 4.0], 2);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_faer_layout_gemm_into_writes_output_without_allocating_temporary() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut c = vec![0.0f32; 4];

        faer_gemm_layout_into(
            MatrixLayout::column_major(&a, 2, 2),
            MatrixLayout::column_major(&b, 2, 2),
            &mut c,
        );
        c.fill(-1.0);

        let ((), allocations) = allocation_counting::with_allocation_counting(|| {
            faer_gemm_layout_into(
                MatrixLayout::column_major(&a, 2, 2),
                MatrixLayout::column_major(&b, 2, 2),
                &mut c,
            );
        });

        assert_eq!(c, vec![7.0, 10.0, 15.0, 22.0]);
        assert_eq!(
            allocations, 0,
            "faer_gemm_f32_layout_into should write into the provided output slice"
        );
    }

    #[test]
    fn test_faer_layout_gemm_f64_into_writes_output_without_allocating_temporary() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0];
        let mut c = vec![0.0f64; 4];

        faer_gemm_layout_into(
            MatrixLayout::column_major(&a, 2, 2),
            MatrixLayout::column_major(&b, 2, 2),
            &mut c,
        );
        c.fill(-1.0);

        let ((), allocations) = allocation_counting::with_allocation_counting(|| {
            faer_gemm_layout_into(
                MatrixLayout::column_major(&a, 2, 2),
                MatrixLayout::column_major(&b, 2, 2),
                &mut c,
            );
        });

        assert_eq!(c, vec![7.0, 10.0, 15.0, 22.0]);
        assert_eq!(
            allocations, 0,
            "faer_gemm_f64_layout_into should write into the provided output slice"
        );
    }

    #[test]
    fn test_complex_batched_gemm_contiguous_matches_generic() {
        let cpu = Cpu;
        let (batch_size, m, k, n) = (2usize, 11usize, 10usize, 10usize);
        let a32: Vec<Complex32> = (0..batch_size * m * k)
            .map(|index| {
                Complex32::new(
                    (index % 17) as f32 * 0.0625 - 0.5,
                    (index % 11) as f32 * 0.03125 - 0.125,
                )
            })
            .collect();
        let b32: Vec<Complex32> = (0..batch_size * k * n)
            .map(|index| {
                Complex32::new(
                    (index % 13) as f32 * -0.05 + 0.3,
                    (index % 7) as f32 * 0.04 - 0.1,
                )
            })
            .collect();
        let actual32 =
            cpu.gemm_batched_internal::<Standard<Complex32>>(&a32, batch_size, m, k, &b32, n);
        let expected32 =
            generic_batched_gemm_for_test::<Standard<Complex32>>(&a32, batch_size, m, k, &b32, n);
        for (actual, expected) in actual32.iter().zip(&expected32) {
            assert!((*actual - *expected).norm() <= 1e-4);
        }

        let a64: Vec<Complex64> = a32[..m * k]
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let b64: Vec<Complex64> = b32[..k * n]
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let actual64 = cpu.gemm_batched_internal::<Standard<Complex64>>(&a64, 1, m, k, &b64, n);
        let expected64 =
            generic_batched_gemm_for_test::<Standard<Complex64>>(&a64, 1, m, k, &b64, n);
        for (actual, expected) in actual64.iter().zip(&expected64) {
            assert!((*actual - *expected).norm() <= 1e-12);
        }

        assert!(cpu
            .gemm_batched_internal::<Standard<Complex32>>(&[], 0, m, k, &[], n)
            .is_empty());
    }

    #[test]
    fn test_complex_batched_layout_gemm_accepts_interleaved_and_transposed_views() {
        let cpu = Cpu;
        let (batch_size, m, k, n) = (3usize, 2usize, 3usize, 2usize);
        let a32: Vec<Complex32> = (0..batch_size * m * k)
            .map(|index| Complex32::new(index as f32 * 0.1 - 0.4, index as f32 * 0.03 - 0.2))
            .collect();
        let b32: Vec<Complex32> = (0..batch_size * k * n)
            .map(|index| Complex32::new(index as f32 * -0.07 + 0.5, index as f32 * 0.02))
            .collect();
        let expected32 =
            generic_batched_gemm_for_test::<Standard<Complex32>>(&a32, batch_size, m, k, &b32, n);

        let mut a_interleaved = vec![Complex32::new(0.0, 0.0); a32.len()];
        let mut b_interleaved = vec![Complex32::new(0.0, 0.0); b32.len()];
        for batch in 0..batch_size {
            for col in 0..k {
                for row in 0..m {
                    a_interleaved[batch + row * batch_size + col * batch_size * m] =
                        a32[batch * m * k + col * m + row];
                }
            }
            for col in 0..n {
                for row in 0..k {
                    b_interleaved[batch + row * batch_size + col * batch_size * k] =
                        b32[batch * k * n + col * k + row];
                }
            }
        }
        let actual32 = cpu
            .gemm_batched_standard_layout_internal::<Standard<Complex32>>(
                batch_size,
                MatrixLayout {
                    data: &a_interleaved,
                    rows: m,
                    cols: k,
                    row_stride: batch_size as isize,
                    col_stride: (batch_size * m) as isize,
                },
                MatrixLayout {
                    data: &b_interleaved,
                    rows: k,
                    cols: n,
                    row_stride: batch_size as isize,
                    col_stride: (batch_size * k) as isize,
                },
            )
            .expect("interleaved Complex32 batches should use faer");
        for (actual, expected) in actual32.iter().zip(&expected32) {
            assert!((*actual - *expected).norm() <= 1e-5);
        }

        let a64: Vec<Complex64> = a32
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let b64: Vec<Complex64> = b32
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let expected64 =
            generic_batched_gemm_for_test::<Standard<Complex64>>(&a64, batch_size, m, k, &b64, n);
        let mut a_transposed = vec![Complex64::new(0.0, 0.0); a64.len()];
        let mut b_transposed = vec![Complex64::new(0.0, 0.0); b64.len()];
        for batch in 0..batch_size {
            for col in 0..k {
                for row in 0..m {
                    a_transposed[batch + row * batch_size * k + col * batch_size] =
                        a64[batch * m * k + col * m + row];
                }
            }
            for col in 0..n {
                for row in 0..k {
                    b_transposed[batch + row * batch_size * n + col * batch_size] =
                        b64[batch * k * n + col * k + row];
                }
            }
        }
        let actual64 = cpu
            .gemm_batched_standard_layout_internal::<Standard<Complex64>>(
                batch_size,
                MatrixLayout {
                    data: &a_transposed,
                    rows: m,
                    cols: k,
                    row_stride: (batch_size * k) as isize,
                    col_stride: batch_size as isize,
                },
                MatrixLayout {
                    data: &b_transposed,
                    rows: k,
                    cols: n,
                    row_stride: (batch_size * n) as isize,
                    col_stride: batch_size as isize,
                },
            )
            .expect("transposed Complex64 batches should use faer");
        for (actual, expected) in actual64.iter().zip(&expected64) {
            assert!((*actual - *expected).norm() <= 1e-12);
        }
    }

    #[test]
    fn test_complex_batched_layout_gemm_does_not_allocate_temporary_outputs() {
        let cpu = Cpu;
        let (batch_size, dim) = (3usize, 64usize);
        let matrix_len = dim * dim;
        let a = vec![Complex32::new(1.0, 0.5); batch_size * matrix_len];
        let b = vec![Complex32::new(0.5, -1.0); batch_size * matrix_len];
        let a_layout = MatrixLayout {
            data: &a,
            rows: dim,
            cols: dim,
            row_stride: batch_size as isize,
            col_stride: (batch_size * dim) as isize,
        };
        let b_layout = MatrixLayout {
            data: &b,
            rows: dim,
            cols: dim,
            row_stride: batch_size as isize,
            col_stride: (batch_size * dim) as isize,
        };
        let mut one_output = vec![Complex32::new(0.0, 0.0); matrix_len];

        // Warm faer's dispatch before measuring its per-call workspace behavior.
        faer_gemm_layout_into(a_layout, b_layout, &mut one_output);
        drop(
            cpu.gemm_batched_standard_layout_internal::<Standard<Complex32>>(
                batch_size, a_layout, b_layout,
            ),
        );

        let ((), workspace_allocations) = allocation_counting::with_allocation_counting(|| {
            faer_gemm_layout_into(a_layout, b_layout, &mut one_output);
        });
        let (result, batched_allocations) = allocation_counting::with_allocation_counting(|| {
            cpu.gemm_batched_standard_layout_internal::<Standard<Complex32>>(
                batch_size, a_layout, b_layout,
            )
            .expect("Complex32 batches should use faer")
        });

        assert_eq!(result.len(), batch_size * matrix_len);
        assert_eq!(
            batched_allocations,
            1 + batch_size * workspace_allocations,
            "batched GEMM should allocate one result plus only faer's per-batch workspace"
        );
    }

    #[test]
    fn test_gemm_batched_standard_layout_internal_accepts_batch_major_views() {
        let cpu = Cpu;
        let a = vec![1.0f32, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];
        let b = vec![1.0f32, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0];

        let c = cpu
            .gemm_batched_standard_layout_internal::<Standard<f32>>(
                2,
                MatrixLayout {
                    data: &a,
                    rows: 2,
                    cols: 2,
                    row_stride: 2,
                    col_stride: 4,
                },
                MatrixLayout {
                    data: &b,
                    rows: 2,
                    cols: 2,
                    row_stride: 2,
                    col_stride: 4,
                },
            )
            .expect("standard layout helper should handle batch-major inputs");

        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn test_should_use_standard_batched_gemm_requires_enough_work_per_batch() {
        assert!(!should_use_standard_batched_gemm(4096, 2, 2, 2));
        assert!(should_use_standard_batched_gemm(4, 16, 16, 16));
    }

    #[test]
    fn test_gemm_batched_internal_standard_f32_matches_column_major_batches() {
        let cpu = Cpu;
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0];

        let c = cpu.gemm_batched_internal::<Standard<f32>>(&a, 2, 2, 2, &b, 2);

        assert_eq!(c, vec![7.0, 10.0, 15.0, 22.0, 5.0, 6.0, 7.0, 8.0]);
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

    // ------------------------------------------------------------------
    // Layout-bug repro for `try_tropical_gemm` (see issue).
    //
    // Same pattern as `test_tropical_gemm_optimized_maxplus`: call both the
    // dispatch path (which goes through `try_tropical_gemm` when
    // `tropical-kernels` is on) and the hand-written column-major oracle
    // `generic_gemm`, assert element-wise equality.
    //
    // The existing test uses m = k = n = 64 AND `a == b` bytewise. Under
    // those conditions the row-major vs column-major transpose algebraically
    // cancels out, so the byte-compare happens to pass. The tests below break
    // either condition and expose the bug.
    // ------------------------------------------------------------------

    #[cfg(feature = "tropical-kernels")]
    #[test]
    fn layout_bug_repro_nonsquare_m_ne_n() {
        let cpu = Cpu;
        let (m, k, n) = (3usize, 4, 5);
        let a: Vec<f64> = (0..m * k).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.25).collect();
        let c_opt = cpu.gemm_internal::<MaxPlus<f64>>(&a, m, k, &b, n);
        let c_gen = generic_gemm::<MaxPlus<f64>>(&a, m, k, &b, n);
        eprintln!("m={m} k={k} n={n}");
        eprintln!("dispatch (SIMD path): {:?}", c_opt);
        eprintln!("oracle   (generic)  : {:?}", c_gen);
        for (i, (o, g)) in c_opt.iter().zip(c_gen.iter()).enumerate() {
            assert!(
                (o - g).abs() < 1e-9,
                "m={m} k={k} n={n}: mismatch at idx {i}: dispatch={o}, oracle={g}",
            );
        }
    }

    /// Regression test for the `gemm_batched_internal` dispatch gap: batched
    /// MaxPlus GEMM must agree element-by-element with generic_gemm-per-batch,
    /// across a non-square shape (which is where the layout bug hides).
    #[cfg(feature = "tropical-kernels")]
    #[test]
    fn batched_dispatch_matches_generic_nonsquare() {
        let cpu = Cpu;
        let (batch_size, m, k, n) = (3usize, 3, 4, 5);
        let a: Vec<f64> = (0..batch_size * m * k).map(|i| (i as f64) * 0.5).collect();
        let b: Vec<f64> = (0..batch_size * k * n)
            .map(|i| (i as f64) * 0.25 + 0.1)
            .collect();

        let c_batched = cpu.gemm_batched_internal::<MaxPlus<f64>>(&a, batch_size, m, k, &b, n);

        // Oracle: concatenated generic_gemm per batch.
        let mut c_ref = Vec::with_capacity(batch_size * m * n);
        for batch in 0..batch_size {
            let a_slice = &a[batch * m * k..(batch + 1) * m * k];
            let b_slice = &b[batch * k * n..(batch + 1) * k * n];
            c_ref.extend(generic_gemm::<MaxPlus<f64>>(a_slice, m, k, b_slice, n));
        }

        for (i, (o, g)) in c_batched.iter().zip(c_ref.iter()).enumerate() {
            assert!(
                (o - g).abs() < 1e-9,
                "batched dispatch diverges from generic at idx {i}: {o} vs {g}",
            );
        }
    }

    #[cfg(feature = "tropical-kernels")]
    #[test]
    fn layout_bug_repro_nonsquare_with_neg_inf() {
        let cpu = Cpu;
        let (m, k, n) = (3usize, 4, 5);
        let mut a: Vec<f64> = (0..m * k).map(|i| i as f64).collect();
        a[2 * m + 1] = f64::NEG_INFINITY;
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.25).collect();
        let c_opt = cpu.gemm_internal::<MaxPlus<f64>>(&a, m, k, &b, n);
        let c_gen = generic_gemm::<MaxPlus<f64>>(&a, m, k, &b, n);
        eprintln!("dispatch: {:?}", c_opt);
        eprintln!("oracle  : {:?}", c_gen);
        let eq = |o: f64, g: f64| {
            (o.is_infinite() && g.is_infinite() && o.is_sign_negative() == g.is_sign_negative())
                || (o - g).abs() < 1e-9
        };
        for (i, (o, g)) in c_opt.iter().zip(c_gen.iter()).enumerate() {
            assert!(eq(*o, *g), "with -inf: mismatch at idx {i}: {o} vs {g}");
        }
    }
}
