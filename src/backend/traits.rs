//! Backend trait definitions.

use crate::algebra::{Algebra, Scalar};

/// Storage trait for tensor data.
///
/// Abstracts over different storage backends (CPU memory, GPU memory).
pub trait Storage<T: Scalar>: Clone + Send + Sync + Sized {
    /// Number of elements in storage.
    fn len(&self) -> usize;

    /// Check if storage is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get element at index (may be slow for GPU).
    fn get(&self, index: usize) -> T;

    /// Set element at index (may be slow for GPU).
    fn set(&mut self, index: usize, value: T);

    /// Copy all data to a Vec (downloads from GPU if needed).
    fn to_vec(&self) -> Vec<T>;

    /// Create storage from slice.
    fn from_slice(data: &[T]) -> Self;

    /// Create zero-initialized storage.
    fn zeros(len: usize) -> Self;
}

/// Marker trait for scalar types supported by a specific backend.
///
/// This enables compile-time checking that a scalar type is supported
/// by a particular backend (e.g., CUDA only supports f32/f64/complex).
pub trait BackendScalar<B: Backend>: Scalar {}

/// Backend trait for tensor execution.
///
/// Defines how tensor operations are executed on different hardware.
pub trait Backend: Clone + Send + Sync + 'static {
    /// Storage type for this backend.
    type Storage<T: Scalar>: Storage<T>;

    /// Backend name for debugging.
    fn name() -> &'static str;

    /// Synchronize all pending operations.
    fn synchronize(&self);

    /// Allocate storage.
    fn alloc<T: Scalar>(&self, len: usize) -> Self::Storage<T>;

    /// Create storage from slice.
    #[allow(clippy::wrong_self_convention)]
    fn from_slice<T: Scalar>(&self, data: &[T]) -> Self::Storage<T>;

    /// Copy strided data to contiguous storage.
    ///
    /// This is the core operation for making non-contiguous tensors contiguous.
    fn copy_strided<T: Scalar>(
        &self,
        src: &Self::Storage<T>,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Self::Storage<T>;

    /// Binary tensor contraction.
    ///
    /// Computes a generalized tensor contraction: `C[modes_c] = Σ A[modes_a] ⊗ B[modes_b]`
    /// where the sum (using semiring addition) is over indices appearing in both A and B
    /// but not in the output C.
    ///
    /// # Mode Labels
    ///
    /// Each mode (dimension) of the input tensors is labeled with a unique integer identifier.
    /// These labels determine how the contraction is performed:
    ///
    /// - **Contracted indices**: Labels appearing in both `modes_a` and `modes_b` but NOT in
    ///   `modes_c`. These dimensions are summed over (reduced).
    /// - **Free indices from A**: Labels appearing only in `modes_a`. These appear in the output.
    /// - **Free indices from B**: Labels appearing only in `modes_b`. These appear in the output.
    /// - **Batch indices**: Labels appearing in `modes_a`, `modes_b`, AND `modes_c`.
    ///   These dimensions are preserved and processed in parallel.
    ///
    /// # Arguments
    ///
    /// * `a` - Storage for first input tensor
    /// * `shape_a` - Shape (dimensions) of tensor A
    /// * `strides_a` - Strides for tensor A (column-major, supports non-contiguous tensors)
    /// * `modes_a` - Mode labels for tensor A (length must equal `shape_a.len()`)
    /// * `b` - Storage for second input tensor
    /// * `shape_b` - Shape of tensor B
    /// * `strides_b` - Strides for tensor B
    /// * `modes_b` - Mode labels for tensor B (length must equal `shape_b.len()`)
    /// * `shape_c` - Shape of output tensor C (must be consistent with `modes_c`)
    /// * `modes_c` - Mode labels for output tensor C (determines output structure)
    ///
    /// # Returns
    ///
    /// Contiguous storage containing the result tensor with shape `shape_c`.
    ///
    /// # Examples
    ///
    /// ## Matrix multiplication: `C[i,k] = Σⱼ A[i,j] ⊗ B[j,k]`
    ///
    /// ```ignore
    /// // A is 2×3, B is 3×4 -> C is 2×4
    /// let c = backend.contract::<Standard<f32>>(
    ///     &a, &[2, 3], &[1, 2], &[0, 1],  // A[i=0, j=1], shape 2×3
    ///     &b, &[3, 4], &[1, 3], &[1, 2],  // B[j=1, k=2], shape 3×4
    ///     &[2, 4], &[0, 2],               // C[i=0, k=2], shape 2×4
    /// );
    /// ```
    ///
    /// ## Batched matrix multiplication: `C[b,i,k] = Σⱼ A[b,i,j] ⊗ B[b,j,k]`
    ///
    /// ```ignore
    /// // Batch size 8, A is 2×3, B is 3×4 -> C is 8×2×4
    /// let c = backend.contract::<Standard<f32>>(
    ///     &a, &[8, 2, 3], &[1, 8, 16], &[0, 1, 2],  // A[b=0, i=1, j=2]
    ///     &b, &[8, 3, 4], &[1, 8, 24], &[0, 2, 3],  // B[b=0, j=2, k=3]
    ///     &[8, 2, 4], &[0, 1, 3],                    // C[b=0, i=1, k=3]
    /// );
    /// ```
    ///
    /// ## Tropical shortest path (with min-plus semiring)
    ///
    /// ```ignore
    /// // Find shortest paths via matrix multiplication in (min,+) semiring
    /// let distances = backend.contract::<MinPlus<f32>>(
    ///     &graph_a, &[n, n], &[1, n], &[0, 1],
    ///     &graph_b, &[n, n], &[1, n], &[1, 2],
    ///     &[n, n], &[0, 2],
    /// );
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Mode labels have inconsistent sizes across tensors (e.g., if mode 1 has size 3
    ///   in A but size 4 in B)
    /// - The scalar type is not supported by the backend (compile-time check via `BackendScalar`)
    #[allow(clippy::too_many_arguments)]
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
        A::Scalar: BackendScalar<Self>;

    /// Contraction with argmax tracking for tropical backpropagation.
    ///
    /// This is identical to [`Backend::contract`] but additionally returns an argmax
    /// tensor that tracks which contracted index "won" the reduction at each output
    /// position. This is essential for tropical algebra backward passes where gradients
    /// are routed through the winning path only.
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - `result`: The contraction result (same as `contract`)
    /// - `argmax`: Tensor of `u32` indices indicating which contracted index won
    ///   at each output position
    ///
    /// # Use Cases
    ///
    /// - Tropical backpropagation (Viterbi, shortest path)
    /// - Computing attention patterns in max-pooling operations
    /// - Any semiring where addition is idempotent and gradient routing matters
    #[allow(clippy::too_many_arguments)]
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
        A::Scalar: BackendScalar<Self>;
}

// CPU supports all Scalar types
impl<T: Scalar> BackendScalar<crate::backend::Cpu> for T {}

// CUDA supports f32, f64, and CudaComplex types
#[cfg(feature = "cuda")]
impl BackendScalar<crate::backend::Cuda> for f32 {}
#[cfg(feature = "cuda")]
impl BackendScalar<crate::backend::Cuda> for f64 {}
#[cfg(feature = "cuda")]
impl BackendScalar<crate::backend::Cuda> for crate::backend::CudaComplex<f32> {}
#[cfg(feature = "cuda")]
impl BackendScalar<crate::backend::Cuda> for crate::backend::CudaComplex<f64> {}
