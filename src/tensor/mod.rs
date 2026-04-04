//! Stride-based tensor type with zero-copy views.
//!
//! The [`Tensor`] type supports:
//! - Zero-copy `permute` and `reshape` operations
//! - Automatic contiguous copy when needed for GEMM
//! - Generic over algebra and backend

mod ops;
mod view;

use std::sync::Arc;

use crate::algebra::{Algebra, Scalar};
use crate::backend::{Backend, Storage};

pub use view::TensorView;

/// A multi-dimensional tensor with stride-based layout.
///
/// Tensors support zero-copy view operations (permute, reshape) and
/// automatically make data contiguous when needed for operations like GEMM.
///
/// # Type Parameters
///
/// * `T` - The scalar element type (f32, f64, etc.)
/// * `B` - The backend type (Cpu, Cuda)
///
/// # Example
///
/// ```rust
/// use omeinsum::{Tensor, Cpu};
///
/// let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
/// let b = a.permute(&[1, 0]);  // Zero-copy transpose
/// let c = b.contiguous();      // Make contiguous copy
/// ```
#[derive(Clone)]
pub struct Tensor<T: Scalar, B: Backend> {
    /// Shared storage (reference counted)
    storage: Arc<B::Storage<T>>,

    /// Shape of this view
    shape: Vec<usize>,

    /// Strides for each dimension (in elements)
    strides: Vec<usize>,

    /// Offset into storage
    offset: usize,

    /// Backend instance
    backend: B,
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a tensor from data with the given shape.
    ///
    /// Data is assumed to be in column-major (Fortran) order.
    pub fn from_data(data: &[T], shape: &[usize]) -> Self
    where
        B: Default,
    {
        Self::from_data_with_backend(data, shape, B::default())
    }

    /// Create a tensor from data with explicit backend.
    pub fn from_data_with_backend(data: &[T], shape: &[usize], backend: B) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            numel
        );

        let storage = backend.from_slice(data);
        let strides = compute_contiguous_strides(shape);

        Self {
            storage: Arc::new(storage),
            shape: shape.to_vec(),
            strides,
            offset: 0,
            backend,
        }
    }

    /// Create a zero-filled tensor.
    pub fn zeros(shape: &[usize]) -> Self
    where
        B: Default,
    {
        Self::zeros_with_backend(shape, B::default())
    }

    /// Create a zero-filled tensor with explicit backend.
    pub fn zeros_with_backend(shape: &[usize], backend: B) -> Self {
        let numel: usize = shape.iter().product();
        let storage = backend.alloc(numel);
        let strides = compute_contiguous_strides(shape);

        Self {
            storage: Arc::new(storage),
            shape: shape.to_vec(),
            strides,
            offset: 0,
            backend,
        }
    }

    /// Create a tensor from storage with given shape.
    ///
    /// The storage must be contiguous and have exactly `shape.iter().product()` elements.
    pub fn from_storage(storage: B::Storage<T>, shape: &[usize], backend: B) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            storage.len(),
            numel,
            "Storage length {} doesn't match shape {:?} (expected {})",
            storage.len(),
            shape,
            numel
        );

        let strides = compute_contiguous_strides(shape);

        Self {
            storage: Arc::new(storage),
            shape: shape.to_vec(),
            strides,
            offset: 0,
            backend,
        }
    }

    /// Get a reference to the underlying storage.
    ///
    /// Returns `Some(&storage)` only if the tensor is contiguous and has no offset.
    /// For non-contiguous tensors, call `contiguous()` first.
    pub fn storage(&self) -> Option<&B::Storage<T>> {
        if self.is_contiguous() {
            Some(self.storage.as_ref())
        } else {
            None
        }
    }

    // ========================================================================
    // Metadata
    // ========================================================================

    /// Get the shape of the tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides of the tensor.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the backend.
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Check if the tensor is contiguous in memory (row-major).
    pub fn is_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        let expected = compute_contiguous_strides(&self.shape);
        self.strides == expected
    }

    // ========================================================================
    // Data Access
    // ========================================================================

    /// Copy all data to a Vec.
    pub fn to_vec(&self) -> Vec<T> {
        if self.is_contiguous() {
            self.storage.to_vec()
        } else {
            self.contiguous().storage.to_vec()
        }
    }

    /// Get underlying storage (only if contiguous).
    pub fn as_slice(&self) -> Option<&[T]>
    where
        B::Storage<T>: AsRef<[T]>,
    {
        if self.is_contiguous() {
            Some(self.storage.as_ref().as_ref())
        } else {
            None
        }
    }

    /// Get element at linear index (column-major).
    ///
    /// This is an O(ndim) operation that directly accesses storage without
    /// allocating memory. The linear index is interpreted in column-major order.
    ///
    /// # Arguments
    ///
    /// * `index` - Linear index into the flattened tensor (column-major order)
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    ///
    /// # Example
    ///
    /// ```rust
    /// use omeinsum::{Tensor, Cpu};
    ///
    /// let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// assert_eq!(t.get(0), 1.0);
    /// assert_eq!(t.get(3), 4.0);
    /// ```
    pub fn get(&self, index: usize) -> T {
        let numel = self.numel();
        assert!(
            index < numel,
            "Index {} out of bounds for tensor with {} elements (shape {:?})",
            index,
            numel,
            self.shape
        );

        // Convert linear index to multi-dimensional coordinates (column-major)
        // Column-major: first dimension varies fastest
        let mut remaining = index;
        let mut storage_offset = self.offset;

        for dim in 0..self.ndim() {
            let coord = remaining % self.shape[dim];
            remaining /= self.shape[dim];
            storage_offset += coord * self.strides[dim];
        }

        self.storage.get(storage_offset)
    }

    // ========================================================================
    // View Operations (zero-copy)
    // ========================================================================

    /// Permute dimensions (zero-copy).
    ///
    /// # Example
    ///
    /// ```rust
    /// use omeinsum::{Tensor, Cpu};
    ///
    /// let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    /// let a = Tensor::<f32, Cpu>::from_data(&data, &[2, 3, 4]);
    /// let b = a.permute(&[2, 0, 1]);  // Shape becomes [4, 2, 3]
    /// assert_eq!(b.shape(), &[4, 2, 3]);
    /// ```
    pub fn permute(&self, axes: &[usize]) -> Self {
        assert_eq!(
            axes.len(),
            self.ndim(),
            "Permutation axes length {} doesn't match ndim {}",
            axes.len(),
            self.ndim()
        );

        // Check axes are valid and unique
        let mut seen = vec![false; self.ndim()];
        for &ax in axes {
            assert!(
                ax < self.ndim(),
                "Axis {} out of range for ndim {}",
                ax,
                self.ndim()
            );
            assert!(!seen[ax], "Duplicate axis {} in permutation", ax);
            seen[ax] = true;
        }

        let new_shape: Vec<usize> = axes.iter().map(|&i| self.shape[i]).collect();
        let new_strides: Vec<usize> = axes.iter().map(|&i| self.strides[i]).collect();

        Self {
            storage: Arc::clone(&self.storage),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            backend: self.backend.clone(),
        }
    }

    /// Transpose (2D shorthand for permute).
    pub fn t(&self) -> Self {
        assert_eq!(
            self.ndim(),
            2,
            "transpose requires 2D tensor, got {}D",
            self.ndim()
        );
        self.permute(&[1, 0])
    }

    /// Reshape to a new shape (zero-copy if contiguous).
    ///
    /// # Example
    ///
    /// ```rust
    /// use omeinsum::{Tensor, Cpu};
    ///
    /// let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    /// let b = a.reshape(&[6]);      // Flatten
    /// let c = a.reshape(&[3, 2]);   // Different shape, same data
    /// assert_eq!(b.shape(), &[6]);
    /// assert_eq!(c.shape(), &[3, 2]);
    /// ```
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let old_numel: usize = self.shape.iter().product();
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            old_numel, new_numel,
            "Cannot reshape from {:?} ({} elements) to {:?} ({} elements)",
            self.shape, old_numel, new_shape, new_numel
        );

        if self.is_contiguous() {
            // Fast path: just update shape and strides
            Self {
                storage: Arc::clone(&self.storage),
                shape: new_shape.to_vec(),
                strides: compute_contiguous_strides(new_shape),
                offset: self.offset,
                backend: self.backend.clone(),
            }
        } else {
            // Must make contiguous first
            self.contiguous().reshape(new_shape)
        }
    }

    /// Make tensor contiguous in memory.
    ///
    /// If already contiguous, returns a clone (shared storage).
    /// Otherwise, copies data to a new contiguous buffer.
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            self.clone()
        } else {
            let storage =
                self.backend
                    .copy_strided(&self.storage, &self.shape, &self.strides, self.offset);
            Self {
                storage: Arc::new(storage),
                shape: self.shape.clone(),
                strides: compute_contiguous_strides(&self.shape),
                offset: 0,
                backend: self.backend.clone(),
            }
        }
    }

    // ========================================================================
    // Reduction Operations
    // ========================================================================

    /// Sum all elements using the algebra's addition.
    ///
    /// # Type Parameters
    ///
    /// * `A` - The algebra to use for summation
    ///
    /// # Example
    ///
    /// ```rust
    /// use omeinsum::{Tensor, Cpu, Standard};
    ///
    /// let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// let sum = t.sum::<Standard<f32>>();
    /// assert_eq!(sum, 10.0);
    /// ```
    pub fn sum<A: Algebra<Scalar = T>>(&self) -> T {
        let data = self.to_vec();
        let mut acc = A::zero();
        for val in data {
            acc = acc.add(A::from_scalar(val));
        }
        acc.to_scalar()
    }

    /// Sum along a specific axis using the algebra's addition.
    ///
    /// The result has one fewer dimension than the input.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to sum over
    ///
    /// # Panics
    ///
    /// Panics if axis is out of bounds.
    ///
    /// # Example
    ///
    /// ```rust
    /// use omeinsum::{Tensor, Cpu, Standard};
    ///
    /// // Column-major: data [1, 2, 3, 4] with shape [2, 2] represents:
    /// // [[1, 3],
    /// //  [2, 4]]
    /// let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// // Sum over axis 1 (columns): [1+3, 2+4] = [4, 6]
    /// let result = t.sum_axis::<Standard<f32>>(1);
    /// assert_eq!(result.to_vec(), vec![4.0, 6.0]);
    /// ```
    pub fn sum_axis<A: Algebra<Scalar = T>>(&self, axis: usize) -> Self
    where
        B: Default,
    {
        assert!(
            axis < self.ndim(),
            "Axis {} out of bounds for {}D tensor",
            axis,
            self.ndim()
        );

        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(axis);

        // Handle reduction to scalar
        if new_shape.is_empty() {
            let sum = self.sum::<A>();
            return Self::from_data(&[sum], &[1]);
        }

        let data = self.to_vec();
        let output_strides = compute_contiguous_strides(&new_shape);
        let _axis_size = self.shape[axis];

        // Compute output size
        let output_numel: usize = new_shape.iter().product();
        let mut result = vec![A::zero(); output_numel];

        // Iterate over all elements in the input
        for (flat_idx, &val) in data.iter().enumerate() {
            // Convert flat index to multi-dimensional coordinates (column-major)
            let mut coords: Vec<usize> = vec![0; self.ndim()];
            let mut remaining = flat_idx;
            for (dim, coord) in coords.iter_mut().enumerate() {
                *coord = remaining % self.shape[dim];
                remaining /= self.shape[dim];
            }

            // Build output coordinates by removing the summed axis
            let out_coords: Vec<usize> = coords
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &c)| c)
                .collect();

            // Convert output coordinates to flat index (column-major)
            let mut out_flat_idx = 0;
            for (i, &coord) in out_coords.iter().enumerate() {
                out_flat_idx += coord * output_strides[i];
            }

            result[out_flat_idx] = result[out_flat_idx].add(A::from_scalar(val));
        }

        let result_data: Vec<T> = result.into_iter().map(|v| v.to_scalar()).collect();
        Self::from_data(&result_data, &new_shape)
    }

    /// Extract diagonal elements from a 2D tensor.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not 2D or not square.
    ///
    /// # Example
    ///
    /// ```rust
    /// use omeinsum::{Tensor, Cpu};
    ///
    /// let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// let diag = t.diagonal();
    /// assert_eq!(diag.to_vec(), vec![1.0, 4.0]);
    /// ```
    pub fn diagonal(&self) -> Self
    where
        B: Default,
    {
        assert_eq!(
            self.ndim(),
            2,
            "diagonal requires 2D tensor, got {}D",
            self.ndim()
        );
        assert_eq!(
            self.shape[0], self.shape[1],
            "diagonal requires square tensor, got {:?}",
            self.shape
        );

        let n = self.shape[0];
        let data = self.to_vec();
        let diag: Vec<T> = (0..n).map(|i| data[i * n + i]).collect();

        Self::from_data(&diag, &[n])
    }
}

/// Compute contiguous strides for column-major (Fortran) layout.
///
/// For shape [m, n], returns strides [1, m] (first dimension is contiguous).
pub fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }

    let mut strides = vec![1; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1];
    }
    strides
}

impl<T: Scalar, B: Backend> std::fmt::Debug for Tensor<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .field("contiguous", &self.is_contiguous())
            .field("backend", &B::name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;

    #[test]
    fn test_tensor_creation() {
        // Column-major: data [1,2,3,4,5,6] for shape [2,3] represents:
        // [[1, 3, 5],
        //  [2, 4, 6]]
        // Strides for column-major [2, 3] are [1, 2]
        let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.strides(), &[1, 2]); // Column-major strides
        assert!(t.is_contiguous());
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_permute() {
        // Column-major: data [1,2,3,4,5,6] for shape [2,3] represents:
        // [[1, 3, 5],
        //  [2, 4, 6]]
        let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let p = t.permute(&[1, 0]);

        assert_eq!(p.shape(), &[3, 2]);
        assert_eq!(p.strides(), &[2, 1]); // Permuted strides
        assert!(!p.is_contiguous());

        // After making contiguous, data should be transposed
        // Transposed matrix in column-major:
        // [[1, 2],
        //  [3, 4],
        //  [5, 6]] -> column-major data: [1, 3, 5, 2, 4, 6]
        let c = p.contiguous();
        assert!(c.is_contiguous());
        assert_eq!(c.to_vec(), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let r = t.reshape(&[3, 2]);

        assert_eq!(r.shape(), &[3, 2]);
        assert!(r.is_contiguous());
        assert_eq!(r.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_permute_then_reshape() {
        // Column-major: data [1,2,3,4,5,6] for shape [2,3]
        let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let p = t.permute(&[1, 0]); // [3, 2], non-contiguous
        let r = p.reshape(&[6]); // Must make contiguous first

        assert_eq!(r.shape(), &[6]);
        assert!(r.is_contiguous());
        // Transposed and flattened in column-major: [1, 3, 5, 2, 4, 6]
        assert_eq!(r.to_vec(), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_sum() {
        use crate::algebra::Standard;

        let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let sum = t.sum::<Standard<f32>>();
        assert_eq!(sum, 10.0);
    }

    #[test]
    fn test_sum_axis() {
        use crate::algebra::Standard;

        // Column-major: data [1, 2, 3, 4] for shape [2, 2] represents:
        // [[1, 3],
        //  [2, 4]]
        let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        // Sum over axis 1 (columns): [1+3, 2+4] = [4, 6]
        let sum_cols = t.sum_axis::<Standard<f32>>(1);
        assert_eq!(sum_cols.shape(), &[2]);
        assert_eq!(sum_cols.to_vec(), vec![4.0, 6.0]);

        // Sum over axis 0 (rows): [1+2, 3+4] = [3, 7]
        let sum_rows = t.sum_axis::<Standard<f32>>(0);
        assert_eq!(sum_rows.shape(), &[2]);
        assert_eq!(sum_rows.to_vec(), vec![3.0, 7.0]);
    }

    #[test]
    fn test_diagonal() {
        // Matrix: [[1, 2], [3, 4]]
        let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let diag = t.diagonal();

        assert_eq!(diag.shape(), &[2]);
        assert_eq!(diag.to_vec(), vec![1.0, 4.0]);
    }

    #[test]
    fn test_diagonal_3x3() {
        // Matrix: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        let t =
            Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        let diag = t.diagonal();

        assert_eq!(diag.shape(), &[3]);
        assert_eq!(diag.to_vec(), vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn test_get() {
        // Column-major: data [1, 2, 3, 4, 5, 6] for shape [2, 3] represents:
        // [[1, 3, 5],
        //  [2, 4, 6]]
        let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        // Test accessing each element by linear index
        assert_eq!(t.get(0), 1.0);
        assert_eq!(t.get(1), 2.0);
        assert_eq!(t.get(2), 3.0);
        assert_eq!(t.get(3), 4.0);
        assert_eq!(t.get(4), 5.0);
        assert_eq!(t.get(5), 6.0);
    }

    #[test]
    fn test_get_permuted() {
        // Test that get works correctly on permuted (non-contiguous) tensors
        let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let p = t.permute(&[1, 0]); // Shape becomes [3, 2], non-contiguous

        // After transpose, column-major data should be [1, 3, 5, 2, 4, 6]
        assert_eq!(p.get(0), 1.0);
        assert_eq!(p.get(1), 3.0);
        assert_eq!(p.get(2), 5.0);
        assert_eq!(p.get(3), 2.0);
        assert_eq!(p.get(4), 4.0);
        assert_eq!(p.get(5), 6.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_get_out_of_bounds() {
        let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let _ = t.get(4); // Index 4 is out of bounds for 4-element tensor
    }

    #[test]
    fn test_get_3d_tensor() {
        // Test get on a 3D tensor to ensure multi-dimensional indexing works
        // Shape [2, 3, 2], 12 elements
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let t = Tensor::<f32, Cpu>::from_data(&data, &[2, 3, 2]);

        // In column-major, elements are ordered by first dim varying fastest
        for (i, &expected) in data.iter().enumerate() {
            assert_eq!(t.get(i), expected);
        }
    }
}
