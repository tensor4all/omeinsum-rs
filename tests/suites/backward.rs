//! Tests for backward pass / gradient computation.
//!
//! These tests verify that gradients are correctly computed for tensor contractions
//! in both standard and tropical algebras.

use num_complex::Complex64;
use omeinsum::backend::Cpu;
use omeinsum::{einsum_with_grad, Standard, Tensor};

#[cfg(feature = "tropical")]
use omeinsum::MaxPlus;

#[test]
fn test_backward_matmul_standard() {
    // Column-major: [1,2,3,4] for shape [2,2] → [[1,3],[2,4]]
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // Column-major matmul: [[1,3],[2,4]] @ [[1,3],[2,4]] = [[7,15],[10,22]]
    // In column-major storage: [7, 10, 15, 22]
    assert_eq!(result.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);

    let grad_out = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<Standard<f32>>(&grad_out, &[&a, &b]);

    assert_eq!(grads.len(), 2);

    // For C = A @ B, with grad_out = ones [[1,1],[1,1]]:
    // B.T = [[1,2],[3,4]], grad_A = grad_out @ B.T = [[4,6],[4,6]]
    // In column-major: [4, 4, 6, 6]
    // A.T = [[1,2],[3,4]], grad_B = A.T @ grad_out = [[3,3],[7,7]]
    // In column-major: [3, 7, 3, 7]
    let grad_a = &grads[0];
    let grad_b = &grads[1];

    assert_eq!(grad_a.shape(), &[2, 2]);
    assert_eq!(grad_b.shape(), &[2, 2]);

    assert_eq!(grad_a.to_vec(), vec![4.0, 4.0, 6.0, 6.0]);
    assert_eq!(grad_b.to_vec(), vec![3.0, 7.0, 3.0, 7.0]);
}

#[test]
fn test_backward_matmul_identity() {
    // Test with identity matrix to verify gradients
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let identity = Tensor::<f32, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<Standard<f32>, _, _>(&[&a, &identity], &[&[0, 1], &[1, 2]], &[0, 2]);

    // A @ I = A
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

    let grad_out = Tensor::<f32, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<Standard<f32>>(&grad_out, &[&a, &identity]);

    assert_eq!(grads.len(), 2);
    // When grad_out = I, grad_A = I @ I^T = I
    assert_eq!(grads[0].to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_backward_matmul_rectangular() {
    // Test backward with non-square matrices
    // Column-major: [1,2,3,4,5,6] for shape [2,3] → [[1,3,5],[2,4,6]]
    // Column-major: [1,2,3,4,5,6] for shape [3,2] → [[1,4],[2,5],[3,6]]
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // [[1,3,5],[2,4,6]] @ [[1,4],[2,5],[3,6]] = [[22,49],[28,64]]
    // In column-major: [22, 28, 49, 64]
    assert_eq!(result.to_vec(), vec![22.0, 28.0, 49.0, 64.0]);

    let grad_out = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<Standard<f32>>(&grad_out, &[&a, &b]);

    assert_eq!(grads.len(), 2);

    let grad_a = &grads[0];
    let grad_b = &grads[1];

    // grad_A should be [2, 3], grad_B should be [3, 2]
    assert_eq!(grad_a.shape(), &[2, 3]);
    assert_eq!(grad_b.shape(), &[3, 2]);

    // grad_A = grad_out @ B.T = [[1,1],[1,1]] @ [[1,2,3],[4,5,6]] = [[5,7,9],[5,7,9]]
    // In column-major: [5, 5, 7, 7, 9, 9]
    assert_eq!(grad_a.to_vec(), vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);

    // grad_B = A.T @ grad_out = [[1,2],[3,4],[5,6]] @ [[1,1],[1,1]] = [[3,3],[7,7],[11,11]]
    // In column-major: [3, 7, 11, 3, 7, 11]
    assert_eq!(grad_b.to_vec(), vec![3.0, 7.0, 11.0, 3.0, 7.0, 11.0]);
}

#[test]
fn test_backward_matmul_ones() {
    // Test with all ones gradient
    // Column-major: [1,2,3,4] for shape [2,2] → [[1,3],[2,4]]
    // Column-major: [5,6,7,8] for shape [2,2] → [[5,7],[6,8]]
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // [[1,3],[2,4]] @ [[5,7],[6,8]] = [[23,31],[34,46]]
    // In column-major: [23, 34, 31, 46]
    assert_eq!(result.to_vec(), vec![23.0, 34.0, 31.0, 46.0]);

    let grad_out = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<Standard<f32>>(&grad_out, &[&a, &b]);

    assert_eq!(grads.len(), 2);

    // grad_A = grad_out @ B.T = [[1,1],[1,1]] @ [[5,6],[7,8]] = [[12,14],[12,14]]
    // In column-major: [12, 12, 14, 14]
    assert_eq!(grads[0].to_vec(), vec![12.0, 12.0, 14.0, 14.0]);

    // grad_B = A.T @ grad_out = [[1,2],[3,4]] @ [[1,1],[1,1]] = [[3,3],[7,7]]
    // In column-major: [3, 7, 3, 7]
    assert_eq!(grads[1].to_vec(), vec![3.0, 7.0, 3.0, 7.0]);
}

#[test]
fn test_backward_three_tensor_scalar_cycle_standard() {
    // A, B, C are diagonal matrices:
    // A = diag(2, 3), B = diag(5, 7), C = diag(11, 13)
    // einsum "ij,jk,ki->" = sum_i A[i,i] * B[i,i] * C[i,i]
    // = 2*5*11 + 3*7*13 = 383
    let a = Tensor::<f64, Cpu>::from_data(&[2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[5.0, 0.0, 0.0, 7.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[11.0, 0.0, 0.0, 13.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<Standard<f64>, _, _>(&[&a, &b, &c], &[&[0, 1], &[1, 2], &[2, 0]], &[]);

    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.to_vec(), vec![383.0]);

    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0], &[]);
    let grads = grad_fn.backward::<Standard<f64>>(&grad_out, &[&a, &b, &c]);

    assert_eq!(grads.len(), 3);
    assert_eq!(grads[0].shape(), &[2, 2]);
    assert_eq!(grads[1].shape(), &[2, 2]);
    assert_eq!(grads[2].shape(), &[2, 2]);

    // d/dA = diag(B .* C) = diag(55, 91)
    assert_eq!(grads[0].to_vec(), vec![55.0, 0.0, 0.0, 91.0]);
    // d/dB = diag(A .* C) = diag(22, 39)
    assert_eq!(grads[1].to_vec(), vec![22.0, 0.0, 0.0, 39.0]);
    // d/dC = diag(A .* B) = diag(10, 21)
    assert_eq!(grads[2].to_vec(), vec![10.0, 0.0, 0.0, 21.0]);
}

// ============================================================================
// Tropical algebra backward tests (require tropical feature)
// ============================================================================

#[cfg(feature = "tropical")]
#[test]
fn test_backward_matmul_tropical() {
    // Column-major: [1,2,3,4] for shape [2,2] → [[1,3],[2,4]]
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<MaxPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // Column-major A = [[1,3],[2,4]], B = [[1,3],[2,4]]
    // MaxPlus: C[i,k] = max_j (A[i,j] + B[j,k])
    // C[0,0] = max(1+1, 3+2) = 5, C[1,0] = max(2+1, 4+2) = 6
    // C[0,1] = max(1+3, 3+4) = 7, C[1,1] = max(2+3, 4+4) = 8
    // In column-major: [5, 6, 7, 8]
    assert_eq!(result.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);

    let grad_out = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<MaxPlus<f32>>(&grad_out, &[&a, &b]);

    assert_eq!(grads.len(), 2);

    let grad_a = &grads[0];
    let grad_b = &grads[1];

    assert_eq!(grad_a.shape(), &[2, 2]);
    assert_eq!(grad_b.shape(), &[2, 2]);

    // All argmax = 1, grad_A = [[0,2],[0,2]] in column-major: [0, 0, 2, 2]
    assert_eq!(grad_a.to_vec(), vec![0.0, 0.0, 2.0, 2.0]);

    // grad_B = [[0,0],[2,2]] in column-major: [0, 2, 0, 2]
    assert_eq!(grad_b.to_vec(), vec![0.0, 2.0, 0.0, 2.0]);
}

// Skip when tropical-kernels is enabled due to different iteration order in optimized path
#[cfg(all(feature = "tropical", not(feature = "tropical-kernels")))]
#[test]
fn test_backward_tropical_sparse_gradient() {
    // Test that tropical backward produces sparse gradients (only winners)
    // Column-major: [0,10,20,0] for shape [2,2] → A = [[0,20],[10,0]]
    let a = Tensor::<f32, Cpu>::from_data(&[0.0, 10.0, 20.0, 0.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<MaxPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // MaxPlus results - clear winners due to large differences in A
    // A = [[0,20],[10,0]], B = [[1,1],[1,1]]
    // C[0,0] = max(0+1, 20+1) = 21 (j=1 wins)
    // C[1,0] = max(10+1, 0+1) = 11 (j=0 wins)
    // C[0,1] = max(0+1, 20+1) = 21 (j=1 wins)
    // C[1,1] = max(10+1, 0+1) = 11 (j=0 wins)
    // In column-major: [21, 11, 21, 11]
    assert_eq!(result.to_vec(), vec![21.0, 11.0, 21.0, 11.0]);

    let grad_out = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<MaxPlus<f32>>(&grad_out, &[&a, &b]);

    // The gradient should be sparse - only the winning paths get gradients
    let grad_a = &grads[0];
    let _grad_b = &grads[1];

    // Winners: j=1 for C[0,0], C[0,1]; j=0 for C[1,0], C[1,1]
    // grad_A[0,0] = 0 (j=0 never won for row i=0)
    // grad_A[1,0] = grad_C[1,0] + grad_C[1,1] = 2 (j=0 won for row i=1)
    // grad_A[0,1] = grad_C[0,0] + grad_C[0,1] = 2 (j=1 won for row i=0)
    // grad_A[1,1] = 0 (j=1 never won for row i=1)
    // In column-major: [0, 2, 2, 0]
    let grad_a_vec = grad_a.to_vec();
    assert_eq!(grad_a_vec[0], 0.0); // Loser
    assert_eq!(grad_a_vec[1], 2.0); // Winner (twice)
    assert_eq!(grad_a_vec[2], 2.0); // Winner (twice)
    assert_eq!(grad_a_vec[3], 0.0); // Loser
}

// Skip when tropical-kernels is enabled due to different iteration order in optimized path
#[cfg(all(feature = "tropical", not(feature = "tropical-kernels")))]
#[test]
fn test_backward_tropical_different_winners() {
    // Test case where different elements have different winners
    // Column-major: [5,1,1,5] for shape [2,2] → A = [[5,1],[1,5]]
    // Column-major: [1,5,5,1] for shape [2,2] → B = [[1,5],[5,1]]
    let a = Tensor::<f32, Cpu>::from_data(&[5.0, 1.0, 1.0, 5.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 5.0, 5.0, 1.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<MaxPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // A = [[5,1],[1,5]], B = [[1,5],[5,1]]
    // C[0,0] = max(5+1, 1+5) = max(6, 6) = 6, j=0 wins (first on tie)
    // C[1,0] = max(1+1, 5+5) = max(2, 10) = 10, j=1 wins
    // C[0,1] = max(5+5, 1+1) = max(10, 2) = 10, j=0 wins
    // C[1,1] = max(1+5, 5+1) = max(6, 6) = 6, j=0 wins (first on tie)
    // In column-major: [6, 10, 10, 6]
    assert_eq!(result.to_vec(), vec![6.0, 10.0, 10.0, 6.0]);

    let grad_out = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<MaxPlus<f32>>(&grad_out, &[&a, &b]);

    let grad_a = &grads[0];
    let grad_b = &grads[1];

    assert_eq!(grad_a.shape(), &[2, 2]);
    assert_eq!(grad_b.shape(), &[2, 2]);

    // Winners: j=0 for C[0,0], C[0,1], C[1,1]; j=1 for C[1,0]
    // grad_A[0,0] = grad_C[0,0] + grad_C[0,1] = 2 (j=0 won for both)
    // grad_A[1,0] = grad_C[1,1] = 1 (j=0 won for C[1,1])
    // grad_A[0,1] = 0 (j=1 never won for row i=0)
    // grad_A[1,1] = grad_C[1,0] = 1 (j=1 won for C[1,0])
    // In column-major: [2, 1, 0, 1]
    assert_eq!(grad_a.to_vec(), vec![2.0, 1.0, 0.0, 1.0]);

    // grad_B[0,0] = grad_C[0,0] = 1 (j=0 won for C[0,0])
    // grad_B[1,0] = grad_C[1,0] = 1 (j=1 won for C[1,0])
    // grad_B[0,1] = grad_C[0,1] + grad_C[1,1] = 2 (j=0 won for both)
    // grad_B[1,1] = 0 (j=1 never won for col k=1)
    // In column-major: [1, 1, 2, 0]
    assert_eq!(grad_b.to_vec(), vec![1.0, 1.0, 2.0, 0.0]);
}

// ============================================================================
// f64 precision tests
// ============================================================================

#[test]
fn test_backward_matmul_f64() {
    // Test with f64 for higher precision
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // Same as f32 test but with f64
    assert_eq!(result.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);

    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<Standard<f64>>(&grad_out, &[&a, &b]);

    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].to_vec(), vec![4.0, 4.0, 6.0, 6.0]);
    assert_eq!(grads[1].to_vec(), vec![3.0, 7.0, 3.0, 7.0]);
}

#[test]
fn test_backward_large_values_f64() {
    // Test with large values where f64 precision matters
    let a = Tensor::<f64, Cpu>::from_data(&[1e10, 2e10, 3e10, 4e10], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // Verify result is computed correctly
    assert_eq!(result.shape(), &[2, 2]);

    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<Standard<f64>>(&grad_out, &[&a, &b]);

    assert_eq!(grads[0].shape(), &[2, 2]);
    assert_eq!(grads[1].shape(), &[2, 2]);
}

// ============================================================================
// Complex64 tests
// ============================================================================

#[test]
fn test_backward_matmul_complex64() {
    // Test gradient computation with complex numbers
    // A = [[1+i, 2], [3, 4-i]] in column-major: [1+i, 3, 2, 4-i]
    let a = Tensor::<Complex64, Cpu>::from_data(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, -1.0),
        ],
        &[2, 2],
    );
    // B = [[1, 0], [0, 1]] identity in column-major: [1, 0, 0, 1]
    let b = Tensor::<Complex64, Cpu>::from_data(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        &[2, 2],
    );

    let (result, grad_fn) =
        einsum_with_grad::<Standard<Complex64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // A @ I = A
    let result_vec = result.to_vec();
    assert!((result_vec[0].re - 1.0).abs() < 1e-10);
    assert!((result_vec[0].im - 1.0).abs() < 1e-10);
    assert!((result_vec[1].re - 3.0).abs() < 1e-10);
    assert!((result_vec[2].re - 2.0).abs() < 1e-10);
    assert!((result_vec[3].re - 4.0).abs() < 1e-10);
    assert!((result_vec[3].im - (-1.0)).abs() < 1e-10);

    let grad_out = Tensor::<Complex64, Cpu>::from_data(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        &[2, 2],
    );
    let grads = grad_fn.backward::<Standard<Complex64>>(&grad_out, &[&a, &b]);

    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].shape(), &[2, 2]);
    assert_eq!(grads[1].shape(), &[2, 2]);
}

#[test]
fn test_backward_complex64_nontrivial() {
    // Test with non-trivial complex numbers and gradients
    let a = Tensor::<Complex64, Cpu>::from_data(
        &[
            Complex64::new(1.0, 1.0),  // 1+i
            Complex64::new(2.0, -1.0), // 2-i
        ],
        &[2, 1],
    );
    let b = Tensor::<Complex64, Cpu>::from_data(
        &[
            Complex64::new(1.0, 0.0), // 1
            Complex64::new(0.0, 1.0), // i
        ],
        &[1, 2],
    );

    // Outer product: C[i,j] = A[i] * B[j]
    let (result, grad_fn) =
        einsum_with_grad::<Standard<Complex64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // C[0,0] = (1+i) * 1 = 1+i
    // C[1,0] = (2-i) * 1 = 2-i
    // C[0,1] = (1+i) * i = i + i² = i - 1 = -1+i
    // C[1,1] = (2-i) * i = 2i - i² = 2i + 1 = 1+2i
    let result_vec = result.to_vec();
    assert!((result_vec[0] - Complex64::new(1.0, 1.0)).norm() < 1e-10);
    assert!((result_vec[1] - Complex64::new(2.0, -1.0)).norm() < 1e-10);
    assert!((result_vec[2] - Complex64::new(-1.0, 1.0)).norm() < 1e-10);
    assert!((result_vec[3] - Complex64::new(1.0, 2.0)).norm() < 1e-10);

    let grad_out = Tensor::<Complex64, Cpu>::from_data(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        &[2, 2],
    );
    let grads = grad_fn.backward::<Standard<Complex64>>(&grad_out, &[&a, &b]);

    assert_eq!(grads[0].shape(), &[2, 1]);
    assert_eq!(grads[1].shape(), &[1, 2]);
}

// ============================================================================
// Complex64 Unary Gradient Tests
// ============================================================================

#[test]
fn test_backward_complex64_trace() {
    // Test trace gradient with complex numbers: ii->
    let a = Tensor::<Complex64, Cpu>::from_data(
        &[
            Complex64::new(1.0, 1.0),  // A[0,0] = 1+i
            Complex64::new(2.0, -1.0), // A[1,0] = 2-i
            Complex64::new(3.0, 0.5),  // A[0,1] = 3+0.5i
            Complex64::new(4.0, -0.5), // A[1,1] = 4-0.5i
        ],
        &[2, 2],
    );

    let (result, grad_fn) = einsum_with_grad::<Standard<Complex64>, _, _>(&[&a], &[&[0, 0]], &[]);

    // Trace = A[0,0] + A[1,1] = (1+i) + (4-0.5i) = 5 + 0.5i
    let result_vec = result.to_vec();
    assert!((result_vec[0] - Complex64::new(5.0, 0.5)).norm() < 1e-10);

    let grad_out = Tensor::<Complex64, Cpu>::from_data(&[Complex64::new(1.0, 0.0)], &[]);
    let grads = grad_fn.backward::<Standard<Complex64>>(&grad_out, &[&a]);

    // Gradient of trace is identity matrix
    assert_eq!(grads[0].shape(), &[2, 2]);
    let grad_vec = grads[0].to_vec();
    // Should be [[1,0],[0,1]] in col-major: [1, 0, 0, 1]
    assert!((grad_vec[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    assert!((grad_vec[1] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((grad_vec[2] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((grad_vec[3] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
}

#[test]
fn test_backward_complex64_sum() {
    // Test sum gradient with complex numbers: ij->
    let a = Tensor::<Complex64, Cpu>::from_data(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, -0.5),
        ],
        &[2, 2],
    );

    let (result, grad_fn) = einsum_with_grad::<Standard<Complex64>, _, _>(&[&a], &[&[0, 1]], &[]);

    // Sum = (1+i) + (2-i) + (3+0.5i) + (4-0.5i) = 10 + 0i
    let result_vec = result.to_vec();
    assert!((result_vec[0] - Complex64::new(10.0, 0.0)).norm() < 1e-10);

    let grad_out = Tensor::<Complex64, Cpu>::from_data(&[Complex64::new(1.0, 0.0)], &[]);
    let grads = grad_fn.backward::<Standard<Complex64>>(&grad_out, &[&a]);

    // Gradient of sum is all ones
    assert_eq!(grads[0].shape(), &[2, 2]);
    for &g in grads[0].to_vec().iter() {
        assert!((g - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }
}

#[test]
fn test_backward_complex64_transpose() {
    // Test transpose gradient with complex numbers: ij->ji
    let a = Tensor::<Complex64, Cpu>::from_data(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, -0.5),
        ],
        &[2, 2],
    );

    let (result, grad_fn) =
        einsum_with_grad::<Standard<Complex64>, _, _>(&[&a], &[&[0, 1]], &[1, 0]);

    assert_eq!(result.shape(), &[2, 2]);

    let grad_out = Tensor::<Complex64, Cpu>::from_data(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        &[2, 2],
    );
    let grads = grad_fn.backward::<Standard<Complex64>>(&grad_out, &[&a]);

    // Gradient of transpose with all-ones output grad is all ones (transposed back)
    assert_eq!(grads[0].shape(), &[2, 2]);
    for &g in grads[0].to_vec().iter() {
        assert!((g - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }
}

#[test]
fn test_backward_complex64_chain() {
    // Test gradient through chain: (A @ B) where both are complex
    let a = Tensor::<Complex64, Cpu>::from_data(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),  // i
            Complex64::new(0.0, -1.0), // -i
            Complex64::new(1.0, 0.0),
        ],
        &[2, 2],
    );
    let b = Tensor::<Complex64, Cpu>::from_data(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(1.0, -1.0),
        ],
        &[2, 2],
    );

    let (result, grad_fn) =
        einsum_with_grad::<Standard<Complex64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(result.shape(), &[2, 2]);

    let grad_out = Tensor::<Complex64, Cpu>::from_data(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        &[2, 2],
    );
    let grads = grad_fn.backward::<Standard<Complex64>>(&grad_out, &[&a, &b]);

    assert_eq!(grads[0].shape(), a.shape());
    assert_eq!(grads[1].shape(), b.shape());

    // Verify gradients are computed (non-trivial)
    let grad_a_nonzero = grads[0].to_vec().iter().any(|g| g.norm() > 1e-10);
    let grad_b_nonzero = grads[1].to_vec().iter().any(|g| g.norm() > 1e-10);
    assert!(grad_a_nonzero, "Gradient of A should be non-zero");
    assert!(grad_b_nonzero, "Gradient of B should be non-zero");
}

// ============================================================================
// Additional Pattern Coverage Tests
// ============================================================================

#[test]
fn test_backward_rectangular_matmul_variants() {
    // Test all matmul transpose combinations with rectangular matrices

    // A: 2x3, B: 3x4 -> C: 2x4 (standard)
    let a = Tensor::<f64, Cpu>::from_data(&(1..=6).map(|x| x as f64).collect::<Vec<_>>(), &[2, 3]);
    let b = Tensor::<f64, Cpu>::from_data(&(1..=12).map(|x| x as f64).collect::<Vec<_>>(), &[3, 4]);

    let (result, grad_fn) =
        einsum_with_grad::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(result.shape(), &[2, 4]);

    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0; 8], &[2, 4]);
    let grads = grad_fn.backward::<Standard<f64>>(&grad_out, &[&a, &b]);

    assert_eq!(grads[0].shape(), &[2, 3]);
    assert_eq!(grads[1].shape(), &[3, 4]);
}

#[test]
fn test_backward_all_matmul_transposes() {
    // Test ij,jk->ik, ij,kj->ik, ji,jk->ik, ji,kj->ik
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    // ij,jk->ik (standard)
    let (_, grad_fn1) =
        einsum_with_grad::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0; 4], &[2, 2]);
    let grads1 = grad_fn1.backward::<Standard<f64>>(&grad_out, &[&a, &b]);
    assert_eq!(grads1[0].shape(), &[2, 2]);
    assert_eq!(grads1[1].shape(), &[2, 2]);

    // ij,kj->ik (B transposed)
    let (_, grad_fn2) =
        einsum_with_grad::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[2, 1]], &[0, 2]);
    let grads2 = grad_fn2.backward::<Standard<f64>>(&grad_out, &[&a, &b]);
    assert_eq!(grads2[0].shape(), &[2, 2]);
    assert_eq!(grads2[1].shape(), &[2, 2]);

    // ji,jk->ik (A transposed)
    let (_, grad_fn3) =
        einsum_with_grad::<Standard<f64>, _, _>(&[&a, &b], &[&[1, 0], &[1, 2]], &[0, 2]);
    let grads3 = grad_fn3.backward::<Standard<f64>>(&grad_out, &[&a, &b]);
    assert_eq!(grads3[0].shape(), &[2, 2]);
    assert_eq!(grads3[1].shape(), &[2, 2]);

    // ji,kj->ik (both transposed)
    let (_, grad_fn4) =
        einsum_with_grad::<Standard<f64>, _, _>(&[&a, &b], &[&[1, 0], &[2, 1]], &[0, 2]);
    let grads4 = grad_fn4.backward::<Standard<f64>>(&grad_out, &[&a, &b]);
    assert_eq!(grads4[0].shape(), &[2, 2]);
    assert_eq!(grads4[1].shape(), &[2, 2]);
}

#[test]
fn test_backward_3tensor_chain() {
    // Test gradient through 3-tensor chain: A @ B @ C
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[2.0, 0.0, 0.0, 2.0], &[2, 2]);

    // First: A @ B
    let (ab, grad_fn_ab) =
        einsum_with_grad::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // Second: (A @ B) @ C
    let (result, grad_fn_abc) =
        einsum_with_grad::<Standard<f64>, _, _>(&[&ab, &c], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(result.shape(), &[2, 2]);

    // Backprop through second contraction
    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0; 4], &[2, 2]);
    let grads_abc = grad_fn_abc.backward::<Standard<f64>>(&grad_out, &[&ab, &c]);

    // Backprop through first contraction
    let grads_ab = grad_fn_ab.backward::<Standard<f64>>(&grads_abc[0], &[&a, &b]);

    assert_eq!(grads_ab[0].shape(), a.shape());
    assert_eq!(grads_ab[1].shape(), b.shape());
    assert_eq!(grads_abc[1].shape(), c.shape());
}
