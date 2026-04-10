//! Integration tests for omeinsum.
//!
//! These tests focus on matrix multiplication and tensor contraction operations
//! that are the primary supported use cases of omeinsum.

use omeinsum::backend::Cpu;
use omeinsum::{einsum, Standard, Tensor};

#[cfg(feature = "tropical")]
use omeinsum::{MaxPlus, MinPlus};

#[test]
fn test_matmul_standard() {
    // Basic matrix multiplication: A[i,j] x B[j,k] -> C[i,k]
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let c = einsum::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(c.shape(), &[2, 2]);
    // [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
    assert_eq!(c.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);
}

#[test]
fn test_matmul_identity() {
    // Multiplication with identity matrix
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let identity = Tensor::<f32, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let c = einsum::<Standard<f32>, _, _>(&[&a, &identity], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_matmul_chain_standard() {
    // A[i,j] x B[j,k] x C[k,l] -> D[i,l]
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // Identity
    let c = Tensor::<f64, Cpu>::from_data(&[2.0, 0.0, 0.0, 2.0], &[2, 2]); // 2*Identity

    let d = einsum::<Standard<f64>, _, _>(&[&a, &b, &c], &[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);

    // A @ I @ 2I = 2A
    assert_eq!(d.shape(), &[2, 2]);
    // The values should all be present (2*A)
    let d_vec = d.to_vec();
    let mut sorted = d_vec.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(sorted, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_matmul_rectangular() {
    // Non-square matrices: A[2,3] x B[3,2] -> C[2,2]
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

    let c = einsum::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(c.shape(), &[2, 2]);
    // [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
    assert_eq!(c.to_vec(), vec![22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_tensor_contraction_3d() {
    // A[i,j,k] x B[k,l] -> C[i,j,l]
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let c = einsum::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1, 2], &[2, 3]], &[0, 1, 3]);

    assert_eq!(c.shape(), &[2, 2, 2]);
    // Verify the contraction produces correct results
    // For each (i,j,l): sum over k of A[i,j,k] * B[k,l]
    // Column-major for [2,2,2]: strides [1,2,4], so A[i,j,k] at index i + 2j + 4k
    // A[0,0,0]=1, A[0,0,1]=5; B[0,0]=1, B[1,0]=2
    // C[0,0,0] = A[0,0,0]*B[0,0] + A[0,0,1]*B[1,0] = 1*1 + 5*2 = 11
    let c_vec = c.to_vec();
    assert_eq!(c_vec[0], 11.0);
}

#[test]
fn test_batch_matmul() {
    // Batch matmul: A[b,i,j] x B[b,j,k] -> C[b,i,k]
    // Column-major [2,2,2]: element [b,i,j] at position b + 2*i + 4*j
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[2, 2, 2]);

    let c = einsum::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1, 2], &[0, 2, 3]], &[0, 1, 3]);

    assert_eq!(c.shape(), &[2, 2, 2]);
    // Column-major interpretation:
    // A batch 0: [[1,5],[3,7]], A batch 1: [[2,6],[4,8]]
    // B batch 0: [[1,2],[0,0]], B batch 1: [[0,0],[1,2]]
    // C batch 0: [[1,2],[3,6]], C batch 1: [[6,12],[8,16]]
    // Column-major result: [1, 6, 3, 8, 2, 12, 6, 16]
    let c_vec = c.to_vec();
    assert_eq!(c_vec, vec![1.0, 6.0, 3.0, 8.0, 2.0, 12.0, 6.0, 16.0]);
}

#[test]
fn test_einsum_with_different_contraction_axes() {
    // A[i,j,k] x B[j,k,l] -> C[i,l] (contract over two axes)
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);

    let c = einsum::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1, 2], &[1, 2, 3]], &[0, 3]);

    assert_eq!(c.shape(), &[2, 2]);
}

#[test]
fn test_matmul_f64() {
    // Test with f64 precision
    // Column-major: [1,2,3,4] for shape [2,2] → [[1,3],[2,4]]
    // Column-major: [5,6,7,8] for shape [2,2] → [[5,7],[6,8]]
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let c = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(c.shape(), &[2, 2]);
    // [[1,3],[2,4]] @ [[5,7],[6,8]] = [[23,31],[34,46]]
    // In column-major: [23, 34, 31, 46]
    assert_eq!(c.to_vec(), vec![23.0, 34.0, 31.0, 46.0]);
}

#[test]
fn test_matmul_four_tensors() {
    // Chain of four matrices: A @ B @ C @ D
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // Identity
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let c = Tensor::<f32, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // Identity
    let d = Tensor::<f32, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // Identity

    let result = einsum::<Standard<f32>, _, _>(
        &[&a, &b, &c, &d],
        &[&[0, 1], &[1, 2], &[2, 3], &[3, 4]],
        &[0, 4],
    );

    assert_eq!(result.shape(), &[2, 2]);
    // I @ B @ I @ I = B (values should be 1, 2, 3, 4)
    let result_vec = result.to_vec();
    let mut sorted = result_vec.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(sorted, vec![1.0, 2.0, 3.0, 4.0]);
}

// ============================================================================
// Tropical algebra tests (require tropical feature)
// ============================================================================

#[cfg(feature = "tropical")]
#[test]
fn test_matmul_maxplus() {
    // MaxPlus: C[i,k] = max_j (A[i,j] + B[j,k])
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let c = einsum::<MaxPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(c.shape(), &[2, 2]);
    // C[0,0] = max(1+1, 2+3) = max(2, 5) = 5
    // C[0,1] = max(1+2, 2+4) = max(3, 6) = 6
    // C[1,0] = max(3+1, 4+3) = max(4, 7) = 7
    // C[1,1] = max(3+2, 4+4) = max(5, 8) = 8
    assert_eq!(c.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_matmul_minplus() {
    // MinPlus: C[i,k] = min_j (A[i,j] + B[j,k])
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let c = einsum::<MinPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(c.shape(), &[2, 2]);
    // C[0,0] = min(1+1, 2+3) = min(2, 5) = 2
    // C[0,1] = min(1+2, 2+4) = min(3, 6) = 3
    // C[1,0] = min(3+1, 4+3) = min(4, 7) = 4
    // C[1,1] = min(3+2, 4+4) = min(5, 8) = 5
    assert_eq!(c.to_vec(), vec![2.0, 3.0, 4.0, 5.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_matmul_chain_tropical() {
    // Three-matrix chain in tropical algebra
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[0.0, 0.0, 0.0, 0.0], &[2, 2]); // Tropical zero for +
    let c = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

    let d = einsum::<MaxPlus<f32>, _, _>(&[&a, &b, &c], &[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);

    assert_eq!(d.shape(), &[2, 2]);
}
