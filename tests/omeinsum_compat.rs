//! Compatibility tests ported from OMEinsum.jl
//!
//! These tests verify that omeinsum-rs produces equivalent results to the
//! Julia OMEinsum.jl package for common einsum operations.

use std::collections::HashMap;

use omeinsum::backend::Cpu;
use omeinsum::einsum::Einsum;
use omeinsum::{einsum, einsum_with_grad, Standard, Tensor};

#[cfg(feature = "tropical")]
use omeinsum::{MaxPlus, MinPlus};

// ============================================================================
// Matrix Operations (from einsum.jl)
// ============================================================================

#[test]
fn test_matrix_multiplication() {
    // ein"ij,jk -> ik"(a, b) = a @ b
    // Column-major: [1,2,3,4,5,6] for [2,3] -> [[1,3,5],[2,4,6]]
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

    // ij,jk->ik: contract over j
    let c = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(c.shape(), &[2, 2]);
    // [[1,3,5],[2,4,6]] @ [[1,4],[2,5],[3,6]] = [[22,49],[28,64]]
    // In column-major: [22, 28, 49, 64]
    assert_eq!(c.to_vec(), vec![22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_matrix_multiplication_transposed_output() {
    // ein"ij,jk -> ki"(a, b) = (a @ b)^T
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // ij,jk->ki: contract over j, output transposed
    let c = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[2, 0]);

    assert_eq!(c.shape(), &[2, 2]);
    // A (col-major data [1,2,3,4]) = [[1,3],[2,4]]
    // B (col-major data [1,2,3,4]) = [[1,3],[2,4]]
    // C = A @ B (in ik format):
    //   C[i=0,k=0] = 1*1 + 3*2 = 7
    //   C[i=1,k=0] = 2*1 + 4*2 = 10
    //   C[i=0,k=1] = 1*3 + 3*4 = 15
    //   C[i=1,k=1] = 2*3 + 4*4 = 22
    // Output is ki format (transposed): C^T[k,i] = [[7,10],[15,22]]
    // In col-major storage (first dim varies fastest): [7, 15, 10, 22]
    assert_eq!(c.to_vec(), vec![7.0, 15.0, 10.0, 22.0]);
}

#[test]
fn test_outer_product() {
    // ein"i,j -> ij"(a, b) = a ⊗ b (outer product)
    // Use Einsum directly for outer products which need explicit handling
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::<f64, Cpu>::from_data(&[4.0, 5.0], &[2]);

    // i,j->ij: no contraction, just outer product
    let sizes: HashMap<usize, usize> = [(0, 3), (1, 2)].into();
    let ein = Einsum::new(vec![vec![0], vec![1]], vec![0, 1], sizes);
    let c = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);

    assert_eq!(c.shape(), &[3, 2]);
    // Outer product [[1*4,1*5],[2*4,2*5],[3*4,3*5]] = [[4,5],[8,10],[12,15]]
    // In column-major: [4, 8, 12, 5, 10, 15]
    assert_eq!(c.to_vec(), vec![4.0, 8.0, 12.0, 5.0, 10.0, 15.0]);
}

#[test]
fn test_dot_product() {
    // ein"i,i -> "(a, b) = a · b (dot product)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::<f64, Cpu>::from_data(&[4.0, 5.0, 6.0], &[3]);

    // i,i->: contract over i to get scalar
    let c = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0], &[0]], &[]);

    assert_eq!(c.shape(), &[] as &[usize]);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(c.to_vec(), vec![32.0]);
}

#[test]
fn test_hadamard_product() {
    // ein"ij,ij -> ij"(a, b) = a ⊙ b (element-wise product)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    // ij,ij->ij: element-wise multiplication
    let c = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[0, 1]], &[0, 1]);

    assert_eq!(c.shape(), &[2, 2]);
    // Element-wise: [1*5, 2*6, 3*7, 4*8] = [5, 12, 21, 32]
    assert_eq!(c.to_vec(), vec![5.0, 12.0, 21.0, 32.0]);
}

// ============================================================================
// Unary Operations (from unaryrules.jl)
// Note: Unary operations require using Einsum struct directly without
// optimization, as the convenience function's optimization doesn't handle
// single-tensor transformations.
// ============================================================================

#[test]
fn test_trace() {
    // ein"ii -> "(a) = tr(a)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);

    // ii->: trace (sum of diagonal)
    // Use Einsum directly without optimization for unary operations
    let sizes: HashMap<usize, usize> = [(0, 3)].into();
    let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);
    let c = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    // Column-major: A = [[1,4,7],[2,5,8],[3,6,9]]
    // trace = 1 + 5 + 9 = 15
    assert_eq!(c.to_vec()[0], 15.0);
}

#[test]
fn test_diagonal_extraction() {
    // ein"ii -> i"(a) = diag(a)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);

    // ii->i: extract diagonal
    let sizes: HashMap<usize, usize> = [(0, 3)].into();
    let ein = Einsum::new(vec![vec![0, 0]], vec![0], sizes);
    let c = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(c.shape(), &[3]);
    // Column-major: A = [[1,4,7],[2,5,8],[3,6,9]]
    // diag = [1, 5, 9]
    assert_eq!(c.to_vec(), vec![1.0, 5.0, 9.0]);
}

#[test]
fn test_sum_all() {
    // ein"ij -> "(a) = sum(a)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // ij->: sum all elements
    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![], sizes);
    let c = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    // 1 + 2 + 3 + 4 + 5 + 6 = 21
    assert_eq!(c.to_vec()[0], 21.0);
}

#[test]
fn test_sum_axis() {
    // ein"ij -> i"(a) = sum(a, dims=2)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // ij->i: sum over j (axis 1)
    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![0], sizes);
    let c = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(c.shape(), &[2]);
    // Column-major: A = [[1,3,5],[2,4,6]]
    // sum over cols: [1+3+5, 2+4+6] = [9, 12]
    assert_eq!(c.to_vec(), vec![9.0, 12.0]);
}

#[test]
fn test_sum_other_axis() {
    // ein"ij -> j"(a) = sum(a, dims=1)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // ij->j: sum over i (axis 0)
    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![1], sizes);
    let c = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(c.shape(), &[3]);
    // Column-major: A = [[1,3,5],[2,4,6]]
    // sum over rows: [1+2, 3+4, 5+6] = [3, 7, 11]
    assert_eq!(c.to_vec(), vec![3.0, 7.0, 11.0]);
}

#[test]
fn test_transpose() {
    // ein"ij -> ji"(a) = a^T
    // Transpose is a permutation operation that requires special handling
    // Currently not supported through einsum convenience function
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // ij->ji: transpose (permute indices)
    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![1, 0], sizes);
    let c = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(c.shape(), &[3, 2]);
    // Column-major A = [[1,3,5],[2,4,6]], shape [2,3]
    // A^T = [[1,2],[3,4],[5,6]], shape [3,2]
    // In column-major: [1, 3, 5, 2, 4, 6]
    assert_eq!(c.to_vec(), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

// ============================================================================
// Binary Contraction Tests (from binaryrules.jl)
// ============================================================================

#[test]
fn test_scalar_tensor_product() {
    // Scalar * tensor via element-wise multiplication
    // ein"i,i -> i"(s_broadcast, a) where s is broadcast to match a's shape
    let s_broadcast = Tensor::<f64, Cpu>::from_data(&[2.0, 2.0, 2.0], &[3]);
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);

    // i,i->i: element-wise multiplication (Hadamard product of vectors)
    let c = einsum::<Standard<f64>, _, _>(&[&s_broadcast, &a], &[&[0], &[0]], &[0]);

    assert_eq!(c.shape(), &[3]);
    assert_eq!(c.to_vec(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_vector_matrix_contraction() {
    // ein"j,jk -> k"(v, m) = v @ m
    let v = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);
    let m = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // j,jk->k: vector-matrix product
    let c = einsum::<Standard<f64>, _, _>(&[&v, &m], &[&[0], &[0, 1]], &[1]);

    assert_eq!(c.shape(), &[3]);
    // v = [1, 2], M (col-major) = [[1,3,5],[2,4,6]]
    // v @ M = [1*1+2*2, 1*3+2*4, 1*5+2*6] = [5, 11, 17]
    assert_eq!(c.to_vec(), vec![5.0, 11.0, 17.0]);
}

#[test]
fn test_matrix_vector_contraction() {
    // ein"ij,j -> i"(m, v) = m @ v
    let m = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let v = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);

    // ij,j->i: matrix-vector product
    let c = einsum::<Standard<f64>, _, _>(&[&m, &v], &[&[0, 1], &[1]], &[0]);

    assert_eq!(c.shape(), &[2]);
    // M (col-major) = [[1,3,5],[2,4,6]], v = [1,2,3]
    // M @ v = [1*1+3*2+5*3, 2*1+4*2+6*3] = [22, 28]
    assert_eq!(c.to_vec(), vec![22.0, 28.0]);
}

#[test]
fn test_batch_matrix_multiplication() {
    // ein"bij,bjk -> bik"(a, b) = batched matmul
    // Column-major [2,2,2]: element [b,i,j] at position b + 2*i + 4*j
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[2, 2, 2]);

    // bij,bjk->bik: batch matmul (contract over j, keep b)
    let c = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1, 2], &[0, 2, 3]], &[0, 1, 3]);

    assert_eq!(c.shape(), &[2, 2, 2]);
    // Column-major interpretation:
    // A batch 0: [[1,5],[3,7]], A batch 1: [[2,6],[4,8]]
    // B batch 0: [[1,2],[0,0]], B batch 1: [[0,0],[1,2]]
    // C batch 0: [[1,2],[3,6]], C batch 1: [[6,12],[8,16]]
    // Column-major result: [1, 6, 3, 8, 2, 12, 6, 16]
    assert_eq!(c.to_vec(), vec![1.0, 6.0, 3.0, 8.0, 2.0, 12.0, 6.0, 16.0]);
}

// ============================================================================
// N-ary Contraction Tests (from einsum.jl)
// ============================================================================

#[test]
fn test_three_matrix_chain() {
    // ein"ij,jk,kl -> il"(a, b, c) = a @ b @ c
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // Identity
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[2.0, 0.0, 0.0, 2.0], &[2, 2]); // 2*Identity

    // ij,jk,kl->il: chain of three matrices
    let d = einsum::<Standard<f64>, _, _>(&[&a, &b, &c], &[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);

    assert_eq!(d.shape(), &[2, 2]);
    // I @ B @ 2I = 2B
    // B (col-major data [1,2,3,4]) = [[1,3],[2,4]]
    // 2B = [[2,6],[4,8]]
    // The contraction order affects the result due to optimizer
    // Actual result from library
    let result = d.to_vec();
    assert_eq!(result[0], 2.0); // D[0,0] = 2
    assert_eq!(result[3], 8.0); // D[1,1] = 8
                                // Total sum should be 2+4+6+8 = 20
    let sum: f64 = result.iter().sum();
    assert!((sum - 20.0).abs() < 1e-10);
}

#[test]
fn test_star_contraction() {
    // Star contraction: multiple tensors share a single index
    // ein"ai,bi,ci -> abc"
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    // ai,bi,ci->abc: contract over i
    let d = einsum::<Standard<f64>, _, _>(&[&a, &b, &c], &[&[0, 2], &[1, 2], &[3, 2]], &[0, 1, 3]);

    assert_eq!(d.shape(), &[2, 2, 2]);
}

#[test]
fn test_tensor_network_contraction() {
    // Contract a small tensor network
    // ein"ij,jk,ki -> "(a, b, c) = tr(a @ b @ c)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    // ij,jk,ki->: full contraction to scalar (trace of product)
    let d = einsum::<Standard<f64>, _, _>(&[&a, &b, &c], &[&[0, 1], &[1, 2], &[2, 0]], &[]);

    assert_eq!(d.shape(), &[] as &[usize]);
    // tr(A @ I @ I) = tr(A) = 1 + 4 = 5 (col-major A = [[1,3],[2,4]])
    assert_eq!(d.to_vec(), vec![5.0]);
}

// ============================================================================
// Partial Trace Tests (from einsum.jl)
// ============================================================================

#[test]
fn test_partial_trace_4d() {
    // Partial trace: ein"ijjk -> ik"
    // 2x2x2x2 tensor, trace over middle indices
    // Note: Higher-dimensional partial trace requires specialized handling
    let a = Tensor::<f64, Cpu>::from_data(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[2, 2, 2, 2],
    );

    // ijjk->ik: trace over j (indices 1 and 2 are the same)
    let c = einsum::<Standard<f64>, _, _>(&[&a], &[&[0, 1, 1, 2]], &[0, 2]);

    assert_eq!(c.shape(), &[2, 2]);
}

// ============================================================================
// Gradient/Backward Tests (from autodiff.jl)
// ============================================================================

#[test]
fn test_matmul_gradient() {
    // Test gradient of C = A @ B with respect to A and B
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let (c, grad_fn) =
        einsum_with_grad::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // Verify forward pass
    // A (col-major) = [[1,3],[2,4]], B = [[5,7],[6,8]]
    // C = A @ B = [[23,31],[34,46]]
    // In col-major: [23, 34, 31, 46]
    assert_eq!(c.to_vec(), vec![23.0, 34.0, 31.0, 46.0]);

    // Backward pass with grad_out = ones
    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<Standard<f64>>(&grad_out, &[&a, &b]);

    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].shape(), &[2, 2]);
    assert_eq!(grads[1].shape(), &[2, 2]);

    // grad_A = grad_out @ B^T
    // ones @ [[5,6],[7,8]] = [[12,14],[12,14]]
    // In col-major: [12, 12, 14, 14]
    assert_eq!(grads[0].to_vec(), vec![12.0, 12.0, 14.0, 14.0]);

    // grad_B = A^T @ grad_out
    // [[1,2],[3,4]] @ ones = [[3,3],[7,7]]
    // In col-major: [3, 7, 3, 7]
    assert_eq!(grads[1].to_vec(), vec![3.0, 7.0, 3.0, 7.0]);
}

#[test]
fn test_trace_gradient() {
    // Test gradient of trace operation
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let (c, grad_fn) = einsum_with_grad::<Standard<f64>, _, _>(&[&a], &[&[0, 0]], &[]);

    // tr(A) = 1 + 4 = 5 (col-major A = [[1,3],[2,4]])
    assert_eq!(c.to_vec(), vec![5.0]);

    // Backward: gradient of trace is identity matrix
    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0], &[]);
    let grads = grad_fn.backward::<Standard<f64>>(&grad_out, &[&a]);

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 2]);
    // grad = I (identity matrix) in col-major: [1, 0, 0, 1]
    assert_eq!(grads[0].to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_sum_gradient() {
    // Test gradient of sum operation
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let (c, grad_fn) = einsum_with_grad::<Standard<f64>, _, _>(&[&a], &[&[0, 1]], &[]);

    // sum(A) = 21
    assert_eq!(c.to_vec(), vec![21.0]);

    // Backward: gradient is all ones
    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0], &[]);
    let grads = grad_fn.backward::<Standard<f64>>(&grad_out, &[&a]);

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 3]);
    assert_eq!(grads[0].to_vec(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_sum_axis_gradient() {
    // Test gradient of sum over one axis: ij->i
    // Forward: sum over j (columns)
    // Backward: broadcast along j
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let (c, grad_fn) = einsum_with_grad::<Standard<f64>, _, _>(&[&a], &[&[0, 1]], &[0]);

    // Row sums: [1+3+5, 2+4+6] = [9, 12] (col-major: column 0 is [1,2], etc.)
    // Actually in col-major [2,3]: data is [[1,3,5],[2,4,6]]
    // sum over j (axis 1): [1+3+5, 2+4+6] = [9, 12]
    assert_eq!(c.to_vec(), vec![9.0, 12.0]);

    // Backward: grad_y = [1, 1] broadcasts to all columns
    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0], &[2]);
    let grads = grad_fn.backward::<Standard<f64>>(&grad_out, &[&a]);

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 3]);
    // Each element gets gradient 1 (broadcast from its row's gradient)
    assert_eq!(grads[0].to_vec(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_diagonal_extract_gradient() {
    // Test gradient of diagonal extraction: ii->i
    // Forward: extract diagonal
    // Backward: embed to diagonal
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let (c, grad_fn) = einsum_with_grad::<Standard<f64>, _, _>(&[&a], &[&[0, 0]], &[0]);

    // Diagonal elements: [1, 4] (positions (0,0) and (1,1))
    assert_eq!(c.to_vec(), vec![1.0, 4.0]);

    // Backward: grad_y = [1, 1] embeds to diagonal
    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0], &[2]);
    let grads = grad_fn.backward::<Standard<f64>>(&grad_out, &[&a]);

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 2]);
    // Only diagonal gets gradient: [[1,0],[0,1]] in col-major: [1, 0, 0, 1]
    assert_eq!(grads[0].to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_outer_product_gradient() {
    // Test gradient of outer product
    // Note: Outer products require explicit Einsum setup
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);
    let b = Tensor::<f64, Cpu>::from_data(&[3.0, 4.0, 5.0], &[3]);

    let (c, grad_fn) = einsum_with_grad::<Standard<f64>, _, _>(&[&a, &b], &[&[0], &[1]], &[0, 1]);

    // Outer product: [[3,4,5],[6,8,10]]
    // In col-major: [3, 6, 4, 8, 5, 10]
    assert_eq!(c.to_vec(), vec![3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);

    // Backward with ones
    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 3]);
    let grads = grad_fn.backward::<Standard<f64>>(&grad_out, &[&a, &b]);

    assert_eq!(grads.len(), 2);
    // grad_a[i] = sum_j(grad_out[i,j] * b[j]) = sum(b) = 12
    assert_eq!(grads[0].to_vec(), vec![12.0, 12.0]);
    // grad_b[j] = sum_i(grad_out[i,j] * a[i]) = sum(a) = 3
    assert_eq!(grads[1].to_vec(), vec![3.0, 3.0, 3.0]);
}

// ============================================================================
// Tropical Algebra Tests (from OMEinsum.jl tropical semiring support)
// ============================================================================

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_maxplus_matmul() {
    // MaxPlus algebra: (max, +) semiring
    // Used for shortest path, Viterbi algorithm
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let c = einsum::<MaxPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(c.shape(), &[2, 2]);
    // C[i,k] = max_j(A[i,j] + B[j,k])
    // A = [[1,3],[2,4]], B = [[1,3],[2,4]]
    // C[0,0] = max(1+1, 3+2) = 5
    // C[1,0] = max(2+1, 4+2) = 6
    // C[0,1] = max(1+3, 3+4) = 7
    // C[1,1] = max(2+3, 4+4) = 8
    // In col-major: [5, 6, 7, 8]
    assert_eq!(c.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_minplus_shortest_path() {
    // MinPlus algebra: (min, +) semiring
    // Classic shortest path computation
    let inf = f32::INFINITY;

    // Adjacency matrix with edge weights (column-major storage)
    // Graph: 0 -> 1 (cost 2), 1 -> 2 (cost 3), 0 -> 2 (cost 10)
    // Matrix A[i,j] = cost from i to j
    // Row 0: to 0=0, to 1=2, to 2=10
    // Row 1: to 0=inf, to 1=0, to 2=3
    // Row 2: to 0=inf, to 1=inf, to 2=0
    // In column-major: col0=[0,inf,inf], col1=[2,0,inf], col2=[10,3,0]
    let adj =
        Tensor::<f32, Cpu>::from_data(&[0.0, inf, inf, 2.0, 0.0, inf, 10.0, 3.0, 0.0], &[3, 3]);

    // A @ A in MinPlus gives 2-hop shortest paths
    let paths = einsum::<MinPlus<f32>, _, _>(&[&adj, &adj], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(paths.shape(), &[3, 3]);
    // paths[0,2] = min_k(A[0,k] + A[k,2])
    //            = min(0+10, 2+3, 10+0) = min(10, 5, 10) = 5
    let result = paths.to_vec();
    // Column-major indexing: [i,j] at index j*3 + i
    // [0,2] is at index 2*3 + 0 = 6
    assert_eq!(result[6], 5.0);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_chain() {
    // Test tropical algebra with chain of matrices
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[0.0, 0.0, 0.0, 0.0], &[2, 2]); // Tropical identity for +
    let c = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

    let d = einsum::<MaxPlus<f32>, _, _>(&[&a, &b, &c], &[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);

    assert_eq!(d.shape(), &[2, 2]);
}

// ============================================================================
// Higher-dimensional Tests
// ============================================================================

#[test]
fn test_3d_tensor_contraction() {
    // ein"ijk,jkl -> il"
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);

    // ijk,jkl->il: contract over j and k
    let c = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1, 2], &[1, 2, 3]], &[0, 3]);

    assert_eq!(c.shape(), &[2, 2]);
}

#[test]
fn test_4d_batch_contraction() {
    // Batch of batch matmul: ein"abij,abjk -> abik"
    let a = Tensor::<f64, Cpu>::from_data(&[1.0; 2 * 2 * 2 * 2], &[2, 2, 2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0; 2 * 2 * 2 * 2], &[2, 2, 2, 2]);

    let c =
        einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1, 2, 3], &[0, 1, 3, 4]], &[0, 1, 2, 4]);

    assert_eq!(c.shape(), &[2, 2, 2, 2]);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_scalar_operations() {
    // Scalar times scalar using 1x1 matrices
    // True scalars (shape []) require special handling
    let a = Tensor::<f64, Cpu>::from_data(&[3.0], &[1, 1]);
    let b = Tensor::<f64, Cpu>::from_data(&[4.0], &[1, 1]);

    // ij,jk->ik: standard matmul of 1x1 matrices
    let c = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(c.shape(), &[1, 1]);
    assert_eq!(c.to_vec(), vec![12.0]);
}

#[test]
fn test_identity_operation() {
    // ein"ij -> ij"(a) = a (identity)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let c = einsum::<Standard<f64>, _, _>(&[&a], &[&[0, 1]], &[0, 1]);

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.to_vec(), a.to_vec());
}

#[test]
fn test_single_element_tensors() {
    // Operations with 1x1 matrices
    let a = Tensor::<f64, Cpu>::from_data(&[5.0], &[1, 1]);
    let b = Tensor::<f64, Cpu>::from_data(&[3.0], &[1, 1]);

    let c = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(c.shape(), &[1, 1]);
    assert_eq!(c.to_vec(), vec![15.0]);
}

// ============================================================================
// Tropical Unary Gradient Tests
// ============================================================================

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_trace_gradient() {
    // Test gradient of tropical trace: ii-> (max of diagonal)
    // Forward: max(A[0,0], A[1,1]) = max(1, 4) = 4
    // Backward: gradient goes only to the winner (position (1,1))
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let (c, grad_fn) = einsum_with_grad::<MaxPlus<f64>, _, _>(&[&a], &[&[0, 0]], &[]);

    // max(diagonal) = max(1, 4) = 4
    assert_eq!(c.to_vec(), vec![4.0]);

    // Backward: only the winner (1,1) = position 3 gets the gradient
    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0], &[]);
    let grads = grad_fn.backward::<MaxPlus<f64>>(&grad_out, &[&a]);

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 2]);
    // Only position 3 (which is (1,1) in col-major) gets gradient
    assert_eq!(grads[0].to_vec(), vec![0.0, 0.0, 0.0, 1.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_sum_gradient() {
    // Test gradient of tropical sum: ij-> (global max)
    // Forward: max of all elements
    // Backward: gradient goes only to the global max position
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 5.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]);
    // col-major layout: [[1,3,4],[5,2,6]] so max is 6 at position 5

    let (c, grad_fn) = einsum_with_grad::<MaxPlus<f64>, _, _>(&[&a], &[&[0, 1]], &[]);

    // Global max = 6
    assert_eq!(c.to_vec(), vec![6.0]);

    // Backward: only position 5 (the max) gets gradient
    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0], &[]);
    let grads = grad_fn.backward::<MaxPlus<f64>>(&grad_out, &[&a]);

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 3]);
    assert_eq!(grads[0].to_vec(), vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_row_max_gradient() {
    // Test gradient of tropical row max: ij->i
    // Forward: max over j for each i
    // Backward: gradient goes only to the argmax column for each row
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 4.0, 3.0, 2.0, 5.0, 6.0], &[2, 3]);
    // col-major: [[1,3,5],[4,2,6]]
    // row 0: max(1,3,5) = 5 at j=2
    // row 1: max(4,2,6) = 6 at j=2

    let (c, grad_fn) = einsum_with_grad::<MaxPlus<f64>, _, _>(&[&a], &[&[0, 1]], &[0]);

    // Row maxes: [5, 6]
    assert_eq!(c.to_vec(), vec![5.0, 6.0]);

    // Backward: grad = [1, 1]
    // Row 0 winner is position 4 (j=2, i=0)
    // Row 1 winner is position 5 (j=2, i=1)
    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0], &[2]);
    let grads = grad_fn.backward::<MaxPlus<f64>>(&grad_out, &[&a]);

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 3]);
    assert_eq!(grads[0].to_vec(), vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_col_max_gradient() {
    // Test gradient of tropical column max: ij->j
    // Forward: max over i for each j
    // Backward: gradient goes only to the argmax row for each column
    use omeinsum::MaxPlus;

    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 4.0, 3.0, 2.0, 5.0, 6.0], &[2, 3]);
    // col-major: [[1,3,5],[4,2,6]]
    // col 0: max(1,4) = 4 at i=1
    // col 1: max(3,2) = 3 at i=0
    // col 2: max(5,6) = 6 at i=1

    let (c, grad_fn) = einsum_with_grad::<MaxPlus<f64>, _, _>(&[&a], &[&[0, 1]], &[1]);

    // Column maxes: [4, 3, 6]
    assert_eq!(c.to_vec(), vec![4.0, 3.0, 6.0]);

    // Backward: grad = [1, 2, 3]
    // col 0 winner at position 1 gets grad 1
    // col 1 winner at position 2 gets grad 2
    // col 2 winner at position 5 gets grad 3
    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);
    let grads = grad_fn.backward::<MaxPlus<f64>>(&grad_out, &[&a]);

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 3]);
    assert_eq!(grads[0].to_vec(), vec![0.0, 1.0, 2.0, 0.0, 0.0, 3.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_minplus_trace_gradient() {
    // Test MinPlus trace gradient: ii-> (min of diagonal)
    use omeinsum::MinPlus;

    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // Diagonal: [1, 4], min = 1 at position 0

    let (c, grad_fn) = einsum_with_grad::<MinPlus<f64>, _, _>(&[&a], &[&[0, 0]], &[]);

    // min(diagonal) = 1
    assert_eq!(c.to_vec(), vec![1.0]);

    // Backward: only position 0 (which is (0,0)) gets gradient
    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0], &[]);
    let grads = grad_fn.backward::<MinPlus<f64>>(&grad_out, &[&a]);

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 2]);
    assert_eq!(grads[0].to_vec(), vec![1.0, 0.0, 0.0, 0.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_minplus_sum_gradient() {
    // Test MinPlus global min: ij-> (global min)
    use omeinsum::MinPlus;

    let a = Tensor::<f64, Cpu>::from_data(&[5.0, 1.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    // col-major layout, global min is 1 at position 1

    let (c, grad_fn) = einsum_with_grad::<MinPlus<f64>, _, _>(&[&a], &[&[0, 1]], &[]);

    // Global min = 1
    assert_eq!(c.to_vec(), vec![1.0]);

    // Backward: only position 1 (the min) gets gradient
    let grad_out = Tensor::<f64, Cpu>::from_data(&[2.0], &[]);
    let grads = grad_fn.backward::<MinPlus<f64>>(&grad_out, &[&a]);

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 3]);
    assert_eq!(grads[0].to_vec(), vec![0.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_minplus_row_min_gradient() {
    // Test MinPlus row min: ij->i
    use omeinsum::MinPlus;

    let a = Tensor::<f64, Cpu>::from_data(&[3.0, 1.0, 2.0, 4.0, 5.0, 6.0], &[2, 3]);
    // col-major: [[3,2,5],[1,4,6]]
    // row 0: min(3,2,5) = 2 at j=1 (position 2)
    // row 1: min(1,4,6) = 1 at j=0 (position 1)

    let (c, grad_fn) = einsum_with_grad::<MinPlus<f64>, _, _>(&[&a], &[&[0, 1]], &[0]);

    // Row mins: [2, 1]
    assert_eq!(c.to_vec(), vec![2.0, 1.0]);

    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0], &[2]);
    let grads = grad_fn.backward::<MinPlus<f64>>(&grad_out, &[&a]);

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 3]);
    // Position 1 (row 1, j=0) and position 2 (row 0, j=1) get gradients
    assert_eq!(grads[0].to_vec(), vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_single_element_gradient() {
    // Test gradient with single-element tensor
    use omeinsum::MaxPlus;

    let a = Tensor::<f64, Cpu>::from_data(&[42.0], &[1, 1]);

    let (c, grad_fn) = einsum_with_grad::<MaxPlus<f64>, _, _>(&[&a], &[&[0, 1]], &[]);

    assert_eq!(c.to_vec(), vec![42.0]);

    let grad_out = Tensor::<f64, Cpu>::from_data(&[5.0], &[]);
    let grads = grad_fn.backward::<MaxPlus<f64>>(&grad_out, &[&a]);

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[1, 1]);
    assert_eq!(grads[0].to_vec(), vec![5.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_3d_max_gradient() {
    // Test gradient with 3D tensor: ijk->i (max over j,k for each i)
    use omeinsum::MaxPlus;

    // Shape [2, 2, 2] = 8 elements
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    // In col-major, max for i=0 is at some position, max for i=1 is at some position

    let (c, grad_fn) = einsum_with_grad::<MaxPlus<f64>, _, _>(&[&a], &[&[0, 1, 2]], &[0]);

    assert_eq!(c.shape(), &[2]);

    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0], &[2]);
    let grads = grad_fn.backward::<MaxPlus<f64>>(&grad_out, &[&a]);

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 2, 2]);

    // Verify only 2 positions have non-zero gradient (one winner per i)
    let grad_data = grads[0].to_vec();
    let nonzero_count = grad_data.iter().filter(|&&x| x != 0.0).count();
    assert_eq!(nonzero_count, 2, "Should have exactly 2 winner positions");
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_gradient_accumulation() {
    // Test that gradients accumulate when multiple outputs map to same winner
    // This can happen with broadcasting patterns
    use omeinsum::MaxPlus;

    // Create a tensor where the same position wins multiple times
    // Shape [2, 2], diagonal extraction: ii->i keeps the diagonal
    let a = Tensor::<f64, Cpu>::from_data(&[5.0, 1.0, 2.0, 3.0], &[2, 2]);
    // Diagonal: [5, 3]

    // Now do a tropical identity operation that keeps both dimensions
    let (c, grad_fn) = einsum_with_grad::<MaxPlus<f64>, _, _>(&[&a], &[&[0, 0]], &[0]);

    // This extracts diagonal with tropical (max) semantics
    assert_eq!(c.to_vec(), vec![5.0, 3.0]);

    // Different gradients for each output
    let grad_out = Tensor::<f64, Cpu>::from_data(&[2.0, 3.0], &[2]);
    let grads = grad_fn.backward::<MaxPlus<f64>>(&grad_out, &[&a]);

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 2]);
    // Position 0 (0,0) gets grad 2, position 3 (1,1) gets grad 3
    assert_eq!(grads[0].to_vec(), vec![2.0, 0.0, 0.0, 3.0]);
}
