//! Tests for Backend::contract unified API.

use omeinsum::backend::Backend;
use omeinsum::{Cpu, Standard};

#[test]
fn test_cpu_contract_matmul() {
    // ij,jk->ik (matrix multiplication)
    let cpu = Cpu;

    let a = vec![1.0f64, 2.0, 3.0, 4.0]; // 2x2 column-major
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let c = cpu.contract::<Standard<f64>>(
        &a,
        &[2, 2],
        &[1, 2],
        &[0, 1],
        &b,
        &[2, 2],
        &[1, 2],
        &[1, 2],
        &[2, 2],
        &[0, 2],
    );

    // Column-major: [1,2,3,4] for shape [2,2] -> A = [[1,3],[2,4]] (rows)
    // Column-major: [5,6,7,8] for shape [2,2] -> B = [[5,7],[6,8]]
    // A @ B = [[1*5+3*6, 1*7+3*8], [2*5+4*6, 2*7+4*8]]
    //       = [[23, 31], [34, 46]]
    // Column-major: [23, 34, 31, 46]
    assert_eq!(c, vec![23.0, 34.0, 31.0, 46.0]);
}

#[test]
fn test_cpu_contract_inner_product() {
    // i,i-> (inner product)
    let cpu = Cpu;

    let a = vec![1.0f64, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];

    let c = cpu.contract::<Standard<f64>>(&a, &[3], &[1], &[0], &b, &[3], &[1], &[0], &[1], &[]);

    // 1*4 + 2*5 + 3*6 = 32
    assert_eq!(c, vec![32.0]);
}

#[test]
fn test_cpu_contract_outer_product() {
    // i,j->ij (outer product)
    let cpu = Cpu;

    let a = vec![1.0f64, 2.0];
    let b = vec![3.0, 4.0, 5.0];

    let c =
        cpu.contract::<Standard<f64>>(&a, &[2], &[1], &[0], &b, &[3], &[1], &[1], &[2, 3], &[0, 1]);

    // [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]] = [[3,4,5], [6,8,10]]
    // Column-major: [3, 6, 4, 8, 5, 10]
    assert_eq!(c, vec![3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);
}

#[test]
fn test_cpu_contract_batched() {
    // bij,bjk->bik (batched matmul)
    let cpu = Cpu;

    // 2 batches of 2x2 matrices
    let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];

    let c = cpu.contract::<Standard<f64>>(
        &a,
        &[2, 2, 2],
        &[1, 2, 4],
        &[0, 1, 2],
        &b,
        &[2, 2, 2],
        &[1, 2, 4],
        &[0, 2, 3],
        &[2, 2, 2],
        &[0, 1, 3],
    );

    // Batch 0: identity @ [[1,2],[3,4]] = [[1,2],[3,4]]
    // Batch 1: 2*identity @ [[5,6],[7,8]] = [[10,12],[14,16]]
    assert_eq!(c.len(), 8);
}

#[cfg(feature = "tropical")]
#[test]
fn test_cpu_contract_tropical() {
    use omeinsum::MaxPlus;

    let cpu = Cpu;

    let a = vec![1.0f64, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];

    let c = cpu.contract::<MaxPlus<f64>>(
        &a,
        &[2, 2],
        &[1, 2],
        &[0, 1],
        &b,
        &[2, 2],
        &[1, 2],
        &[1, 2],
        &[2, 2],
        &[0, 2],
    );

    // Column-major: [1,2,3,4] -> A = [[1,3],[2,4]]
    // MaxPlus: C[i,k] = max_j(A[i,j] + B[j,k])
    // C[0,0] = max(A[0,0]+B[0,0], A[0,1]+B[1,0]) = max(1+1, 3+2) = 5
    // C[1,0] = max(A[1,0]+B[0,0], A[1,1]+B[1,0]) = max(2+1, 4+2) = 6
    // C[0,1] = max(A[0,0]+B[0,1], A[0,1]+B[1,1]) = max(1+3, 3+4) = 7
    // C[1,1] = max(A[1,0]+B[0,1], A[1,1]+B[1,1]) = max(2+3, 4+4) = 8
    // Column-major: [5, 6, 7, 8]
    assert_eq!(c, vec![5.0, 6.0, 7.0, 8.0]);
}

// ============================================================================
// Tests for strided (non-contiguous) inputs
// ============================================================================

#[test]
fn test_cpu_contract_strided_input() {
    // Test that non-contiguous inputs are handled correctly
    let cpu = Cpu;

    // Create a 3x3 matrix in column-major
    // [[1,4,7], [2,5,8], [3,6,9]]
    // Column-major storage: [1,2,3,4,5,6,7,8,9]
    let a_full = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    // Use strided access to get the first 2x2 submatrix
    // Shape [2,2], strides [1,3] (non-contiguous column stride)
    // This should give [[1,4], [2,5]]
    let b = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity

    let c = cpu.contract::<Standard<f64>>(
        &a_full,
        &[2, 2],
        &[1, 3],
        &[0, 1], // Strided access to a
        &b,
        &[2, 2],
        &[1, 2],
        &[1, 2],
        &[2, 2],
        &[0, 2],
    );

    // Result should be the 2x2 submatrix times identity = the submatrix
    // [[1,4], [2,5]] in column-major = [1, 2, 4, 5]
    assert_eq!(c, vec![1.0, 2.0, 4.0, 5.0]);
}

#[test]
fn test_cpu_contract_both_strided() {
    // Both inputs are strided
    let cpu = Cpu;

    // 4x4 matrix, we'll use strided views of 2x2 submatrices
    let data_a = vec![
        1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let data_b = vec![
        9.0f64, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ];

    // Access the 2x2 submatrix starting at (0,0) with stride [1,4]
    // [[1,5], [2,6]]
    // Access the 2x2 submatrix starting at (0,0) with stride [1,4]
    // [[9,13], [10,14]]
    let c = cpu.contract::<Standard<f64>>(
        &data_a,
        &[2, 2],
        &[1, 4],
        &[0, 1],
        &data_b,
        &[2, 2],
        &[1, 4],
        &[1, 2],
        &[2, 2],
        &[0, 2],
    );

    // [[1,5], [2,6]] @ [[9,13], [10,14]]
    // C[0,0] = 1*9 + 5*10 = 59
    // C[1,0] = 2*9 + 6*10 = 78
    // C[0,1] = 1*13 + 5*14 = 83
    // C[1,1] = 2*13 + 6*14 = 110
    // Column-major: [59, 78, 83, 110]
    assert_eq!(c, vec![59.0, 78.0, 83.0, 110.0]);
}

// ============================================================================
// Tests for output permutation
// ============================================================================

#[test]
fn test_cpu_contract_output_permuted() {
    // ij,jk->ki (output is transposed)
    let cpu = Cpu;

    let a = vec![1.0f64, 2.0, 3.0, 4.0]; // 2x2
    let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2

    // Normal matmul gives C[i,k], but we want C[k,i] (transposed)
    let c = cpu.contract::<Standard<f64>>(
        &a,
        &[2, 2],
        &[1, 2],
        &[0, 1],
        &b,
        &[2, 2],
        &[1, 2],
        &[1, 2],
        &[2, 2],
        &[2, 0], // Output indices are [k, i] instead of [i, k]
    );

    // Normal result would be [23, 34, 31, 46]
    // Transposed: [[23, 34], [31, 46]]^T = [[23, 31], [34, 46]]
    // Column-major: [23, 31, 34, 46]
    assert_eq!(c, vec![23.0, 31.0, 34.0, 46.0]);
}

#[test]
fn test_cpu_contract_batched_output_permuted() {
    // bij,bjk->kib (complex permutation)
    let cpu = Cpu;

    // Simple 2x2x2 tensors for easy manual verification
    // A: batch=2, i=2, j=2
    let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // B: batch=2, j=2, k=2 (identity matrices)
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];

    // Result should be A with axes permuted to [k, i, b]
    let c = cpu.contract::<Standard<f64>>(
        &a,
        &[2, 2, 2],
        &[1, 2, 4],
        &[0, 1, 2], // b, i, j
        &b,
        &[2, 2, 2],
        &[1, 2, 4],
        &[0, 2, 3], // b, j, k
        &[2, 2, 2],
        &[3, 1, 0], // k, i, b
    );

    assert_eq!(c.len(), 8);
    // Since B is identity, result is A with permuted axes
}

// ============================================================================
// Tests for contract_with_argmax
// ============================================================================

#[cfg(feature = "tropical")]
#[test]
fn test_cpu_contract_with_argmax() {
    use omeinsum::MaxPlus;

    let cpu = Cpu;

    let a = vec![1.0f64, 2.0, 3.0, 4.0]; // 2x2
    let b = vec![1.0, 2.0, 3.0, 4.0]; // 2x2

    let (c, argmax) = cpu.contract_with_argmax::<MaxPlus<f64>>(
        &a,
        &[2, 2],
        &[1, 2],
        &[0, 1],
        &b,
        &[2, 2],
        &[1, 2],
        &[1, 2],
        &[2, 2],
        &[0, 2],
    );

    // Same as test_cpu_contract_tropical
    assert_eq!(c, vec![5.0, 6.0, 7.0, 8.0]);
    // All winners should be k=1 (second column/row wins)
    assert_eq!(argmax, vec![1, 1, 1, 1]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_cpu_contract_with_argmax_strided() {
    use omeinsum::MaxPlus;

    let cpu = Cpu;

    // 3x3 matrix, use 2x2 submatrix via strides
    let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity-ish for MaxPlus

    let (c, argmax) = cpu.contract_with_argmax::<MaxPlus<f64>>(
        &a,
        &[2, 2],
        &[1, 3],
        &[0, 1], // Strided
        &b,
        &[2, 2],
        &[1, 2],
        &[1, 2],
        &[2, 2],
        &[0, 2],
    );

    assert_eq!(c.len(), 4);
    assert_eq!(argmax.len(), 4);
}

#[cfg(feature = "tropical")]
#[test]
fn test_cpu_contract_with_argmax_output_permuted() {
    use omeinsum::MaxPlus;

    let cpu = Cpu;

    let a = vec![1.0f64, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];

    // Output permuted: ki instead of ik
    let (c, argmax) = cpu.contract_with_argmax::<MaxPlus<f64>>(
        &a,
        &[2, 2],
        &[1, 2],
        &[0, 1],
        &b,
        &[2, 2],
        &[1, 2],
        &[1, 2],
        &[2, 2],
        &[2, 0], // Permuted output
    );

    // Result should be transposed version of [5, 6, 7, 8]
    assert_eq!(c, vec![5.0, 7.0, 6.0, 8.0]);
    // Argmax should also be permuted
    assert_eq!(argmax.len(), 4);
}

#[cfg(feature = "tropical")]
#[test]
fn test_cpu_contract_with_argmax_batched() {
    use omeinsum::MaxPlus;

    let cpu = Cpu;

    // Batched tropical contraction
    let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2x2x2
    let b = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]; // 2x2x2

    let (c, argmax) = cpu.contract_with_argmax::<MaxPlus<f64>>(
        &a,
        &[2, 2, 2],
        &[1, 2, 4],
        &[0, 1, 2], // b, i, j
        &b,
        &[2, 2, 2],
        &[1, 2, 4],
        &[0, 2, 3], // b, j, k
        &[2, 2, 2],
        &[0, 1, 3], // b, i, k
    );

    assert_eq!(c.len(), 8);
    assert_eq!(argmax.len(), 8);
}

// ============================================================================
// Tests for f32 (additional coverage for faer path)
// ============================================================================

#[test]
fn test_cpu_contract_f32() {
    let cpu = Cpu;

    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];

    let c = cpu.contract::<Standard<f32>>(
        &a,
        &[2, 2],
        &[1, 2],
        &[0, 1],
        &b,
        &[2, 2],
        &[1, 2],
        &[1, 2],
        &[2, 2],
        &[0, 2],
    );

    assert_eq!(c, vec![23.0f32, 34.0, 31.0, 46.0]);
}
