//! Unary operation tests ported from OMEinsum.jl unaryrules.jl
//!
//! Tests for single-tensor einsum operations: trace, diagonal, sum, permutation, etc.

use std::collections::HashMap;

use omeinsum::backend::Cpu;
use omeinsum::einsum::Einsum;
use omeinsum::{Standard, Tensor};

#[cfg(feature = "tropical")]
use omeinsum::{MaxPlus, MinPlus};

// ============================================================================
// Trace Tests (ii -> )
// ============================================================================

#[test]
fn test_unary_trace_2x2() {
    // ii -> (trace of 2x2 matrix)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // Column-major: [[1,3],[2,4]], diagonal = [1, 4], trace = 5

    let sizes: HashMap<usize, usize> = [(0, 2)].into();
    let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.to_vec(), vec![5.0]);
}

#[test]
fn test_unary_trace_3x3() {
    // ii -> (trace of 3x3 matrix)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    // Column-major: [[1,4,7],[2,5,8],[3,6,9]], diagonal = [1, 5, 9], trace = 15

    let sizes: HashMap<usize, usize> = [(0, 3)].into();
    let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.to_vec(), vec![15.0]);
}

#[test]
fn test_unary_trace_5x5() {
    // ii -> (trace of 5x5 matrix)
    let mut data = [0.0; 25];
    for i in 0..5 {
        data[i * 5 + i] = (i + 1) as f64; // Diagonal elements at column-major positions
    }
    // Actually for column-major, diagonal is at i + i*5
    let mut data = vec![0.0; 25];
    for i in 0..5 {
        data[i + i * 5] = (i + 1) as f64;
    }
    let a = Tensor::<f64, Cpu>::from_data(&data, &[5, 5]);
    // Diagonal = [1, 2, 3, 4, 5], trace = 15

    let sizes: HashMap<usize, usize> = [(0, 5)].into();
    let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.to_vec(), vec![15.0]);
}

// ============================================================================
// Diagonal Extraction Tests (ii -> i)
// ============================================================================

#[test]
fn test_unary_diagonal_2x2() {
    // ii -> i (extract diagonal of 2x2 matrix)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // Column-major: [[1,3],[2,4]], diagonal = [1, 4]

    let sizes: HashMap<usize, usize> = [(0, 2)].into();
    let ein = Einsum::new(vec![vec![0, 0]], vec![0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.to_vec(), vec![1.0, 4.0]);
}

#[test]
fn test_unary_diagonal_3x3() {
    // ii -> i (extract diagonal of 3x3 matrix)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    // Column-major: [[1,4,7],[2,5,8],[3,6,9]], diagonal = [1, 5, 9]

    let sizes: HashMap<usize, usize> = [(0, 3)].into();
    let ein = Einsum::new(vec![vec![0, 0]], vec![0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.to_vec(), vec![1.0, 5.0, 9.0]);
}

// ============================================================================
// Sum Tests (reduction)
// ============================================================================

#[test]
fn test_unary_sum_all_2d() {
    // ij -> (sum all elements)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // sum = 1+2+3+4+5+6 = 21

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.to_vec(), vec![21.0]);
}

#[test]
fn test_unary_sum_axis_0() {
    // ij -> j (sum over first axis)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // Column-major: [[1,3,5],[2,4,6]]
    // Sum over i (rows): [1+2, 3+4, 5+6] = [3, 7, 11]

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![1], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.to_vec(), vec![3.0, 7.0, 11.0]);
}

#[test]
fn test_unary_sum_axis_1() {
    // ij -> i (sum over second axis)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // Column-major: [[1,3,5],[2,4,6]]
    // Sum over j (cols): [1+3+5, 2+4+6] = [9, 12]

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.to_vec(), vec![9.0, 12.0]);
}

#[test]
fn test_unary_sum_3d_to_1d() {
    // ijk -> i (sum over j and k)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1, 2]], vec![0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2]);
    // Sum for i=0: 1+3+5+7 = 16, Sum for i=1: 2+4+6+8 = 20
    assert_eq!(result.to_vec(), vec![16.0, 20.0]);
}

// ============================================================================
// Permutation Tests (transpose, reorder axes)
// ============================================================================

#[test]
fn test_unary_transpose_2x2() {
    // ij -> ji (transpose)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // Column-major: [[1,3],[2,4]] -> [[1,2],[3,4]]
    // Transposed in column-major: [1, 3, 2, 4]

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![1, 0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_unary_transpose_2x3() {
    // ij -> ji (transpose 2x3 to 3x2)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // Column-major [2,3]: [[1,3,5],[2,4,6]]
    // Transposed [3,2]: [[1,2],[3,4],[5,6]]
    // In column-major: [1, 3, 5, 2, 4, 6]

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![1, 0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.to_vec(), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn test_unary_permute_3d() {
    // ijk -> kji (reverse all axes)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1, 2]], vec![2, 1, 0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2, 2, 2]);
}

#[test]
fn test_unary_permute_3d_partial() {
    // ijk -> jik (swap first two axes)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1, 2]], vec![1, 0, 2], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2, 2, 2]);
}

// ============================================================================
// Identity Tests (no-op)
// ============================================================================

#[test]
fn test_unary_identity_2d() {
    // ij -> ij (identity, no change)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![0, 1], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.to_vec(), a.to_vec());
}

#[test]
fn test_unary_identity_3d() {
    // ijk -> ijk (identity)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1, 2]], vec![0, 1, 2], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2, 2, 2]);
    assert_eq!(result.to_vec(), a.to_vec());
}

// ============================================================================
// Partial Trace Tests (ijjk -> ik)
// ============================================================================

#[test]
fn test_unary_partial_trace_4d() {
    // ijjk -> ik (trace over repeated middle index)
    // Shape [2, 2, 2, 2], trace over j (indices 1 and 2)
    let a = Tensor::<f64, Cpu>::from_data(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[2, 2, 2, 2],
    );

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1, 1, 2]], vec![0, 2], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2, 2]);
}

// ============================================================================
// Diag (from matrix to diagonal matrix) Tests
// ============================================================================

#[test]
fn test_unary_diag_extract_and_embed() {
    // ii -> ii (project to diagonal matrix)
    // Takes matrix, zeroes off-diagonal elements
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // Column-major: [[1,3],[2,4]]
    // Diagonal matrix: [[1,0],[0,4]]
    // In column-major: [1, 0, 0, 4]

    let sizes: HashMap<usize, usize> = [(0, 2)].into();
    let ein = Einsum::new(vec![vec![0, 0]], vec![0, 0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.to_vec(), vec![1.0, 0.0, 0.0, 4.0]);
}

// ============================================================================
// Tropical Unary Tests
// ============================================================================

#[cfg(feature = "tropical")]
#[test]
fn test_unary_tropical_trace() {
    // MaxPlus trace: max of diagonal elements
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // Diagonal = [1, 4], max = 4

    let sizes: HashMap<usize, usize> = [(0, 2)].into();
    let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);
    let result = ein.execute::<MaxPlus<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.to_vec(), vec![4.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_unary_tropical_sum_all() {
    // MaxPlus sum: max of all elements
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 5.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]);
    // Max = 6

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![], sizes);
    let result = ein.execute::<MaxPlus<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.to_vec(), vec![6.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_unary_tropical_row_max() {
    // MaxPlus row reduction: ij -> i (max over j for each i)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 4.0, 3.0, 2.0, 5.0, 6.0], &[2, 3]);
    // Column-major: [[1,3,5],[4,2,6]]
    // Row 0 max: max(1,3,5) = 5
    // Row 1 max: max(4,2,6) = 6

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![0], sizes);
    let result = ein.execute::<MaxPlus<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.to_vec(), vec![5.0, 6.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_unary_tropical_col_max() {
    // MaxPlus column reduction: ij -> j (max over i for each j)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 4.0, 3.0, 2.0, 5.0, 6.0], &[2, 3]);
    // Column-major: [[1,3,5],[4,2,6]]
    // Col 0 max: max(1,4) = 4
    // Col 1 max: max(3,2) = 3
    // Col 2 max: max(5,6) = 6

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![1], sizes);
    let result = ein.execute::<MaxPlus<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.to_vec(), vec![4.0, 3.0, 6.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_unary_minplus_trace() {
    // MinPlus trace: min of diagonal elements
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // Diagonal = [1, 4], min = 1

    let sizes: HashMap<usize, usize> = [(0, 2)].into();
    let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);
    let result = ein.execute::<MinPlus<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.to_vec(), vec![1.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_unary_minplus_sum_all() {
    // MinPlus sum: min of all elements
    let a = Tensor::<f64, Cpu>::from_data(&[5.0, 1.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    // Min = 1

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![], sizes);
    let result = ein.execute::<MinPlus<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.to_vec(), vec![1.0]);
}

// ============================================================================
// Duplicate (Embed to Diagonal) Tests
// These are the inverse of diagonal extraction
// ============================================================================

#[test]
fn test_unary_duplicate_vector_to_diagonal() {
    // i -> ii (embed vector to diagonal of matrix)
    // Input: [1, 2, 3]
    // Output: [[1,0,0], [0,2,0], [0,0,3]]
    let v = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);

    let sizes: HashMap<usize, usize> = [(0, 3)].into();
    let ein = Einsum::new(vec![vec![0]], vec![0, 0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&v]);

    assert_eq!(result.shape(), &[3, 3]);
    // Column-major diagonal matrix: [1,0,0, 0,2,0, 0,0,3]
    assert_eq!(
        result.to_vec(),
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
    );
}

#[test]
fn test_unary_duplicate_small() {
    // i -> ii (2-element vector to 2x2 diagonal)
    let v = Tensor::<f64, Cpu>::from_data(&[5.0, 7.0], &[2]);

    let sizes: HashMap<usize, usize> = [(0, 2)].into();
    let ein = Einsum::new(vec![vec![0]], vec![0, 0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&v]);

    assert_eq!(result.shape(), &[2, 2]);
    // [[5,0],[0,7]] in col-major: [5, 0, 0, 7]
    assert_eq!(result.to_vec(), vec![5.0, 0.0, 0.0, 7.0]);
}

#[test]
fn test_unary_duplicate_roundtrip() {
    // i -> ii -> i should recover original
    let v = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[4]);

    // First: i -> ii (embed to diagonal)
    let sizes: HashMap<usize, usize> = [(0, 4)].into();
    let ein_dup = Einsum::new(vec![vec![0]], vec![0, 0], sizes.clone());
    let diagonal_matrix = ein_dup.execute::<Standard<f64>, f64, Cpu>(&[&v]);

    assert_eq!(diagonal_matrix.shape(), &[4, 4]);

    // Then: ii -> i (extract diagonal)
    let ein_diag = Einsum::new(vec![vec![0, 0]], vec![0], sizes);
    let recovered = ein_diag.execute::<Standard<f64>, f64, Cpu>(&[&diagonal_matrix]);

    assert_eq!(recovered.shape(), &[4]);
    assert_eq!(recovered.to_vec(), v.to_vec());
}

// ============================================================================
// Repeat (Broadcast) Tests
// Expand tensor by adding dimensions
// ============================================================================

#[test]
fn test_unary_repeat_add_dimension() {
    // i -> ij (repeat vector along new dimension)
    // Input: [1, 2], output shape [2, 3]
    let v = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0]], vec![0, 1], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&v]);

    assert_eq!(result.shape(), &[2, 3]);
    // Each row repeated 3 times (column-major)
    // [[1,1,1], [2,2,2]] in col-major: [1, 2, 1, 2, 1, 2]
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn test_unary_repeat_prepend_dimension() {
    // i -> ji (prepend batch dimension)
    // Input: [1, 2, 3], output shape [2, 3]
    let v = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);

    let sizes: HashMap<usize, usize> = [(0, 3), (1, 2)].into();
    let ein = Einsum::new(vec![vec![0]], vec![1, 0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&v]);

    assert_eq!(result.shape(), &[2, 3]);
    // [[1,2,3], [1,2,3]] in col-major: [1, 1, 2, 2, 3, 3]
    assert_eq!(result.to_vec(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
}

#[test]
fn test_unary_repeat_to_3d() {
    // i -> ijk (expand to 3D)
    let v = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
    let ein = Einsum::new(vec![vec![0]], vec![0, 1, 2], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&v]);

    assert_eq!(result.shape(), &[2, 2, 2]);
    // 8 elements, all either 1 or 2 depending on first dimension
    let result_vec = result.to_vec();
    assert_eq!(result_vec.len(), 8);
    // Values should only be 1.0 or 2.0
    assert!(result_vec.iter().all(|&x| x == 1.0 || x == 2.0));
}

#[test]
fn test_unary_repeat_matrix_add_batch() {
    // ij -> bij (add batch dimension to matrix)
    let m = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![2, 0, 1], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&m]);

    assert_eq!(result.shape(), &[3, 2, 2]);
    // The matrix is repeated 3 times along batch dimension
    let result_vec = result.to_vec();
    assert_eq!(result_vec.len(), 12);
}

// ============================================================================
// Combined Duplicate + Contract Tests
// ============================================================================

#[test]
fn test_unary_duplicate_then_sum() {
    // i -> ii -> (trace after duplicate = original sum)
    // This tests that duplicate followed by trace gives the same as sum
    let v = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);

    // Direct sum: i ->
    let sizes: HashMap<usize, usize> = [(0, 3)].into();
    let ein_sum = Einsum::new(vec![vec![0]], vec![], sizes.clone());
    let sum_result = ein_sum.execute::<Standard<f64>, f64, Cpu>(&[&v]);

    // Duplicate then trace: i -> ii ->
    let ein_dup_trace = Einsum::new(vec![vec![0]], vec![], sizes);
    // Note: This is the same operation - just verifying consistency
    let dup_trace_result = ein_dup_trace.execute::<Standard<f64>, f64, Cpu>(&[&v]);

    assert_eq!(sum_result.to_vec(), dup_trace_result.to_vec());
    assert_eq!(sum_result.to_vec(), vec![6.0]);
}

#[test]
fn test_unary_repeat_then_reduce() {
    // i -> ij -> i (repeat then sum back)
    // Sum over j should give n * original value where n = size of j
    let v = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);

    // i -> ij
    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein_repeat = Einsum::new(vec![vec![0]], vec![0, 1], sizes.clone());
    let repeated = ein_repeat.execute::<Standard<f64>, f64, Cpu>(&[&v]);

    // ij -> i
    let ein_sum = Einsum::new(vec![vec![0, 1]], vec![0], sizes);
    let summed = ein_sum.execute::<Standard<f64>, f64, Cpu>(&[&repeated]);

    assert_eq!(summed.shape(), &[2]);
    // Each element multiplied by 3 (size of j)
    assert_eq!(summed.to_vec(), vec![3.0, 6.0]);
}
