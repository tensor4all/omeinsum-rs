//! Core einsum tests ported from OMEinsum.jl einsum.jl
//!
//! Comprehensive tests covering various einsum patterns and edge cases.

use std::collections::HashMap;

use omeinsum::backend::Cpu;
use omeinsum::einsum::Einsum;
use omeinsum::{einsum, Standard, Tensor};

// ============================================================================
// Matrix and Vector Multiplication (from Julia einsum.jl)
// ============================================================================

#[test]
fn test_einsum_identity_4d() {
    // ijkl -> ijkl (4D identity)
    let t = Tensor::<f64, Cpu>::from_data(&[1.0; 16], &[2, 2, 2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1, 2, 3]], vec![0, 1, 2, 3], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&t]);

    assert_eq!(result.shape(), &[2, 2, 2, 2]);
    assert_eq!(result.to_vec(), t.to_vec());
}

#[test]
fn test_einsum_three_matrix_chain_to_14() {
    // ij,jk,kl -> il
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[2.0, 0.0, 0.0, 2.0], &[2, 2]);

    let result =
        einsum::<Standard<f64>, _, _>(&[&a, &b, &c], &[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);

    assert_eq!(result.shape(), &[2, 2]);
    // I @ A @ 2I = 2A
}

#[test]
fn test_einsum_three_matrix_chain_to_41() {
    // ij,jk,kl -> li (transposed output)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let result =
        einsum::<Standard<f64>, _, _>(&[&a, &b, &c], &[&[0, 1], &[1, 2], &[2, 3]], &[3, 0]);

    assert_eq!(result.shape(), &[2, 2]);
}

#[test]
fn test_einsum_matrix_vector() {
    // ij,j -> i
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let v = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0], &[2]);

    let result = einsum::<Standard<f64>, _, _>(&[&a, &v], &[&[0, 1], &[1]], &[0]);

    assert_eq!(result.shape(), &[2]);
    // A (col-major) = [[1,3],[2,4]], v = [1,1]
    // A @ v = [1+3, 2+4] = [4, 6]
    assert_eq!(result.to_vec(), vec![4.0, 6.0]);
}

// ============================================================================
// Contract to 0-dim Array
// ============================================================================

#[test]
fn test_einsum_contract_to_scalar() {
    // ij,ij -> (Frobenius inner product)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let result = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[0, 1]], &[]);

    assert_eq!(result.shape(), &[] as &[usize]);
    // sum(a .* a) = 1 + 4 + 9 + 16 = 30
    assert_eq!(result.to_vec(), vec![30.0]);
}

// ============================================================================
// Trace Operations
// ============================================================================

#[test]
fn test_einsum_trace_2x2() {
    // ii -> (trace)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2)].into();
    let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[] as &[usize]);
    // trace = 1 + 4 = 5
    assert_eq!(result.to_vec(), vec![5.0]);
}

#[test]
fn test_einsum_trace_4d() {
    // ijji -> (double trace)
    // Shape [2, 4, 4, 2] requires 2*4*4*2 = 64 elements
    let aa = Tensor::<f64, Cpu>::from_data(&vec![1.0; 64], &[2, 4, 4, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 4)].into();
    let ein = Einsum::new(vec![vec![0, 1, 1, 0]], vec![], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&aa]);

    assert_eq!(result.shape(), &[] as &[usize]);
}

// ============================================================================
// Partial Trace
// ============================================================================

#[test]
fn test_einsum_partial_trace() {
    // ijjk -> ik
    let aa = Tensor::<f64, Cpu>::from_data(&[1.0; 16], &[2, 2, 2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1, 1, 2]], vec![0, 2], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&aa]);

    assert_eq!(result.shape(), &[2, 2]);
}

#[test]
fn test_einsum_partial_trace_permuted() {
    // ijjk -> ki (with permutation)
    let aa = Tensor::<f64, Cpu>::from_data(&[1.0; 16], &[2, 2, 2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1, 1, 2]], vec![2, 0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&aa]);

    assert_eq!(result.shape(), &[2, 2]);
}

// ============================================================================
// Diagonal Operations
// ============================================================================

#[test]
fn test_einsum_diag_extract() {
    // ijjk -> ijk (extract "diagonal" along middle indices)
    let aa = Tensor::<f64, Cpu>::from_data(&[1.0; 16], &[2, 2, 2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1, 1, 2]], vec![0, 1, 2], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&aa]);

    assert_eq!(result.shape(), &[2, 2, 2]);
}

// ============================================================================
// Permutation Operations
// ============================================================================

#[test]
fn test_einsum_permute_2d() {
    // ij -> ji
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![1, 0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2, 2]);
    // Transpose of [[1,3],[2,4]] = [[1,2],[3,4]]
    // In col-major: [1, 3, 2, 4]
    assert_eq!(result.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_einsum_permute_4d() {
    // ijkl -> jkil
    let t = Tensor::<f64, Cpu>::from_data(&[1.0; 16], &[2, 2, 2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1, 2, 3]], vec![1, 2, 0, 3], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&t]);

    assert_eq!(result.shape(), &[2, 2, 2, 2]);
}

// ============================================================================
// Tensor Contraction
// ============================================================================

#[test]
fn test_einsum_tensor_contraction_4d_2d() {
    // ijkl,jk -> il
    let t = Tensor::<f64, Cpu>::from_data(&[1.0; 16], &[2, 2, 2, 2]);
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let result = einsum::<Standard<f64>, _, _>(&[&t, &a], &[&[0, 1, 2, 3], &[1, 2]], &[0, 3]);

    assert_eq!(result.shape(), &[2, 2]);
}

#[test]
fn test_einsum_tensor_contraction_permuted() {
    // lkji,jk -> il (permuted input)
    let t = Tensor::<f64, Cpu>::from_data(&[1.0; 16], &[2, 2, 2, 2]);
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let result = einsum::<Standard<f64>, _, _>(&[&t, &a], &[&[3, 2, 1, 0], &[1, 2]], &[0, 3]);

    assert_eq!(result.shape(), &[2, 2]);
}

// ============================================================================
// Star Contraction
// ============================================================================

#[test]
fn test_einsum_star_contraction() {
    // ai,ai,ai -> a (sum over shared index)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    // Using indices: a=0, i=1
    let result = einsum::<Standard<f64>, _, _>(&[&a, &b, &c], &[&[0, 1], &[0, 1], &[0, 1]], &[0]);

    assert_eq!(result.shape(), &[2]);
}

#[test]
fn test_einsum_star_to_output() {
    // ai,bi,ci -> abc
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    // a=0, b=1, c=2, i=3
    let result =
        einsum::<Standard<f64>, _, _>(&[&a, &b, &c], &[&[0, 3], &[1, 3], &[2, 3]], &[0, 1, 2]);

    assert_eq!(result.shape(), &[2, 2, 2]);
}

#[test]
fn test_einsum_star_and_contract() {
    // ai,ai,al -> l
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    // a=0, i=1, l=2
    let result = einsum::<Standard<f64>, _, _>(&[&a, &b, &c], &[&[0, 1], &[0, 1], &[0, 2]], &[2]);

    assert_eq!(result.shape(), &[2]);
}

// ============================================================================
// Index Sum (Reduction)
// ============================================================================

#[test]
fn test_einsum_index_sum() {
    // ijk -> ij (sum over k)
    let a = Tensor::<f64, Cpu>::from_data(&vec![1.0; 30], &[2, 3, 5]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3), (2, 5)].into();
    let ein = Einsum::new(vec![vec![0, 1, 2]], vec![0, 1], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2, 3]);
    // Each element should be sum of 5 ones = 5
    for &v in result.to_vec().iter() {
        assert_eq!(v, 5.0);
    }
}

#[test]
fn test_einsum_index_sum_with_permute() {
    // ijk -> ki (sum over j, permute)
    let a = Tensor::<f64, Cpu>::from_data(&vec![1.0; 30], &[2, 3, 5]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3), (2, 5)].into();
    let ein = Einsum::new(vec![vec![0, 1, 2]], vec![2, 0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[5, 2]);
    // Each element should be sum of 3 ones = 3
    for &v in result.to_vec().iter() {
        assert_eq!(v, 3.0);
    }
}

#[test]
fn test_einsum_sum_all() {
    // ijk -> (sum all)
    let a = Tensor::<f64, Cpu>::from_data(&vec![1.0; 30], &[2, 3, 5]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3), (2, 5)].into();
    let ein = Einsum::new(vec![vec![0, 1, 2]], vec![], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.to_vec(), vec![30.0]);
}

// ============================================================================
// Hadamard Product
// ============================================================================

#[test]
fn test_einsum_hadamard() {
    // ij,ij -> ij
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::<f64, Cpu>::from_data(&[2.0, 2.0, 2.0, 2.0, 2.0, 2.0], &[2, 3]);

    let result = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[0, 1]], &[0, 1]);

    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
}

// ============================================================================
// Outer Product
// ============================================================================

#[test]
fn test_einsum_outer_product() {
    // ij,kl -> ijkl
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 3]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3), (2, 2), (3, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1], vec![2, 3]], vec![0, 1, 2, 3], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);

    assert_eq!(result.shape(), &[2, 3, 2, 3]);
}

// ============================================================================
// Diagonal to Diagonal (Project to diagonal)
// ============================================================================

#[test]
fn test_einsum_project_to_diag() {
    // ii -> ii (project matrix to diagonal matrix)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2)].into();
    let ein = Einsum::new(vec![vec![0, 0]], vec![0, 0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2, 2]);
    // [[1,0],[0,4]] in col-major: [1, 0, 0, 4]
    assert_eq!(result.to_vec(), vec![1.0, 0.0, 0.0, 4.0]);
}

// ============================================================================
// Combined Operations
// ============================================================================

#[test]
fn test_einsum_combined_trace_and_diag() {
    // iijj -> (double trace)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0; 16], &[2, 2, 2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
    let ein = Einsum::new(vec![vec![0, 0, 1, 1]], vec![], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[] as &[usize]);
}

#[test]
fn test_einsum_tensor_network_cycle() {
    // ijkl,jkmn -> ilmn (tensor contraction)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0; 16], &[2, 2, 2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0; 16], &[2, 2, 2, 2]);

    let result =
        einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1, 2, 3], &[1, 2, 4, 5]], &[0, 3, 4, 5]);

    assert_eq!(result.shape(), &[2, 2, 2, 2]);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_einsum_single_element() {
    // 1x1 matrix operations
    let a = Tensor::<f64, Cpu>::from_data(&[5.0], &[1, 1]);

    let sizes: HashMap<usize, usize> = [(0, 1)].into();
    let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.to_vec(), vec![5.0]);
}

#[test]
fn test_einsum_large_contraction() {
    // Large tensor contraction for stress testing
    let a = Tensor::<f64, Cpu>::from_data(&vec![1.0; 1000], &[10, 10, 10]);
    let b = Tensor::<f64, Cpu>::from_data(&vec![1.0; 1000], &[10, 10, 10]);

    let result = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1, 2], &[1, 2, 3]], &[0, 3]);

    assert_eq!(result.shape(), &[10, 10]);
    // Each element should be sum over 10*10 = 100 ones
    for &v in result.to_vec().iter() {
        assert_eq!(v, 100.0);
    }
}

// ============================================================================
// Issue Regression Tests (from Julia)
// ============================================================================

#[test]
fn test_einsum_issue_136_like() {
    // From Julia issue 136: edge case with singleton dimension
    let a = Tensor::<f64, Cpu>::from_data(&[1.0; 4], &[2, 2, 1]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0; 2], &[2]);

    let result = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1, 2], &[1]], &[0, 2]);

    assert_eq!(result.shape(), &[2, 1]);
}

#[test]
fn test_einsum_empty_contraction_result() {
    // Contraction resulting in size-0 dimension (if shape allows)
    // This tests handling of edge cases
    let a = Tensor::<f64, Cpu>::from_data(&[1.0; 4], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0; 4], &[2, 2]);

    // Normal contraction, just verifying it works
    let result = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
    assert_eq!(result.shape(), &[2, 2]);
}

// ============================================================================
// Argument Validation Tests
// ============================================================================

#[test]
#[should_panic(expected = "assertion")]
fn test_einsum_tensor_count_mismatch() {
    // Provide 2 tensors but only 1 index specification
    let a = Tensor::<f64, Cpu>::from_data(&[1.0; 4], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0; 4], &[2, 2]);

    // This should panic due to mismatched tensor/index count
    let _ = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1]], &[0, 1]);
}

#[test]
#[should_panic(expected = "assertion")]
fn test_einsum_index_dimension_mismatch() {
    // Tensor has 2 dimensions but 3 indices specified
    let a = Tensor::<f64, Cpu>::from_data(&[1.0; 4], &[2, 2]);

    // This should panic due to index count != tensor dimensions
    let _ = einsum::<Standard<f64>, _, _>(&[&a], &[&[0, 1, 2]], &[0, 1]);
}

#[test]
fn test_einsum_valid_edge_cases() {
    // Scalar output from vector
    let v = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);
    let result = einsum::<Standard<f64>, _, _>(&[&v], &[&[0]], &[]);
    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.to_vec(), vec![6.0]);
}

#[test]
fn test_einsum_single_element_contraction() {
    // Single element tensor contraction
    let a = Tensor::<f64, Cpu>::from_data(&[5.0], &[1]);
    let b = Tensor::<f64, Cpu>::from_data(&[3.0], &[1]);

    // Contraction of single-element tensors to scalar
    let result = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0], &[0]], &[]);
    assert_eq!(result.to_vec(), vec![15.0]);
}

#[test]
fn test_einsum_identity_operation() {
    // i -> i (identity, no actual operation)
    let v = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let result = einsum::<Standard<f64>, _, _>(&[&v], &[&[0]], &[0]);
    assert_eq!(result.to_vec(), v.to_vec());
}

#[test]
fn test_einsum_large_index_values() {
    // Test with larger index values (not 0-based sequential)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // Using indices 10, 11, 12 instead of 0, 1, 2
    let sizes: HashMap<usize, usize> = [(10, 2), (11, 2), (12, 2)].into();
    let ein = Einsum::new(vec![vec![10, 11], vec![11, 12]], vec![10, 12], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);

    assert_eq!(result.shape(), &[2, 2]);
}

#[test]
fn test_einsum_repeated_input_index() {
    // ij,ij->ij (Hadamard product - same indices for both inputs)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let result = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[0, 1]], &[0, 1]);
    assert_eq!(result.to_vec(), vec![5.0, 12.0, 21.0, 32.0]);
}

#[test]
fn test_einsum_all_indices_contracted() {
    // ij,jk,ki-> (complete contraction, scalar result)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1], vec![1, 2], vec![2, 0]], vec![], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b, &c]);

    assert_eq!(result.shape(), &[] as &[usize]);
    // trace(A @ I @ I) = trace(A) = 1 + 4 = 5
    assert_eq!(result.to_vec(), vec![5.0]);
}

#[test]
fn test_einsum_no_contraction() {
    // i,j->ij (outer product, no shared indices)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);
    let b = Tensor::<f64, Cpu>::from_data(&[3.0, 4.0, 5.0], &[3]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0], vec![1]], vec![0, 1], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);

    assert_eq!(result.shape(), &[2, 3]);
}

#[test]
fn test_einsum_size_consistency() {
    // Verify that the same index must have consistent sizes
    // This tests that einsum correctly uses the size_dict
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);

    // Both tensors share index 0 with size 3
    let result = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0], &[0]], &[0]);
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.to_vec(), vec![1.0, 4.0, 9.0]);
}

// ============================================================================
// Comprehensive Einsum Pattern Coverage Tests (Batch 3)
// ============================================================================

#[test]
fn test_einsum_batch_matrix_vector() {
    // bij,bj -> bi (batched matrix-vector multiply)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0; 24], &[2, 3, 4]); // batch=2, rows=3, cols=4
    let v = Tensor::<f64, Cpu>::from_data(&[1.0; 8], &[2, 4]); // batch=2, cols=4

    let result = einsum::<Standard<f64>, _, _>(&[&a, &v], &[&[0, 1, 2], &[0, 2]], &[0, 1]);
    assert_eq!(result.shape(), &[2, 3]);
    // Each output is sum of 4 ones = 4
    for &v in result.to_vec().iter() {
        assert_eq!(v, 4.0);
    }
}

#[test]
fn test_einsum_einsum_bilinear() {
    // ij,jk,il -> lk (bilinear contraction)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0; 4], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0; 4], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[1.0; 4], &[2, 2]);

    let result =
        einsum::<Standard<f64>, _, _>(&[&a, &b, &c], &[&[0, 1], &[1, 2], &[0, 3]], &[3, 2]);
    assert_eq!(result.shape(), &[2, 2]);
}

#[test]
fn test_einsum_tensor_train_core() {
    // al,lir,rb -> aib (tensor train/MPS contraction)
    let left = Tensor::<f64, Cpu>::from_data(&[1.0; 6], &[2, 3]); // a=2, l=3
    let core = Tensor::<f64, Cpu>::from_data(&[1.0; 24], &[3, 2, 4]); // l=3, i=2, r=4
    let right = Tensor::<f64, Cpu>::from_data(&[1.0; 8], &[4, 2]); // r=4, b=2

    let result = einsum::<Standard<f64>, _, _>(
        &[&left, &core, &right],
        &[&[0, 1], &[1, 2, 3], &[3, 4]],
        &[0, 2, 4],
    );
    assert_eq!(result.shape(), &[2, 2, 2]);
    // sum over l=3, r=4 gives 12 for each element
    for &v in result.to_vec().iter() {
        assert_eq!(v, 12.0);
    }
}

#[test]
fn test_einsum_khatri_rao() {
    // ij,kj -> ikj (Khatri-Rao-like product)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], &[2, 3]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 3)].into();
    let ein = Einsum::new(vec![vec![0, 2], vec![1, 2]], vec![0, 1, 2], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);

    assert_eq!(result.shape(), &[2, 2, 3]);
}

#[test]
fn test_einsum_attention_weights() {
    // bhqk,bhvk -> bhqv (attention-like weighted combination)
    let weights = Tensor::<f64, Cpu>::from_data(&vec![1.0; 48], &[2, 3, 2, 4]); // b=2,h=3,q=2,k=4
    let values = Tensor::<f64, Cpu>::from_data(&vec![1.0; 48], &[2, 3, 2, 4]); // b=2,h=3,v=2,k=4

    let result = einsum::<Standard<f64>, _, _>(
        &[&weights, &values],
        &[&[0, 1, 2, 3], &[0, 1, 4, 3]],
        &[0, 1, 2, 4],
    );
    assert_eq!(result.shape(), &[2, 3, 2, 2]);
    // Each element sums over k=4
    for &v in result.to_vec().iter() {
        assert_eq!(v, 4.0);
    }
}

#[test]
fn test_einsum_triple_contraction() {
    // ij,jk,ki -> (cyclic trace product)
    // Uses three matrices in a cyclic contraction pattern
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[3.0, 0.0, 0.0, 4.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[5.0, 0.0, 0.0, 6.0], &[2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1], vec![1, 2], vec![2, 0]], vec![], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b, &c]);

    assert_eq!(result.shape(), &[] as &[usize]);
    // trace(A @ B @ C) - diagonal matrices so result is diagonal product trace
}

#[test]
fn test_einsum_diagonal_sum() {
    // ii -> (sum of diagonal)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);

    let sizes: HashMap<usize, usize> = [(0, 3)].into();
    let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[] as &[usize]);
    // col-major: diag = [1, 5, 9]
    assert_eq!(result.to_vec(), vec![15.0]);
}

#[test]
fn test_einsum_matrix_to_row_sums() {
    // ij -> i (sum rows)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![0], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[2]);
    // col-major [[1,3,5],[2,4,6]]: row sums = [1+3+5, 2+4+6] = [9, 12]
    assert_eq!(result.to_vec(), vec![9.0, 12.0]);
}

#[test]
fn test_einsum_matrix_to_col_sums() {
    // ij -> j (sum columns)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
    let ein = Einsum::new(vec![vec![0, 1]], vec![1], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a]);

    assert_eq!(result.shape(), &[3]);
    // col-major [[1,3,5],[2,4,6]]: col sums = [1+2, 3+4, 5+6] = [3, 7, 11]
    assert_eq!(result.to_vec(), vec![3.0, 7.0, 11.0]);
}

#[test]
fn test_einsum_five_tensor_contraction() {
    // ai,bi,ci,di,ei -> abcde (5-tensor star contraction)
    let tensors: Vec<_> = (0..5)
        .map(|_| Tensor::<f64, Cpu>::from_data(&[1.0; 4], &[2, 2]))
        .collect();

    // Each tensor has shape [2,2] with first index being its own, second shared
    let result = einsum::<Standard<f64>, _, _>(
        &[
            &tensors[0],
            &tensors[1],
            &tensors[2],
            &tensors[3],
            &tensors[4],
        ],
        &[&[0, 5], &[1, 5], &[2, 5], &[3, 5], &[4, 5]],
        &[0, 1, 2, 3, 4],
    );
    assert_eq!(result.shape(), &[2, 2, 2, 2, 2]);
    // Each element sums over shared index with size 2
    for &v in result.to_vec().iter() {
        assert_eq!(v, 2.0);
    }
}

#[test]
fn test_einsum_kronecker_product() {
    // ij,kl -> ikjl (Kronecker product)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1], vec![2, 3]], vec![0, 2, 1, 3], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);

    assert_eq!(result.shape(), &[2, 2, 2, 2]);
}

#[test]
fn test_einsum_self_contraction() {
    // ijk,ijk -> (self dot product)
    let a = Tensor::<f64, Cpu>::from_data(&[2.0; 24], &[2, 3, 4]);

    let result = einsum::<Standard<f64>, _, _>(&[&a, &a], &[&[0, 1, 2], &[0, 1, 2]], &[]);
    assert_eq!(result.shape(), &[] as &[usize]);
    // sum(a^2) = 24 * 4 = 96
    assert_eq!(result.to_vec(), vec![96.0]);
}

#[test]
fn test_einsum_cyclic_contraction() {
    // ij,jk,ki -> (cyclic trace)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[9.0, 10.0, 11.0, 12.0], &[2, 2]);

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1], vec![1, 2], vec![2, 0]], vec![], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b, &c]);

    assert_eq!(result.shape(), &[] as &[usize]);
    // This is trace(A @ B @ C)
}

#[test]
fn test_einsum_reshape_via_permute() {
    // ijkl -> iljk (complex permutation)
    let t = Tensor::<f64, Cpu>::from_data(
        &(1..=24).map(|x| x as f64).collect::<Vec<_>>(),
        &[2, 3, 2, 2],
    );

    let sizes: HashMap<usize, usize> = [(0, 2), (1, 3), (2, 2), (3, 2)].into();
    let ein = Einsum::new(vec![vec![0, 1, 2, 3]], vec![0, 3, 1, 2], sizes);
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&t]);

    assert_eq!(result.shape(), &[2, 2, 3, 2]);
    assert_eq!(result.to_vec().len(), 24);
}

#[test]
fn test_einsum_mixed_batch_contraction() {
    // bijk,bkl -> bijl (batch matmul with extra dimensions)
    let a = Tensor::<f64, Cpu>::from_data(&vec![1.0; 48], &[2, 2, 3, 4]);
    let b = Tensor::<f64, Cpu>::from_data(&vec![1.0; 40], &[2, 4, 5]);

    let result =
        einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1, 2, 3], &[0, 3, 4]], &[0, 1, 2, 4]);
    assert_eq!(result.shape(), &[2, 2, 3, 5]);
    // Each element is sum over k=4
    for &v in result.to_vec().iter() {
        assert_eq!(v, 4.0);
    }
}
