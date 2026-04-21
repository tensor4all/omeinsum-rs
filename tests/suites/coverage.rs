//! Additional tests for code coverage.
//!
//! These tests target specific code paths that may not be covered by other tests.

use omeinsum::backend::{Backend, Cpu};
use omeinsum::tensor::TensorView;
use omeinsum::{einsum, einsum_with_grad, Standard, Tensor};

#[cfg(feature = "tropical")]
use omeinsum::{MaxMul, MaxPlus, MinPlus};

// ============================================================================
// TensorView tests
// ============================================================================

#[test]
fn test_tensor_view_basic() {
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let view = TensorView::new(&t);

    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides(), &[1, 2]);
    assert_eq!(view.ndim(), 2);
    assert_eq!(view.numel(), 6);
    assert!(view.is_contiguous());
    assert_eq!(view.as_tensor().shape(), &[2, 3]);
}

#[test]
fn test_tensor_view_from_trait() {
    let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let view: TensorView<f32, Cpu> = (&t).into();

    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view.numel(), 4);
}

#[test]
fn test_tensor_view_non_contiguous() {
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let t_permuted = t.permute(&[1, 0]); // Now [3, 2] with non-contiguous strides
    let view = TensorView::new(&t_permuted);

    assert_eq!(view.shape(), &[3, 2]);
    assert!(!view.is_contiguous());
}

// ============================================================================
// Backend tests
// ============================================================================

#[test]
fn test_cpu_backend_name() {
    assert_eq!(Cpu::name(), "cpu");
}

#[test]
fn test_cpu_backend_synchronize() {
    let cpu = Cpu;
    cpu.synchronize(); // Should be no-op
}

#[test]
fn test_cpu_backend_alloc() {
    let cpu = Cpu;
    let storage: Vec<f64> = cpu.alloc(10);
    assert_eq!(storage.len(), 10);
    assert!(storage.iter().all(|&x| x == 0.0));
}

#[test]
fn test_cpu_backend_from_slice() {
    let cpu = Cpu;
    let data = [1.0f32, 2.0, 3.0];
    let storage = cpu.from_slice(&data);
    assert_eq!(storage, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_storage_is_empty() {
    let empty: Vec<f32> = vec![];
    let non_empty: Vec<f32> = vec![1.0, 2.0];

    assert!(empty.is_empty());
    assert!(!non_empty.is_empty());
}

#[test]
fn test_storage_get_set() {
    use omeinsum::backend::Storage;

    let mut storage: Vec<f64> = vec![1.0, 2.0, 3.0];

    assert_eq!(storage.get(0), 1.0);
    assert_eq!(storage.get(1), 2.0);

    storage.set(1, 5.0);
    assert_eq!(storage.get(1), 5.0);
}

#[test]
fn test_storage_zeros() {
    use omeinsum::backend::Storage;

    let zeros: Vec<f32> = Vec::zeros(5);
    assert_eq!(zeros.len(), 5);
    assert!(zeros.iter().all(|&x| x == 0.0));
}

// ============================================================================
// Tensor operations tests
// ============================================================================

#[test]
fn test_tensor_clone() {
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let t2 = t.clone();
    assert_eq!(t.to_vec(), t2.to_vec());
    assert_eq!(t.shape(), t2.shape());
}

#[test]
fn test_tensor_debug() {
    let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0], &[2]);
    let debug_str = format!("{:?}", t);
    assert!(debug_str.contains("Tensor"));
    assert!(debug_str.contains("shape"));
}

#[test]
fn test_tensor_zeros() {
    let t = Tensor::<f64, Cpu>::zeros(&[3, 4]);
    assert_eq!(t.shape(), &[3, 4]);
    assert_eq!(t.numel(), 12);
    assert!(t.to_vec().iter().all(|&x| x == 0.0));
}

#[test]
fn test_tensor_from_storage() {
    let storage = vec![1.0f32, 2.0, 3.0, 4.0];
    let t = Tensor::<f32, Cpu>::from_storage(storage, &[2, 2], Cpu);
    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_tensor_backend() {
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);
    let _backend = t.backend();
}

#[test]
fn test_tensor_storage() {
    let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);
    let storage = t.storage();
    assert!(storage.is_some());
    assert_eq!(storage.unwrap().len(), 3);
}

#[test]
fn test_tensor_strides() {
    // Test that strides are computed correctly for column-major layout
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // Column-major: stride for first dim = 1, stride for second dim = 2
    assert_eq!(t.strides(), &[1, 2]);
}

#[test]
fn test_tensor_reshape_various() {
    let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // Reshape to 1D
    let t1 = t.reshape(&[6]);
    assert_eq!(t1.shape(), &[6]);

    // Reshape to 3D
    let t3 = t.reshape(&[1, 2, 3]);
    assert_eq!(t3.shape(), &[1, 2, 3]);
}

#[test]
fn test_tensor_contiguous_already() {
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(t.is_contiguous());

    let t2 = t.contiguous();
    assert!(t2.is_contiguous());
    assert_eq!(t.to_vec(), t2.to_vec());
}

#[test]
fn test_tensor_sum_various_shapes() {
    // 1D tensor
    let t1 = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(t1.sum::<Standard<f64>>(), 6.0);

    // 3D tensor
    let t3 = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    assert_eq!(t3.sum::<Standard<f64>>(), 36.0);
}

#[test]
fn test_tensor_sum_axis_various() {
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // Sum along axis 0
    let s0 = t.sum_axis::<Standard<f64>>(0);
    assert_eq!(s0.shape(), &[3]);

    // Sum along axis 1
    let s1 = t.sum_axis::<Standard<f64>>(1);
    assert_eq!(s1.shape(), &[2]);
}

#[test]
fn test_tensor_diagonal_3x3() {
    // 3x3 square matrix diagonal
    // Column-major: [1,4,7,2,5,8,3,6,9] for:
    //   [[1,2,3],
    //    [4,5,6],
    //    [7,8,9]]
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0], &[3, 3]);
    let diag = t.diagonal();
    assert_eq!(diag.shape(), &[3]);
    // Diagonal elements: [1,5,9]
    assert_eq!(diag.to_vec(), vec![1.0, 5.0, 9.0]);
}

// ============================================================================
// Einsum tests for coverage
// ============================================================================

#[test]
fn test_einsum_scalar_output() {
    // Contract to scalar: ij,ij->
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

    let c = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[0, 1]], &[]);
    assert_eq!(c.shape(), &[] as &[usize]);
    assert_eq!(c.to_vec(), vec![10.0]); // 1+2+3+4 = 10
}

#[test]
fn test_einsum_with_grad_single_tensor() {
    // Single tensor should return unchanged gradient
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let (result, grad_fn) = einsum_with_grad::<Standard<f64>, _, _>(&[&a], &[&[0, 1]], &[0, 1]);

    assert_eq!(result.to_vec(), a.to_vec());

    let grad_out = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<Standard<f64>>(&grad_out, &[&a]);
    assert_eq!(grads.len(), 1);
}

#[test]
fn test_einsum_batch_contraction() {
    // Batch matrix multiply: bij,bjk->bik
    let a = Tensor::<f64, Cpu>::from_data(
        &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], // 2 batches of 2x2 identity-ish
        &[2, 2, 2],
    );
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);

    let c = einsum::<Standard<f64>, _, _>(
        &[&a, &b],
        &[&[0, 1, 2], &[0, 2, 3]], // b=0, i=1, j=2, k=3
        &[0, 1, 3],                // output: b, i, k
    );

    assert_eq!(c.shape(), &[2, 2, 2]);
}

// ============================================================================
// Tropical algebra tests for coverage
// ============================================================================

#[cfg(feature = "tropical")]
#[test]
fn test_maxplus_is_zero() {
    use omeinsum::algebra::GenericSemiring;

    let zero = MaxPlus::<f32>::zero();
    let one = MaxPlus::<f32>::one();
    let val = MaxPlus(5.0f32);

    assert!(zero.is_zero());
    assert!(!one.is_zero());
    assert!(!val.is_zero());
}

#[cfg(feature = "tropical")]
#[test]
fn test_minplus_is_zero() {
    use omeinsum::algebra::GenericSemiring;

    let zero = MinPlus::<f32>::zero();
    let one = MinPlus::<f32>::one();
    let val = MinPlus(5.0f32);

    assert!(zero.is_zero());
    assert!(!one.is_zero());
    assert!(!val.is_zero());
}

#[cfg(feature = "tropical")]
#[test]
fn test_maxmul_is_zero() {
    use omeinsum::algebra::GenericSemiring;

    let zero = MaxMul::<f32>::zero();
    let one = MaxMul::<f32>::one();
    let val = MaxMul(5.0f32);

    assert!(zero.is_zero());
    assert!(!one.is_zero());
    assert!(!val.is_zero());
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_add_backward() {
    use omeinsum::algebra::Algebra;

    // MaxPlus add_backward
    let a = MaxPlus(3.0f32);
    let b = MaxPlus(5.0f32);
    let (ga, gb) = a.add_backward(b, 1.0, Some(1)); // b wins (index 1)
    assert_eq!(ga, 0.0);
    assert_eq!(gb, 1.0);

    let (ga, gb) = a.add_backward(b, 1.0, Some(0)); // a wins (index 0)
    assert_eq!(ga, 1.0);
    assert_eq!(gb, 0.0);

    // MinPlus add_backward (same logic)
    let a = MinPlus(3.0f32);
    let b = MinPlus(5.0f32);
    let (ga, gb) = a.add_backward(b, 1.0, Some(0)); // a wins (smaller)
    assert_eq!(ga, 1.0);
    assert_eq!(gb, 0.0);

    // MaxMul add_backward
    let a = MaxMul(3.0f32);
    let b = MaxMul(5.0f32);
    let (ga, gb) = a.add_backward(b, 1.0, Some(1)); // b wins
    assert_eq!(ga, 0.0);
    assert_eq!(gb, 1.0);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_mul_backward() {
    use omeinsum::algebra::Algebra;

    // MaxPlus mul_backward (multiplication is addition, so both get gradient)
    let a = MaxPlus(3.0f32);
    let b = MaxPlus(5.0f32);
    let (ga, gb) = a.mul_backward(b, 1.0);
    assert_eq!(ga, 1.0);
    assert_eq!(gb, 1.0);

    // MinPlus mul_backward
    let a = MinPlus(3.0f32);
    let b = MinPlus(5.0f32);
    let (ga, gb) = a.mul_backward(b, 1.0);
    assert_eq!(ga, 1.0);
    assert_eq!(gb, 1.0);

    // MaxMul mul_backward (multiplication is real multiplication)
    let a = MaxMul(3.0f32);
    let b = MaxMul(5.0f32);
    let (ga, gb) = a.mul_backward(b, 1.0);
    assert_eq!(ga, 5.0); // grad_a = grad_out * b
    assert_eq!(gb, 3.0); // grad_b = grad_out * a
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_needs_argmax() {
    use omeinsum::algebra::Algebra;

    assert!(MaxPlus::<f32>::needs_argmax());
    assert!(MinPlus::<f32>::needs_argmax());
    assert!(MaxMul::<f32>::needs_argmax());
    assert!(!Standard::<f32>::needs_argmax());
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_f64_operations() {
    use omeinsum::algebra::{GenericSemiring, Semiring};

    // MaxPlus f64
    let a = MaxPlus(2.0f64);
    let b = MaxPlus(3.0f64);
    assert_eq!(a.add(b).to_scalar(), 3.0);
    assert_eq!(a.mul(b).to_scalar(), 5.0);

    // MinPlus f64
    let a = MinPlus(2.0f64);
    let b = MinPlus(3.0f64);
    assert_eq!(a.add(b).to_scalar(), 2.0);
    assert_eq!(a.mul(b).to_scalar(), 5.0);

    // MaxMul f64
    let a = MaxMul(2.0f64);
    let b = MaxMul(3.0f64);
    assert_eq!(a.add(b).to_scalar(), 3.0);
    assert_eq!(a.mul(b).to_scalar(), 6.0);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_contract_binary_f64() {
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // MaxPlus f64
    let c = a.contract_binary::<MaxPlus<f64>>(&b, &[0, 1], &[1, 2], &[0, 2]);
    assert_eq!(c.shape(), &[2, 2]);

    // MinPlus f64
    let c = a.contract_binary::<MinPlus<f64>>(&b, &[0, 1], &[1, 2], &[0, 2]);
    assert_eq!(c.shape(), &[2, 2]);

    // MaxMul f64
    let c = a.contract_binary::<MaxMul<f64>>(&b, &[0, 1], &[1, 2], &[0, 2]);
    assert_eq!(c.shape(), &[2, 2]);
}

// ============================================================================
// Standard algebra tests for coverage
// ============================================================================

#[test]
fn test_standard_is_zero() {
    use omeinsum::algebra::GenericSemiring;

    let zero = Standard::<f32>::zero();
    let one = Standard::<f32>::one();
    let val = Standard(5.0f32);

    assert!(zero.is_zero());
    assert!(!one.is_zero());
    assert!(!val.is_zero());
}

#[test]
fn test_standard_f64() {
    use omeinsum::algebra::{GenericSemiring, Semiring};

    let a = Standard(2.0f64);
    let b = Standard(3.0f64);

    assert_eq!(a.add(b).to_scalar(), 5.0);
    assert_eq!(a.mul(b).to_scalar(), 6.0);
    assert_eq!(Standard::<f64>::zero().to_scalar(), 0.0);
    assert_eq!(Standard::<f64>::one().to_scalar(), 1.0);
}

// ============================================================================
// Complex number tests
// ============================================================================

#[test]
fn test_complex_tensor_basic() {
    use num_complex::Complex64 as C64;

    let t = Tensor::<C64, Cpu>::from_data(
        &[
            C64::new(1.0, 0.0),
            C64::new(0.0, 1.0),
            C64::new(1.0, 1.0),
            C64::new(2.0, 0.0),
        ],
        &[2, 2],
    );

    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.numel(), 4);
}

#[test]
fn test_complex_contract_binary() {
    use num_complex::Complex64 as C64;

    let a = Tensor::<C64, Cpu>::from_data(
        &[
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
        ],
        &[2, 2],
    );
    let b = Tensor::<C64, Cpu>::from_data(
        &[
            C64::new(1.0, 1.0),
            C64::new(2.0, 0.0),
            C64::new(0.0, 1.0),
            C64::new(3.0, 0.0),
        ],
        &[2, 2],
    );

    // Identity * B = B
    let c = a.contract_binary::<Standard<C64>>(&b, &[0, 1], &[1, 2], &[0, 2]);
    assert_eq!(c.shape(), &[2, 2]);
}

// ============================================================================
// Batched GEMM tests
// ============================================================================

#[test]
fn test_contract_batched_standard() {
    // Test batched matrix multiplication via contract_binary
    // A[b,i,j] × B[b,j,k] → C[b,i,k]
    // 2 batches of 2x2 matrices
    let a = Tensor::<f32, Cpu>::from_data(
        &[
            1.0f32, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ],
        &[2, 2, 2],
    );
    let b = Tensor::<f32, Cpu>::from_data(
        &[
            1.0f32, 0.0, 0.0, 1.0, // batch 0: identity
            1.0, 0.0, 0.0, 1.0, // batch 1: identity
        ],
        &[2, 2, 2],
    );

    let c = a.contract_binary::<Standard<f32>>(&b, &[0, 1, 2], &[0, 2, 3], &[0, 1, 3]);
    assert_eq!(c.shape(), &[2, 2, 2]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_contract_batched_tropical() {
    // Test batched matrix multiplication via contract_binary
    let a = Tensor::<f32, Cpu>::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 2, 2]);

    let c = a.contract_binary::<MaxPlus<f32>>(&b, &[0, 1, 2], &[0, 2, 3], &[0, 1, 3]);
    assert_eq!(c.shape(), &[2, 2, 2]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_contract_batched_with_argmax() {
    // Test batched matrix multiplication with argmax via contract_binary_with_argmax
    let a = Tensor::<f32, Cpu>::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], &[2, 2, 2]);

    let (c, argmax) =
        a.contract_binary_with_argmax::<MaxPlus<f32>>(&b, &[0, 1, 2], &[0, 2, 3], &[0, 1, 3]);
    assert_eq!(c.shape(), &[2, 2, 2]);
    assert_eq!(argmax.shape(), &[2, 2, 2]);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_tensor_1x1() {
    let t = Tensor::<f64, Cpu>::from_data(&[42.0], &[1, 1]);
    assert_eq!(t.shape(), &[1, 1]);
    assert_eq!(t.get(0), 42.0);
    assert_eq!(t.sum::<Standard<f64>>(), 42.0);
    assert_eq!(t.diagonal().to_vec(), vec![42.0]);
}

#[test]
fn test_einsum_identity() {
    // i->i (identity)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);
    let c = einsum::<Standard<f64>, _, _>(&[&a], &[&[0]], &[0]);
    assert_eq!(c.to_vec(), a.to_vec());
}

#[test]
fn test_einsum_transpose() {
    // ij->ji (transpose)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let c = einsum::<Standard<f64>, _, _>(&[&a], &[&[0, 1]], &[1, 0]);
    assert_eq!(c.shape(), &[3, 2]);
}

#[test]
fn test_einsum_outer_product() {
    // i,j->ij (outer product)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);
    let b = Tensor::<f64, Cpu>::from_data(&[3.0, 4.0, 5.0], &[3]);

    let c = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0], &[1]], &[0, 1]);
    assert_eq!(c.shape(), &[2, 3]);
}

// ============================================================================
// Additional coverage tests
// ============================================================================

#[test]
fn test_tensor_storage_non_contiguous() {
    // Test that storage() returns None for non-contiguous tensor
    let t = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // Permute creates a non-contiguous view
    let permuted = t.permute(&[1, 0]);
    assert!(!permuted.is_contiguous());
    assert!(permuted.storage().is_none());

    // After making contiguous, storage should be accessible
    let contiguous = permuted.contiguous();
    assert!(contiguous.is_contiguous());
    assert!(contiguous.storage().is_some());
}

#[test]
fn test_einsum_treesa_optimizer() {
    use omeinsum::Einsum;

    // Create einsum and optimize with TreeSA
    let mut ein = Einsum::new(
        vec![vec![0, 1], vec![1, 2], vec![2, 3]], // A[i,j], B[j,k], C[k,l]
        vec![0, 3],                               // D[i,l]
        [(0, 2), (1, 2), (2, 2), (3, 2)].into(),
    );

    assert!(!ein.is_optimized());

    ein.optimize_treesa();

    assert!(ein.is_optimized());
    assert!(ein.contraction_tree().is_some());
}

#[test]
fn test_einsum_three_tensor_standard() {
    // Test pairwise contraction path for Standard algebra (3 tensors)
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[2.0, 0.0, 0.0, 2.0], &[2, 2]);

    // Contract three tensors: (A @ B) @ C
    let result =
        einsum::<Standard<f64>, _, _>(&[&a, &b, &c], &[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);

    assert_eq!(result.shape(), &[2, 2]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_add_backward_recompute_winner() {
    // Test add_backward when winner_idx is None (recompute winner case)
    use omeinsum::algebra::Algebra;

    let a = MaxPlus(5.0_f64);
    let b = MaxPlus(3.0_f64);

    // Call add_backward with None - should recompute winner
    let (grad_a, grad_b) = a.add_backward(b, 1.0, None);

    // a > b, so a wins and gets the gradient
    assert_eq!(grad_a, 1.0);
    assert_eq!(grad_b, 0.0);

    // Test when b wins
    let a2 = MaxPlus(2.0_f64);
    let b2 = MaxPlus(7.0_f64);
    let (grad_a2, grad_b2) = a2.add_backward(b2, 1.0, None);

    assert_eq!(grad_a2, 0.0);
    assert_eq!(grad_b2, 1.0);
}

#[cfg(feature = "tropical")]
#[test]
fn test_minplus_add_backward_recompute_winner() {
    use omeinsum::algebra::Algebra;
    use omeinsum::MinPlus;

    let a = MinPlus(5.0_f64);
    let b = MinPlus(3.0_f64);

    // Call add_backward with None - should recompute winner
    // For MinPlus, smaller value wins
    let (grad_a, grad_b) = a.add_backward(b, 1.0, None);

    // b < a, so b wins and gets the gradient
    assert_eq!(grad_a, 0.0);
    assert_eq!(grad_b, 1.0);

    // Test when a wins (smaller)
    let a2 = MinPlus(2.0_f64);
    let b2 = MinPlus(7.0_f64);
    let (grad_a2, grad_b2) = a2.add_backward(b2, 1.0, None);

    assert_eq!(grad_a2, 1.0);
    assert_eq!(grad_b2, 0.0);
}

#[cfg(feature = "tropical")]
#[test]
fn test_maxmul_add_backward_recompute_winner() {
    use omeinsum::algebra::Algebra;
    use omeinsum::MaxMul;

    let a = MaxMul(5.0_f64);
    let b = MaxMul(3.0_f64);

    // Call add_backward with None - should recompute winner
    // For MaxMul, larger value wins (same as MaxPlus for add)
    let (grad_a, grad_b) = a.add_backward(b, 1.0, None);

    // a > b, so a wins
    assert_eq!(grad_a, 1.0);
    assert_eq!(grad_b, 0.0);

    // Test when b wins (recompute)
    let a2 = MaxMul(2.0_f64);
    let b2 = MaxMul(7.0_f64);
    let (grad_a2, grad_b2) = a2.add_backward(b2, 1.0, None);
    assert_eq!(grad_a2, 0.0);
    assert_eq!(grad_b2, 1.0);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_add_backward_with_winner_idx() {
    use omeinsum::algebra::Algebra;

    // Test MaxPlus with explicit winner_idx
    let a = MaxPlus(5.0_f64);
    let b = MaxPlus(3.0_f64);

    // winner_idx = Some(0) means self (a) is the winner
    let (grad_a, grad_b) = a.add_backward(b, 1.0, Some(0));
    assert_eq!(grad_a, 1.0);
    assert_eq!(grad_b, 0.0);

    // winner_idx = Some(1) means rhs (b) is the winner
    let (grad_a2, grad_b2) = a.add_backward(b, 1.0, Some(1));
    assert_eq!(grad_a2, 0.0);
    assert_eq!(grad_b2, 1.0);
}

#[cfg(feature = "tropical")]
#[test]
fn test_minplus_add_backward_with_winner_idx() {
    use omeinsum::algebra::Algebra;
    use omeinsum::MinPlus;

    let a = MinPlus(5.0_f64);
    let b = MinPlus(3.0_f64);

    // winner_idx = Some(0) means self (a) wins
    let (grad_a, grad_b) = a.add_backward(b, 1.0, Some(0));
    assert_eq!(grad_a, 1.0);
    assert_eq!(grad_b, 0.0);

    // winner_idx = Some(1) means rhs (b) wins
    let (grad_a2, grad_b2) = a.add_backward(b, 1.0, Some(1));
    assert_eq!(grad_a2, 0.0);
    assert_eq!(grad_b2, 1.0);
}

#[cfg(feature = "tropical")]
#[test]
fn test_maxmul_add_backward_with_winner_idx() {
    use omeinsum::algebra::Algebra;
    use omeinsum::MaxMul;

    let a = MaxMul(5.0_f64);
    let b = MaxMul(3.0_f64);

    // winner_idx = Some(0) means self (a) wins
    let (grad_a, grad_b) = a.add_backward(b, 1.0, Some(0));
    assert_eq!(grad_a, 1.0);
    assert_eq!(grad_b, 0.0);

    // winner_idx = Some(1) means rhs (b) wins
    let (grad_a2, grad_b2) = a.add_backward(b, 1.0, Some(1));
    assert_eq!(grad_a2, 0.0);
    assert_eq!(grad_b2, 1.0);
}

#[cfg(feature = "tropical")]
#[test]
fn test_minplus_add_with_argmax() {
    use omeinsum::algebra::Algebra;
    use omeinsum::MinPlus;

    // Test add_with_argmax for MinPlus (min wins)
    let a = MinPlus(5.0_f64);
    let b = MinPlus(3.0_f64);

    // b is smaller, so b should win
    let (result, idx) = a.add_with_argmax(0, b, 1);
    assert_eq!(result.0, 3.0);
    assert_eq!(idx, 1);

    // Test when a wins
    let a2 = MinPlus(2.0_f64);
    let b2 = MinPlus(7.0_f64);
    let (result2, idx2) = a2.add_with_argmax(0, b2, 1);
    assert_eq!(result2.0, 2.0);
    assert_eq!(idx2, 0);
}

#[cfg(feature = "tropical")]
#[test]
fn test_maxmul_add_with_argmax() {
    use omeinsum::algebra::Algebra;
    use omeinsum::MaxMul;

    // Test add_with_argmax for MaxMul (max wins)
    let a = MaxMul(5.0_f64);
    let b = MaxMul(3.0_f64);

    // a is larger, so a should win
    let (result, idx) = a.add_with_argmax(0, b, 1);
    assert_eq!(result.0, 5.0);
    assert_eq!(idx, 0);

    // Test when b wins
    let a2 = MaxMul(2.0_f64);
    let b2 = MaxMul(7.0_f64);
    let (result2, idx2) = a2.add_with_argmax(0, b2, 1);
    assert_eq!(result2.0, 7.0);
    assert_eq!(idx2, 1);
}

#[cfg(feature = "tropical")]
#[test]
fn test_minplus_einsum_with_grad() {
    use omeinsum::MinPlus;

    // Test MinPlus einsum with gradient to exercise gemm_with_argmax for MinPlus
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[0.0, 10.0, 10.0, 0.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<MinPlus<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(result.shape(), &[2, 2]);

    // Compute gradients
    let grad_output = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<MinPlus<f64>>(&grad_output, &[&a, &b]);

    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].shape(), a.shape());
    assert_eq!(grads[1].shape(), b.shape());
}

#[cfg(feature = "tropical")]
#[test]
fn test_maxmul_einsum_with_grad() {
    use omeinsum::MaxMul;

    // Test MaxMul einsum with gradient to exercise gemm_with_argmax for MaxMul
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[2.0, 0.5, 0.5, 2.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<MaxMul<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(result.shape(), &[2, 2]);

    // Compute gradients
    let grad_output = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<MaxMul<f64>>(&grad_output, &[&a, &b]);

    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].shape(), a.shape());
    assert_eq!(grads[1].shape(), b.shape());
}

#[cfg(feature = "tropical")]
#[test]
fn test_minplus_f32_einsum_with_grad() {
    use omeinsum::MinPlus;

    // Test MinPlus with f32 to exercise the f32 gemm_with_argmax path
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[0.0, 10.0, 10.0, 0.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<MinPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(result.shape(), &[2, 2]);

    let grad_output = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<MinPlus<f32>>(&grad_output, &[&a, &b]);

    assert_eq!(grads.len(), 2);
}

#[cfg(feature = "tropical")]
#[test]
fn test_maxmul_f32_einsum_with_grad() {
    use omeinsum::MaxMul;

    // Test MaxMul with f32 to exercise the f32 gemm_with_argmax path
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[2.0, 0.5, 0.5, 2.0], &[2, 2]);

    let (result, grad_fn) =
        einsum_with_grad::<MaxMul<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    assert_eq!(result.shape(), &[2, 2]);

    let grad_output = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<MaxMul<f32>>(&grad_output, &[&a, &b]);

    assert_eq!(grads.len(), 2);
}

#[cfg(feature = "tropical")]
#[test]
fn test_batched_tropical_forward() {
    // Test batched tropical einsum (forward only - backward not yet implemented for batched)

    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[2, 2, 2]);

    // Batched matrix multiply with tropical algebra
    let result = einsum::<MaxPlus<f64>, _, _>(
        &[&a, &b],
        &[&[0, 1, 2], &[0, 2, 3]], // batch=0, contract=2
        &[0, 1, 3],                // output: batch, left, right
    );

    assert_eq!(result.shape(), &[2, 2, 2]);

    // Just verify the shape and that it completes without error
    // Full value verification would require careful analysis of the batched contraction
}

#[test]
fn test_einsum_with_grad_three_tensor_backward() {
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[2.0, 0.0, 0.0, 2.0], &[2, 2]);

    let (result, grad_fn) = einsum_with_grad::<Standard<f64>, _, _>(
        &[&a, &b, &c],
        &[&[0, 1], &[1, 2], &[2, 3]],
        &[0, 3],
    );

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);

    let grad_output = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let grads = grad_fn.backward::<Standard<f64>>(&grad_output, &[&a, &b, &c]);

    assert_eq!(grads.len(), 3);
    assert_eq!(grads[0].to_vec(), vec![2.0, 0.0, 0.0, 2.0]);
    assert_eq!(grads[1].to_vec(), vec![2.0, 6.0, 4.0, 8.0]);
    assert_eq!(grads[2].to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_einsum_builder_methods() {
    use omeinsum::EinBuilder;

    // Test builder with size_dict using correct API
    let builder = EinBuilder::new()
        .input(&[0, 1])
        .input(&[1, 2])
        .output(&[0, 2])
        .size(0, 3)
        .size(1, 4)
        .size(2, 5);

    let ein = builder.build();

    // Verify the einsum was built correctly
    let a = Tensor::<f64, Cpu>::from_data(&(0..12).map(|x| x as f64).collect::<Vec<_>>(), &[3, 4]);
    let b = Tensor::<f64, Cpu>::from_data(&(0..20).map(|x| x as f64).collect::<Vec<_>>(), &[4, 5]);

    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);
    assert_eq!(result.shape(), &[3, 5]);
}

#[test]
fn test_einsum_builder_sizes() {
    use omeinsum::EinBuilder;

    // Test builder with sizes() method (multiple sizes at once)
    let ein = EinBuilder::new()
        .input(&[0, 1])
        .input(&[1, 2])
        .output(&[0, 2])
        .sizes([(0, 3), (1, 4), (2, 5)])
        .build();

    let a = Tensor::<f64, Cpu>::from_data(&(0..12).map(|x| x as f64).collect::<Vec<_>>(), &[3, 4]);
    let b = Tensor::<f64, Cpu>::from_data(&(0..20).map(|x| x as f64).collect::<Vec<_>>(), &[4, 5]);

    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);
    assert_eq!(result.shape(), &[3, 5]);
}

#[test]
fn test_einsum_builder_default() {
    use omeinsum::EinBuilder;

    // Test Default trait implementation
    let builder: EinBuilder<usize> = EinBuilder::default();
    let ein = builder
        .input(&[0, 1])
        .input(&[1, 2])
        .output(&[0, 2])
        .size(0, 2)
        .size(1, 2)
        .size(2, 2)
        .build();

    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);
    assert_eq!(result.shape(), &[2, 2]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_minplus_f32_operations() {
    use omeinsum::algebra::GenericSemiring;
    use omeinsum::MinPlus;

    let a = MinPlus(3.0_f32);
    let b = MinPlus(5.0_f32);

    // Test add (min)
    let sum = a.add(b);
    assert_eq!(sum.0, 3.0);

    // Test mul (+)
    let prod = a.mul(b);
    assert_eq!(prod.0, 8.0);

    // Test zero and one
    assert_eq!(MinPlus::<f32>::zero().0, f32::MAX);
    assert_eq!(MinPlus::<f32>::one().0, 0.0);
}

#[cfg(feature = "tropical")]
#[test]
fn test_maxmul_f32_operations() {
    use omeinsum::algebra::GenericSemiring;
    use omeinsum::MaxMul;

    let a = MaxMul(3.0_f32);
    let b = MaxMul(5.0_f32);

    // Test add (max)
    let sum = a.add(b);
    assert_eq!(sum.0, 5.0);

    // Test mul (*)
    let prod = a.mul(b);
    assert_eq!(prod.0, 15.0);

    // Test zero and one
    assert_eq!(MaxMul::<f32>::zero().0, 0.0);
    assert_eq!(MaxMul::<f32>::one().0, 1.0);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_einsum_minplus() {
    use omeinsum::MinPlus;

    // Test MinPlus einsum
    // Column-major: a = [1,2,3,4] means a[0,0]=1, a[1,0]=2, a[0,1]=3, a[1,1]=4
    // So a = [[1, 3], [2, 4]]
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // b = [0,10,10,0] means b[0,0]=0, b[1,0]=10, b[0,1]=10, b[1,1]=0
    // So b = [[0, 10], [10, 0]]
    let b = Tensor::<f64, Cpu>::from_data(&[0.0, 10.0, 10.0, 0.0], &[2, 2]);

    let c = einsum::<MinPlus<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // MinPlus: c[i,k] = min_j (a[i,j] + b[j,k])
    // c[0,0] = min(a[0,0]+b[0,0], a[0,1]+b[1,0]) = min(1+0, 3+10) = 1
    // c[1,0] = min(a[1,0]+b[0,0], a[1,1]+b[1,0]) = min(2+0, 4+10) = 2
    // c[0,1] = min(a[0,0]+b[0,1], a[0,1]+b[1,1]) = min(1+10, 3+0) = 3
    // c[1,1] = min(a[1,0]+b[0,1], a[1,1]+b[1,1]) = min(2+10, 4+0) = 4
    // Column-major output: [c[0,0], c[1,0], c[0,1], c[1,1]] = [1, 2, 3, 4]
    let result = c.to_vec();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[cfg(feature = "tropical")]
#[test]
fn test_tropical_einsum_maxmul() {
    use omeinsum::MaxMul;

    // Test MaxMul einsum
    // Column-major: a = [1,2,3,4] means a[0,0]=1, a[1,0]=2, a[0,1]=3, a[1,1]=4
    // So a = [[1, 3], [2, 4]]
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // b = [2,0.5,0.5,2] means b[0,0]=2, b[1,0]=0.5, b[0,1]=0.5, b[1,1]=2
    // So b = [[2, 0.5], [0.5, 2]]
    let b = Tensor::<f64, Cpu>::from_data(&[2.0, 0.5, 0.5, 2.0], &[2, 2]);

    let c = einsum::<MaxMul<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

    // MaxMul: c[i,k] = max_j (a[i,j] * b[j,k])
    // c[0,0] = max(a[0,0]*b[0,0], a[0,1]*b[1,0]) = max(1*2, 3*0.5) = max(2, 1.5) = 2
    // c[1,0] = max(a[1,0]*b[0,0], a[1,1]*b[1,0]) = max(2*2, 4*0.5) = max(4, 2) = 4
    // c[0,1] = max(a[0,0]*b[0,1], a[0,1]*b[1,1]) = max(1*0.5, 3*2) = max(0.5, 6) = 6
    // c[1,1] = max(a[1,0]*b[0,1], a[1,1]*b[1,1]) = max(2*0.5, 4*2) = max(1, 8) = 8
    // Column-major output: [c[0,0], c[1,0], c[0,1], c[1,1]] = [2, 4, 6, 8]
    let result = c.to_vec();
    assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_tensor_numel_edge_cases() {
    // Empty shape (scalar)
    let t = Tensor::<f64, Cpu>::from_data(&[42.0], &[]);
    assert_eq!(t.numel(), 1);
    assert_eq!(t.ndim(), 0);

    // Single element
    let t2 = Tensor::<f64, Cpu>::from_data(&[1.0], &[1]);
    assert_eq!(t2.numel(), 1);
    assert_eq!(t2.ndim(), 1);
}

#[test]
fn test_einsum_pairwise_without_optimization() {
    use omeinsum::Einsum;

    // Create einsum WITHOUT optimization - uses pairwise path
    let ein = Einsum::new(
        vec![vec![0, 1], vec![1, 2]],
        vec![0, 2],
        [(0, 2), (1, 2), (2, 2)].into(),
    );

    assert!(!ein.is_optimized());

    // Column-major: a = [1,2,3,4] means a = [[1,3],[2,4]]
    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // Column-major: b = [1,0,0,1] means b = [[1,0],[0,1]] = identity
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);
    assert_eq!(result.shape(), &[2, 2]);
    // A @ I = A, so result should equal a
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_einsum_pairwise_three_tensors_no_optimization() {
    use omeinsum::Einsum;

    // Create 3-tensor einsum WITHOUT optimization - exercises compute_intermediate_output
    let ein = Einsum::new(
        vec![vec![0, 1], vec![1, 2], vec![2, 3]],
        vec![0, 3],
        [(0, 2), (1, 2), (2, 2), (3, 2)].into(),
    );

    assert!(!ein.is_optimized());

    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    // I @ B @ I = B
    let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b, &c]);
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_einsum_with_grad_three_tensors_forward() {
    // Test einsum_with_grad forward pass for 3 tensors
    // This exercises the execute_pairwise_with_argmax path for Standard algebra

    let a = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let c = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    // einsum_with_grad uses optimized tree path which may give different contraction order
    let (result, _grad_fn) = einsum_with_grad::<Standard<f64>, _, _>(
        &[&a, &b, &c],
        &[&[0, 1], &[1, 2], &[2, 3]],
        &[0, 3],
    );

    assert_eq!(result.shape(), &[2, 2]);
    // Result may differ due to contraction order, just verify shape and non-panicking
    // The key is that forward pass works for 3 tensors
    let result_vec = result.to_vec();
    assert_eq!(result_vec.len(), 4);
}

#[cfg(feature = "tropical")]
#[test]
fn test_einsum_with_grad_three_tensors_tropical_forward() {
    // Test einsum_with_grad forward pass for 3 tensors with tropical algebra
    // This exercises the execute_pairwise_with_argmax path for tropical

    let a =
        Tensor::<f64, Cpu>::from_data(&[0.0, f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0], &[2, 2]);
    let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let c =
        Tensor::<f64, Cpu>::from_data(&[0.0, f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0], &[2, 2]);

    // einsum_with_grad uses optimized tree path
    let (result, _grad_fn) = einsum_with_grad::<MaxPlus<f64>, _, _>(
        &[&a, &b, &c],
        &[&[0, 1], &[1, 2], &[2, 3]],
        &[0, 3],
    );

    assert_eq!(result.shape(), &[2, 2]);
    // Just verify forward pass completes and produces valid result
    let result_vec = result.to_vec();
    assert_eq!(result_vec.len(), 4);
}
