//! Backward pass implementation for einsum gradients.
//!
//! This module provides gradient computation for tensor contractions:
//!
//! - **Standard algebra**: Uses the index-exchange trick for both unary and binary operations
//! - **Tropical algebras**: Uses argmax routing for binary operations (unary not yet supported)
//!
//! ## Index-Exchange Trick (Standard Algebra)
//!
//! For unary einsum operations:
//! - Forward: `y = einsum(ix -> iy, x)`
//! - Backward: `grad_x = einsum(iy -> ix, grad_y)`
//!
//! This elegantly handles trace, sum, diagonal, transpose, and their gradients.

use crate::algebra::{Algebra, Scalar};
use crate::backend::{Backend, BackendScalar, Storage};
use crate::tensor::Tensor;
use std::collections::HashMap;

use super::engine::{execute_unary_naive, normalize_binary_operand};

/// Compute gradient for a unary einsum operation.
///
/// Uses the index-exchange trick: backward(ix -> iy) = forward(iy -> ix).
///
/// # Arguments
///
/// * `grad_y` - Gradient of the output tensor
/// * `ix` - Input index labels
/// * `iy` - Output index labels
/// * `size_dict` - Mapping from index labels to sizes
///
/// # Returns
///
/// Gradient tensor with the same shape as the original input.
pub fn contract_unary_backward<A, T, B>(
    grad_y: &Tensor<T, B>,
    ix: &[usize],
    iy: &[usize],
    size_dict: &HashMap<usize, usize>,
) -> Tensor<T, B>
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar,
    B: Backend,
{
    // The elegant insight: gradient is just einsum with swapped indices!
    // Forward: y = einsum(ix -> iy, x)
    // Backward: grad_x = einsum(iy -> ix, grad_y)
    execute_unary_naive::<A, T, B>(grad_y, iy, ix, size_dict)
}

/// Compute gradient for a tropical unary einsum operation.
///
/// For tropical algebras, gradients are routed through the argmax:
/// only the "winning" input positions get the gradient.
///
/// # Arguments
///
/// * `grad_y` - Gradient of the output tensor
/// * `argmax` - Argmax tensor from forward pass (stores linear indices into input)
/// * `input_shape` - Shape of the original input tensor
///
/// # Returns
///
/// Gradient tensor with the same shape as the original input.
pub fn tropical_unary_backward<T, B>(
    grad_y: &Tensor<T, B>,
    argmax: &Tensor<u32, B>,
    input_shape: &[usize],
) -> Tensor<T, B>
where
    T: Scalar,
    B: Backend,
{
    // Create zero tensor with input shape (Scalar requires Default which gives 0 for numerics)
    let input_size: usize = input_shape.iter().product();
    let mut grad_data = vec![T::default(); input_size];

    // Scatter gradients to winner positions
    let grad_y_data = grad_y.to_vec();
    let argmax_data = argmax.to_vec();

    for (out_idx, &winner_pos) in argmax_data.iter().enumerate() {
        let grad_val = grad_y_data[out_idx];
        // For tropical, each output maps to exactly one winner, but multiple
        // outputs can share the same winner (e.g., in broadcasting).
        // We need AddAssign which Scalar provides.
        grad_data[winner_pos as usize] += grad_val;
    }

    Tensor::from_data_with_backend(&grad_data, input_shape, grad_y.backend().clone())
}

/// Compute gradients for a binary contraction.
///
/// Given the gradient of the output (grad_c), compute gradients for both inputs (a, b).
///
/// # Arguments
///
/// * `grad_c` - Gradient of the output tensor
/// * `a` - First input tensor
/// * `b` - Second input tensor
/// * `argmax` - Optional argmax tensor for tropical algebras
/// * `ia` - Index labels for first input
/// * `ib` - Index labels for second input
/// * `iy` - Output index labels
///
/// # Returns
///
/// Tuple of (grad_a, grad_b) gradients for the two inputs.
pub fn contract_binary_backward<A, T, B>(
    grad_c: &Tensor<T, B>,
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    argmax: Option<&Tensor<u32, B>>,
    ia: &[usize],
    ib: &[usize],
    iy: &[usize],
) -> (Tensor<T, B>, Tensor<T, B>)
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar + BackendScalar<B>,
    B: Backend,
{
    if A::needs_argmax() {
        // Tropical backward: route gradients through argmax
        let argmax = argmax.expect("Tropical backward requires argmax");
        tropical_backward::<A, T, B>(grad_c, a, b, argmax, ia, ib, iy)
    } else {
        // Standard backward: grad_a = grad_c @ b.T, grad_b = a.T @ grad_c
        standard_backward::<A, T, B>(grad_c, a, b, ia, ib, iy)
    }
}

/// Backward pass for standard (non-tropical) algebra.
///
/// For a contraction C = A @ B (with proper index handling):
/// - grad_a = grad_c @ b.T  (contracted with b's transpose)
/// - grad_b = a.T @ grad_c  (a's transpose contracted with grad_c)
fn standard_backward<A, T, B>(
    grad_c: &Tensor<T, B>,
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    ia: &[usize],
    ib: &[usize],
    iy: &[usize],
) -> (Tensor<T, B>, Tensor<T, B>)
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar + BackendScalar<B>,
    B: Backend,
{
    // For C[iy] = A[ia] @ B[ib] (contraction over shared indices in ia and ib not in iy):
    //
    // grad_A[ia] = grad_C[iy] @ B[ib] contracted appropriately
    // grad_B[ib] = A[ia] @ grad_C[iy] contracted appropriately
    //
    // The key insight is:
    // - To get grad_A, we contract grad_C with B, but now the contracted indices
    //   are the "right" indices of iy that came from ib, and the output should be ia
    // - To get grad_B, we contract A with grad_C, and the output should be ib

    // Find contracted indices (in both ia and ib, but not in iy)
    let contracted: Vec<usize> = ia
        .iter()
        .filter(|&i| ib.contains(i) && !iy.contains(i))
        .copied()
        .collect();

    // For grad_a: contract grad_c with b to get shape of a
    // grad_c has indices iy, b has indices ib
    // We want result with indices ia
    // The contraction should be over indices that are in both iy and ib (right indices)
    let grad_a = grad_c.contract_binary::<A>(b, iy, ib, ia);

    // For grad_b: contract a with grad_c to get shape of b
    // a has indices ia, grad_c has indices iy
    // We want result with indices ib
    // The contraction should be over indices that are in both ia and iy (left indices)
    let grad_b = a.contract_binary::<A>(grad_c, ia, iy, ib);

    let _ = contracted; // Mark as used (for documentation clarity)

    (grad_a, grad_b)
}

/// Backward pass for tropical (max/min-plus) algebra.
///
/// For tropical algebras, gradients are routed through the argmax:
/// only the "winning" element gets the gradient.
#[allow(clippy::extra_unused_type_parameters)]
fn tropical_backward<A, T, B>(
    grad_c: &Tensor<T, B>,
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    argmax: &Tensor<u32, B>,
    ia: &[usize],
    ib: &[usize],
    iy: &[usize],
) -> (Tensor<T, B>, Tensor<T, B>)
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar,
    B: Backend,
{
    // For tropical backward, we need to use the argmax to route gradients.
    // The argmax tells us which k index "won" for each output element.
    //
    // For a simple matmul C[i,j] = max_k (A[i,k] + B[k,j]):
    // - grad_A[i, argmax[i,j]] += grad_C[i,j] for each (i,j)
    // - grad_B[argmax[i,j], j] += grad_C[i,j] for each (i,j)

    // Get the shapes we need
    let a_shape = a.shape();
    let b_shape = b.shape();

    // For the simple 2D matmul case: C[i,j] = max_k (A[i,k] + B[k,j])
    // We need to scatter gradients using the argmax.
    //
    // Generic implementation using tensor indexing
    // (CPU backend has optimized internal methods, but this works for any backend)

    if a.ndim() == 2 && b.ndim() == 2 && grad_c.ndim() == 2 {
        // Matmul case: C[m,n] = A[m,k] * B[k,n]
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        // Make tensors contiguous for indexing
        let grad_c_contig = grad_c.contiguous();
        let argmax_contig = argmax.contiguous();

        // Get data as vectors for generic implementation
        let grad_c_vec = grad_c_contig.to_vec();
        let argmax_vec = argmax_contig.to_vec();

        // Initialize gradient storage
        let mut grad_a_vec = vec![T::default(); m * k];
        let mut grad_b_vec = vec![T::default(); k * n];

        // Route gradients through argmax (column-major indexing)
        // Column-major: element (i, j) is at index j * nrows + i
        for j in 0..n {
            for i in 0..m {
                let idx = j * m + i;
                let winner_k = argmax_vec[idx] as usize;
                let gc = grad_c_vec[idx];

                // grad_a[i, winner_k] += grad_c[i, j]
                grad_a_vec[winner_k * m + i] += gc;

                // grad_b[winner_k, j] += grad_c[i, j]
                grad_b_vec[j * k + winner_k] += gc;
            }
        }

        let grad_a = Tensor::from_storage(
            B::Storage::from_slice(&grad_a_vec),
            a_shape,
            a.backend().clone(),
        );
        let grad_b = Tensor::from_storage(
            B::Storage::from_slice(&grad_b_vec),
            b_shape,
            b.backend().clone(),
        );

        (grad_a, grad_b)
    } else {
        // For higher-dimensional cases, we would need more complex logic
        // For now, this handles the common matmul case
        let _ = (ia, ib, iy); // Mark as used
        unimplemented!("Tropical backward only implemented for 2D matmul currently");
    }
}

// ============================================================================
// CacheTree and cost_and_gradient implementation
// ============================================================================
//
// Port of Julia OMEinsum's bp.jl for computing gradients efficiently.
// The key insight is to cache intermediate contraction results during forward
// pass and reuse them during backward pass.

use omeco::NestedEinsum;

use crate::einsum::Einsum;

/// Cache tree for storing intermediate contraction results.
///
/// Isomorphic to the contraction tree structure from `NestedEinsum`.
/// Each node stores the tensor computed at that step, plus optional
/// argmax for tropical algebras.
///
/// # Example
///
/// For a contraction `(A @ B) @ C`:
/// ```text
/// CacheTree {
///     content: result of (A @ B) @ C
///     argmax: Some(...) if tropical
///     siblings: [
///         CacheTree { content: result of A @ B, siblings: [A_cache, B_cache] },
///         CacheTree { content: C, siblings: [] }
///     ]
/// }
/// ```
pub struct CacheTree<T: Scalar, B: Backend> {
    /// The cached tensor result at this node
    pub content: Tensor<T, B>,
    /// Argmax tensor for tropical algebras (optional)
    pub argmax: Option<Tensor<u32, B>>,
    /// Child cache trees (called "siblings" in Julia for tree siblings)
    pub siblings: Vec<CacheTree<T, B>>,
}

impl<T: Scalar, B: Backend> CacheTree<T, B> {
    /// Create a leaf cache (for input tensors).
    pub fn leaf(tensor: Tensor<T, B>) -> Self {
        CacheTree {
            content: tensor,
            argmax: None,
            siblings: Vec::new(),
        }
    }

    /// Create an internal node cache.
    pub fn node(
        content: Tensor<T, B>,
        argmax: Option<Tensor<u32, B>>,
        siblings: Vec<Self>,
    ) -> Self {
        CacheTree {
            content,
            argmax,
            siblings,
        }
    }
}

/// Compute einsum with caching for backward pass.
///
/// Recursively contracts tensors following the contraction tree,
/// caching intermediate results for gradient computation.
///
/// # Arguments
///
/// * `code` - The contraction tree from optimization
/// * `tensors` - Input tensors
/// * `size_dict` - Dimension sizes
///
/// # Returns
///
/// CacheTree containing all intermediate results.
#[allow(clippy::only_used_in_recursion)]
pub fn cached_einsum<A, T, B>(
    code: &NestedEinsum<usize>,
    tensors: &[&Tensor<T, B>],
    size_dict: &HashMap<usize, usize>,
) -> CacheTree<T, B>
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar + BackendScalar<B>,
    B: Backend,
{
    match code {
        NestedEinsum::Leaf { tensor_index } => {
            // For leaf nodes, just cache the input tensor
            CacheTree::leaf(tensors[*tensor_index].clone())
        }
        NestedEinsum::Node { args, eins } => {
            // Recursively cache children
            let children: Vec<CacheTree<T, B>> = args
                .iter()
                .map(|arg| cached_einsum::<A, T, B>(arg, tensors, size_dict))
                .collect();

            // Contract the children's results
            assert_eq!(args.len(), 2, "Expected binary contraction tree");

            let left = &children[0].content;
            let right = &children[1].content;
            let ia = &eins.ixs[0];
            let ib = &eins.ixs[1];
            let iy = &eins.iy;
            let (left, ia) = normalize_binary_operand::<A, T, B>(left, ia, ib, iy, size_dict);
            let (right, ib) =
                normalize_binary_operand::<A, T, B>(right, ib, ia.as_slice(), iy, size_dict);

            let (result, argmax) = if A::needs_argmax() {
                let (r, am) = left.contract_binary_with_argmax::<A>(&right, &ia, &ib, iy);
                (r, Some(am))
            } else {
                (left.contract_binary::<A>(&right, &ia, &ib, iy), None)
            };

            CacheTree::node(result, argmax, children)
        }
    }
}

/// Back-propagate gradients through the cache tree.
///
/// Given a gradient on the output, propagates it backward through the tree,
/// computing gradients for all intermediate nodes.
///
/// # Arguments
///
/// * `code` - The contraction tree
/// * `cache` - Cached forward results
/// * `dy` - Gradient on the output
/// * `size_dict` - Dimension sizes
///
/// # Returns
///
/// CacheTree where each node's `content` is the gradient at that position.
#[allow(clippy::only_used_in_recursion)]
pub fn back_propagate<A, T, B>(
    code: &NestedEinsum<usize>,
    cache: &CacheTree<T, B>,
    dy: &Tensor<T, B>,
    size_dict: &HashMap<usize, usize>,
) -> CacheTree<T, B>
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar + BackendScalar<B>,
    B: Backend,
{
    match code {
        NestedEinsum::Leaf { .. } => {
            // For leaves, the gradient is just dy
            CacheTree::leaf(dy.clone())
        }
        NestedEinsum::Node { args, eins } => {
            assert_eq!(args.len(), 2, "Expected binary contraction tree");

            // Get cached forward values
            let left = &cache.siblings[0].content;
            let right = &cache.siblings[1].content;
            let argmax = cache.argmax.as_ref();

            let ia = &eins.ixs[0];
            let ib = &eins.ixs[1];
            let iy = &eins.iy;

            // Compute gradients for left and right children
            let (grad_left, grad_right) =
                contract_binary_backward::<A, T, B>(dy, left, right, argmax, ia, ib, iy);

            // Recursively back-propagate through children
            let child_grads: Vec<CacheTree<T, B>> = vec![
                back_propagate::<A, T, B>(&args[0], &cache.siblings[0], &grad_left, size_dict),
                back_propagate::<A, T, B>(&args[1], &cache.siblings[1], &grad_right, size_dict),
            ];

            CacheTree::node(dy.clone(), None, child_grads)
        }
    }
}

/// Extract gradients from leaf nodes of a gradient tree.
///
/// # Arguments
///
/// * `code` - The contraction tree
/// * `grad_tree` - The gradient tree from back_propagate
/// * `num_inputs` - Total number of input tensors; used to pre-allocate the
///   result vector and index gradients by their original tensor positions.
///
/// # Returns
///
/// Vector of gradients, one per input tensor in original order.
pub fn extract_leaves<T: Scalar, B: Backend>(
    code: &NestedEinsum<usize>,
    grad_tree: &CacheTree<T, B>,
    num_inputs: usize,
) -> Vec<Tensor<T, B>> {
    let mut result: Vec<Option<Tensor<T, B>>> = vec![None; num_inputs];
    extract_leaves_impl(code, grad_tree, &mut result);
    result.into_iter().map(|opt| opt.unwrap()).collect()
}

fn extract_leaves_impl<T: Scalar, B: Backend>(
    code: &NestedEinsum<usize>,
    grad_tree: &CacheTree<T, B>,
    result: &mut [Option<Tensor<T, B>>],
) {
    match code {
        NestedEinsum::Leaf { tensor_index } => {
            result[*tensor_index] = Some(grad_tree.content.clone());
        }
        NestedEinsum::Node { args, .. } => {
            for (arg, sibling) in args.iter().zip(grad_tree.siblings.iter()) {
                extract_leaves_impl(arg, sibling, result);
            }
        }
    }
}

/// Compute cost and gradients in a single optimized pass.
///
/// Like Julia's `OMEinsum.cost_and_gradient`, this caches intermediate
/// contraction results during forward pass and reuses them for backward.
///
/// # Arguments
///
/// * `code` - The einsum specification (must be optimized)
/// * `xs` - Input tensors
/// * `dy` - Optional gradient seed. If `None`, output must be scalar and `1.0` is used.
///
/// # Returns
///
/// `(cost, grads)` where:
/// - `cost` - The forward contraction result
/// - `grads` - Vector of gradients, one per input tensor
///
/// # Panics
///
/// Panics if `dy` is `None` and the output is not a scalar.
///
/// # Example
///
/// ```rust
/// use omeinsum::{Einsum, Tensor, Cpu, cost_and_gradient};
/// use omeinsum::algebra::Standard;
/// use std::collections::HashMap;
///
/// // Trace: A[i,i] -> scalar
/// let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
///
/// let sizes: HashMap<usize, usize> = [(0, 2)].into();
/// let mut ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);
/// ein.optimize_greedy();
///
/// let (cost, grads) = cost_and_gradient::<Standard<f64>, _, _>(&ein, &[&a], None);
/// assert_eq!(grads.len(), 1);
/// assert!((cost.to_vec()[0] - 5.0).abs() < 1e-10); // trace = 1 + 4 = 5
/// ```
pub fn cost_and_gradient<A, T, B>(
    code: &Einsum<usize>,
    xs: &[&Tensor<T, B>],
    dy: Option<&Tensor<T, B>>,
) -> (Tensor<T, B>, Vec<Tensor<T, B>>)
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar + BackendScalar<B>,
    B: Backend,
{
    // Require optimization
    let tree = code
        .contraction_tree()
        .expect("cost_and_gradient requires optimized Einsum. Call optimize_greedy() first.");

    // Handle single tensor case specially
    if xs.len() == 1 {
        return cost_and_gradient_unary::<A, T, B>(code, xs[0], dy);
    }

    // Forward pass with caching
    let cache = cached_einsum::<A, T, B>(tree, xs, &code.size_dict);

    // Initialize dy if not provided
    let dy_tensor = match dy {
        Some(d) => d.clone(),
        None => {
            // Output must be scalar
            assert!(
                code.iy.is_empty(),
                "cost_and_gradient: output must be scalar when dy is None. Got output indices: {:?}",
                code.iy
            );
            Tensor::from_data_with_backend(
                &[A::one().to_scalar()],
                &[],
                cache.content.backend().clone(),
            )
        }
    };

    // Backward pass through the tree
    let grad_tree = back_propagate::<A, T, B>(tree, &cache, &dy_tensor, &code.size_dict);

    // Extract leaf gradients
    let grads = extract_leaves(tree, &grad_tree, xs.len());

    (cache.content, grads)
}

/// Handle unary case for cost_and_gradient.
fn cost_and_gradient_unary<A, T, B>(
    code: &Einsum<usize>,
    x: &Tensor<T, B>,
    dy: Option<&Tensor<T, B>>,
) -> (Tensor<T, B>, Vec<Tensor<T, B>>)
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar + BackendScalar<B>,
    B: Backend,
{
    // Forward pass
    let (result, argmax) = if A::needs_argmax() {
        super::engine::execute_unary_with_argmax::<A, T, B>(
            x,
            &code.ixs[0],
            &code.iy,
            &code.size_dict,
        )
    } else {
        (
            super::engine::execute_unary_naive::<A, T, B>(
                x,
                &code.ixs[0],
                &code.iy,
                &code.size_dict,
            ),
            Tensor::from_data_with_backend(&[0u32], &[], x.backend().clone()), // Dummy, won't be used
        )
    };

    // Initialize dy if not provided
    let dy_tensor = match dy {
        Some(d) => d.clone(),
        None => {
            assert!(
                code.iy.is_empty(),
                "cost_and_gradient: output must be scalar when dy is None"
            );
            Tensor::from_data_with_backend(&[A::one().to_scalar()], &[], x.backend().clone())
        }
    };

    // Backward pass
    let grad = if A::needs_argmax() {
        tropical_unary_backward::<T, B>(&dy_tensor, &argmax, x.shape())
    } else {
        contract_unary_backward::<A, T, B>(&dy_tensor, &code.ixs[0], &code.iy, &code.size_dict)
    };

    (result, vec![grad])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Standard;
    use crate::backend::Cpu;

    #[cfg(feature = "tropical")]
    use crate::algebra::MaxPlus;

    #[test]
    fn test_standard_backward_matmul() {
        // Test backward pass for standard matmul
        // C[i,k] = sum_j A[i,j] * B[j,k]
        //
        // For forward: A[2,3] @ B[3,2] -> C[2,2]
        // grad_A = grad_C @ B.T  -> [2,2] @ [2,3] -> [2,3]
        // grad_B = A.T @ grad_C  -> [3,2] @ [2,2] -> [3,2]

        // Column-major: data [1,2,3,4,5,6] for shape [2,3] represents:
        // A = [[1, 3, 5],
        //      [2, 4, 6]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // Column-major: data [1,2,3,4,5,6] for shape [3,2] represents:
        // B = [[1, 4],
        //      [2, 5],
        //      [3, 6]]
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

        // grad_c is all ones [2, 2] -> [[1, 1], [1, 1]]
        let grad_c = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

        let ia = &[0, 1]; // i, j
        let ib = &[1, 2]; // j, k
        let iy = &[0, 2]; // i, k

        let (grad_a, grad_b) =
            contract_binary_backward::<Standard<f32>, _, _>(&grad_c, &a, &b, None, ia, ib, iy);

        // grad_A = grad_C @ B.T
        // B.T = [[1, 2, 3], [4, 5, 6]]
        // grad_A = [[1,1],[1,1]] @ [[1,2,3],[4,5,6]] = [[5,7,9],[5,7,9]]
        // In column-major: [5, 5, 7, 7, 9, 9]
        assert_eq!(grad_a.shape(), &[2, 3]);
        assert_eq!(grad_a.to_vec(), vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);

        // grad_B = A.T @ grad_C
        // A.T = [[1, 2], [3, 4], [5, 6]]
        // grad_B = [[1,2],[3,4],[5,6]] @ [[1,1],[1,1]] = [[3,3],[7,7],[11,11]]
        // In column-major: [3, 7, 11, 3, 7, 11]
        assert_eq!(grad_b.shape(), &[3, 2]);
        assert_eq!(grad_b.to_vec(), vec![3.0, 7.0, 11.0, 3.0, 7.0, 11.0]);
    }

    #[test]
    fn test_standard_backward_square_matmul() {
        // Simpler case: 2x2 matrices
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        // grad_c is identity matrix
        let grad_c = Tensor::<f32, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

        let ia = &[0, 1];
        let ib = &[1, 2];
        let iy = &[0, 2];

        let (grad_a, grad_b) =
            contract_binary_backward::<Standard<f32>, _, _>(&grad_c, &a, &b, None, ia, ib, iy);

        // grad_A = grad_C @ B.T = [[1,0],[0,1]] @ [[5,7],[6,8]] = [[5,7],[6,8]]
        assert_eq!(grad_a.shape(), &[2, 2]);
        assert_eq!(grad_a.to_vec(), vec![5.0, 7.0, 6.0, 8.0]);

        // grad_B = A.T @ grad_C = [[1,3],[2,4]] @ [[1,0],[0,1]] = [[1,3],[2,4]]
        assert_eq!(grad_b.shape(), &[2, 2]);
        assert_eq!(grad_b.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_tropical_backward_matmul() {
        // Test backward pass for MaxPlus matmul
        // C[i,j] = max_k (A[i,k] + B[k,j])
        //
        // Column-major: [1,2,3,4] for shape [2,2] -> [[1,3],[2,4]]
        // A = [[1, 3], [2, 4]], B = [[1, 3], [2, 4]]
        // C[0,0] = max(1+1, 3+2) = max(2, 5) = 5, argmax = 1
        // C[1,0] = max(2+1, 4+2) = max(3, 6) = 6, argmax = 1
        // C[0,1] = max(1+3, 3+4) = max(4, 7) = 7, argmax = 1
        // C[1,1] = max(2+3, 4+4) = max(5, 8) = 8, argmax = 1

        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        // Compute forward using contract_binary_with_argmax
        let (c, argmax) =
            a.contract_binary_with_argmax::<MaxPlus<f32>>(&b, &[0, 1], &[1, 2], &[0, 2]);
        assert_eq!(c.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(argmax.to_vec(), vec![1, 1, 1, 1]);

        // grad_c is all ones
        let grad_c = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

        let ia = &[0, 1];
        let ib = &[1, 2];
        let iy = &[0, 2];

        let (grad_a, grad_b) = contract_binary_backward::<MaxPlus<f32>, _, _>(
            &grad_c,
            &a,
            &b,
            Some(&argmax),
            ia,
            ib,
            iy,
        );

        // For tropical backward with argmax all = 1:
        // grad_A[i, argmax[i,j]] += grad_C[i,j]
        // grad_A[0,1] = grad_C[0,0] + grad_C[0,1] = 2
        // grad_A[1,1] = grad_C[1,0] + grad_C[1,1] = 2
        // grad_A = [[0, 2], [0, 2]] in column-major: [0, 0, 2, 2]
        assert_eq!(grad_a.shape(), &[2, 2]);
        assert_eq!(grad_a.to_vec(), vec![0.0, 0.0, 2.0, 2.0]);

        // grad_B[argmax[i,j], j] += grad_C[i,j]
        // grad_B[1,0] = grad_C[0,0] + grad_C[1,0] = 2
        // grad_B[1,1] = grad_C[0,1] + grad_C[1,1] = 2
        // grad_B = [[0, 0], [2, 2]] in column-major: [0, 2, 0, 2]
        assert_eq!(grad_b.shape(), &[2, 2]);
        assert_eq!(grad_b.to_vec(), vec![0.0, 2.0, 0.0, 2.0]);
    }

    // Test only with tropical feature, not tropical-kernels, because the optimized
    // tropical-gemm kernels may have different iteration order for small matrices
    #[cfg(all(feature = "tropical", not(feature = "tropical-kernels")))]
    #[test]
    fn test_tropical_backward_different_winners() {
        // Test case where different elements have different winners
        // Column-major: [5,1,1,5] for shape [2,2] -> [[5,1],[1,5]]
        // Column-major: [1,5,5,1] for shape [2,2] -> [[1,5],[5,1]]
        // A = [[5, 1], [1, 5]], B = [[1, 5], [5, 1]]
        // C[0,0] = max(5+1, 1+5) = max(6, 6) = 6, argmax = 0 (first wins on tie)
        // C[1,0] = max(1+1, 5+5) = max(2, 10) = 10, argmax = 1
        // C[0,1] = max(5+5, 1+1) = max(10, 2) = 10, argmax = 0
        // C[1,1] = max(1+5, 5+1) = max(6, 6) = 6, argmax = 0

        let a = Tensor::<f32, Cpu>::from_data(&[5.0, 1.0, 1.0, 5.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 5.0, 5.0, 1.0], &[2, 2]);

        let (c, argmax) =
            a.contract_binary_with_argmax::<MaxPlus<f32>>(&b, &[0, 1], &[1, 2], &[0, 2]);
        // Column-major: [6, 10, 10, 6]
        assert_eq!(c.to_vec(), vec![6.0, 10.0, 10.0, 6.0]);
        // Column-major argmax: [0, 1, 0, 0]
        // argmax[0,0]=0, argmax[1,0]=1, argmax[0,1]=0, argmax[1,1]=0
        assert_eq!(argmax.to_vec(), vec![0, 1, 0, 0]);

        let grad_c = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

        let ia = &[0, 1];
        let ib = &[1, 2];
        let iy = &[0, 2];

        let (grad_a, grad_b) = contract_binary_backward::<MaxPlus<f32>, _, _>(
            &grad_c,
            &a,
            &b,
            Some(&argmax),
            ia,
            ib,
            iy,
        );

        // argmax (column-major) = [0, 1, 0, 0]
        // grad_A[i, argmax[i,k]] += grad_C[i,k]
        // idx=0 (i=0,k=0): argmax=0, grad_A[0,0] += 1
        // idx=1 (i=1,k=0): argmax=1, grad_A[1,1] += 1
        // idx=2 (i=0,k=1): argmax=0, grad_A[0,0] += 1
        // idx=3 (i=1,k=1): argmax=0, grad_A[1,0] += 1
        // grad_A = [[2, 0], [1, 1]] in column-major: [2, 1, 0, 1]
        assert_eq!(grad_a.shape(), &[2, 2]);
        assert_eq!(grad_a.to_vec(), vec![2.0, 1.0, 0.0, 1.0]);

        // grad_B[argmax[i,k], k] += grad_C[i,k]
        // idx=0: argmax=0, grad_B[0,0] += 1
        // idx=1: argmax=1, grad_B[1,0] += 1
        // idx=2: argmax=0, grad_B[0,1] += 1
        // idx=3: argmax=0, grad_B[0,1] += 1
        // grad_B = [[1, 2], [1, 0]] in column-major: [1, 1, 2, 0]
        assert_eq!(grad_b.shape(), &[2, 2]);
        assert_eq!(grad_b.to_vec(), vec![1.0, 1.0, 2.0, 0.0]);
    }

    // ========================================================================
    // bpcheck: Finite-difference gradient validation utility
    // ========================================================================
    //
    // Port of Julia OMEinsum's bpcheck function for verifying gradients.
    // Uses the relation: f(x - η*g) ≈ f(x) - η|g|² for gradient verification.

    use crate::einsum;
    use crate::einsum::einsum_with_grad;

    /// Check unary gradient via finite differences.
    ///
    /// Returns (passed, max_error) where passed is true if max_error < tol.
    fn bpcheck_unary(
        input: &Tensor<f64, Cpu>,
        ix: &[usize],
        iy: &[usize],
        eta: f64,
        tol: f64,
    ) -> (bool, f64) {
        // Forward and backward pass
        let (result, grad_fn) = einsum_with_grad::<Standard<f64>, _, _>(&[input], &[ix], iy);

        // Create gradient output of all ones
        let grad_output =
            Tensor::<f64, Cpu>::from_data(&vec![1.0; result.to_vec().len()], result.shape());
        let grads = grad_fn.backward::<Standard<f64>>(&grad_output, &[input]);
        let grad = &grads[0];

        // Compute |g|² and expected change
        let grad_vec = grad.to_vec();
        let grad_sq_sum: f64 = grad_vec.iter().map(|x| x * x).sum();
        let expected_change = eta * grad_sq_sum;

        // Compute f(x) and f(x - η*g)
        let f_x: f64 = result.to_vec().iter().sum();

        let input_vec = input.to_vec();
        let perturbed_vec: Vec<f64> = input_vec
            .iter()
            .zip(grad_vec.iter())
            .map(|(x, g)| x - eta * g)
            .collect();
        let perturbed = Tensor::<f64, Cpu>::from_data(&perturbed_vec, input.shape());
        let f_x_perturbed: f64 = einsum::<Standard<f64>, _, _>(&[&perturbed], &[ix], iy)
            .to_vec()
            .iter()
            .sum();

        let actual_change = f_x - f_x_perturbed;
        let error = (actual_change - expected_change).abs();
        let passed = error < tol || (expected_change > 0.0 && error / expected_change < 0.01);

        (passed, error)
    }

    /// Check binary gradient via finite differences.
    ///
    /// Returns (passed, max_error_a, max_error_b).
    fn bpcheck_binary(
        a: &Tensor<f64, Cpu>,
        b: &Tensor<f64, Cpu>,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
        eta: f64,
        tol: f64,
    ) -> (bool, f64, f64) {
        let (result, grad_fn) = einsum_with_grad::<Standard<f64>, _, _>(&[a, b], &[ia, ib], iy);

        let grad_output =
            Tensor::<f64, Cpu>::from_data(&vec![1.0; result.to_vec().len()], result.shape());
        let grads = grad_fn.backward::<Standard<f64>>(&grad_output, &[a, b]);

        // Check gradient of A
        let grad_a_vec = grads[0].to_vec();
        let a_vec = a.to_vec();
        let grad_a_sq_sum: f64 = grad_a_vec.iter().map(|x| x * x).sum();
        let expected_a = eta * grad_a_sq_sum;

        let perturbed_a_vec: Vec<f64> = a_vec
            .iter()
            .zip(grad_a_vec.iter())
            .map(|(x, g)| x - eta * g)
            .collect();
        let perturbed_a = Tensor::<f64, Cpu>::from_data(&perturbed_a_vec, a.shape());

        let f_orig: f64 = result.to_vec().iter().sum();
        let f_perturbed_a: f64 = einsum::<Standard<f64>, _, _>(&[&perturbed_a, b], &[ia, ib], iy)
            .to_vec()
            .iter()
            .sum();

        let actual_a = f_orig - f_perturbed_a;
        let error_a = (actual_a - expected_a).abs();

        // Check gradient of B
        let grad_b_vec = grads[1].to_vec();
        let b_vec = b.to_vec();
        let grad_b_sq_sum: f64 = grad_b_vec.iter().map(|x| x * x).sum();
        let expected_b = eta * grad_b_sq_sum;

        let perturbed_b_vec: Vec<f64> = b_vec
            .iter()
            .zip(grad_b_vec.iter())
            .map(|(x, g)| x - eta * g)
            .collect();
        let perturbed_b = Tensor::<f64, Cpu>::from_data(&perturbed_b_vec, b.shape());

        let f_perturbed_b: f64 = einsum::<Standard<f64>, _, _>(&[a, &perturbed_b], &[ia, ib], iy)
            .to_vec()
            .iter()
            .sum();

        let actual_b = f_orig - f_perturbed_b;
        let error_b = (actual_b - expected_b).abs();

        let passed_a = error_a < tol || (expected_a > 0.0 && error_a / expected_a < 0.01);
        let passed_b = error_b < tol || (expected_b > 0.0 && error_b / expected_b < 0.01);

        (passed_a && passed_b, error_a, error_b)
    }

    // ========================================================================
    // Unary Gradient Tests (ported from Julia OMEinsum autodiff.jl)
    // ========================================================================

    #[test]
    fn test_bpcheck_trace() {
        // Trace: ii -> (sum of diagonal)
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let (passed, error) = bpcheck_unary(&a, &[0, 0], &[], 1e-5, 1e-8);
        assert!(passed, "Trace gradient failed with error {}", error);
    }

    #[test]
    fn test_bpcheck_trace_4d() {
        // 4D trace: ijji -> (trace over both pairs)
        let a = Tensor::<f64, Cpu>::from_data(
            &(1..=16).map(|x| x as f64).collect::<Vec<_>>(),
            &[2, 2, 2, 2],
        );
        let (passed, error) = bpcheck_unary(&a, &[0, 1, 1, 0], &[], 1e-5, 1e-8);
        assert!(passed, "4D trace gradient failed with error {}", error);
    }

    #[test]
    fn test_bpcheck_partial_trace() {
        // Partial trace: ijji -> i (trace over j, keep i)
        // Actually: ibbj -> ij (trace over repeated b)
        let a = Tensor::<f64, Cpu>::from_data(
            &(1..=16).map(|x| x as f64).collect::<Vec<_>>(),
            &[2, 2, 2, 2],
        );
        let (passed, error) = bpcheck_unary(&a, &[0, 1, 1, 2], &[0, 2], 1e-5, 1e-8);
        assert!(passed, "Partial trace gradient failed with error {}", error);
    }

    #[test]
    fn test_bpcheck_diagonal() {
        // Diagonal extraction: ibbj -> ibj (extract diagonal along b)
        let a = Tensor::<f64, Cpu>::from_data(
            &(1..=16).map(|x| x as f64).collect::<Vec<_>>(),
            &[2, 2, 2, 2],
        );
        let (passed, error) = bpcheck_unary(&a, &[0, 1, 1, 2], &[0, 1, 2], 1e-5, 1e-8);
        assert!(passed, "Diagonal gradient failed with error {}", error);
    }

    #[test]
    fn test_bpcheck_permutation() {
        // Permutation: ij -> ji (transpose)
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let (passed, error) = bpcheck_unary(&a, &[0, 1], &[1, 0], 1e-5, 1e-8);
        assert!(passed, "Transpose gradient failed with error {}", error);
    }

    #[test]
    fn test_bpcheck_permutation_4d() {
        // 4D permutation: ijkl -> klij
        let a = Tensor::<f64, Cpu>::from_data(
            &(1..=16).map(|x| x as f64).collect::<Vec<_>>(),
            &[2, 2, 2, 2],
        );
        let (passed, error) = bpcheck_unary(&a, &[0, 1, 2, 3], &[2, 3, 0, 1], 1e-5, 1e-8);
        assert!(
            passed,
            "4D permutation gradient failed with error {}",
            error
        );
    }

    #[test]
    fn test_bpcheck_sum_reduction() {
        // Sum: ijk -> ij (sum over k)
        let a = Tensor::<f64, Cpu>::from_data(
            &(1..=8).map(|x| x as f64).collect::<Vec<_>>(),
            &[2, 2, 2],
        );
        let (passed, error) = bpcheck_unary(&a, &[0, 1, 2], &[0, 1], 1e-5, 1e-8);
        assert!(passed, "Sum reduction gradient failed with error {}", error);
    }

    #[test]
    fn test_bpcheck_sum_all() {
        // Sum all: ij -> (scalar)
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let (passed, error) = bpcheck_unary(&a, &[0, 1], &[], 1e-5, 1e-8);
        assert!(passed, "Sum all gradient failed with error {}", error);
    }

    // ========================================================================
    // Binary Gradient Tests (ported from Julia OMEinsum autodiff.jl)
    // ========================================================================

    #[test]
    fn test_bpcheck_matmul() {
        // Matrix multiplication: ij,jk -> ik
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let (passed, err_a, err_b) = bpcheck_binary(&a, &b, &[0, 1], &[1, 2], &[0, 2], 1e-5, 1e-8);
        assert!(
            passed,
            "Matmul gradient failed: err_a={}, err_b={}",
            err_a, err_b
        );
    }

    #[test]
    fn test_bpcheck_matmul_chain() {
        // Matrix chain: ij,jk,kl -> il
        // Note: bpcheck_binary only tests two inputs at a time
        // Here we test the first pair
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let (passed, err_a, err_b) = bpcheck_binary(&a, &b, &[0, 1], &[1, 2], &[0, 2], 1e-5, 1e-8);
        assert!(
            passed,
            "Matmul chain gradient failed: err_a={}, err_b={}",
            err_a, err_b
        );
    }

    #[test]
    fn test_bpcheck_hadamard() {
        // Hadamard product: ij,ij -> ij (element-wise)
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let (passed, err_a, err_b) = bpcheck_binary(&a, &b, &[0, 1], &[0, 1], &[0, 1], 1e-5, 1e-8);
        assert!(
            passed,
            "Hadamard gradient failed: err_a={}, err_b={}",
            err_a, err_b
        );
    }

    #[test]
    fn test_bpcheck_outer_product() {
        // Outer product: ij,kl -> ijkl
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let (passed, err_a, err_b) =
            bpcheck_binary(&a, &b, &[0, 1], &[2, 3], &[0, 1, 2, 3], 1e-5, 1e-8);
        assert!(
            passed,
            "Outer product gradient failed: err_a={}, err_b={}",
            err_a, err_b
        );
    }

    #[test]
    fn test_bpcheck_contract_to_scalar() {
        // Contract to scalar: ij,ij -> (full contraction)
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let (passed, err_a, err_b) = bpcheck_binary(&a, &b, &[0, 1], &[0, 1], &[], 1e-5, 1e-8);
        assert!(
            passed,
            "Contract to scalar gradient failed: err_a={}, err_b={}",
            err_a, err_b
        );
    }

    #[test]
    fn test_bpcheck_star_contraction() {
        // Star contraction: ai,bi,ci -> abc (3 tensors share index i)
        // Testing first two tensors: ai,bi -> ab
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        // a has indices (0, 1) meaning "a=0, i=1"
        // b has indices (2, 1) meaning "b=2, i=1"
        // output is (0, 2) meaning "ab"
        let (passed, err_a, err_b) = bpcheck_binary(&a, &b, &[0, 1], &[2, 1], &[0, 2], 1e-5, 1e-8);
        assert!(
            passed,
            "Star contraction gradient failed: err_a={}, err_b={}",
            err_a, err_b
        );
    }

    #[test]
    fn test_bpcheck_tensor_contraction() {
        // Tensor contraction: ijkl,kl -> ij
        let t = Tensor::<f64, Cpu>::from_data(
            &(1..=16).map(|x| x as f64).collect::<Vec<_>>(),
            &[2, 2, 2, 2],
        );
        let m = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let (passed, err_t, err_m) =
            bpcheck_binary(&t, &m, &[0, 1, 2, 3], &[2, 3], &[0, 1], 1e-5, 1e-8);
        assert!(
            passed,
            "Tensor contraction gradient failed: err_t={}, err_m={}",
            err_t, err_m
        );
    }

    #[test]
    fn test_bpcheck_batched_matmul() {
        // Batched matmul: bij,bjk -> bik
        // Use smaller values to reduce numerical error accumulation
        let a = Tensor::<f64, Cpu>::from_data(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            &[2, 2, 3],
        );
        let b = Tensor::<f64, Cpu>::from_data(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            &[2, 3, 2],
        );
        // Use looser tolerance for 3D tensors due to accumulated numerical error
        let (passed, err_a, err_b) =
            bpcheck_binary(&a, &b, &[0, 1, 2], &[0, 2, 3], &[0, 1, 3], 1e-5, 1e-4);
        assert!(
            passed,
            "Batched matmul gradient failed: err_a={}, err_b={}",
            err_a, err_b
        );
    }

    #[test]
    fn test_bpcheck_matvec() {
        // Matrix-vector: ij,j -> i
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let v = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);
        let (passed, err_a, err_v) = bpcheck_binary(&a, &v, &[0, 1], &[1], &[0], 1e-5, 1e-8);
        assert!(
            passed,
            "Matvec gradient failed: err_a={}, err_v={}",
            err_a, err_v
        );
    }

    // ========================================================================
    // Nested/Chain Gradient Tests
    // ========================================================================

    #[test]
    fn test_nested_chain_gradient_via_composition() {
        // Test gradient through a chain: (A @ B) @ C
        // First compute A @ B, then multiply by C
        // Verify gradient composition works correctly
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[0.5, 0.6, 0.7, 0.8], &[2, 2]);
        let c = Tensor::<f64, Cpu>::from_data(&[0.1, 0.2, 0.3, 0.4], &[2, 2]);

        // Step 1: A @ B -> AB
        let ab = einsum::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

        // Step 2: AB @ C -> result
        let (result, grad_fn) =
            einsum_with_grad::<Standard<f64>, _, _>(&[&ab, &c], &[&[0, 1], &[1, 2]], &[0, 2]);

        // Gradient with respect to final output
        let grad_output = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let grads = grad_fn.backward::<Standard<f64>>(&grad_output, &[&ab, &c]);

        // Verify gradient shapes
        assert_eq!(
            grads[0].shape(),
            ab.shape(),
            "Gradient of AB should match AB shape"
        );
        assert_eq!(
            grads[1].shape(),
            c.shape(),
            "Gradient of C should match C shape"
        );

        // Verify gradients are non-zero
        let grad_ab_sum: f64 = grads[0].to_vec().iter().sum();
        let grad_c_sum: f64 = grads[1].to_vec().iter().sum();
        assert!(grad_ab_sum.abs() > 0.0, "Gradient of AB should be non-zero");
        assert!(grad_c_sum.abs() > 0.0, "Gradient of C should be non-zero");

        // Verify result is computed
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_nested_trace_chain() {
        // Test: trace(A @ B) - unary after binary
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

        // A @ B -> AB
        let (ab, grad_fn_ab) =
            einsum_with_grad::<Standard<f64>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);

        // trace(AB) -> scalar
        let (trace_result, grad_fn_trace) =
            einsum_with_grad::<Standard<f64>, _, _>(&[&ab], &[&[0, 0]], &[]);

        // Gradient of trace is identity matrix
        let grad_scalar = Tensor::<f64, Cpu>::from_data(&[1.0], &[]);
        let grads_trace = grad_fn_trace.backward::<Standard<f64>>(&grad_scalar, &[&ab]);

        // Gradient of AB from trace
        let grad_ab = &grads_trace[0];
        assert_eq!(grad_ab.shape(), &[2, 2]);
        // Trace gradient is identity: [[1,0],[0,1]]
        assert_eq!(grad_ab.to_vec(), vec![1.0, 0.0, 0.0, 1.0]);

        // Now propagate back to A and B
        let grads_matmul = grad_fn_ab.backward::<Standard<f64>>(grad_ab, &[&a, &b]);
        assert_eq!(grads_matmul[0].shape(), a.shape());
        assert_eq!(grads_matmul[1].shape(), b.shape());

        // Verify the trace result
        assert_eq!(trace_result.to_vec(), vec![5.0]); // trace(A @ I) = trace(A) = 1 + 4 = 5
    }

    #[test]
    fn test_nested_sum_after_outer() {
        // Test: sum(outer(a, b)) - should equal sum(a) * sum(b)
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0], &[2]);

        // Outer product: a ⊗ b
        let sizes: HashMap<usize, usize> = [(0, 3), (1, 2)].into();
        let ein_outer =
            crate::einsum::Einsum::new(vec![vec![0], vec![1]], vec![0, 1], sizes.clone());
        let outer = ein_outer.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);

        // Sum all elements
        let ein_sum = crate::einsum::Einsum::new(vec![vec![0, 1]], vec![], sizes);
        let sum_result = ein_sum.execute::<Standard<f64>, f64, Cpu>(&[&outer]);

        // sum(a) = 6, sum(b) = 2, product = 12
        assert_eq!(sum_result.to_vec(), vec![12.0]);
    }

    #[test]
    fn test_gradient_4d_tensor_contract() {
        // Test gradient for higher-dimensional contraction
        // ijkl,klmn->ijmn
        let a = Tensor::<f64, Cpu>::from_data(&[0.1; 16], &[2, 2, 2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[0.2; 16], &[2, 2, 2, 2]);

        let (result, grad_fn) = einsum_with_grad::<Standard<f64>, _, _>(
            &[&a, &b],
            &[&[0, 1, 2, 3], &[2, 3, 4, 5]],
            &[0, 1, 4, 5],
        );

        assert_eq!(result.shape(), &[2, 2, 2, 2]);

        let grad_output = Tensor::<f64, Cpu>::from_data(&[1.0; 16], &[2, 2, 2, 2]);
        let grads = grad_fn.backward::<Standard<f64>>(&grad_output, &[&a, &b]);

        assert_eq!(grads[0].shape(), a.shape());
        assert_eq!(grads[1].shape(), b.shape());

        // Verify gradients are computed (non-zero for non-zero inputs)
        let grad_a_sum: f64 = grads[0].to_vec().iter().sum();
        let grad_b_sum: f64 = grads[1].to_vec().iter().sum();
        assert!(grad_a_sum > 0.0);
        assert!(grad_b_sum > 0.0);
    }

    #[test]
    fn test_gradient_symmetry() {
        // For symmetric operation ij,ji->, grad_a and grad_b should relate to each other
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let (result, grad_fn) = einsum_with_grad::<Standard<f64>, _, _>(
            &[&a, &b],
            &[&[0, 1], &[1, 0]], // ij,ji->
            &[],
        );

        // This is sum_ij A[i,j] * B[j,i] = trace(A @ B.T)
        let grad_output = Tensor::<f64, Cpu>::from_data(&[1.0], &[]);
        let grads = grad_fn.backward::<Standard<f64>>(&grad_output, &[&a, &b]);

        assert_eq!(grads[0].shape(), &[2, 2]);
        assert_eq!(grads[1].shape(), &[2, 2]);

        // For this symmetric case, grad_a[i,j] = b[j,i] and grad_b[j,i] = a[i,j]
        // So grad_a should be b transposed
        // b = [[1,3],[2,4]], b.T = [[1,2],[3,4]] col-major: [1,3,2,4]
        // But b in col-major is already [1,2,3,4] = [[1,3],[2,4]]
        // b.T col-major: [1,2,3,4] -> need to verify actual values
        let grad_a_vec = grads[0].to_vec();
        let grad_b_vec = grads[1].to_vec();

        // Both gradients should have same sum
        let sum_a: f64 = grad_a_vec.iter().sum();
        let sum_b: f64 = grad_b_vec.iter().sum();
        assert!(
            (sum_a - sum_b).abs() < 1e-10,
            "Symmetric gradients should have equal sum"
        );

        // Verify result: trace(A @ B) = 1*1 + 3*2 + 2*3 + 4*4 = 1 + 6 + 6 + 16 = 29
        // Wait, ij,ji-> means A[i,j] * B[j,i], not A[i,j] * B[i,j]
        // A = [[1,3],[2,4]], B = [[1,3],[2,4]]
        // sum_ij A[i,j] * B[j,i] = 1*1 + 3*2 + 2*3 + 4*4 = 1 + 6 + 6 + 16 = 29
        assert_eq!(result.to_vec(), vec![29.0]);
    }

    #[test]
    fn test_gradient_broadcast_pattern() {
        // Test gradient for i,j->ij (outer product) then ij-> (sum)
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);
        let b = Tensor::<f64, Cpu>::from_data(&[3.0, 4.0, 5.0], &[3]);

        // Outer product: a ⊗ b -> [2, 3]
        let _sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
        let (outer, grad_fn) =
            einsum_with_grad::<Standard<f64>, _, _>(&[&a, &b], &[&[0], &[1]], &[0, 1]);

        assert_eq!(outer.shape(), &[2, 3]);

        // Gradient output all ones
        let grad_output = Tensor::<f64, Cpu>::from_data(&[1.0; 6], &[2, 3]);
        let grads = grad_fn.backward::<Standard<f64>>(&grad_output, &[&a, &b]);

        // For outer product with all-ones gradient:
        // grad_a[i] = sum_j grad_output[i,j] * b[j] = sum_j b[j] = 3+4+5 = 12
        // grad_b[j] = sum_i grad_output[i,j] * a[i] = sum_i a[i] = 1+2 = 3
        let grad_a_expected: f64 = b.to_vec().iter().sum();
        let grad_b_expected: f64 = a.to_vec().iter().sum();

        assert!(grads[0]
            .to_vec()
            .iter()
            .all(|&x| (x - grad_a_expected).abs() < 1e-10));
        assert!(grads[1]
            .to_vec()
            .iter()
            .all(|&x| (x - grad_b_expected).abs() < 1e-10));
    }

    // ========================================================================
    // cost_and_gradient tests
    // ========================================================================

    use crate::Einsum;

    /// Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13
    #[test]
    #[ignore = "Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_cost_and_gradient_matmul_to_scalar() {
        // A[i,j] @ B[j,k] -> scalar (full contraction)
        // A[2,2], B[2,2] -> scalar by summing over all indices
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        // A @ B contracted fully: sum_{i,j,k} A[i,j] * B[j,k]
        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 1], vec![1, 2]], vec![], sizes);
        ein.optimize_greedy();

        let (cost, grads) = cost_and_gradient::<Standard<f64>, _, _>(&ein, &[&a, &b], None);

        // Verify we get 2 gradients
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), a.shape());
        assert_eq!(grads[1].shape(), b.shape());

        // Cost should be the sum of A @ B
        // A @ B = [[7, 15], [10, 22]] (column-major)
        // sum = 7 + 10 + 15 + 22 = 54
        assert!((cost.to_vec()[0] - 54.0).abs() < 1e-10);
    }

    /// Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13
    #[test]
    #[ignore = "Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_cost_and_gradient_chain_3_tensors() {
        // A[i,j] @ B[j,k] @ C[k] -> scalar
        // Tests multi-tensor gradient computation
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0], &[2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 1], vec![1, 2], vec![2]], vec![], sizes);
        ein.optimize_greedy();

        let (cost, grads) = cost_and_gradient::<Standard<f64>, _, _>(&ein, &[&a, &b, &c], None);

        // Verify we get 3 gradients
        assert_eq!(grads.len(), 3);
        assert_eq!(grads[0].shape(), a.shape());
        assert_eq!(grads[1].shape(), b.shape());
        assert_eq!(grads[2].shape(), c.shape());

        // Verify cost is scalar
        assert_eq!(cost.shape(), &[] as &[usize]);
    }

    #[test]
    fn test_cost_and_gradient_unary_trace() {
        // Trace: A[i,i] -> scalar
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);
        ein.optimize_greedy();

        let (cost, grads) = cost_and_gradient::<Standard<f64>, _, _>(&ein, &[&a], None);

        // Trace = 1 + 4 = 5
        assert!((cost.to_vec()[0] - 5.0).abs() < 1e-10);

        // Gradient of trace is identity on diagonal, 0 elsewhere
        // In column-major [2,2]: [1, 0, 0, 1]
        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].shape(), &[2, 2]);
        assert_eq!(grads[0].to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_cost_and_gradient_unary_sum() {
        // Sum all: A[i,j] -> scalar
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 1]], vec![], sizes);
        ein.optimize_greedy();

        let (cost, grads) = cost_and_gradient::<Standard<f64>, _, _>(&ein, &[&a], None);

        // Sum = 1 + 2 + 3 + 4 = 10
        assert!((cost.to_vec()[0] - 10.0).abs() < 1e-10);

        // Gradient of sum is all ones
        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].to_vec(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    /// bpcheck for cost_and_gradient with multi-tensor contractions.
    fn bpcheck_cost_and_gradient(
        tensors: &[Tensor<f64, Cpu>],
        ixs: &[Vec<usize>],
        sizes: HashMap<usize, usize>,
        eta: f64,
        tol: f64,
    ) -> bool {
        let mut ein = Einsum::new(ixs.to_vec(), vec![], sizes);
        ein.optimize_greedy();

        let tensor_refs: Vec<&Tensor<f64, Cpu>> = tensors.iter().collect();
        let (cost, grads) = cost_and_gradient::<Standard<f64>, _, _>(&ein, &tensor_refs, None);
        let f_x = cost.to_vec()[0];

        // For each tensor, check gradient via finite differences
        for (i, (tensor, grad)) in tensors.iter().zip(grads.iter()).enumerate() {
            let grad_vec = grad.to_vec();
            let tensor_vec = tensor.to_vec();

            // Compute |g|² and expected change
            let grad_sq_sum: f64 = grad_vec.iter().map(|x| x * x).sum();
            let expected_change = eta * grad_sq_sum;

            // Perturb this tensor
            let perturbed_vec: Vec<f64> = tensor_vec
                .iter()
                .zip(grad_vec.iter())
                .map(|(x, g)| x - eta * g)
                .collect();
            let perturbed = Tensor::<f64, Cpu>::from_data(&perturbed_vec, tensor.shape());

            // Recompute with perturbed tensor
            let mut perturbed_refs = tensor_refs.clone();
            perturbed_refs[i] = &perturbed;
            let (cost_perturbed, _) =
                cost_and_gradient::<Standard<f64>, _, _>(&ein, &perturbed_refs, None);
            let f_perturbed = cost_perturbed.to_vec()[0];

            let actual_change = f_x - f_perturbed;
            let error = (actual_change - expected_change).abs();
            let passed = error < tol || (expected_change > 0.0 && error / expected_change < 0.01);

            if !passed {
                eprintln!(
                    "Tensor {} failed: expected={}, actual={}, error={}",
                    i, expected_change, actual_change, error
                );
                return false;
            }
        }

        true
    }

    /// Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13
    #[test]
    #[ignore = "Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_bpcheck_cost_and_gradient_matmul() {
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        assert!(bpcheck_cost_and_gradient(
            &[a, b],
            &[vec![0, 1], vec![1, 2]],
            sizes,
            1e-5,
            1e-8
        ));
    }

    /// Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13
    #[test]
    #[ignore = "Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_bpcheck_cost_and_gradient_3_tensor_chain() {
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        assert!(bpcheck_cost_and_gradient(
            &[a, b, c],
            &[vec![0, 1], vec![1, 2], vec![2]],
            sizes,
            1e-5,
            1e-8
        ));
    }

    #[test]
    fn test_bpcheck_cost_and_gradient_star_contraction() {
        // Star contraction: A[i,j], B[j,k], C[k,i] -> scalar
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = Tensor::<f64, Cpu>::from_data(&[9.0, 10.0, 11.0, 12.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        assert!(bpcheck_cost_and_gradient(
            &[a, b, c],
            &[vec![0, 1], vec![1, 2], vec![2, 0]],
            sizes,
            1e-5,
            1e-8
        ));
    }

    /// Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13
    #[test]
    #[ignore = "Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_bpcheck_cost_and_gradient_4_tensors() {
        // 4-tensor chain: A[i,j] @ B[j,k] @ C[k,l] @ D[l] -> scalar
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = Tensor::<f64, Cpu>::from_data(&[9.0, 10.0, 11.0, 12.0], &[2, 2]);
        let d = Tensor::<f64, Cpu>::from_data(&[1.0, 1.0], &[2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
        assert!(bpcheck_cost_and_gradient(
            &[a, b, c, d],
            &[vec![0, 1], vec![1, 2], vec![2, 3], vec![3]],
            sizes,
            1e-5,
            1e-8
        ));
    }

    /// Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13
    #[cfg(feature = "tropical")]
    #[test]
    #[ignore = "Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_cost_and_gradient_tropical_matmul() {
        // MaxPlus matmul: C[i,k] = max_j (A[i,j] + B[j,k])
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 1], vec![1, 2]], vec![], sizes);
        ein.optimize_greedy();

        let (cost, grads) = cost_and_gradient::<MaxPlus<f64>, _, _>(&ein, &[&a, &b], None);

        // MaxPlus: (A + B)_ik = max_j(A_ij + B_jk)
        // Then max over i,k: max(5, 6, 7, 8) = 8
        assert!((cost.to_vec()[0] - 8.0).abs() < 1e-10);
        assert_eq!(grads.len(), 2);
    }

    // ========================================================================
    // Julia OMEinsum.jl compatibility tests
    // Port of tests from test/bp.jl and test/autodiff.jl
    // ========================================================================

    /// Julia test: (ij, jk), ki -> scalar (triangle contraction)
    /// Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13
    #[test]
    #[ignore = "Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_julia_bp_triangle() {
        // A[2,3], B[3,4], C[4,2] -> scalar
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::<f64, Cpu>::from_data(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        );
        let c = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 3), (2, 4)].into();
        assert!(bpcheck_cost_and_gradient(
            &[a, b, c],
            &[vec![0, 1], vec![1, 2], vec![2, 0]],
            sizes,
            1e-5,
            1e-8
        ));
    }

    /// Julia test: matrix-vector product ij, j -> i then sum to scalar
    /// Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13
    #[test]
    #[ignore = "Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_julia_bp_matvec() {
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let v = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0], &[2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        assert!(bpcheck_cost_and_gradient(
            &[a, v],
            &[vec![0, 1], vec![1]],
            sizes,
            1e-5,
            1e-8
        ));
    }

    /// Julia test: contract to 0-dim array (ij, ij) -> scalar
    /// Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13
    #[test]
    #[ignore = "Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_julia_bp_contract_to_scalar() {
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        assert!(bpcheck_cost_and_gradient(
            &[a, b],
            &[vec![0, 1], vec![0, 1]],
            sizes,
            1e-5,
            1e-8
        ));
    }

    /// Julia test: trace (ii) -> scalar
    #[test]
    fn test_julia_bp_trace() {
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);
        ein.optimize_greedy();

        let tensor_refs = vec![&a];
        let (cost, grads) = cost_and_gradient::<Standard<f64>, _, _>(&ein, &tensor_refs, None);

        // Verify gradient via finite difference
        let grad = &grads[0];
        let eta = 1e-5;
        let grad_sq_sum: f64 = grad.to_vec().iter().map(|x| x * x).sum();
        let expected_change = eta * grad_sq_sum;

        let a_vec = a.to_vec();
        let grad_vec = grad.to_vec();
        let perturbed_vec: Vec<f64> = a_vec
            .iter()
            .zip(grad_vec.iter())
            .map(|(x, g)| x - eta * g)
            .collect();
        let perturbed = Tensor::<f64, Cpu>::from_data(&perturbed_vec, a.shape());

        let (cost_perturbed, _) =
            cost_and_gradient::<Standard<f64>, _, _>(&ein, &[&perturbed], None);
        let actual_change = cost.to_vec()[0] - cost_perturbed.to_vec()[0];

        assert!(
            (actual_change - expected_change).abs() < 1e-8,
            "Trace gradient check failed"
        );
    }

    /// Julia test: 4D trace (ijji) -> scalar
    #[test]
    fn test_julia_bp_trace_4d() {
        let a = Tensor::<f64, Cpu>::from_data(
            &(1..=16).map(|x| x as f64).collect::<Vec<_>>(),
            &[2, 2, 2, 2],
        );

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 1, 1, 0]], vec![], sizes);
        ein.optimize_greedy();

        // Just verify it runs and produces gradients of correct shape
        let (cost, grads) = cost_and_gradient::<Standard<f64>, _, _>(&ein, &[&a], None);
        assert_eq!(cost.shape(), &[] as &[usize]);
        assert_eq!(grads[0].shape(), &[2, 2, 2, 2]);
    }

    /// Julia test: partial trace (ijjk) -> (ik)
    #[test]
    fn test_julia_bp_partial_trace() {
        let a = Tensor::<f64, Cpu>::from_data(
            &(1..=16).map(|x| x as f64).collect::<Vec<_>>(),
            &[2, 2, 2, 2],
        );

        // For scalar output in cost_and_gradient, we need to reduce to scalar
        // So we test: ijjk -> scalar (trace over i,j,k keeping none)
        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 1, 1, 2]], vec![], sizes);
        ein.optimize_greedy();

        let (cost, grads) = cost_and_gradient::<Standard<f64>, _, _>(&ein, &[&a], None);
        assert_eq!(cost.shape(), &[] as &[usize]);
        assert_eq!(grads[0].shape(), &[2, 2, 2, 2]);
    }

    /// Julia test: permutation ij -> ji (transpose)
    #[test]
    fn test_julia_bp_permutation() {
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        // For cost_and_gradient: ij -> scalar via transpose then sum
        // Actually, we need scalar output, so test with ij -> scalar
        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 1]], vec![], sizes);
        ein.optimize_greedy();

        let (cost, grads) = cost_and_gradient::<Standard<f64>, _, _>(&ein, &[&a], None);

        // Sum = 10, gradient is all ones
        assert!((cost.to_vec()[0] - 10.0).abs() < 1e-10);
        assert!(grads[0].to_vec().iter().all(|&x| (x - 1.0).abs() < 1e-10));
    }

    /// Julia test: tensor contraction (abcd, bc) -> (ad) then sum to scalar
    /// Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13
    #[test]
    #[ignore = "Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_julia_bp_tensor_contraction() {
        let t = Tensor::<f64, Cpu>::from_data(
            &(1..=16).map(|x| x as f64).collect::<Vec<_>>(),
            &[2, 2, 2, 2],
        );
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
        assert!(bpcheck_cost_and_gradient(
            &[t, a],
            &[vec![0, 1, 2, 3], vec![1, 2]],
            sizes,
            1e-5,
            1e-8
        ));
    }

    /// Julia test: star contraction (ia, ib, ic) -> (abc) then sum to scalar
    /// Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13
    #[test]
    #[ignore = "Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_julia_bp_star() {
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = Tensor::<f64, Cpu>::from_data(&[9.0, 10.0, 11.0, 12.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
        assert!(bpcheck_cost_and_gradient(
            &[a, b, c],
            &[vec![0, 1], vec![0, 2], vec![0, 3]],
            sizes,
            1e-5,
            1e-8
        ));
    }

    /// Julia test: Hadamard product (ij, ij) -> ij then sum to scalar
    /// Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13
    #[test]
    #[ignore = "Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_julia_bp_hadamard() {
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        // Hadamard to scalar: ij, ij -> ()
        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        assert!(bpcheck_cost_and_gradient(
            &[a, b],
            &[vec![0, 1], vec![0, 1]],
            sizes,
            1e-5,
            1e-8
        ));
    }

    /// Julia test: outer product (ij, kl) -> (ijkl) then sum to scalar
    /// Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13
    #[test]
    #[ignore = "Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_julia_bp_outer() {
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        // Outer then sum to scalar
        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
        assert!(bpcheck_cost_and_gradient(
            &[a, b],
            &[vec![0, 1], vec![2, 3]],
            sizes,
            1e-5,
            1e-8
        ));
    }

    /// Julia test: chain ij, jk, kl -> il then sum to scalar
    /// Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13
    #[test]
    #[ignore = "Requires omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_julia_bp_chain_3() {
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = Tensor::<f64, Cpu>::from_data(&[9.0, 10.0, 11.0, 12.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
        assert!(bpcheck_cost_and_gradient(
            &[a, b, c],
            &[vec![0, 1], vec![1, 2], vec![2, 3]],
            sizes,
            1e-5,
            1e-8
        ));
    }

    // ========================================================================
    // omeco issue tracking tests
    // See: https://github.com/GiggleLiu/omeco/issues/13
    // ========================================================================

    /// Test that omeco's optimize_code respects the final output indices.
    /// Currently fails because omeco returns a tree with iy=[0,2] instead of iy=[].
    /// Tracked in: https://github.com/GiggleLiu/omeco/issues/13
    #[test]
    #[ignore = "Waiting for omeco fix: https://github.com/GiggleLiu/omeco/issues/13"]
    fn test_omeco_respects_final_iy() {
        use omeco::{optimize_code, EinCode, GreedyMethod, NestedEinsum};

        // A[i,j] @ B[j,k] -> scalar (empty output)
        let code = EinCode::new(vec![vec![0, 1], vec![1, 2]], vec![]);
        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let optimizer = GreedyMethod::new(0.0, 0.0);

        if let Some(tree) = optimize_code(&code, &sizes, &optimizer) {
            match &tree {
                NestedEinsum::Node { eins, .. } => {
                    // This currently fails: eins.iy is [0, 2], not []
                    assert_eq!(
                        eins.iy, code.iy,
                        "omeco should respect the final output indices. \
                         Expected iy={:?}, got iy={:?}",
                        code.iy, eins.iy
                    );
                }
                NestedEinsum::Leaf { .. } => panic!("Expected Node, got Leaf"),
            }
        } else {
            panic!("optimize_code returned None");
        }
    }
}
