//! CPU tensor contraction via reshape→GEMM→reshape.

use std::collections::HashSet;

/// Classify modes into batch, left-only, right-only, and contracted.
///
/// - batch: in both A and B, and in output C
/// - left: only in A (free indices from A)
/// - right: only in B (free indices from B)
/// - contracted: in both A and B, but NOT in output C
pub(super) fn classify_modes(
    modes_a: &[i32],
    modes_b: &[i32],
    modes_c: &[i32],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let a_set: HashSet<i32> = modes_a.iter().copied().collect();
    let b_set: HashSet<i32> = modes_b.iter().copied().collect();
    let c_set: HashSet<i32> = modes_c.iter().copied().collect();

    let mut batch = Vec::new();
    let mut left = Vec::new();
    let mut contracted = Vec::new();

    for &m in modes_a {
        if b_set.contains(&m) && c_set.contains(&m) {
            if !batch.contains(&m) {
                batch.push(m);
            }
        } else if b_set.contains(&m) && !c_set.contains(&m) {
            if !contracted.contains(&m) {
                contracted.push(m);
            }
        } else if !left.contains(&m) {
            left.push(m);
        }
    }

    let right: Vec<i32> = modes_b
        .iter()
        .filter(|m| !a_set.contains(m))
        .copied()
        .collect();

    (batch, left, right, contracted)
}

/// Find the position of a mode in a modes array.
pub(super) fn mode_position(modes: &[i32], mode: i32) -> usize {
    modes
        .iter()
        .position(|&m| m == mode)
        .expect("mode not found")
}

/// Compute the product of dimensions for given modes.
pub(super) fn product_of_dims(modes: &[i32], all_modes: &[i32], shape: &[usize]) -> usize {
    modes
        .iter()
        .map(|&m| shape[mode_position(all_modes, m)])
        .product::<usize>()
        .max(1)
}

/// Compute permutation to reorder modes to [first..., second..., third...].
pub(super) fn compute_permutation(
    current: &[i32],
    first: &[i32],
    second: &[i32],
    third: &[i32],
) -> Vec<usize> {
    let target: Vec<i32> = first
        .iter()
        .chain(second.iter())
        .chain(third.iter())
        .copied()
        .collect();

    target.iter().map(|m| mode_position(current, *m)).collect()
}

use crate::algebra::Algebra;
use crate::backend::Cpu;
use crate::tensor::compute_contiguous_strides;

/// Sum (reduce) over specified modes in contiguous tensor data, removing those dimensions.
///
/// Uses `A::add` for accumulation so it works correctly with all algebras
/// (standard addition for `Standard`, max for `MaxPlus`, etc.).
fn reduce_trace_modes<A: Algebra>(
    data: &[A::Scalar],
    shape: &[usize],
    all_modes: &[i32],
    trace_modes: &[i32],
) -> (Vec<A::Scalar>, Vec<usize>, Vec<i32>)
where
    A::Scalar: crate::algebra::Scalar,
{
    if trace_modes.is_empty() {
        return (data.to_vec(), shape.to_vec(), all_modes.to_vec());
    }

    let trace_positions: HashSet<usize> = trace_modes
        .iter()
        .map(|m| mode_position(all_modes, *m))
        .collect();

    // New shape and modes without the trace dimensions
    let new_shape: Vec<usize> = (0..shape.len())
        .filter(|i| !trace_positions.contains(i))
        .map(|i| shape[i])
        .collect();
    let new_modes: Vec<i32> = (0..all_modes.len())
        .filter(|i| !trace_positions.contains(i))
        .map(|i| all_modes[i])
        .collect();

    let new_size = new_shape.iter().product::<usize>().max(1);
    let mut result: Vec<A::Scalar> = vec![A::zero().to_scalar(); new_size];

    let new_strides = compute_contiguous_strides(&new_shape);
    let old_size = shape.iter().product::<usize>().max(1);

    for old_linear in 0..old_size {
        // Compute old multi-index (column-major)
        let mut remaining = old_linear;
        let mut new_linear = 0usize;
        let mut new_dim = 0usize;
        for i in 0..shape.len() {
            let coord = remaining % shape[i];
            remaining /= shape[i];
            if !trace_positions.contains(&i) {
                new_linear += coord * new_strides[new_dim];
                new_dim += 1;
            }
        }

        let acc = A::from_scalar(result[new_linear]);
        let val = A::from_scalar(data[old_linear]);
        result[new_linear] = acc.add(val).to_scalar();
    }

    (result, new_shape, new_modes)
}

/// Execute tensor contraction on CPU via reshape→GEMM→reshape.
#[allow(clippy::too_many_arguments)]
pub(super) fn contract<A: Algebra>(
    cpu: &Cpu,
    a: &[A::Scalar],
    shape_a: &[usize],
    strides_a: &[usize],
    modes_a: &[i32],
    b: &[A::Scalar],
    shape_b: &[usize],
    strides_b: &[usize],
    modes_b: &[i32],
    shape_c: &[usize],
    modes_c: &[i32],
) -> Vec<A::Scalar>
where
    A::Scalar: crate::algebra::Scalar,
{
    // 1. Make inputs contiguous if needed
    let a_contig = ensure_contiguous(a, shape_a, strides_a);
    let b_contig = ensure_contiguous(b, shape_b, strides_b);

    // 2. Classify modes
    let (batch, left, right, contracted) = classify_modes(modes_a, modes_b, modes_c);

    // 3. Handle trace modes: modes in only one input that are NOT in the output.
    //    GEMM can only contract modes shared by both inputs. Single-input modes
    //    not in the output must be summed over (traced) before GEMM.
    let c_set: HashSet<i32> = modes_c.iter().copied().collect();
    let left_trace: Vec<i32> = left.iter().filter(|m| !c_set.contains(m)).copied().collect();
    let right_trace: Vec<i32> = right.iter().filter(|m| !c_set.contains(m)).copied().collect();

    let (a_data, a_shape, a_modes) = if !left_trace.is_empty() {
        reduce_trace_modes::<A>(&a_contig, shape_a, modes_a, &left_trace)
    } else {
        (a_contig, shape_a.to_vec(), modes_a.to_vec())
    };
    let (b_data, b_shape, b_modes) = if !right_trace.is_empty() {
        reduce_trace_modes::<A>(&b_contig, shape_b, modes_b, &right_trace)
    } else {
        (b_contig, shape_b.to_vec(), modes_b.to_vec())
    };

    // Free modes (left/right modes that ARE in the output)
    let left_free: Vec<i32> = left.iter().filter(|m| c_set.contains(m)).copied().collect();
    let right_free: Vec<i32> = right.iter().filter(|m| c_set.contains(m)).copied().collect();

    // 4. Compute dimension sizes (using reduced inputs)
    let batch_size = product_of_dims(&batch, &a_modes, &a_shape);
    let left_size = product_of_dims(&left_free, &a_modes, &a_shape);
    let right_size = product_of_dims(&right_free, &b_modes, &b_shape);
    let contract_size = product_of_dims(&contracted, &a_modes, &a_shape);

    // 5. Permute A to [left_free, contracted, batch] - batch LAST for correct memory layout
    let a_perm = compute_permutation(&a_modes, &left_free, &contracted, &batch);
    let a_permuted = permute_data(&a_data, &a_shape, &a_perm);

    // 6. Permute B to [contracted, right_free, batch] - batch LAST
    let b_perm = compute_permutation(&b_modes, &contracted, &right_free, &batch);
    let b_permuted = permute_data(&b_data, &b_shape, &b_perm);

    // 7. Call GEMM
    let c_data = if batch.is_empty() {
        cpu.gemm_internal::<A>(
            &a_permuted,
            left_size,
            contract_size,
            &b_permuted,
            right_size,
        )
    } else {
        cpu.gemm_batched_internal::<A>(
            &a_permuted,
            batch_size,
            left_size,
            contract_size,
            &b_permuted,
            right_size,
        )
    };

    // 8. Permute result to output order
    // Result is in [left_free, right_free, batch] order
    let current_order: Vec<i32> = left_free
        .iter()
        .chain(right_free.iter())
        .chain(batch.iter())
        .copied()
        .collect();

    if current_order == modes_c {
        c_data
    } else {
        let c_shape_current: Vec<usize> = current_order
            .iter()
            .map(|&m| shape_c[mode_position(modes_c, m)])
            .collect();
        let out_perm: Vec<usize> = modes_c
            .iter()
            .map(|m| current_order.iter().position(|x| x == m).unwrap())
            .collect();
        permute_data(&c_data, &c_shape_current, &out_perm)
    }
}

/// Ensure data is contiguous (copy if strided).
fn ensure_contiguous<T: Copy + Default>(data: &[T], shape: &[usize], strides: &[usize]) -> Vec<T> {
    let expected_strides = compute_contiguous_strides(shape);
    if strides == expected_strides {
        data.to_vec()
    } else {
        // Copy with stride handling
        let numel: usize = shape.iter().product();
        let mut result = vec![T::default(); numel];
        copy_strided_to_contiguous(data, &mut result, shape, strides);
        result
    }
}

/// Copy strided data to contiguous buffer.
fn copy_strided_to_contiguous<T: Copy>(
    src: &[T],
    dst: &mut [T],
    shape: &[usize],
    strides: &[usize],
) {
    let numel: usize = shape.iter().product();

    for (i, dst_elem) in dst.iter_mut().enumerate().take(numel) {
        // Convert linear index to multi-index
        let mut remaining = i;
        let mut src_offset = 0;
        for dim in 0..shape.len() {
            let coord = remaining % shape[dim];
            remaining /= shape[dim];
            src_offset += coord * strides[dim];
        }
        *dst_elem = src[src_offset];
    }
}

/// Permute data according to axis permutation.
fn permute_data<T: Copy + Default>(data: &[T], shape: &[usize], perm: &[usize]) -> Vec<T> {
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        return data.to_vec(); // Already in correct order
    }

    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let numel: usize = shape.iter().product();
    let mut result = vec![T::default(); numel];

    let old_strides = compute_contiguous_strides(shape);

    for (new_idx, result_elem) in result.iter_mut().enumerate().take(numel) {
        // Convert new linear index to new multi-index
        let mut remaining = new_idx;
        let mut new_coords = vec![0; shape.len()];
        for dim in 0..new_shape.len() {
            new_coords[dim] = remaining % new_shape[dim];
            remaining /= new_shape[dim];
        }

        // Map to old coordinates via inverse permutation
        let mut old_idx = 0;
        for (new_dim, &old_dim) in perm.iter().enumerate() {
            old_idx += new_coords[new_dim] * old_strides[old_dim];
        }

        *result_elem = data[old_idx];
    }

    result
}

/// Execute tensor contraction with argmax tracking.
#[allow(clippy::too_many_arguments)]
pub(super) fn contract_with_argmax<A: Algebra<Index = u32>>(
    cpu: &Cpu,
    a: &[A::Scalar],
    shape_a: &[usize],
    strides_a: &[usize],
    modes_a: &[i32],
    b: &[A::Scalar],
    shape_b: &[usize],
    strides_b: &[usize],
    modes_b: &[i32],
    shape_c: &[usize],
    modes_c: &[i32],
) -> (Vec<A::Scalar>, Vec<u32>)
where
    A::Scalar: crate::algebra::Scalar,
{
    // Same setup as contract
    let a_contig = ensure_contiguous(a, shape_a, strides_a);
    let b_contig = ensure_contiguous(b, shape_b, strides_b);
    let (batch, left, right, contracted) = classify_modes(modes_a, modes_b, modes_c);

    // Handle trace modes (same as contract)
    let c_set: HashSet<i32> = modes_c.iter().copied().collect();
    let left_trace: Vec<i32> = left.iter().filter(|m| !c_set.contains(m)).copied().collect();
    let right_trace: Vec<i32> = right.iter().filter(|m| !c_set.contains(m)).copied().collect();

    let (a_data, a_shape, a_modes) = if !left_trace.is_empty() {
        reduce_trace_modes::<A>(&a_contig, shape_a, modes_a, &left_trace)
    } else {
        (a_contig, shape_a.to_vec(), modes_a.to_vec())
    };
    let (b_data, b_shape, b_modes) = if !right_trace.is_empty() {
        reduce_trace_modes::<A>(&b_contig, shape_b, modes_b, &right_trace)
    } else {
        (b_contig, shape_b.to_vec(), modes_b.to_vec())
    };

    let left_free: Vec<i32> = left.iter().filter(|m| c_set.contains(m)).copied().collect();
    let right_free: Vec<i32> = right.iter().filter(|m| c_set.contains(m)).copied().collect();

    let batch_size = product_of_dims(&batch, &a_modes, &a_shape);
    let left_size = product_of_dims(&left_free, &a_modes, &a_shape);
    let right_size = product_of_dims(&right_free, &b_modes, &b_shape);
    let contract_size = product_of_dims(&contracted, &a_modes, &a_shape);

    // Permute with batch LAST for correct memory layout
    let a_perm = compute_permutation(&a_modes, &left_free, &contracted, &batch);
    let a_permuted = permute_data(&a_data, &a_shape, &a_perm);
    let b_perm = compute_permutation(&b_modes, &contracted, &right_free, &batch);
    let b_permuted = permute_data(&b_data, &b_shape, &b_perm);

    // Call GEMM with argmax
    let (c_data, argmax) = if batch.is_empty() {
        cpu.gemm_with_argmax_internal::<A>(
            &a_permuted,
            left_size,
            contract_size,
            &b_permuted,
            right_size,
        )
    } else {
        cpu.gemm_batched_with_argmax_internal::<A>(
            &a_permuted,
            batch_size,
            left_size,
            contract_size,
            &b_permuted,
            right_size,
        )
    };

    // Permute result - result is in [left_free, right_free, batch] order
    let current_order: Vec<i32> = left_free
        .iter()
        .chain(right_free.iter())
        .chain(batch.iter())
        .copied()
        .collect();

    if current_order == modes_c {
        (c_data, argmax)
    } else {
        let c_shape_current: Vec<usize> = current_order
            .iter()
            .map(|&m| shape_c[mode_position(modes_c, m)])
            .collect();
        let out_perm: Vec<usize> = modes_c
            .iter()
            .map(|m| current_order.iter().position(|x| x == m).unwrap())
            .collect();
        (
            permute_data(&c_data, &c_shape_current, &out_perm),
            permute_data(&argmax, &c_shape_current, &out_perm),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_modes_matmul() {
        // ij,jk->ik
        let (batch, left, right, contracted) = classify_modes(&[0, 1], &[1, 2], &[0, 2]);

        assert!(batch.is_empty());
        assert_eq!(left, vec![0]);
        assert_eq!(right, vec![2]);
        assert_eq!(contracted, vec![1]);
    }

    #[test]
    fn test_classify_modes_batched() {
        // bij,bjk->bik
        let (batch, left, right, contracted) = classify_modes(&[0, 1, 2], &[0, 2, 3], &[0, 1, 3]);

        assert_eq!(batch, vec![0]);
        assert_eq!(left, vec![1]);
        assert_eq!(right, vec![3]);
        assert_eq!(contracted, vec![2]);
    }

    #[test]
    fn test_product_of_dims() {
        let modes = &[0, 1, 2];
        let shape = &[2, 3, 4];

        assert_eq!(product_of_dims(&[0], modes, shape), 2);
        assert_eq!(product_of_dims(&[1, 2], modes, shape), 12);
        assert_eq!(product_of_dims(&[], modes, shape), 1);
    }

    #[test]
    fn test_compute_permutation() {
        // Current: [0, 1, 2], want: [0, 2, 1]
        let perm = compute_permutation(&[0, 1, 2], &[0], &[2], &[1]);
        assert_eq!(perm, vec![0, 2, 1]);
    }
}
