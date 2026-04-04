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
    modes.iter().position(|&m| m == mode).expect("mode not found")
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

    target
        .iter()
        .map(|m| mode_position(current, *m))
        .collect()
}

use crate::algebra::Algebra;
use crate::backend::Cpu;
use crate::tensor::compute_contiguous_strides;

/// Look up dimension sizes for each mode in `current_order` from the input mode/shape arrays.
fn get_current_shape(current_order: &[i32], modes_a: &[i32], shape_a: &[usize], modes_b: &[i32], shape_b: &[usize]) -> Vec<usize> {
    current_order.iter().map(|&m| {
        if let Some(pos) = modes_a.iter().position(|&x| x == m) {
            shape_a[pos]
        } else {
            shape_b[mode_position(modes_b, m)]
        }
    }).collect()
}

/// Reorder GEMM result to output mode order, summing (via `Algebra::add`) over
/// any modes present in the result but absent from the output.
fn reorder_and_reduce<A: Algebra>(
    data: &[A::Scalar],
    current_order: &[i32],
    current_shape: &[usize],
    modes_c: &[i32],
    shape_c: &[usize],
) -> Vec<A::Scalar>
where
    A::Scalar: crate::algebra::Scalar,
{
    // Fast path: no extra modes to reduce
    let c_set: HashSet<i32> = modes_c.iter().copied().collect();
    if current_order.iter().all(|m| c_set.contains(m)) {
        if *current_order == *modes_c {
            return data.to_vec();
        }
        let c_shape_current: Vec<usize> = current_order
            .iter()
            .map(|&m| shape_c[mode_position(modes_c, m)])
            .collect();
        let out_perm: Vec<usize> = modes_c
            .iter()
            .map(|m| mode_position(current_order, *m))
            .collect();
        return permute_data(data, &c_shape_current, &out_perm);
    }

    // Slow path: reduce extra modes via algebra addition
    let out_numel: usize = shape_c.iter().product::<usize>().max(1);
    let mut result = vec![A::zero().to_scalar(); out_numel];

    // Precompute loop-invariant data
    let out_strides = compute_contiguous_strides(shape_c);
    let mode_map: Vec<usize> = modes_c
        .iter()
        .map(|&cm| mode_position(current_order, cm))
        .collect();
    let ndim = current_shape.len();
    let mut coords = vec![0usize; ndim];

    let numel: usize = current_shape.iter().product::<usize>().max(1);
    for idx in 0..numel {
        // Convert linear index to multi-index (column-major)
        let mut remaining = idx;
        for dim in 0..ndim {
            coords[dim] = remaining % current_shape[dim];
            remaining /= current_shape[dim];
        }

        // Compute output linear index using only modes in modes_c
        let mut out_idx = 0;
        for (ci, &pos) in mode_map.iter().enumerate() {
            out_idx += coords[pos] * out_strides[ci];
        }

        let sum = A::from_scalar(result[out_idx]).add(A::from_scalar(data[idx]));
        result[out_idx] = sum.to_scalar();
    }

    result
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

    // 3. Compute dimension sizes
    let batch_size = product_of_dims(&batch, modes_a, shape_a);
    let left_size = product_of_dims(&left, modes_a, shape_a);
    let right_size = product_of_dims(&right, modes_b, shape_b);
    let contract_size = product_of_dims(&contracted, modes_a, shape_a);

    // 4. Permute A to [left, contracted, batch] - batch LAST for correct memory layout
    // In column-major, the last dimension has the largest stride, so batch elements
    // are contiguous blocks rather than interleaved.
    let a_perm = compute_permutation(modes_a, &left, &contracted, &batch);
    let a_permuted = permute_data(&a_contig, shape_a, &a_perm);

    // 5. Permute B to [contracted, right, batch] - batch LAST
    let b_perm = compute_permutation(modes_b, &contracted, &right, &batch);
    let b_permuted = permute_data(&b_contig, shape_b, &b_perm);

    // 6. Call GEMM
    let c_data = if batch.is_empty() {
        cpu.gemm_internal::<A>(&a_permuted, left_size, contract_size, &b_permuted, right_size)
    } else {
        cpu.gemm_batched_internal::<A>(
            &a_permuted, batch_size, left_size, contract_size,
            &b_permuted, right_size,
        )
    };

    // 7. Reduce and permute result to output order
    // Result is in [left, right, batch] order
    let current_order: Vec<i32> = left.iter()
        .chain(right.iter())
        .chain(batch.iter())
        .copied()
        .collect();

    let current_shape = get_current_shape(&current_order, modes_a, shape_a, modes_b, shape_b);
    reorder_and_reduce::<A>(&c_data, &current_order, &current_shape, modes_c, shape_c)
}

/// Ensure data is contiguous (copy if strided).
fn ensure_contiguous<T: Copy + Default>(
    data: &[T],
    shape: &[usize],
    strides: &[usize],
) -> Vec<T> {
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
fn permute_data<T: Copy + Default>(
    data: &[T],
    shape: &[usize],
    perm: &[usize],
) -> Vec<T> {
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
    let batch_size = product_of_dims(&batch, modes_a, shape_a);
    let left_size = product_of_dims(&left, modes_a, shape_a);
    let right_size = product_of_dims(&right, modes_b, shape_b);
    let contract_size = product_of_dims(&contracted, modes_a, shape_a);

    // Permute with batch LAST for correct memory layout
    let a_perm = compute_permutation(modes_a, &left, &contracted, &batch);
    let a_permuted = permute_data(&a_contig, shape_a, &a_perm);
    let b_perm = compute_permutation(modes_b, &contracted, &right, &batch);
    let b_permuted = permute_data(&b_contig, shape_b, &b_perm);

    // Call GEMM with argmax
    let (c_data, argmax) = if batch.is_empty() {
        cpu.gemm_with_argmax_internal::<A>(
            &a_permuted, left_size, contract_size,
            &b_permuted, right_size,
        )
    } else {
        cpu.gemm_batched_with_argmax_internal::<A>(
            &a_permuted, batch_size, left_size, contract_size,
            &b_permuted, right_size,
        )
    };

    // Permute result - result is in [left, right, batch] order
    let current_order: Vec<i32> = left.iter()
        .chain(right.iter())
        .chain(batch.iter())
        .copied()
        .collect();

    let current_shape = get_current_shape(&current_order, modes_a, shape_a, modes_b, shape_b);
    let c_data = reorder_and_reduce::<A>(&c_data, &current_order, &current_shape, modes_c, shape_c);

    // Argmax: permute only (reduction over argmax indices is undefined)
    if current_order == modes_c {
        (c_data, argmax)
    } else if modes_c.is_empty() {
        (c_data, vec![0u32])
    } else {
        let out_perm: Vec<usize> = modes_c
            .iter()
            .map(|&m| mode_position(&current_order, m))
            .collect();
        (c_data, permute_data(&argmax, &current_shape, &out_perm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_modes_matmul() {
        // ij,jk->ik
        let (batch, left, right, contracted) =
            classify_modes(&[0, 1], &[1, 2], &[0, 2]);

        assert!(batch.is_empty());
        assert_eq!(left, vec![0]);
        assert_eq!(right, vec![2]);
        assert_eq!(contracted, vec![1]);
    }

    #[test]
    fn test_classify_modes_batched() {
        // bij,bjk->bik
        let (batch, left, right, contracted) =
            classify_modes(&[0, 1, 2], &[0, 2, 3], &[0, 1, 3]);

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
