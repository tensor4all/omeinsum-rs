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

#[derive(Debug, Clone, PartialEq, Eq)]
enum MaterializationPlan {
    NoCopy,
    MakeContiguous,
    Permute { perm: Vec<usize>, shape: Vec<usize> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ContractionLayoutPlan {
    batch_modes: Vec<i32>,
    left_modes: Vec<i32>,
    right_modes: Vec<i32>,
    contracted_modes: Vec<i32>,
    left_materialization: MaterializationPlan,
    right_materialization: MaterializationPlan,
    output_perm: Option<Vec<usize>>,
    batch_size: usize,
    left_size: usize,
    right_size: usize,
    contract_size: usize,
}

fn physical_axis_order(strides: &[usize]) -> Vec<usize> {
    let mut axes: Vec<usize> = (0..strides.len()).collect();
    axes.sort_by_key(|&axis| (strides[axis], axis));
    axes
}

fn is_flattenable_group(
    shape: &[usize],
    strides: &[usize],
    axes: &[usize],
    require_unit_base: bool,
) -> bool {
    if axes.is_empty() {
        return true;
    }

    let mut expected = if require_unit_base { 1 } else { strides[axes[0]] };
    if require_unit_base && strides[axes[0]] != 1 {
        return false;
    }

    for &axis in axes {
        if strides[axis] != expected {
            return false;
        }
        expected = expected.saturating_mul(shape[axis].max(1));
    }

    true
}

fn is_permutation_like(shape: &[usize], strides: &[usize], physical_order: &[usize]) -> bool {
    let mut expected = 1usize;
    for &axis in physical_order {
        if strides[axis] != expected {
            return false;
        }
        expected = expected.saturating_mul(shape[axis].max(1));
    }
    true
}

fn analyze_operand_materialization(
    shape: &[usize],
    strides: &[usize],
    modes: &[i32],
    batch_modes: &[i32],
    row_modes: &[i32],
    col_modes: &[i32],
) -> MaterializationPlan {
    let batch_axes: Vec<usize> = batch_modes
        .iter()
        .map(|&mode| mode_position(modes, mode))
        .collect();
    let row_axes: Vec<usize> = row_modes.iter().map(|&mode| mode_position(modes, mode)).collect();
    let col_axes: Vec<usize> = col_modes.iter().map(|&mode| mode_position(modes, mode)).collect();
    let physical_order = physical_axis_order(strides);

    let mut no_copy_orders = vec![batch_axes
        .iter()
        .chain(row_axes.iter())
        .chain(col_axes.iter())
        .copied()
        .collect::<Vec<_>>()];
    if row_axes != col_axes {
        no_copy_orders.push(
            batch_axes
                .iter()
                .chain(col_axes.iter())
                .chain(row_axes.iter())
                .copied()
                .collect(),
        );
    }

    let batch_len = batch_axes.len();
    let row_len = row_axes.len();
    for order in &no_copy_orders {
        let rows = &order[batch_len..batch_len + row_len];
        let cols = &order[batch_len + row_len..];
        if *order == physical_order
            && is_flattenable_group(shape, strides, &batch_axes, !batch_axes.is_empty())
            && is_flattenable_group(shape, strides, rows, false)
            && is_flattenable_group(shape, strides, cols, false)
        {
            return MaterializationPlan::NoCopy;
        }
    }

    if is_permutation_like(shape, strides, &physical_order) {
        let target_axes: Vec<usize> = batch_axes
            .iter()
            .chain(row_axes.iter())
            .chain(col_axes.iter())
            .copied()
            .collect();
        let perm: Vec<usize> = target_axes
            .iter()
            .map(|axis| {
                physical_order
                    .iter()
                    .position(|physical_axis| physical_axis == axis)
                    .expect("axis must exist in physical order")
            })
            .collect();
        let physical_shape: Vec<usize> = physical_order.iter().map(|&axis| shape[axis]).collect();
        return MaterializationPlan::Permute {
            perm,
            shape: physical_shape,
        };
    }

    MaterializationPlan::MakeContiguous
}

fn analyze_contraction_layout(
    shape_a: &[usize],
    strides_a: &[usize],
    modes_a: &[i32],
    shape_b: &[usize],
    strides_b: &[usize],
    modes_b: &[i32],
    modes_c: &[i32],
) -> ContractionLayoutPlan {
    let (batch_modes, left_candidates, right_candidates, contracted_modes) =
        classify_modes(modes_a, modes_b, modes_c);
    let output_set: HashSet<i32> = modes_c.iter().copied().collect();
    let left_modes: Vec<i32> = left_candidates
        .into_iter()
        .filter(|mode| output_set.contains(mode))
        .collect();
    let right_modes: Vec<i32> = right_candidates
        .into_iter()
        .filter(|mode| output_set.contains(mode))
        .collect();

    let left_materialization = analyze_operand_materialization(
        shape_a,
        strides_a,
        modes_a,
        &batch_modes,
        &left_modes,
        &contracted_modes,
    );
    let right_materialization = analyze_operand_materialization(
        shape_b,
        strides_b,
        modes_b,
        &batch_modes,
        &contracted_modes,
        &right_modes,
    );

    let current_output: Vec<i32> = left_modes
        .iter()
        .chain(right_modes.iter())
        .chain(batch_modes.iter())
        .copied()
        .collect();
    let output_perm = (current_output != modes_c).then(|| {
        modes_c
            .iter()
            .map(|mode| {
                current_output
                    .iter()
                    .position(|current_mode| current_mode == mode)
                    .expect("output mode must exist in current output")
            })
            .collect()
    });

    ContractionLayoutPlan {
        batch_size: product_of_dims(&batch_modes, modes_a, shape_a),
        left_size: product_of_dims(&left_modes, modes_a, shape_a),
        right_size: product_of_dims(&right_modes, modes_b, shape_b),
        contract_size: product_of_dims(&contracted_modes, modes_a, shape_a),
        batch_modes,
        left_modes,
        right_modes,
        contracted_modes,
        left_materialization,
        right_materialization,
        output_perm,
    }
}

use super::buffer_pool::ScratchPool;
use super::MatrixLayout;
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

    for (old_linear, scalar) in data.iter().copied().enumerate().take(old_size) {
        // Compute old multi-index (column-major)
        let mut remaining = old_linear;
        let mut new_linear = 0usize;
        let mut new_dim = 0usize;
        for (i, dim_size) in shape.iter().copied().enumerate() {
            let coord = remaining % dim_size;
            remaining /= dim_size;
            if !trace_positions.contains(&i) {
                new_linear += coord * new_strides[new_dim];
                new_dim += 1;
            }
        }

        let acc = A::from_scalar(result[new_linear]);
        let val = A::from_scalar(scalar);
        result[new_linear] = acc.add(val).to_scalar();
    }

    (result, new_shape, new_modes)
}

fn group_base_stride(modes: &[i32], strides: &[usize], group_modes: &[i32]) -> isize {
    group_modes
        .first()
        .map(|&mode| strides[mode_position(modes, mode)] as isize)
        .unwrap_or(0)
}

fn matrix_layout_from_operand<'a, T>(
    data: &'a [T],
    shape: &[usize],
    strides: &[usize],
    modes: &[i32],
    row_modes: &[i32],
    col_modes: &[i32],
) -> MatrixLayout<'a, T> {
    MatrixLayout {
        data,
        rows: product_of_dims(row_modes, modes, shape),
        cols: product_of_dims(col_modes, modes, shape),
        row_stride: group_base_stride(modes, strides, row_modes),
        col_stride: group_base_stride(modes, strides, col_modes),
    }
}

struct MaterializedMatrixOperand<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    modes: Vec<i32>,
}

fn materialize_matrix_operand<T: Copy + Default>(
    data: &[T],
    shape: &[usize],
    strides: &[usize],
    modes: &[i32],
    batch_modes: &[i32],
    row_modes: &[i32],
    col_modes: &[i32],
) -> MaterializedMatrixOperand<T> {
    let perm = compute_permutation(modes, batch_modes, row_modes, col_modes);
    let target_modes: Vec<i32> = batch_modes
        .iter()
        .chain(row_modes.iter())
        .chain(col_modes.iter())
        .copied()
        .collect();
    let target_shape: Vec<usize> = target_modes
        .iter()
        .map(|&mode| shape[mode_position(modes, mode)])
        .collect();
    let mut scratch = Vec::new();

    MaterializedMatrixOperand {
        data: materialize_with_permutation_into(data, shape, strides, &perm, &mut scratch)
            .as_slice()
            .to_vec(),
        strides: compute_contiguous_strides(&target_shape),
        shape: target_shape,
        modes: target_modes,
    }
}

fn finalize_contraction_output<T: Copy + Default>(
    c_data: Vec<T>,
    plan: &ContractionLayoutPlan,
    shape_c: &[usize],
    modes_c: &[i32],
) -> Vec<T> {
    if let Some(output_perm) = &plan.output_perm {
        let current_order: Vec<i32> = plan
            .left_modes
            .iter()
            .chain(plan.right_modes.iter())
            .chain(plan.batch_modes.iter())
            .copied()
            .collect();
        let c_shape_current: Vec<usize> = current_order
            .iter()
            .map(|&mode| shape_c[mode_position(modes_c, mode)])
            .collect();
        permute_data(&c_data, &c_shape_current, output_perm)
    } else {
        c_data
    }
}

enum MaterializedSlice<'a, T> {
    Borrowed(&'a [T]),
    Scratch(&'a [T]),
}

impl<'a, T> MaterializedSlice<'a, T> {
    fn as_slice(&self) -> &'a [T] {
        match self {
            Self::Borrowed(data) | Self::Scratch(data) => data,
        }
    }
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
    // 1. Classify modes
    let (_batch, left, right, _contracted) = classify_modes(modes_a, modes_b, modes_c);

    // 2. Handle trace modes: modes in only one input that are NOT in the output.
    //    GEMM can only contract modes shared by both inputs. Single-input modes
    //    not in the output must be summed over (traced) before GEMM.
    let c_set: HashSet<i32> = modes_c.iter().copied().collect();
    let left_trace: Vec<i32> = left
        .iter()
        .filter(|m| !c_set.contains(m))
        .copied()
        .collect();
    let right_trace: Vec<i32> = right
        .iter()
        .filter(|m| !c_set.contains(m))
        .copied()
        .collect();

    if left_trace.is_empty() && right_trace.is_empty() {
        let plan =
            analyze_contraction_layout(shape_a, strides_a, modes_a, shape_b, strides_b, modes_b, modes_c);
        let left_nocopy = matches!(plan.left_materialization, MaterializationPlan::NoCopy);
        let right_nocopy = matches!(plan.right_materialization, MaterializationPlan::NoCopy);
        if left_nocopy || right_nocopy {
            let left_materialized;
            let a_layout = if left_nocopy {
                matrix_layout_from_operand(
                    a,
                    shape_a,
                    strides_a,
                    modes_a,
                    &plan.left_modes,
                    &plan.contracted_modes,
                )
            } else {
                left_materialized = materialize_matrix_operand(
                    a,
                    shape_a,
                    strides_a,
                    modes_a,
                    &plan.batch_modes,
                    &plan.left_modes,
                    &plan.contracted_modes,
                );
                matrix_layout_from_operand(
                    &left_materialized.data,
                    &left_materialized.shape,
                    &left_materialized.strides,
                    &left_materialized.modes,
                    &plan.left_modes,
                    &plan.contracted_modes,
                )
            };

            let right_materialized;
            let b_layout = if right_nocopy {
                matrix_layout_from_operand(
                    b,
                    shape_b,
                    strides_b,
                    modes_b,
                    &plan.contracted_modes,
                    &plan.right_modes,
                )
            } else {
                right_materialized = materialize_matrix_operand(
                    b,
                    shape_b,
                    strides_b,
                    modes_b,
                    &plan.batch_modes,
                    &plan.contracted_modes,
                    &plan.right_modes,
                );
                matrix_layout_from_operand(
                    &right_materialized.data,
                    &right_materialized.shape,
                    &right_materialized.strides,
                    &right_materialized.modes,
                    &plan.contracted_modes,
                    &plan.right_modes,
                )
            };

            let c_data = if plan.batch_modes.is_empty() {
                cpu.gemm_standard_layout_internal::<A>(a_layout, b_layout)
            } else {
                cpu.gemm_batched_standard_layout_internal::<A>(plan.batch_size, a_layout, b_layout)
            };
            if let Some(c_data) = c_data {
                return finalize_contraction_output(c_data, &plan, shape_c, modes_c);
            }
        }
    }

    let mut a_pool = ScratchPool::<A::Scalar>::default();
    let mut b_pool = ScratchPool::<A::Scalar>::default();

    // 3. Reduce trace modes before the generic GEMM fallback.
    let (a_reduced, a_shape, a_modes, a_strides) = if !left_trace.is_empty() {
        let (a_data, a_shape, a_modes) = {
            let mut a_contig = a_pool.acquire(shape_a.iter().product::<usize>().max(1));
            let a_contig = ensure_contiguous_into(a, shape_a, strides_a, a_contig.as_mut_vec());
            reduce_trace_modes::<A>(a_contig.as_slice(), shape_a, modes_a, &left_trace)
        };
        let a_strides = compute_contiguous_strides(&a_shape);
        (Some(a_data), a_shape, a_modes, a_strides)
    } else {
        (None, shape_a.to_vec(), modes_a.to_vec(), strides_a.to_vec())
    };
    let (b_reduced, b_shape, b_modes, b_strides) = if !right_trace.is_empty() {
        let (b_data, b_shape, b_modes) = {
            let mut b_contig = b_pool.acquire(shape_b.iter().product::<usize>().max(1));
            let b_contig = ensure_contiguous_into(b, shape_b, strides_b, b_contig.as_mut_vec());
            reduce_trace_modes::<A>(b_contig.as_slice(), shape_b, modes_b, &right_trace)
        };
        let b_strides = compute_contiguous_strides(&b_shape);
        (Some(b_data), b_shape, b_modes, b_strides)
    } else {
        (None, shape_b.to_vec(), modes_b.to_vec(), strides_b.to_vec())
    };

    let plan = analyze_contraction_layout(
        &a_shape,
        &a_strides,
        &a_modes,
        &b_shape,
        &b_strides,
        &b_modes,
        modes_c,
    );

    // 5. Permute A to [left_free, contracted, batch] - batch LAST for current GEMM layout
    let a_perm = compute_permutation(
        &a_modes,
        &plan.left_modes,
        &plan.contracted_modes,
        &plan.batch_modes,
    );
    let mut a_permuted_scratch = a_pool.acquire(a_shape.iter().product::<usize>().max(1));
    let a_permuted = if let Some(ref a_data) = a_reduced {
        permute_data_into(a_data, &a_shape, &a_perm, a_permuted_scratch.as_mut_vec())
    } else {
        materialize_with_permutation_into(a, &a_shape, &a_strides, &a_perm, a_permuted_scratch.as_mut_vec())
    };

    // 6. Permute B to [contracted, right_free, batch] - batch LAST
    let b_perm = compute_permutation(
        &b_modes,
        &plan.contracted_modes,
        &plan.right_modes,
        &plan.batch_modes,
    );
    let mut b_permuted_scratch = b_pool.acquire(b_shape.iter().product::<usize>().max(1));
    let b_permuted = if let Some(ref b_data) = b_reduced {
        permute_data_into(b_data, &b_shape, &b_perm, b_permuted_scratch.as_mut_vec())
    } else {
        materialize_with_permutation_into(b, &b_shape, &b_strides, &b_perm, b_permuted_scratch.as_mut_vec())
    };

    // 7. Call GEMM
    let c_data = if plan.batch_modes.is_empty() {
        cpu.gemm_internal::<A>(
            a_permuted.as_slice(),
            plan.left_size,
            plan.contract_size,
            b_permuted.as_slice(),
            plan.right_size,
        )
    } else {
        cpu.gemm_batched_internal::<A>(
            a_permuted.as_slice(),
            plan.batch_size,
            plan.left_size,
            plan.contract_size,
            b_permuted.as_slice(),
            plan.right_size,
        )
    };

    finalize_contraction_output(c_data, &plan, shape_c, modes_c)
}

/// Ensure data is contiguous (copy if strided).
fn ensure_contiguous<T: Copy + Default>(data: &[T], shape: &[usize], strides: &[usize]) -> Vec<T> {
    let mut scratch = Vec::new();
    ensure_contiguous_into(data, shape, strides, &mut scratch)
        .as_slice()
        .to_vec()
}

fn ensure_contiguous_into<'a, T: Copy + Default>(
    data: &'a [T],
    shape: &[usize],
    strides: &[usize],
    scratch: &'a mut Vec<T>,
) -> MaterializedSlice<'a, T> {
    let identity: Vec<usize> = (0..shape.len()).collect();
    materialize_with_permutation_into(data, shape, strides, &identity, scratch)
}

/// Permute data according to axis permutation.
fn permute_data<T: Copy + Default>(data: &[T], shape: &[usize], perm: &[usize]) -> Vec<T> {
    let mut scratch = Vec::new();
    permute_data_into(data, shape, perm, &mut scratch)
        .as_slice()
        .to_vec()
}

fn permute_data_into<'a, T: Copy + Default>(
    data: &'a [T],
    shape: &[usize],
    perm: &[usize],
    scratch: &'a mut Vec<T>,
) -> MaterializedSlice<'a, T> {
    let strides = compute_contiguous_strides(shape);
    materialize_with_permutation_into(data, shape, &strides, perm, scratch)
}

fn materialize_with_permutation_into<'a, T: Copy + Default>(
    data: &'a [T],
    shape: &[usize],
    strides: &[usize],
    perm: &[usize],
    scratch: &'a mut Vec<T>,
) -> MaterializedSlice<'a, T> {
    let expected_strides = compute_contiguous_strides(shape);
    if strides == expected_strides && perm.iter().enumerate().all(|(i, &p)| i == p) {
        return MaterializedSlice::Borrowed(data);
    }

    scratch.clear();
    let numel: usize = shape.iter().product();
    scratch.resize(numel, T::default());
    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();

    for (new_idx, result_elem) in scratch.iter_mut().enumerate().take(numel) {
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
            old_idx += new_coords[new_dim] * strides[old_dim];
        }

        *result_elem = data[old_idx];
    }

    MaterializedSlice::Scratch(scratch.as_slice())
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
    let left_trace: Vec<i32> = left
        .iter()
        .filter(|m| !c_set.contains(m))
        .copied()
        .collect();
    let right_trace: Vec<i32> = right
        .iter()
        .filter(|m| !c_set.contains(m))
        .copied()
        .collect();

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
    let right_free: Vec<i32> = right
        .iter()
        .filter(|m| c_set.contains(m))
        .copied()
        .collect();

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

    #[test]
    fn test_analyze_contraction_layout_detects_copy_free_matmul() {
        let plan = analyze_contraction_layout(
            &[2, 3],
            &[1, 2],
            &[0, 1],
            &[3, 4],
            &[1, 3],
            &[1, 2],
            &[0, 2],
        );

        assert!(plan.batch_modes.is_empty());
        assert_eq!(plan.left_modes, vec![0]);
        assert_eq!(plan.right_modes, vec![2]);
        assert_eq!(plan.contracted_modes, vec![1]);
        assert!(matches!(plan.left_materialization, MaterializationPlan::NoCopy));
        assert!(matches!(plan.right_materialization, MaterializationPlan::NoCopy));
    }

    #[test]
    fn test_analyze_contraction_layout_marks_single_side_materialization() {
        let plan = analyze_contraction_layout(
            &[2, 2, 2],
            &[1, 2, 4],
            &[0, 1, 2],
            &[2, 2, 2],
            &[2, 1, 4],
            &[0, 2, 3],
            &[0, 1, 3],
        );

        assert!(matches!(plan.left_materialization, MaterializationPlan::NoCopy));
        assert!(matches!(
            plan.right_materialization,
            MaterializationPlan::Permute { .. }
        ));
    }

    #[test]
    fn test_analyze_contraction_layout_preserves_batch_tail() {
        let plan = analyze_contraction_layout(
            &[2, 2, 2],
            &[1, 2, 4],
            &[0, 1, 2],
            &[2, 2, 2],
            &[1, 2, 4],
            &[0, 2, 3],
            &[0, 1, 3],
        );

        assert_eq!(plan.batch_modes, vec![0]);
        assert_eq!(plan.batch_size, 2);
        assert!(matches!(plan.left_materialization, MaterializationPlan::NoCopy));
        assert!(matches!(plan.right_materialization, MaterializationPlan::NoCopy));
    }
}
