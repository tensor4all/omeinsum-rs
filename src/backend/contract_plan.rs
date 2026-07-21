//! Backend-neutral contraction *planning* (plus a couple of shared host helpers).
//!
//! Most of these helpers compute *what* a binary einsum contraction must do —
//! purely from the mode (index) labels and shapes — without touching any tensor
//! data or backend storage. They are the "planner" half of the planner/executor
//! split: the CPU and CUDA executors both consume this plan and then move data +
//! run the GEMM in their own backend-specific way.
//!
//! Mirrors the index logic of OMEinsum.jl's `binaryrules.jl`: classify modes into
//! batch / left / right / contracted, reduce the contraction to a (batched) matmul
//! of sizes `left_size × contract_size × right_size`, then permute the result back
//! to the requested output order.
//!
//! A small number of *data-touching but backend-neutral* host helpers also live
//! here so both executors can share them: [`reduce_trace`] (semiring sum over
//! trace modes) is used by the CPU executor and the CUDA tropical executor;
//! [`materialize_strided`] / [`gather_contiguous`] back the CUDA tropical path's
//! host-side operand layout (the CPU backend keeps its own zero-alloc materializer
//! on its hot path).

use std::collections::HashSet;

use crate::algebra::Algebra;
use crate::tensor::compute_contiguous_strides;

/// Classify modes into batch, left-only, right-only, and contracted.
///
/// - batch: in both A and B, and in output C
/// - left: only in A (free indices from A) — *may* include trace modes not in C
/// - right: only in B (free indices from B) — *may* include trace modes not in C
/// - contracted: in both A and B, but NOT in output C
pub(crate) fn classify_modes(
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
pub(crate) fn mode_position(modes: &[i32], mode: i32) -> usize {
    modes
        .iter()
        .position(|&m| m == mode)
        .expect("mode not found")
}

/// Compute the product of dimensions for given modes.
pub(crate) fn product_of_dims(modes: &[i32], all_modes: &[i32], shape: &[usize]) -> usize {
    modes
        .iter()
        .map(|&m| shape[mode_position(all_modes, m)])
        .product::<usize>()
        .max(1)
}

/// Compute permutation to reorder modes to [first..., second..., third...].
pub(crate) fn compute_permutation(
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

// ============================================================================
// Shared host materialization helpers (backend-neutral, data-touching)
// ============================================================================

/// Sum (via the semiring add `A::add`) over `trace_modes`, removing those
/// dimensions from a contiguous column-major buffer.
///
/// "Trace" modes are single-operand modes absent from the output: GEMM can only
/// contract modes shared by both inputs, so these must be reduced beforehand.
/// Works for every algebra (standard `+`, `MaxPlus` `max`, …) because it
/// accumulates through `A`. Shared by the CPU executor and the CUDA tropical
/// executor. Input must be contiguous column-major; returns the reduced data
/// plus its new shape and mode list.
pub(crate) fn reduce_trace<A: Algebra>(
    data: &[A::Scalar],
    shape: &[usize],
    all_modes: &[i32],
    trace_modes: &[i32],
) -> (Vec<A::Scalar>, Vec<usize>, Vec<i32>) {
    if trace_modes.is_empty() {
        return (data.to_vec(), shape.to_vec(), all_modes.to_vec());
    }

    let trace_positions: HashSet<usize> = trace_modes
        .iter()
        .map(|&m| mode_position(all_modes, m))
        .collect();

    let new_shape: Vec<usize> = (0..shape.len())
        .filter(|i| !trace_positions.contains(i))
        .map(|i| shape[i])
        .collect();
    let new_modes: Vec<i32> = (0..all_modes.len())
        .filter(|i| !trace_positions.contains(i))
        .map(|i| all_modes[i])
        .collect();

    let new_size = new_shape.iter().product::<usize>().max(1);
    let new_strides = compute_contiguous_strides(&new_shape);
    let mut result: Vec<A::Scalar> = vec![A::zero().to_scalar(); new_size];

    let old_size = shape.iter().product::<usize>().max(1);
    for (old_linear, scalar) in data.iter().copied().enumerate().take(old_size) {
        // Decode the column-major multi-index, keeping only the surviving axes.
        let mut remaining = old_linear;
        let mut new_linear = 0usize;
        let mut new_dim = 0usize;
        for (axis, &dim) in shape.iter().enumerate() {
            let coord = remaining % dim;
            remaining /= dim;
            if !trace_positions.contains(&axis) {
                new_linear += coord * new_strides[new_dim];
                new_dim += 1;
            }
        }
        result[new_linear] = A::from_scalar(result[new_linear])
            .add(A::from_scalar(scalar))
            .to_scalar();
    }

    (result, new_shape, new_modes)
}

/// Gather a (possibly strided) column-major host tensor into a contiguous
/// column-major buffer reordered by `perm` (`new_shape[i] = shape[perm[i]]`).
///
/// Correctness over speed (the CPU backend keeps its own zero-alloc materializer
/// for its hot path); used by the CUDA tropical executor to lay operands out.
#[cfg(any(feature = "cuda-tropical", test))]
pub(crate) fn materialize_strided<T: Copy + Default>(
    data: &[T],
    shape: &[usize],
    strides: &[usize],
    perm: &[usize],
) -> Vec<T> {
    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let numel: usize = new_shape.iter().product::<usize>();
    if numel == 0 {
        return Vec::new();
    }
    let mut out = vec![T::default(); numel];

    let mut coords = vec![0usize; new_shape.len()];
    for slot in out.iter_mut() {
        let src: usize = coords
            .iter()
            .enumerate()
            .map(|(axis, &c)| c * strides[perm[axis]])
            .sum();
        *slot = data[src];

        // Increment the multi-index in column-major order.
        for axis in 0..new_shape.len() {
            coords[axis] += 1;
            if coords[axis] < new_shape[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
    out
}

/// Gather a (possibly strided) column-major host tensor into a contiguous
/// column-major buffer in the *same* axis order (the identity permutation).
#[cfg(any(feature = "cuda-tropical", test))]
pub(crate) fn gather_contiguous<T: Copy + Default>(
    data: &[T],
    shape: &[usize],
    strides: &[usize],
) -> Vec<T> {
    let identity: Vec<usize> = (0..shape.len()).collect();
    materialize_strided(data, shape, strides, &identity)
}

/// Backend-neutral plan for a single binary contraction.
///
/// Produced by [`plan_contraction`]. Carries the canonical mode groups, the
/// matmul sizes, the per-operand permutations needed to reach the canonical
/// `[left, contracted, batch]` / `[contracted, right, batch]` layouts, and the
/// permutation that maps the GEMM output (`[left, right, batch]`) back to the
/// requested `modes_c` order. Any trace modes (single-operand modes not in the
/// output) are reported separately and must be summed out *before* the GEMM.
///
/// Consumed by the CUDA tropical executor (the CPU backend keeps its own richer
/// layout plan, and the cuTENSOR `cuda` path lowers via cuTENSOR directly), so it
/// is compiled only under `cuda-tropical` (and in tests).
#[cfg(any(feature = "cuda-tropical", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ContractionPlan {
    pub batch_modes: Vec<i32>,
    /// Free A modes that appear in the output.
    pub left_modes: Vec<i32>,
    /// Free B modes that appear in the output.
    pub right_modes: Vec<i32>,
    pub contracted_modes: Vec<i32>,
    /// A-only modes NOT in the output — must be reduced (summed) before GEMM.
    pub left_trace: Vec<i32>,
    /// B-only modes NOT in the output — must be reduced (summed) before GEMM.
    pub right_trace: Vec<i32>,
    pub batch_size: usize,
    pub left_size: usize,
    pub right_size: usize,
    pub contract_size: usize,
    /// Maps the canonical output order `[left, right, batch]` to `modes_c`.
    /// `None` when they already coincide (no final permute needed).
    pub output_perm: Option<Vec<usize>>,
}

#[cfg(any(feature = "cuda-tropical", test))]
impl ContractionPlan {
    /// Whether the contraction has trace modes that require a pre-reduction.
    ///
    /// The CUDA executor inspects `left_trace`/`right_trace` directly (it reduces
    /// each side independently), so this convenience predicate is currently only
    /// exercised by the planner unit tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn has_trace(&self) -> bool {
        !self.left_trace.is_empty() || !self.right_trace.is_empty()
    }

    /// Permutation bringing operand A into the canonical `[left, contracted, batch]`
    /// layout (batch last), given A's current mode order.
    pub(crate) fn a_permutation(&self, modes_a: &[i32]) -> Vec<usize> {
        compute_permutation(
            modes_a,
            &self.left_modes,
            &self.contracted_modes,
            &self.batch_modes,
        )
    }

    /// Permutation bringing operand B into the canonical `[contracted, right, batch]`
    /// layout (batch last), given B's current mode order.
    pub(crate) fn b_permutation(&self, modes_b: &[i32]) -> Vec<usize> {
        compute_permutation(
            modes_b,
            &self.contracted_modes,
            &self.right_modes,
            &self.batch_modes,
        )
    }
}

/// Build the backend-neutral [`ContractionPlan`] from mode labels and shapes.
///
/// Pure index/shape arithmetic — no tensor data is touched. The sizes are read
/// from whichever operand carries each mode (`modes_a`/`shape_a` for batch, left,
/// and contracted; `modes_b`/`shape_b` for right).
#[cfg(any(feature = "cuda-tropical", test))]
pub(crate) fn plan_contraction(
    modes_a: &[i32],
    shape_a: &[usize],
    modes_b: &[i32],
    shape_b: &[usize],
    modes_c: &[i32],
) -> ContractionPlan {
    let (batch_modes, left_candidates, right_candidates, contracted_modes) =
        classify_modes(modes_a, modes_b, modes_c);
    let output_set: HashSet<i32> = modes_c.iter().copied().collect();

    let mut left_modes = Vec::new();
    let mut left_trace = Vec::new();
    for m in left_candidates {
        if output_set.contains(&m) {
            left_modes.push(m);
        } else {
            left_trace.push(m);
        }
    }
    let mut right_modes = Vec::new();
    let mut right_trace = Vec::new();
    for m in right_candidates {
        if output_set.contains(&m) {
            right_modes.push(m);
        } else {
            right_trace.push(m);
        }
    }

    // Canonical GEMM output order is [left, right, batch]; map it to modes_c.
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
                    .expect("output mode must exist in canonical output")
            })
            .collect()
    });

    ContractionPlan {
        batch_size: product_of_dims(&batch_modes, modes_a, shape_a),
        left_size: product_of_dims(&left_modes, modes_a, shape_a),
        right_size: product_of_dims(&right_modes, modes_b, shape_b),
        contract_size: product_of_dims(&contracted_modes, modes_a, shape_a),
        batch_modes,
        left_modes,
        right_modes,
        contracted_modes,
        left_trace,
        right_trace,
        output_perm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_matmul_ik_kj_to_ij() {
        // "ik,kj->ij": i=0,k=1,j=2 ; A[i,k] (2x3), B[k,j] (3x4) -> C[i,j] (2x4)
        let plan = plan_contraction(&[0, 1], &[2, 3], &[1, 2], &[3, 4], &[0, 2]);
        assert_eq!(plan.left_modes, vec![0]);
        assert_eq!(plan.contracted_modes, vec![1]);
        assert_eq!(plan.right_modes, vec![2]);
        assert!(plan.batch_modes.is_empty());
        assert_eq!(
            (plan.left_size, plan.contract_size, plan.right_size),
            (2, 3, 4)
        );
        assert!(!plan.has_trace());
        assert_eq!(plan.output_perm, None); // canonical [left,right] == modes_c
        assert_eq!(plan.a_permutation(&[0, 1]), vec![0, 1]); // already [left, contracted]
        assert_eq!(plan.b_permutation(&[1, 2]), vec![0, 1]); // already [contracted, right]
    }

    #[test]
    fn batched_matmul_bik_bkj_to_bij() {
        // "bik,bkj->bij": b=0,i=1,k=2,j=3 ; batch=8, A 8x2x3, B 8x3x4 -> 8x2x4
        let plan = plan_contraction(&[0, 1, 2], &[8, 2, 3], &[0, 2, 3], &[8, 3, 4], &[0, 1, 3]);
        assert_eq!(plan.batch_modes, vec![0]);
        assert_eq!(plan.left_modes, vec![1]);
        assert_eq!(plan.contracted_modes, vec![2]);
        assert_eq!(plan.right_modes, vec![3]);
        assert_eq!(plan.batch_size, 8);
        assert_eq!(
            (plan.left_size, plan.contract_size, plan.right_size),
            (2, 3, 4)
        );
        // canonical output [left,right,batch] = [1,3,0]; modes_c = [0,1,3] -> needs permute
        assert_eq!(plan.output_perm, Some(vec![2, 0, 1]));
    }

    #[test]
    fn transposed_operand_ki_kj_to_ij_needs_permutation() {
        // "ki,kj->ij": k=0,i=1,j=2 ; A stored [k,i] must permute to [i,k]
        let plan = plan_contraction(&[0, 1], &[3, 2], &[0, 2], &[3, 4], &[1, 2]);
        assert_eq!(plan.left_modes, vec![1]);
        assert_eq!(plan.contracted_modes, vec![0]);
        assert_eq!(
            (plan.left_size, plan.contract_size, plan.right_size),
            (2, 3, 4)
        );
        // A is [k,i]; canonical [left=i, contracted=k] -> permutation [1,0]
        assert_eq!(plan.a_permutation(&[0, 1]), vec![1, 0]);
    }

    #[test]
    fn trace_mode_detected() {
        // "ikl,kj->ij": i=0,k=1,l=2,j=3 ; l only in A and not in output -> left trace
        let plan = plan_contraction(&[0, 1, 2], &[2, 3, 5], &[1, 3], &[3, 4], &[0, 3]);
        assert!(plan.has_trace());
        assert_eq!(plan.left_trace, vec![2]);
        assert!(plan.right_trace.is_empty());
        assert_eq!(plan.left_modes, vec![0]);
        assert_eq!(plan.contracted_modes, vec![1]);
    }

    // --- Shared host helpers (run without a GPU) -------------------------------

    #[test]
    fn gather_contiguous_is_identity_for_contiguous_input() {
        // 2x3 column-major contiguous: strides [1,2].
        let data: Vec<i32> = vec![0, 1, 2, 3, 4, 5];
        let out = gather_contiguous(&data, &[2, 3], &[1, 2]);
        assert_eq!(out, data);
    }

    #[test]
    fn gather_contiguous_materializes_a_transpose_view() {
        // Logical A is [k=3, i=2] but stored as the transpose of a 2x3 row block:
        // here we view a 2x3 contiguous buffer (strides [1,2]) as shape [3,2] with
        // strides [2,1] (i.e. transposed) and gather it contiguous.
        let data: Vec<i32> = vec![0, 1, 2, 3, 4, 5]; // col-major 2x3: cols (0,1),(2,3),(4,5)
        let out = gather_contiguous(&data, &[3, 2], &[2, 1]);
        // new[r + 3*c] = data[c*1 + r*2]; r in 0..3, c in 0..2
        // c=0: data[0],data[2],data[4] = 0,2,4 ; c=1: data[1],data[3],data[5] = 1,3,5
        assert_eq!(out, vec![0, 2, 4, 1, 3, 5]);
    }

    #[test]
    fn materialize_strided_permutes_axes() {
        // 2x3 column-major contiguous, permute to [j, i] (shape 3x2).
        let data: Vec<i32> = vec![0, 1, 2, 3, 4, 5]; // strides [1,2]
        let out = materialize_strided(&data, &[2, 3], &[1, 2], &[1, 0]);
        // out is shape [3,2] col-major: out[r + 3*c] = data[perm-applied]
        // element (j=r, i=c) -> data[c*1 + r*2]
        assert_eq!(out, vec![0, 2, 4, 1, 3, 5]);
    }

    #[test]
    fn materialize_strided_scalar_and_empty() {
        // 0-dim scalar: empty shape -> single element copied.
        let s: Vec<i32> = vec![42];
        assert_eq!(materialize_strided(&s, &[], &[], &[]), vec![42]);
        // size-0 dimension -> empty output.
        let e: Vec<i32> = vec![];
        assert_eq!(
            materialize_strided(&e, &[0, 3], &[1, 0], &[0, 1]),
            Vec::<i32>::new()
        );
    }

    #[test]
    fn reduce_trace_standard_sums_over_axis() {
        use crate::algebra::Standard;
        // A shape [i=2, l=3] col-major; reduce mode l (label 1), keep i (label 0).
        // Standard add = +, so result[i] = sum_l A[i,l].
        // data (col-major, strides [1,2]): A[0,0]=0,A[1,0]=1,A[0,1]=2,A[1,1]=3,A[0,2]=4,A[1,2]=5
        let data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (out, shape, modes) = reduce_trace::<Standard<f64>>(&data, &[2, 3], &[0, 1], &[1]);
        assert_eq!(shape, vec![2]);
        assert_eq!(modes, vec![0]);
        // i=0: 0+2+4=6 ; i=1: 1+3+5=9
        assert_eq!(out, vec![6.0, 9.0]);
    }

    #[test]
    fn reduce_trace_empty_is_noop() {
        use crate::algebra::Standard;
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let (out, shape, modes) = reduce_trace::<Standard<f64>>(&data, &[2, 2], &[0, 1], &[]);
        assert_eq!(out, data);
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(modes, vec![0, 1]);
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn reduce_trace_maxplus_takes_max_over_axis() {
        use crate::algebra::MaxPlus;
        // Same layout as the Standard test; MaxPlus add = max, so result[i] = max_l A[i,l].
        let data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (out, shape, _modes) = reduce_trace::<MaxPlus<f64>>(&data, &[2, 3], &[0, 1], &[1]);
        assert_eq!(shape, vec![2]);
        // i=0: max(0,2,4)=4 ; i=1: max(1,3,5)=5
        assert_eq!(out, vec![4.0, 5.0]);
    }
}
