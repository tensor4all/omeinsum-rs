# Contraction Performance Improvement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce CPU contraction overhead on the standardized `binary` and `network` benchmarks by adding stride-aware Standard fast paths, reusing temporary buffers, and tightening root-result handling without rewriting the architecture.

**Architecture:** Keep the public tensor and einsum APIs stable. Add an internal layout-analysis layer in the CPU contraction kernel, teach the Standard `f32`/`f64` GEMM path to consume strided/transposed views without unconditional permutation, introduce a typed scratch pool for temporary materialization, then let the optimized execution tree contract directly into final root order when that is only a permutation. Non-Standard and argmax-sensitive paths remain on conservative fallbacks until the Standard fast path is proven.

**Tech Stack:** Rust 2021, `faer`, Criterion, `make check`, `cargo bench`, `python3 benchmarks/compare.py`, `samply`.

---

**Execution Notes**

- Use `@superpowers/test-driven-development` for Tasks 1 through 5.
- If any contraction, argmax, or layout bug appears, stop and use `@superpowers/systematic-debugging` before changing more code.
- Before any completion claim, use `@superpowers/verification-before-completion` with the full benchmark and profiler loop from Task 6.
- Current line ranges below are anchors from branch `pr-29` at commit `00b23f2`; adjust only when earlier tasks have already moved the code.

## Chunk 1: Layout-Aware CPU Contraction

### Task 1: Extract contraction layout analysis with conservative fallback decisions

**Files:**
- Modify: `src/backend/cpu/contract.rs:11-79`
- Modify: `src/backend/cpu/contract.rs:145-329`
- Test: `src/backend/cpu/contract.rs:442-484`

- [ ] **Step 1: Write failing unit tests for the new planning layer**

Add unit tests near the existing `contract.rs` tests for a new internal planner API:

```rust
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
    assert!(matches!(plan.right_materialization, MaterializationPlan::Permute { .. }));
}
```

- [ ] **Step 2: Run the new tests to confirm the planner does not exist yet**

Run: `cargo test test_analyze_contraction_layout_detects_copy_free_matmul --lib`
Expected: FAIL with a missing `analyze_contraction_layout` / `MaterializationPlan` error.

- [ ] **Step 3: Implement the minimal planning types and helper**

Add an internal planning layer in `src/backend/cpu/contract.rs` with concrete types along these lines:

```rust
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

fn analyze_contraction_layout(
    shape_a: &[usize],
    strides_a: &[usize],
    modes_a: &[i32],
    shape_b: &[usize],
    strides_b: &[usize],
    modes_b: &[i32],
    modes_c: &[i32],
) -> ContractionLayoutPlan
```

The implementation in this task should stay conservative:

- classify batch, left, right, contracted modes once
- compute target operand order once
- detect when current strides already match the target logical order
- return `Permute` only for the side that actually needs it
- do not change `contract()` behavior yet beyond switching to the shared planner

- [ ] **Step 4: Re-run the planner tests**

Run: `cargo test test_analyze_contraction_layout_detects_copy_free_matmul --lib`
Expected: PASS

- [ ] **Step 5: Commit the planning groundwork**

```bash
git add src/backend/cpu/contract.rs
git commit -m "perf(cpu): add contraction layout analysis"
```

### Task 2: Add an unbatched Standard `f32`/`f64` fast path that accepts strided views

**Files:**
- Modify: `src/backend/cpu/mod.rs:21-54`
- Modify: `src/backend/cpu/mod.rs:333-390`
- Modify: `src/backend/cpu/contract.rs:147-260`
- Test: `src/backend/cpu/mod.rs`
- Test: `tests/suites/backend_contract.rs:129-230`

- [ ] **Step 1: Write failing tests for layout-aware Standard GEMM**

Add focused CPU-backend tests for a new Standard-only GEMM helper:

```rust
#[test]
fn test_faer_layout_gemm_accepts_rhs_transpose_view() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![1.0f32, 2.0, 3.0, 4.0];

    let c = faer_gemm_f32_layout(
        MatrixLayout::column_major(&a, 2, 2),
        MatrixLayout::column_major_transposed(&b, 2, 2),
    );

    let expected = faer_gemm_f32(&a, 2, 2, &[1.0, 3.0, 2.0, 4.0], 2);
    assert_eq!(c, expected);
}
```

Add one end-to-end regression in `tests/suites/backend_contract.rs` that uses a strided right-hand input and expects the same result as the contiguous case.

- [ ] **Step 2: Run the new tests to confirm the helper is missing**

Run: `cargo test test_faer_layout_gemm_accepts_rhs_transpose_view --lib`
Expected: FAIL with a missing `MatrixLayout` / `faer_gemm_f32_layout` error.

- [ ] **Step 3: Implement the Standard layout-aware GEMM helper**

In `src/backend/cpu/mod.rs`, add a Standard-only layout adapter:

```rust
#[derive(Clone, Copy)]
pub(crate) struct MatrixLayout<'a, T> {
    pub data: &'a [T],
    pub rows: usize,
    pub cols: usize,
    pub row_stride: isize,
    pub col_stride: isize,
}

pub(crate) fn gemm_standard_layout_internal<A: Algebra>(
    &self,
    a: MatrixLayout<'_, A::Scalar>,
    b: MatrixLayout<'_, A::Scalar>,
) -> Option<Vec<A::Scalar>>
```

Implementation requirements:

- support only `Standard<f32>` and `Standard<f64>`
- translate layout metadata into `faer` matrix views instead of rebuilding full matrices through `permute_data`
- return `None` for all other algebras so the caller can use the existing fallback
- preserve the current `gemm_internal()` API for callers that already have contiguous canonical matrices

- [ ] **Step 4: Thread the planner into `contract()` for the unbatched Standard case**

Update `src/backend/cpu/contract.rs` so the unbatched Standard path follows this order:

1. use `analyze_contraction_layout(...)`
2. if both operands are `NoCopy`, build `MatrixLayout` views and call `gemm_standard_layout_internal`
3. if only one side needs a permute/copy, materialize only that side
4. if the Standard fast path declines, fall back to the current fully materialized path unchanged

Do not change `contract_with_argmax()` in this task.

- [ ] **Step 5: Run focused regressions and the binary benchmark**

Run: `cargo test --test main backend_contract::test_cpu_contract_both_strided`
Expected: PASS

Run: `cargo test --test main backend_contract::test_cpu_contract_output_permuted`
Expected: PASS

Run: `cargo bench --bench binary`
Expected: PASS and at least the non-batched scenarios improve relative to the `0a20e38` baseline.

- [ ] **Step 6: Commit the unbatched fast path**

```bash
git add src/backend/cpu/mod.rs src/backend/cpu/contract.rs tests/suites/backend_contract.rs
git commit -m "perf(cpu): add standard strided gemm fast path"
```

### Task 3: Extend the Standard fast path to batched contractions while preserving argmax fallbacks

**Files:**
- Modify: `src/backend/cpu/contract.rs:331-440`
- Modify: `src/backend/cpu/mod.rs:138-200`
- Test: `src/backend/cpu/contract.rs:442-484`
- Test: `tests/suites/backend_contract.rs:66-90`
- Test: `tests/suites/backend_contract.rs:233-260`

- [ ] **Step 1: Add failing tests for batched layout planning**

Add planner-level tests for batched contractions:

```rust
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
```

Keep the existing tropical backend tests untouched so they continue to verify the generic / argmax route.

- [ ] **Step 2: Run the new batched planner test**

Run: `cargo test test_analyze_contraction_layout_preserves_batch_tail --lib`
Expected: FAIL until the batched planner fields are wired through.

- [ ] **Step 3: Implement a batched Standard layout helper**

Extend `src/backend/cpu/mod.rs` with a batched Standard helper that reuses the single-batch layout logic:

```rust
pub(crate) fn gemm_batched_standard_layout_internal<A: Algebra>(
    &self,
    batch_size: usize,
    a: MatrixLayout<'_, A::Scalar>,
    b: MatrixLayout<'_, A::Scalar>,
) -> Option<Vec<A::Scalar>>
```

Requirements:

- loop over batch slices using the planner’s batch metadata
- reuse a single output allocation for the full batched result
- preserve `gemm_batched_internal()` as the generic fallback for non-Standard algebras

- [ ] **Step 4: Use the new helper only on the Standard branch**

Update `contract()` so batched Standard contractions use the layout-aware helper when possible, but keep:

- `contract_with_argmax()` on the current materialized path
- `gemm_batched_with_argmax_internal()` unchanged
- tropical and other non-Standard paths unchanged

- [ ] **Step 5: Run correctness and benchmark gates**

Run: `cargo test --test main backend_contract::test_cpu_contract_batched`
Expected: PASS

Run: `cargo test --test main backend_contract::test_cpu_contract_batched_output_permuted`
Expected: PASS

Run: `cargo test --features tropical --test main backend_contract::test_cpu_contract_tropical`
Expected: PASS

Run: `cargo bench --bench network`
Expected: PASS and some movement on `small`, `medium`, or `large` away from the fully materialized baseline.

- [ ] **Step 6: Commit the batched Standard path**

```bash
git add src/backend/cpu/mod.rs src/backend/cpu/contract.rs tests/suites/backend_contract.rs
git commit -m "perf(cpu): extend standard fast path to batched contractions"
```

## Chunk 2: Scratch Buffer Reuse

### Task 4: Add a typed CPU scratch pool for temporary materialization

**Files:**
- Create: `src/backend/cpu/buffer_pool.rs`
- Modify: `src/backend/cpu/mod.rs:1-13`
- Modify: `src/backend/cpu/contract.rs:262-329`
- Test: `src/backend/cpu/buffer_pool.rs`

- [ ] **Step 1: Write failing tests for scratch reuse**

Create `src/backend/cpu/buffer_pool.rs` with tests first:

```rust
#[test]
fn test_scratch_pool_reuses_released_capacity() {
    let mut pool = ScratchPool::<f32>::default();
    let mut first = pool.acquire(32);
    first.as_mut_slice().fill(1.0);
    drop(first);

    let second = pool.acquire(16);
    assert!(second.capacity() >= 32);
}

#[test]
fn test_scratch_pool_grows_when_requested_capacity_is_larger() {
    let mut pool = ScratchPool::<f32>::default();
    let small = pool.acquire(8);
    drop(small);

    let large = pool.acquire(128);
    assert!(large.capacity() >= 128);
}
```

- [ ] **Step 2: Run the tests to confirm the module is missing**

Run: `cargo test test_scratch_pool_reuses_released_capacity --lib`
Expected: FAIL because `src/backend/cpu/buffer_pool.rs` is not wired into the module tree yet.

- [ ] **Step 3: Implement the minimal typed pool**

Add a small internal pool API:

```rust
#[derive(Default)]
pub(crate) struct ScratchPool<T> {
    free: Vec<Vec<T>>,
}

pub(crate) struct ScratchBuffer<'a, T> {
    buf: Vec<T>,
    pool: &'a mut ScratchPool<T>,
}
```

Required behavior:

- capacity-based reuse only
- no public API exposure
- fully overwrite temporary buffers before any read
- no unsafe code for the initial version

- [ ] **Step 4: Switch materialization helpers to accept reusable buffers**

Refactor `ensure_contiguous()` and `permute_data()` into internal variants that can write into caller-owned scratch storage, for example:

```rust
enum MaterializedSlice<'a, T> {
    Borrowed(&'a [T]),
    Scratch(&'a [T]),
}

fn ensure_contiguous_into<'a, T: Copy + Default>(
    data: &[T],
    shape: &[usize],
    strides: &[usize],
    scratch: &'a mut Vec<T>,
) -> MaterializedSlice<'a, T>

fn permute_data_into<'a, T: Copy + Default>(
    data: &[T],
    shape: &[usize],
    perm: &[usize],
    scratch: &'a mut Vec<T>,
) -> MaterializedSlice<'a, T>
```

Keep compatibility wrappers if needed so earlier tests stay readable.

- [ ] **Step 5: Run the pool tests and core contract regressions**

Run: `cargo test test_scratch_pool_reuses_released_capacity --lib`
Expected: PASS

Run: `cargo test --test main backend_contract::test_cpu_contract_strided_input`
Expected: PASS

Run: `cargo test --test main backend_contract::test_cpu_contract_output_permuted`
Expected: PASS

- [ ] **Step 6: Commit the scratch-pool integration**

```bash
git add src/backend/cpu/buffer_pool.rs src/backend/cpu/mod.rs src/backend/cpu/contract.rs
git commit -m "perf(cpu): reuse scratch buffers for contraction materialization"
```

## Chunk 3: Root-Result Execution Cleanup And Full Verification

### Task 5: Let optimized tree execution contract directly into final root order when safe

**Files:**
- Modify: `src/einsum/engine.rs:134-261`
- Modify: `src/einsum/engine.rs:350-570`
- Modify: `src/tensor/ops.rs:76-133`
- Test: `src/einsum/engine.rs:825-1524`

- [ ] **Step 1: Add failing engine tests for root-output planning**

Add helper-focused tests in `src/einsum/engine.rs` for a new root-output decision:

```rust
#[test]
fn test_root_output_plan_uses_final_order_for_pure_permutation() {
    let tree_output = vec![2, 0, 1];
    let final_output = vec![0, 1, 2];

    assert!(can_emit_final_root_output(&tree_output, &final_output));
}

#[test]
fn test_root_output_plan_rejects_non_permutation_finalize_cases() {
    let tree_output = vec![0, 1, 2];
    let final_output = vec![0, 0, 2];

    assert!(!can_emit_final_root_output(&tree_output, &final_output));
}
```

- [ ] **Step 2: Run the new tests**

Run: `cargo test test_root_output_plan_uses_final_order_for_pure_permutation --lib`
Expected: FAIL because the helper does not exist yet.

- [ ] **Step 3: Add a private execution option for binary contraction**

Extend `src/tensor/ops.rs` with a private options struct used only inside the crate:

```rust
#[derive(Default)]
pub(crate) struct BinaryContractOptions {
    pub preferred_output_indices: Option<Vec<usize>>,
}
```

Add a private helper that keeps the public API unchanged:

```rust
fn contract_binary_impl_with_options<A: Algebra<Scalar = T, Index = u32>>(
    &self,
    other: &Self,
    ia: &[usize],
    ib: &[usize],
    iy: &[usize],
    track_argmax: bool,
    options: &BinaryContractOptions,
) -> (Self, Option<Tensor<u32, B>>)
```

Public `contract_binary()` and `contract_binary_with_argmax()` should call it with default options.

- [ ] **Step 4: Refactor optimized execution to distinguish root and intermediate nodes**

In `src/einsum/engine.rs`:

- add `can_emit_final_root_output(tree_output, final_output)`
- change `execute_tree()` / `execute_tree_with_argmax()` to pass final root output order directly into the last binary contraction only when the finalize step would otherwise be a pure permutation
- keep the existing `finalize_optimized_result()` fallback for cases that still require unary trace/diagonal/permutation logic

This keeps the change small and avoids a broader IR rewrite.

- [ ] **Step 5: Run optimized-tree regressions**

Run: `cargo test test_einsum_matmul --lib`
Expected: PASS

Run: `cargo test test_einsum_transpose_optimized --lib`
Expected: PASS

Run: `cargo test test_outer_product_optimized --lib`
Expected: PASS

Run: `cargo test --test main integration::test_matmul_four_tensors`
Expected: PASS

Run: `cargo test --test main einsum_core::test_einsum_triple_contraction`
Expected: PASS

- [ ] **Step 6: Commit the root-output cleanup**

```bash
git add src/einsum/engine.rs src/tensor/ops.rs
git commit -m "perf(einsum): contract optimized roots in final order"
```

### Task 6: Run the full correctness, benchmark, and profiler loop and report the result

**Files:**
- Verify: `target/criterion/**`
- Verify: `benchmarks/data/*.json`
- Optional report: PR comment on `tensor4all/omeinsum-rs#29`

- [ ] **Step 1: Run the full non-GPU correctness gate**

Run: `make check`
Expected: PASS

- [ ] **Step 2: Re-run the Rust benchmarks**

Run: `cargo bench --bench binary`
Expected: PASS and refreshed `target/criterion/binary/**`

Run: `make bench-network`
Expected: PASS and refreshed `target/criterion/network/**` plus `benchmarks/data/rust_network_timings.json`

- [ ] **Step 3: Recompute the Rust-vs-Julia comparison**

Run: `python3 benchmarks/compare.py`
Expected: PASS with a comparison table that shows reduced slowdown for `binary` and improved `large` / `3reg_150` network cases relative to the `0a20e38` baseline.

- [ ] **Step 4: Re-profile the heavy network case**

Run: `samply record cargo run --release --example profile_network -- --scenario 3reg_150 --iterations 1 --output /tmp/omeinsum-profile-network-after.json`
Expected: PASS and a trace where `permute_data` and materialization overhead are lower than the earlier profile.

- [ ] **Step 5: Summarize the benchmark delta and profiler delta**

Capture, at minimum:

- the before/after `binary` slowdown range versus Julia
- the before/after `network/large` and `network/3reg_150` ratios
- whether the hottest self-time frame moved away from `contract::permute_data`

If the slowdown remains large, note the next bottleneck explicitly rather than guessing.

- [ ] **Step 6: Commit the final state and post the PR update**

```bash
git add src/backend/cpu src/einsum/engine.rs src/tensor/ops.rs tests/suites/backend_contract.rs
git commit -m "perf(cpu): reduce contraction materialization overhead"
```

Then post or update the PR comment with the new benchmark table and profiler findings.
