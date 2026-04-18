# Contraction Performance Improvement Design

Date: 2026-04-12
Branch: `pr-29`

## Goal

Reduce the CPU contraction overhead that now dominates the new benchmark suite,
starting from the concrete hotspots observed on PR `#29`:

- `omeinsum::backend::cpu::Cpu::gemm_batched_internal`
- `omeinsum::backend::cpu::contract::permute_data`
- the surrounding `execute_tree -> contract_binary -> contract` path

The target is not a rewrite of `omeinsum-rs` into a new IR/compiler model. The
goal is a staged performance pass within the current architecture that:

1. removes avoidable copies before GEMM,
2. reuses temporary buffers instead of allocating fresh `Vec`s at each binary
   step, and
3. reduces tree-execution allocation overhead at the root and along
   intermediate contraction paths.

## Non-Goals

- porting `tenferro-rs` as an architecture
- introducing a `DotGeneral` graph IR or optimizer pipeline in this PR
- redesigning the public tensor API around borrowed views
- changing tensor semantics, storage order, or algebra behavior
- optimizing CUDA paths
- adding strict benchmark thresholds to CI in this pass

## Current Evidence

### Benchmark Evidence

Fresh comparison results on this branch show:

- `binary`: Rust is slower than Julia on all 8 scenarios
- `network`: Rust is faster on `small` and `medium`, but slower on `large`
  and dramatically slower on `3reg_150`

Representative numbers from the current comparison table:

- `binary/high_d_15x15_contract_5_batch_5`: Rust `584349 ns`, Julia `35000 ns`
  (`+1569.57%` slowdown)
- `binary/high_d_20x20_contract_9`: Rust `16763430 ns`, Julia `12573417 ns`
  (`+33.32%` slowdown)
- `network/large`: Rust `5075392 ns`, Julia `1720750 ns`
  (`+194.95%` slowdown)
- `network/3reg_150`: Rust `78631096917 ns`, Julia `853717541 ns`
  (`+9110.43%` slowdown)

### Profiling Evidence

`samply` profiling of the heavy network path
(`profile_network --scenario 3reg_150 --iterations 1`) shows the main-thread
hot path is:

- `Einsum::execute_tree`
- `Tensor::contract_binary`
- `backend::cpu::contract::contract`

The dominant self-time hotspots are:

- `Cpu::gemm_batched_internal`
- `contract::permute_data`

This matters because the current CPU contraction path does not only pay for
arithmetic. It also pays repeatedly for:

- making inputs contiguous,
- permuting both operands into canonical GEMM order,
- allocating fresh output/intermediate buffers,
- then permuting the result into output order.

### Reference Comparison

`reference/tenferro-rs` is relevant because it already encodes several ideas
that line up directly with the profiled hotspots:

- contiguous column-major tensors as the execution baseline
- stride-aware GEMM analysis in
  `tenferro-tensor/src/cpu/gemm/mod.rs`
- BLAS/FAER GEMM entry points that can consume strided inputs using transpose
  flags instead of forcing materialized transposes
- typed temporary buffer reuse
- explicit documentation that physical copies should be avoided when dimension
  groups are already fusible

The important lesson is not "copy tenferro's architecture". The useful lesson
is that `omeinsum-rs` should stop treating every non-canonical contraction as
"copy both sides, then GEMM".

## Current Implementation Problems

### 1. Unconditional materialization in CPU contraction

`src/backend/cpu/contract.rs` currently:

- forces both operands contiguous with `ensure_contiguous`
- computes a canonical permutation for each operand
- materializes both permutations with `permute_data`
- runs GEMM
- potentially materializes another permutation on the output

That is simple and correct, but it means even a layout that is already
compatible with GEMM still pays for full physical copies.

### 2. Batched GEMM is arithmetic-only and layout-blind

`src/backend/cpu/mod.rs` currently has:

- contiguous-only `gemm_internal`
- contiguous-only `gemm_batched_internal`

The standard `f32`/`f64` path uses faer only after the inputs have already been
packed into contiguous canonical matrices. The generic batched path loops over
batches and allocates/copies full batch outputs.

### 3. Tree execution always returns owned intermediates

`src/einsum/engine.rs` recursively clones leaves and returns owned
`Tensor<T, B>` results for every intermediate contraction. The root path then
returns another owned tensor. There is no direct "write final result into the
known destination layout" path, and there is no explicit reuse of freed
intermediate storage.

### 4. No CPU scratch reuse

The current contraction pipeline allocates fresh `Vec`s for:

- contiguous copies,
- permutations,
- batched outputs,
- intermediate contraction results

This amplifies the cost of the recursive execution tree, especially on the
large network workloads.

## Proposed Design

Use a layered performance plan within the existing architecture.

### Stage 1: Stride-Aware CPU Contraction Fast Path

Add a new internal analysis layer to the CPU contraction path that decides
whether a binary contraction can reach GEMM without materializing both operands.

#### Design

Split the current `contract.rs` flow into distinct phases:

1. mode classification and trace pre-reduction
2. contraction layout analysis
3. fast-path GEMM dispatch when the existing layout is fusible
4. partial materialization fallback for unfusible groups
5. output writeback / final permutation

The layout-analysis phase should derive:

- free / contracted / batch mode groups
- `m`, `n`, `k`, and batch product
- canonical target order for each operand
- whether the current operand strides already represent the required GEMM view
- whether one side can be handled via transpose/leading-dimension metadata
  instead of physical permutation

#### Fast-path scope

The first fast path should focus on the benchmark-critical standard numeric CPU
case:

- `Standard<f32>`
- `Standard<f64>`

All other algebras keep the current correct fallback until the fast path is
proven out.

That keeps the initial change targeted and aligned with the benchmark evidence.

#### Fallback policy

Do not require "all or nothing" materialization.

The intended fallback order is:

1. no materialization if both operand layouts are already fusible
2. materialize only the unfusible operand or dim group when possible
3. fall back to the current fully materialized path if needed

This is the most important lesson from the `tenferro-rs` comparison.

### Stage 2: Typed Scratch Buffer Reuse

Add a small backend-owned temporary buffer facility under `src/backend/cpu/`.

#### Design

Use a typed pool keyed by capacity, similar in spirit to
`tenferro-tensor/src/buffer_pool.rs`, but sized to the current repo:

- backend-owned, not global
- CPU-only
- separated by scalar type
- used only for temporary materialization and intermediate outputs

The pool should support:

- acquire scratch buffer with at least `N` capacity
- release scratch buffer after the last use
- safe zero-length / empty-buffer handling

The initial call sites should be:

- contiguous-copy fallback
- permutation materialization
- batched GEMM output staging
- contraction intermediate outputs

#### Constraints

This pool must not change public tensor ownership semantics. It is an internal
execution detail. The pool should only hand out buffers that are fully
overwritten before read.

### Stage 3: Tree Execution Output And Intermediate Policy

Reduce allocation overhead above the kernel level by changing how the execution
tree manages results.

#### Design

Introduce a staged execution path in `src/einsum/engine.rs` that distinguishes:

- leaf passthrough,
- intermediate contraction,
- root/final contraction

The root path should be able to write directly into its final output layout
when:

- the optimized tree root output matches the final output shape/order, or
- the engine can precompute the final normalized layout and hand it into the
  last binary contraction without a trailing result permutation

Intermediates should be treated as recyclable temporaries rather than
implicitly permanent owned values. This stage does not require a public
borrowed-view API, but it should stop assuming every binary contraction result
must allocate a fresh final buffer with no reuse path.

#### Scope boundary

This stage is intentionally smaller than a full borrowed-view or ExecIR
architecture. The aim is:

- direct root write when possible
- cleaner separation between temporary and final outputs
- less intermediate allocation churn

## File-Level Impact

Expected primary files:

- Modify: `src/backend/cpu/contract.rs`
  Introduce analysis helpers, fast-path dispatch, and partial materialization.
- Modify: `src/backend/cpu/mod.rs`
  Add lower-level GEMM helpers that can consume analyzed layout metadata.
- Create: `src/backend/cpu/buffer_pool.rs` or similar
  CPU scratch reuse for temporary materialization and outputs.
- Modify: `src/einsum/engine.rs`
  Root-write and intermediate-output policy changes.
- Modify: `src/tensor/ops.rs`
  Only if needed to thread new contraction helpers or explicit temporary-aware
  hooks through binary contraction.

Expected test surfaces:

- unit tests near CPU contraction/layout code
- integration tests in the existing `tests/` structure
- benchmark reruns for `binary` and `network`

## Verification Strategy

### Correctness

For each stage, verify equivalence against the current behavior on:

- standard matmul
- batched contractions
- contractions with trace/pre-reduction on one side
- non-contiguous inputs
- output permutations
- tropical argmax paths where the fast path is not yet used

### Benchmarking

Minimum performance verification:

1. `make check`
2. `cargo bench --bench binary`
3. `make bench-network`
4. `python3 benchmarks/compare.py`

The point is not "all slowdowns disappear in one patch". The point is to make
each stage measurable and to confirm that changes move the profiled hotspots in
the right direction.

## Risks

### Layout-analysis bugs

The highest risk is silent wrong answers from incorrect stride/dimension
analysis. This must be guarded with strong equivalence tests and conservative
fallbacks.

### Temporary reuse bugs

Incorrect scratch reuse can produce stale-data or aliasing bugs. Internal pool
APIs must make overwrite-before-read assumptions explicit.

### Scope drift

The `tenferro-rs` comparison can tempt the implementation toward an IR rewrite.
That would be the wrong move for this PR. The work should stay rooted in the
current backend and execution engine.

## Recommendation

Implement the performance work as a layered plan:

1. stride-aware contraction fast path first,
2. typed scratch reuse second,
3. root/intermediate execution cleanup third.

This order is justified by the current benchmark data, the `samply` hotspots,
and the strongest directly transferable lessons from `tenferro-rs`.
