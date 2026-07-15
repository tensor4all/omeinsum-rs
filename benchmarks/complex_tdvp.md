# Optimized CPU GEMM for standard complex contractions

## Status

- Branch: `perf/complex-gemm`
- Owner workload: `rydbergsim-rs` issue #272 (Keesling 2019 TDVP reproduction)
- Target repository: `tensor4all/omeinsum-rs`
- Tracking issue: [`tensor4all/omeinsum-rs#53`](https://github.com/tensor4all/omeinsum-rs/issues/53)
- Current state: CPU implementation and upstream validation complete; downstream validation pending

## Problem

OMEinsum supports `Standard<Complex32>` and `Standard<Complex64>` semantically, but its CPU GEMM dispatch only sends `Standard<f32>` and `Standard<f64>` to faer. Complex contractions therefore fall through to `generic_gemm`, a scalar triple loop.

The contraction planner is not the problem. It already:

1. classifies free, contracted, and batch labels;
2. recognizes directly usable matrix layouts;
3. materializes permutations when required;
4. lowers binary contractions to GEMM.

The missing piece is optimized complex dispatch at the final GEMM boundary.

This gap dominates local Hamiltonian applications in matrix-product-state TDVP. The motivating finite-range Rydberg Hamiltonian has MPO bond dimension `D=9`; the expensive contractions repeatedly multiply complex matrices whose dimensions scale with MPS bond dimension `chi`.

## Scope

### In scope

- Optimized faer GEMM for `Standard<Complex32>` and `Standard<Complex64>` on CPU.
- Contiguous, strided/transposed, and batched execution paths.
- Concrete value tests against the generic implementation.
- A deterministic benchmark reproducing the TDVP contraction shapes.
- Before/after measurements on one pinned machine and toolchain.
- Downstream validation in `rydbergsim-rs` without its local mini-einsum workaround.

### Out of scope

- Contraction-order changes.
- Tropical algebra.
- CUDA/cuTENSOR.
- A new public API.
- TDVP-specific logic inside OMEinsum.
- Thread-pool policy changes; initial implementation should match the existing real-valued faer paths and use `Par::Seq`.

## Target computations

The benchmark at `benches/complex_tdvp.rs` calls the public `Tensor::contract_binary` path. It fixes the MPO bond dimension at `D=9` and exercises `chi in {32, 64, 128}`.

| Case | Left shape | Right shape | Contracted modes | Equivalent GEMM work |
|---|---|---|---|---|
| `h1-left-environment` | `[chi,D,chi]` | `[chi,2,chi]` | one `chi` mode | `(D chi x chi) @ (chi x 2 chi)` |
| `h1-right-environment` | `[chi,chi,2,D]` | `[chi,D,chi]` | `chi*D` | `(2 chi x chi D) @ (chi D x chi)` |
| `h2-left-environment` | `[chi,D,chi]` | `[chi,2,2,chi]` | one `chi` mode | `(D chi x chi) @ (chi x 4 chi)` |
| `h2-right-environment` | `[chi,2,chi,2,D]` | `[chi,D,chi]` | `chi*D` | `(4 chi x chi D) @ (chi D x chi)` |

These are library-level tensor contractions, not a paper simulator embedded in OMEinsum. The labels and shapes come from the downstream workload; the benchmark data are deterministic synthetic complex values.

## Existing dispatch gap

Four CPU entry points need complex support:

1. `Cpu::gemm_internal` — contiguous GEMM.
2. `Cpu::gemm_standard_layout_internal` — directly usable strided/transposed layouts.
3. `Cpu::gemm_batched_internal` — materialized batched GEMM.
4. `Cpu::gemm_batched_standard_layout_internal` — direct batched layouts.

Today each recognizes only `Standard<f32>` and `Standard<f64>`. Complex types return `None` from the layout fast path and eventually reach `generic_gemm`.

## Implementation route

### Stage 0 — establish the baseline

Run each benchmark filter separately in release mode. The scalar baseline can be slow at `chi=128`, so do not start with the whole matrix.

Required order:

1. `chi=32` sanity run.
2. `chi=64` primary baseline.
3. `chi=128` paper-scale kernel baseline after the first two complete.

All performance runs must follow this repository's runscribe protocol. Declare a goal and hypothesis, fill its `Why this path`, obtain owner approval, then wrap every command with `runscribe run`.

Example command after a hypothesis code exists:

```bash
runscribe run --hyp <code> --tag baseline-chi64 -- \
  cargo bench --bench complex_tdvp -- 'chi64-d9'
```

Record the run directory, median time, dispersion, and throughput for every case.

### Stage 1 — contiguous complex GEMM

Add faer helpers for `Complex32` and `Complex64`, then dispatch them from `gemm_internal` using the same `TypeId` pattern as real scalars.

Prefer borrowed column-major `MatRef` inputs and one owned output allocation. Do not copy inputs into `Mat::from_fn` unless faer's type constraints make borrowing impossible.

Correctness tests:

- hand-checkable 2x2 complex products;
- rectangular products;
- both complex widths;
- comparison with `generic_gemm` on deterministic inputs;
- zero/degenerate dimensions if currently supported by the real path.

### Stage 2 — direct layout fast path

Add complex equivalents of the existing layout helpers and dispatch them from `gemm_standard_layout_internal`.

Required layouts:

- ordinary column-major;
- transposed right operand;
- non-unit positive strides already accepted by `faer_mat_ref`;
- negative strides if the existing layout contract permits them.

The output should require one allocation. Inputs must remain borrowed.

### Stage 3 — batched paths

Add complex dispatch to both batched entry points.

Allocate the complete output once and write each batch into its destination slice. Avoid the current pattern of allocating a temporary result vector per batch and copying it into the final output; if this cleanup is generalized to real scalars, benchmark real paths to prove no regression.

Test both layouts used by OMEinsum:

- column-major batch views;
- batch-major views;
- transposed operand within a batch;
- batch size one and multiple batches.

### Stage 4 — remove accidental duplication

Only after all four paths work, inspect whether the f32/f64/c32/c64 helpers can share a small generic implementation. Do not build a trait hierarchy merely to avoid four short dispatch arms. The simple version wins unless genericization clearly reduces unsafe casts and layout code.

### Stage 5 — downstream validation

In `rydbergsim-rs`:

1. temporarily pin OMEinsum to the candidate commit;
2. remove the local `contract_binary_faer` implementation and restore ordinary OMEinsum calls;
3. run the TDVP kernel reference tests;
4. run the N=21, chi=64 saturated-step benchmark;
5. rerun the partial N=51, chi=128 probe only after the kernel result justifies its cost.

This confirms that the upstream optimization survives contraction planning, environment updates, and Krylov repetition.

## Benchmark protocol

### Controlled variables

- Same host, CPU governor, Rust toolchain, commit mode, and feature set.
- Release profile.
- `D=9` and identical deterministic input values.
- Criterion sample size, warm-up, and measurement duration fixed by the benchmark.
- Run one benchmark process at a time.

### Measurements

For every case and `chi`:

- Criterion median latency and confidence interval.
- Scalar fused multiply-add count reported as throughput.
- Candidate/baseline speedup.
- Peak resident memory for the complete benchmark process when practical.
- Allocation count in focused unit tests for direct-layout helpers.

## Validation record (2026-07-15)

Measurements ran on remote host `6xa800` (2-socket Intel Xeon Platinum 8378A,
128 logical CPUs) with Rust 1.88.0. Baseline and candidate used the same release
profile, benchmark inputs, and Criterion settings. Criterion artifacts are under
`~/projects/omeinsum-rs/target/criterion/tdvp-complex-binary` on that host.

The repository-mandated `runscribe` executable was unavailable locally and on
the remote host; the similarly named PyPI package is an unrelated terminal
recorder. With the owner-requested remote workflow, runs were submitted through
`easy-ssh submit` and their full job output was captured instead. There is
therefore no runscribe run directory for these measurements.

Times below are Criterion point estimates in milliseconds; parenthesized ranges
are the default 95% confidence intervals. Speedup is baseline divided by
candidate.

| Case | chi | Scalar baseline | faer candidate | Speedup | Candidate throughput |
|---|---:|---:|---:|---:|---:|
| `h1-left-environment` | 32 | 1.0503 (1.0419-1.0651) | 0.084151 (0.084054-0.084254) | 12.48x | 7.0091 Gelem/s |
| `h1-right-environment` | 32 | 1.1197 (1.1158-1.1286) | 0.13741 (0.13709-0.13759) | 8.15x | 4.2924 Gelem/s |
| `h2-left-environment` | 32 | 1.8304 (1.8032-1.9023) | 0.15972 (0.15964-0.15988) | 11.46x | 7.3856 Gelem/s |
| `h2-right-environment` | 32 | 2.4375 (2.3791-2.5324) | 0.30246 (0.30238-0.30255) | 8.06x | 3.9001 Gelem/s |
| `h1-left-environment` | 64 | 8.0044 (7.9638-8.1163) | 0.61142 (0.61091-0.61199) | 13.09x | 7.7175 Gelem/s |
| `h1-right-environment` | 64 | 12.281 (11.996-12.757) | 0.96001 (0.95767-0.96541) | 12.79x | 4.9151 Gelem/s |
| `h2-left-environment` | 64 | 16.475 (15.980-17.505) | 1.1750 (1.1745-1.1756) | 14.02x | 8.0319 Gelem/s |
| `h2-right-environment` | 64 | 33.899 (32.550-35.217) | 1.9024 (1.8967-1.9183) | 17.82x | 4.9606 Gelem/s |
| `h1-left-environment` | 128 | 92.408 (88.993-95.673) | 4.5194 (4.4687-4.6194) | 20.45x | 8.3525 Gelem/s |
| `h1-right-environment` | 128 | 130.87 (129.61-132.46) | 5.9194 (5.9064-5.9326) | 22.11x | 6.3772 Gelem/s |
| `h2-left-environment` | 128 | 164.62 (163.87-166.52) | 8.6727 (8.6685-8.6766) | 18.98x | 8.7051 Gelem/s |
| `h2-right-environment` | 128 | 248.54 (247.66-249.35) | 11.647 (11.623-11.670) | 21.34x | 6.4822 Gelem/s |

All four chi=64 cases exceed the required 5x threshold. The existing real f32
`binary` benchmark suite was also compared against a clean `eaf29fe` worktree.
Two noisy initial outliers were rerun: `high_d_12x12_contract_6` measured 40.35
microseconds versus 40.21 microseconds (+0.34%), and
`high_d_20x20_contract_9` measured 28.51 milliseconds versus 28.83 milliseconds
(-1.1%). No repeatable real-valued regression exceeded 5%.

`make check` passed after the final change: clippy with `tropical parallel`, 158
library tests passed (11 ignored), 333 integration tests passed, and 16 doctests
passed (4 ignored).

### Acceptance criteria

Correctness is mandatory:

- `make check` passes.
- Complex fast paths agree with generic GEMM within dtype-appropriate relative and absolute tolerances.
- Existing f32/f64, tropical, repeated-label, scalar-output, and backward tests remain green.

Performance acceptance for the motivating workload:

- At least 5x lower median latency than the scalar baseline for all four `chi=64, D=9` contractions.
- No target case regresses relative to baseline.
- `chi=128` shows the same direction of improvement and completes without pathological memory growth.
- Existing real-valued binary benchmarks do not regress by more than 5% beyond measurement noise.

The 5x threshold is a minimum useful result, not a promise that it makes the full Keesling grid feasible. End-to-end TDVP timing decides that separately.

## Risks and controls

| Risk | Control |
|---|---|
| `TypeId` dispatch and `transmute` mismatch scalar storage | Mirror existing real dispatch, keep casts adjacent to checked type IDs, and test both complex widths. |
| Strided `MatRef` points outside backing storage | Reuse `layout_offset_bounds`/`faer_mat_ref`; add transposed and negative-stride regressions. |
| Complex convention accidentally conjugates an operand | Test against plain sum-product GEMM with nontrivial imaginary values; use ordinary views, never adjoints. |
| Batched offsets are wrong | Compare every batch element against independent unbatched contractions. |
| Faer parallelism oversubscribes downstream workloads | Keep `Par::Seq` in this change; treat parallel policy as a separate measured hypothesis. |
| Benchmark only measures synthetic GEMM | Validate the candidate commit in the downstream TDVP step before claiming user-visible speedup. |
| Local workaround and upstream path drift | Delete the downstream workaround once the candidate passes; maintain one implementation in OMEinsum. |

## Deliverables

- `benches/complex_tdvp.rs` with the four target contractions.
- Complex faer dispatch in all four CPU GEMM routes.
- Unit and integration regressions with concrete complex values.
- runscribe baseline and candidate runs, including the surviving alternative analysis.
- Before/after summary in the pull request.
- Downstream `rydbergsim-rs` validation result and candidate revision.

## Decision points

1. After baseline: confirm the scalar fallback is reproduced at target shapes.
2. After contiguous/layout implementation: decide whether batched cleanup belongs in the same PR.
3. After OMEinsum benchmarks: decide whether downstream N=51 probing is justified.
4. After downstream validation: merge, revise, or abandon based on measured end-to-end gain.
