# Benchmark Standardization Design

Date: 2026-04-11
Branch: `pr-29`

## Goal

Bring the PR's benchmark work up to the `~/rcode/yao-rs` standard by:

- using proper Rust bench targets instead of ad hoc example timers for comparison workloads
- introducing an isolated Julia benchmark environment under the repository
- producing compact machine-readable JSON results
- adding a reproducible Rust-vs-Julia comparison step that reports any Rust slowdown relative to Julia

The scope covers both benchmark families currently introduced by PR `#29`:

- binary contraction benchmarks
- tensor-network contraction benchmarks

## Non-Goals

- rewriting the library's contraction engine for performance in this change
- adding broad performance assertions to CI
- benchmarking CUDA paths
- preserving the current example-based benchmark UX if it conflicts with a cleaner benchmark layout

## Current Problems

1. `examples/bench_binary.rs` and `examples/bench_binary.jl` are manual runners, not structured benchmarks.
2. `benches/profile_network.rs` mixes data generation, JSON serialization, and timing in one ad hoc executable-style benchmark.
3. Julia benchmarking depends on an external environment (`~/.julia/dev/OMEinsum`) instead of a repo-local project.
4. Generated JSON currently uses pretty printing in at least one path, which violates the requirement to avoid line breaks.
5. There is no stable comparison pipeline that reads Rust and Julia results and reports the performance delta.

## Proposed Layout

Add a benchmark structure modeled after `yao-rs`:

- `benches/binary.rs`
  Rust Criterion benchmark for binary contraction scenarios.
- `benches/network.rs`
  Rust Criterion benchmark for fixed network benchmark scenarios.
- `benchmarks/compare.py`
  Reads Julia JSON results plus Criterion estimates and prints a comparison table.
- `benchmarks/data/`
  Stores Julia-generated benchmark timing JSON.
- `benchmarks/julia/Project.toml`
  Repo-local Julia benchmark environment.
- `benchmarks/julia/generate_timings.jl`
  Runs Julia benchmarks and writes compact JSON timing files.

Existing manual example/profiling files can be removed or reduced to thin exploratory helpers if they still add value, but the benchmark standard should be defined by the Criterion + Julia pipeline above.

## Benchmark Model

### Rust

Rust benchmarks will use Criterion so results land under `target/criterion/` and can be consumed the same way as in `yao-rs`.

#### Binary Benchmarks

Each current binary case becomes a named Criterion scenario, for example:

- `matmul_10x10`
- `batched_matmul_8x8_batch_4`
- `high_d_12x12_contract_6`
- `high_d_15x15_contract_7`
- `high_d_18x18_contract_8`
- `high_d_20x20_contract_9`
- `high_d_12x12_contract_4_batch_4`
- `high_d_15x15_contract_5_batch_5`

Each benchmark will:

- build deterministic input tensors once per scenario
- warm up through Criterion rather than manual loops
- benchmark only the contraction call
- use stable scenario names so the comparison script can join Rust and Julia results

#### Network Benchmarks

Network benchmarks will target fixed datasets instead of generating fresh graphs during timing:

- `small`
- `medium`
- `large`
- `3reg_150`

The benchmark path will:

- load compact JSON network data from the committed files
- reconstruct the contraction tree
- build deterministic tensors once
- benchmark only the execution of the contraction

Graph generation remains a data-preparation concern, not part of the timed workload.

### Julia

Julia benchmarks will live under `benchmarks/julia/` with their own `Project.toml`, following the same pattern as `yao-rs`.

The Julia script will:

- instantiate repo-local dependencies
- benchmark the same named scenarios as Rust
- write compact JSON timing results to `benchmarks/data/`

The script must not depend on `~/.julia/dev/OMEinsum` or any manually prepared global environment.

## Shared Scenario Definitions

The benchmark names and workload definitions must stay aligned between Rust and Julia.

The simplest acceptable approach is to duplicate the small scenario tables in Rust and Julia, as long as:

- names match exactly
- tensor shapes and contraction semantics match exactly
- network file names match exactly

If duplication starts drifting, a follow-up can extract a shared scenario manifest. That is not required for this change.

## JSON Output

All generated JSON in the benchmark pipeline must be compact.

Requirements:

- use `serde_json::to_writer`, not `to_writer_pretty`, on the Rust side
- Julia should use compact JSON emission as well
- scenario names should be keys or stable string fields
- timings should be stored in nanoseconds for direct comparison with Criterion estimates

This applies to:

- generated network benchmark data from Rust utilities
- Julia timing outputs
- any auxiliary benchmark metadata produced by the new workflow

## Comparison Output

`benchmarks/compare.py` will read:

- Julia timing JSON from `benchmarks/data/`
- Rust Criterion median point estimates from `target/criterion/.../estimates.json`

It will print a table with one row per scenario:

`benchmark | scenario | julia_ns | rust_ns | julia_over_rust | drop_vs_julia`

Interpretation:

- `julia_over_rust > 1` means Rust is faster
- `julia_over_rust < 1` means Rust is slower
- `drop_vs_julia` reports Rust slowdown relative to Julia when Rust is slower, otherwise `0%` or an equivalent no-drop value

Missing data on either side should show as `N/A`, not disappear silently.

## Makefile Changes

Add benchmark targets analogous to `yao-rs`:

- `bench`
- `bench-binary`
- `bench-network`
- `bench-julia`
- `bench-compare`

Expected behavior:

- `make bench` runs the Rust Criterion benches
- `make bench-julia` instantiates `benchmarks/julia/` and generates Julia timing JSON
- `make bench-compare` prints the Rust-vs-Julia comparison

## Data Flow

1. Rust benchmark code runs with `cargo bench`.
2. Criterion writes Rust timing estimates to `target/criterion/`.
3. Julia benchmark script runs in `benchmarks/julia/`.
4. Julia writes compact timing JSON to `benchmarks/data/`.
5. Comparison script reads both sources and prints a joined report.

## Error Handling

- Missing Julia timing files should produce a clear error that tells the user to run `make bench-julia`.
- Missing Criterion estimates should produce a clear error or `N/A` rows that tell the user to run the relevant Rust bench target.
- Mismatched scenario names should surface as missing rows in the comparison rather than being silently normalized.
- Network benchmark loading errors should fail early with the exact missing file path.

## Verification Plan

Minimum verification for the implementation:

1. `make check`
2. `cargo bench --bench binary`
3. `cargo bench --bench network`
4. `make bench-julia`
5. `make bench-compare`

The comparison step must produce enough data to answer whether Rust shows a performance drop versus Julia for both benchmark families.

## Risks

### Benchmark runtime

Criterion plus Julia benchmarking may be slow for the largest scenarios. If runtime becomes excessive, tune sample sizes or Criterion configuration without changing the scenario set or output format.

### Scenario mismatch

The biggest correctness risk is accidental drift between Rust and Julia scenario definitions. Stable names and explicit tables reduce that risk.

### Network benchmark semantics

The network benchmark must measure execution only, not random graph generation or topology search. Keeping timing inputs fixed avoids invalid comparisons.

## Recommendation

Implement the benchmark pipeline in the `yao-rs` style with Criterion for Rust, a dedicated Julia environment under `benchmarks/julia/`, compact JSON outputs everywhere, and a comparison script that reports Rust-vs-Julia slowdown for both binary and network workloads.
