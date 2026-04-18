# Benchmark Standardization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the PR's ad hoc Rust and Julia benchmark runners with a `yao-rs`-style benchmark pipeline that uses Criterion on the Rust side, a repo-local Julia environment, compact JSON outputs, and a reproducible Rust-vs-Julia comparison for both binary and network workloads.

**Architecture:** Rust benchmark execution will move into explicit Criterion bench targets under `benches/`, with small shared support modules for scenario definitions and network loading. Julia benchmarking will live under `benchmarks/julia/` and emit compact timing JSON under `benchmarks/data/`. A Python comparison script will join Julia timing files with Criterion `estimates.json` files and report slowdown versus Julia. Existing manual benchmark examples can be removed or reduced once the structured pipeline is in place.

**Tech Stack:** Rust 2021, Criterion, serde/serde_json, Python 3, Julia with `BenchmarkTools` and `OMEinsum`.

---

## Chunk 1: Rust Benchmark Infrastructure

### Task 1: Wire Criterion and benchmark entry points

**Files:**
- Modify: `Cargo.toml`
- Modify: `Makefile`

- [ ] **Step 1: Confirm the current Rust benchmark entry points are missing**

Run: `cargo bench --bench binary --no-run`
Expected: FAIL with a missing bench target error.

- [ ] **Step 2: Add Criterion and explicit bench target definitions**

Update `Cargo.toml` to:

- add `criterion = { version = "0.5", features = ["html_reports"] }` under `[dev-dependencies]`
- add explicit `[[bench]]` entries with `harness = false` for:
  - `binary`
  - `network`

Expected `Cargo.toml` fragment:

```toml
[dev-dependencies]
rand = "0.9"
approx = "0.5"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "binary"
harness = false

[[bench]]
name = "network"
harness = false
```

- [ ] **Step 3: Add Makefile targets matching the `yao-rs` style**

Update `Makefile` help text and targets to include:

- `bench`
- `bench-binary`
- `bench-network`
- `bench-julia`
- `bench-compare`

The final targets should follow this shape:

```make
bench: bench-binary bench-network
	@echo "All Rust benchmarks complete. Results in target/criterion/"

bench-binary:
	cargo bench --bench binary

bench-network:
	cargo bench --bench network

bench-julia:
	cd benchmarks/julia && julia --project=. -e 'using Pkg; Pkg.instantiate()' && julia --project=. generate_timings.jl

bench-compare:
	python3 benchmarks/compare.py
```

- [ ] **Step 4: Run the compile-only benchmark gate again**

Run: `cargo bench --bench binary --no-run`
Expected: FAIL again, but now because `benches/binary.rs` does not exist yet.

- [ ] **Step 5: Commit the wiring**

```bash
git add Cargo.toml Makefile
git commit -m "build(bench): add criterion benchmark targets"
```

### Task 2: Create the Rust binary Criterion benchmark

**Files:**
- Create: `benches/binary.rs`
- Create: `benches/support/mod.rs`
- Create: `benches/support/binary.rs`
- Delete: `examples/bench_binary.rs`

- [ ] **Step 1: Write the shared binary scenario table first**

Create `benches/support/binary.rs` with:

- a `BinaryScenario` struct
- a `binary_scenarios() -> &'static [BinaryScenario]` function
- helpers to build deterministic tensor data, index arrays, and scenario names

Use the current PR scenarios exactly:

```rust
BinaryScenario::new("matmul_10x10", 10, 10, 5, 0)
BinaryScenario::new("batched_matmul_8x8_batch_4", 8, 8, 4, 4)
BinaryScenario::new("high_d_12x12_contract_6", 12, 12, 6, 0)
BinaryScenario::new("high_d_15x15_contract_7", 15, 15, 7, 0)
BinaryScenario::new("high_d_18x18_contract_8", 18, 18, 8, 0)
BinaryScenario::new("high_d_20x20_contract_9", 20, 20, 9, 0)
BinaryScenario::new("high_d_12x12_contract_4_batch_4", 12, 12, 4, 4)
BinaryScenario::new("high_d_15x15_contract_5_batch_5", 15, 15, 5, 5)
```

- [ ] **Step 2: Add the Criterion bench harness**

Create `benches/binary.rs` using the `yao-rs` pattern:

```rust
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

#[path = "support/mod.rs"]
mod support;

fn bench_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary");
    for scenario in support::binary::binary_scenarios() {
        let prepared = support::binary::prepare_binary_case(scenario);
        group.bench_with_input(BenchmarkId::new("einsum", scenario.name), &prepared, |b, prepared| {
            b.iter(|| support::binary::run_binary_case(black_box(prepared)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_binary);
criterion_main!(benches);
```

- [ ] **Step 3: Verify the new bench compiles**

Run: `cargo bench --bench binary --no-run`
Expected: PASS

- [ ] **Step 4: Run the binary bench once**

Run: `cargo bench --bench binary`
Expected: PASS and create `target/criterion/binary/...`

- [ ] **Step 5: Remove the obsolete ad hoc Rust example**

Delete `examples/bench_binary.rs` after the Criterion bench is working so there is only one supported Rust path for this workload.

- [ ] **Step 6: Commit the binary bench**

```bash
git add benches/binary.rs benches/support/mod.rs benches/support/binary.rs examples/bench_binary.rs
git commit -m "feat(bench): add criterion binary benchmark"
```

### Task 3: Create the Rust network Criterion benchmark and compact JSON path

**Files:**
- Create: `benches/network.rs`
- Create: `benches/support/network.rs`
- Modify: `benches/support/mod.rs`
- Modify: `benches/profile_network.rs`
- Modify: `benches/network_small.json`
- Modify: `benches/network_medium.json`
- Modify: `benches/network_large.json`
- Modify: `benches/network_3reg_150.json`
- Modify: `benches/network_benchmark.json`

- [ ] **Step 1: Move reusable network logic behind support helpers**

Create `benches/support/network.rs` with:

- `NetworkScenario` definitions for `small`, `medium`, `large`, and `3reg_150`
- loader functions for committed JSON benchmark files
- helpers to reconstruct `NestedEinsum`, build deterministic tensors, and execute the contraction

Keep generation/loading separate from the timed benchmark loop.

- [ ] **Step 2: Replace the ad hoc timed runner with a Criterion bench**

Create `benches/network.rs` using the same Criterion pattern as `yao-rs`:

```rust
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

#[path = "support/mod.rs"]
mod support;

fn bench_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("network");
    for scenario in support::network::network_scenarios() {
        let prepared = support::network::prepare_network_case(scenario);
        group.bench_with_input(BenchmarkId::new("einsum", scenario.name), &prepared, |b, prepared| {
            b.iter(|| support::network::run_network_case(black_box(prepared)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_network);
criterion_main!(benches);
```

- [ ] **Step 3: Fix compact JSON serialization at the source**

Update `benches/profile_network.rs` so any generated JSON uses:

```rust
serde_json::to_writer(writer, &network)
```

not `to_writer_pretty`.

- [ ] **Step 4: Regenerate or rewrite the committed network JSON files to compact form**

After switching the writer, regenerate or compact these files so the repository no longer stores line-broken benchmark JSON:

- `benches/network_small.json`
- `benches/network_medium.json`
- `benches/network_large.json`
- `benches/network_3reg_150.json`
- `benches/network_benchmark.json`

- [ ] **Step 5: Verify the compact JSON requirement explicitly**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
for path in [
    "benches/network_small.json",
    "benches/network_medium.json",
    "benches/network_large.json",
    "benches/network_3reg_150.json",
    "benches/network_benchmark.json",
]:
    text = Path(path).read_text()
    assert "\n" not in text, f"{path} still contains line breaks"
print("compact json ok")
PY
```

Expected: PASS

- [ ] **Step 6: Verify the network bench compiles and runs**

Run: `cargo bench --bench network --no-run`
Expected: PASS

Run: `cargo bench --bench network`
Expected: PASS and create `target/criterion/network/...`

- [ ] **Step 7: Commit the network bench**

```bash
git add benches/network.rs benches/support/network.rs benches/support/mod.rs benches/profile_network.rs benches/network_small.json benches/network_medium.json benches/network_large.json benches/network_3reg_150.json benches/network_benchmark.json
git commit -m "feat(bench): add criterion network benchmark"
```

## Chunk 2: Julia Environment And Comparison Pipeline

### Task 4: Add the repo-local Julia benchmark environment

**Files:**
- Create: `benchmarks/julia/Project.toml`
- Create: `benchmarks/julia/generate_timings.jl`
- Create: `benchmarks/data/.gitkeep`
- Delete: `examples/bench_binary.jl`

- [ ] **Step 1: Create the dedicated Julia project**

Create `benchmarks/julia/Project.toml` with explicit dependencies, following the `yao-rs` pattern:

```toml
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
OMEinsum = "ebe7aa44-baf0-506c-a96f-8464559b3922"

[compat]
BenchmarkTools = "1"
JSON = "0.21"
OMEinsum = "0.9"
```

Use a repo-local package setup or explicit package add in `generate_timings.jl`; do not reference `~/.julia/dev/OMEinsum`.

- [ ] **Step 2: Write the Julia binary timing generator**

In `benchmarks/julia/generate_timings.jl`, add:

- the same binary scenario table as the Rust bench
- deterministic tensor construction
- `@benchmark` / `@belapsed` timing in nanoseconds
- compact JSON writing to `benchmarks/data/binary_timings.json`

Expected output shape:

```json
{"matmul_10x10":12345,"batched_matmul_8x8_batch_4":23456}
```

- [ ] **Step 3: Extend the Julia script for network timings**

Add the same fixed network scenarios as Rust:

- `small`
- `medium`
- `large`
- `3reg_150`

Write compact JSON to `benchmarks/data/network_timings.json`.

- [ ] **Step 4: Remove the obsolete Julia example runner**

Delete `examples/bench_binary.jl` once `benchmarks/julia/generate_timings.jl` covers the supported Julia benchmark workflow.

- [ ] **Step 5: Instantiate and run the Julia benchmark script**

Run: `make bench-julia`
Expected: PASS and create:

- `benchmarks/data/binary_timings.json`
- `benchmarks/data/network_timings.json`

- [ ] **Step 6: Verify the Julia timing JSON is compact**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
for path in [
    "benchmarks/data/binary_timings.json",
    "benchmarks/data/network_timings.json",
]:
    text = Path(path).read_text()
    assert "\n" not in text, f"{path} still contains line breaks"
print("compact julia timing json ok")
PY
```

Expected: PASS

- [ ] **Step 7: Commit the Julia environment**

```bash
git add benchmarks/julia/Project.toml benchmarks/julia/generate_timings.jl benchmarks/data/.gitkeep benchmarks/data/binary_timings.json benchmarks/data/network_timings.json examples/bench_binary.jl
git commit -m "feat(bench): add julia benchmark environment"
```

### Task 5: Add the Rust-vs-Julia comparison script

**Files:**
- Create: `benchmarks/compare.py`
- Modify: `Makefile`

- [ ] **Step 1: Write the comparison script with explicit loaders**

Create `benchmarks/compare.py` that:

- loads `benchmarks/data/binary_timings.json`
- loads `benchmarks/data/network_timings.json`
- loads Criterion median point estimates from `target/criterion/`
- prints one table covering both benchmark families

Use a `yao-rs`-style loader:

```python
def load_criterion_estimate(group: str, name: str, param: str | None = None):
    ...
```

Target table columns:

```text
| Benchmark | Scenario | Julia (ns) | Rust (ns) | Julia/Rust | Drop vs Julia |
```

- [ ] **Step 2: Encode the slowdown calculation clearly**

Use:

- `ratio = julia_ns / rust_ns`
- `drop_vs_julia = max((rust_ns - julia_ns) / julia_ns, 0.0)`

Format slowdown as a percentage when Julia is faster; otherwise show `0.0%`.

- [ ] **Step 3: Run the script before all data exists to verify error handling**

If the timing files or Criterion estimates are absent in a clean tree, the script should print a clear message telling the user which `make` target to run next.

Run: `python3 benchmarks/compare.py`
Expected: either PASS with data, or a clear actionable error if data is missing.

- [ ] **Step 4: Run the full comparison after both Rust and Julia data exist**

Run: `make bench-compare`
Expected: PASS and print rows for:

- all binary scenarios
- all network scenarios

- [ ] **Step 5: Commit the comparison tooling**

```bash
git add benchmarks/compare.py Makefile
git commit -m "feat(bench): add rust vs julia comparison report"
```

## Chunk 3: Final Verification And Cleanup

### Task 6: Remove stale benchmark paths and verify the end-to-end workflow

**Files:**
- Review: `Cargo.toml`
- Review: `Makefile`
- Review: `benches/binary.rs`
- Review: `benches/network.rs`
- Review: `benches/support/mod.rs`
- Review: `benches/support/binary.rs`
- Review: `benches/support/network.rs`
- Review: `benches/profile_network.rs`
- Review: `benchmarks/compare.py`
- Review: `benchmarks/julia/Project.toml`
- Review: `benchmarks/julia/generate_timings.jl`
- Review: `benchmarks/data/*.json`
- Review: deletions under `examples/`

- [ ] **Step 1: Run the repo gate**

Run: `make check`
Expected: PASS

- [ ] **Step 2: Run the Rust benches through the supported entry points**

Run: `make bench`
Expected: PASS and populate `target/criterion/`

- [ ] **Step 3: Run the Julia benchmark pipeline**

Run: `make bench-julia`
Expected: PASS and refresh compact timing JSON in `benchmarks/data/`

- [ ] **Step 4: Run the comparison report and inspect the slowdown output**

Run: `make bench-compare`
Expected: PASS and print enough data to answer whether Rust is slower than Julia for any binary or network scenario.

- [ ] **Step 5: Record the observed performance result in the PR summary**

Capture:

- which scenarios show Rust slower than Julia
- which scenarios show Rust faster than Julia
- any missing data or benchmark instability

Do not claim parity if the comparison table shows a slowdown.

- [ ] **Step 6: Commit the final cleanup**

```bash
git add Cargo.toml Makefile benches benchmarks examples
git commit -m "chore(bench): standardize rust and julia benchmark workflow"
```
