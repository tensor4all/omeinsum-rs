# Performance Guide

Tips for getting the best performance from omeinsum-rs.

## Contraction Order

The most important optimization is contraction order:

```rust
// Always optimize for networks with 3+ tensors
let mut ein = Einsum::new(ixs, iy, sizes);
ein.optimize_greedy();  // or optimize_treesa() for large networks
```

Bad contraction order can be exponentially slower.

## Memory Layout

### Keep Tensors Contiguous

Non-contiguous tensors require copies before GEMM:

```rust
// After permute, tensor may be non-contiguous
let t_permuted = t.permute(&[1, 0]);

// Make contiguous if you'll use it multiple times
let t_contig = t_permuted.contiguous();
```

### Avoid Unnecessary Copies

```rust
// Good: zero-copy view
let view = t.permute(&[1, 0]);

// Avoid: unnecessary explicit copy
let bad = t.permute(&[1, 0]).contiguous();  // Only if needed
```

## Parallelization

Enable the optional `parallel` feature for workloads that benefit from Rayon:

```toml
[dependencies]
omeinsum = { version = "0.1", features = ["parallel"] }
```

Omit the feature for single-threaded workloads:

```toml
[dependencies]
omeinsum = "0.1"
```

## Data Types

### Use f32 When Possible

`f32` is typically faster than `f64` due to:
- Smaller memory bandwidth
- Better SIMD utilization

```rust
// Prefer f32
let t = Tensor::<f32, Cpu>::from_data(&data, &shape);

// Use f64 only when precision is critical
let t = Tensor::<f64, Cpu>::from_data(&data, &shape);
```

## Benchmarking

Use the built-in CPU benchmark example in release mode:

```bash
make bench-cpu-contract
```

Tune the benchmark mix or problem size with make variables:

```bash
make bench-cpu-contract BENCH_SCENARIO=rhs-transpose-view BENCH_DIM=192 BENCH_ITERATIONS=60
make bench-cpu-contract BENCH_SCENARIO=column-major-batched BENCH_BATCH=32 BENCH_DIM=128
make bench-cpu-contract BENCH_SCENARIO=batch-major-batched BENCH_BATCH=32 BENCH_DIM=128
make bench-cpu-contract BENCH_SCENARIO=root-output-permutation BENCH_DIM=160
```

To compare two revisions, run the same benchmark command in both checkouts and compare the reported `avg_ms` values.

Profile the benchmark example with:

```bash
cargo build --release --example cpu_contract_bench
perf record ./target/release/examples/cpu_contract_bench --scenario all
perf report
```

## Common Pitfalls

### 1. Forgetting to Optimize

```rust
// Bad: no optimization
let ein = Einsum::new(ixs, iy, sizes);
let result = ein.execute::<A, T, B>(&tensors);

// Good: with optimization
let mut ein = Einsum::new(ixs, iy, sizes);
ein.optimize_greedy();
let result = ein.execute::<A, T, B>(&tensors);
```

### 2. Redundant Contiguous Calls

```rust
// Bad: unnecessary copy
let c = a.contiguous().gemm::<Standard<f32>>(&b.contiguous());

// Good: gemm handles this internally
let c = a.gemm::<Standard<f32>>(&b);
```

### 3. Debug Mode

Debug builds are ~10-50x slower:

```bash
# Bad: debug mode
cargo run --example cpu_contract_bench -- --scenario all

# Good: release mode
cargo run --release --example cpu_contract_bench -- --scenario all
```

## Future Optimizations

Planned performance improvements:
- CUDA backend for GPU acceleration
- Optimized tropical-gemm kernel integration
- Cache-aware blocking
