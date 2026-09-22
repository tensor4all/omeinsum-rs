# omeinsum-rs

> **Deprecated (2026-09-22).** This crate is superseded by
> [tenferro-rs](https://github.com/tensor4all/tenferro-rs) and receives no new
> features or fixes. The last release, `omeinsum` 0.1.2, stays on crates.io and
> keeps working, but new work belongs in tenferro-rs — please file issues and
> pull requests there.

[![CI](https://github.com/tensor4all/omeinsum-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/tensor4all/omeinsum-rs/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/codecov/c/github/tensor4all/omeinsum-rs)](https://codecov.io/gh/tensor4all/omeinsum-rs)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://tensor4all.github.io/omeinsum-rs/)

Einstein summation for tropical and standard tensor networks in Rust. Inspired by [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl).

## Migrating to tenferro-rs

| omeinsum-rs (0.1.x) | tenferro-rs |
|---------------------|-------------|
| `einsum::<Standard<T>, _, _>`, n-ary contractions | [`tenferro-einsum`](https://crates.io/crates/tenferro-einsum) 0.6+ (CPU/CUDA) |
| `Einsum` + `optimize_greedy` / `optimize_treesa` | `tenferro-einsum`: `ContractionTree::optimize`, `ContractionOptimizerOptions` |
| `MaxPlus` / `MinPlus` / `MaxMul` and argmax backward | `tenferro-tropical` (planned); today [`ext/tropical`](https://github.com/tensor4all/tenferro-rs/tree/main/ext/tropical) in-tree as `tenferro-ext-tropical` |
| `GenericSemiring` (non-`Copy` semirings, e.g. configuration enumerators) | not ported yet |
| tropical CUDA backend | not ported yet (`tenferro-gpu` has no tropical path) |
| `omeinsum` CLI | no counterpart yet |

Standard einsum and contraction-order optimization, `tenferro-einsum`:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::Tensor;

let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?;
let b = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12])?;
let mut backend = CpuBackend::new();

let out = [&a, &b].einsum("ij,jk->ik", &mut backend)?;
```

Tropical semirings are being moved out of this repository into the future
`tenferro-tropical` crate. Until it is published, the in-tree
[`ext/tropical`](https://github.com/tensor4all/tenferro-rs/tree/main/ext/tropical)
crate covers CPU fused binary contractions with argmax, traced wrappers and AD
rules:

```rust
use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};

let out = tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
```

## Features

- **Multiple Algebras**: Standard arithmetic, MaxPlus, MinPlus, MaxMul semirings
- **Contraction Optimization**: Uses [omeco](https://github.com/GiggleLiu/omeco) for optimal contraction order
- **Backpropagation Support**: Argmax tracking for tropical gradient computation
- **Flexible Tensors**: Stride-based views with zero-copy permute/reshape

## Installation

```toml
[dependencies]
omeinsum = "0.1"
```

## Quick Start

```rust
use omeinsum::{einsum, Tensor, Cpu};
use omeinsum::algebra::{Standard, MaxPlus};

// Create tensors
let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

// Standard matrix multiplication: C[i,k] = Σ_j A[i,j] × B[j,k]
let c = einsum::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
assert_eq!(c.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);

// Tropical matrix multiplication: C[i,k] = max_j (A[i,j] + B[j,k])
let c = einsum::<MaxPlus<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
assert_eq!(c.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
```

## Documentation

📖 **[User Guide](https://tensor4all.github.io/omeinsum-rs/)** - Installation, tutorials, examples

📚 **[API Reference](https://tensor4all.github.io/omeinsum-rs/api/omeinsum/)** - Rust API documentation

## Development

```bash
make cargo-check
make check
cargo test --test main
cargo test --features tropical
```

`make check` is the canonical non-GPU verification gate. Integration suites live in `tests/suites/` and are wired through `tests/main.rs`.

## Algebras

| Type | ⊕ | ⊗ | Use Case |
|------|---|---|----------|
| `Standard<T>` | + | × | Normal arithmetic |
| `MaxPlus<T>` | max | + | Longest path, Viterbi |
| `MinPlus<T>` | min | + | Shortest path |
| `MaxMul<T>` | max | × | Max probability |

## CLI Tool

Install and use the `omeinsum` CLI for contraction optimization, execution, and autodiff without writing Rust code:

```bash
make cli                    # install to ~/.cargo/bin

omeinsum optimize "ij,jk->ik" --sizes "i=2,j=3,k=2" -o topo.json
omeinsum contract tensors.json -t topo.json
omeinsum autodiff tensors.json --expr "ii->"
```

See the [CLI documentation](https://tensor4all.github.io/omeinsum-rs/cli.html) for JSON formats, parenthesized expressions, and a full walkthrough.

## Contraction Optimization

```rust
use omeinsum::Einsum;
use std::collections::HashMap;

// A[i,j] × B[j,k] × C[k,l] → D[i,l]
let sizes: HashMap<usize, usize> = [(0, 10), (1, 20), (2, 30), (3, 40)].into();

let mut ein = Einsum::new(
    vec![vec![0, 1], vec![1, 2], vec![2, 3]],
    vec![0, 3],
    sizes,
);

// Optimize contraction order (critical for performance!)
ein.optimize_greedy();  // Fast O(n²) algorithm
// ein.optimize_treesa();  // Better for large networks

let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b, &c]);
```

## Related Projects

- [tropical-gemm](https://github.com/TensorBFS/tropical-gemm) - High-performance tropical GEMM
- [omeco](https://github.com/GiggleLiu/omeco) - Contraction order optimization
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) - Julia einsum library

## License

MIT
