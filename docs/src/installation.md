# Installation

## From crates.io

Add to your `Cargo.toml`:

```toml
[dependencies]
omeinsum = "0.1"
```

## From Git

For the latest development version:

```toml
[dependencies]
omeinsum = { git = "https://github.com/tensor4all/omeinsum-rs" }
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `parallel` | Yes | Enable parallel execution with rayon |
| `tropical-kernels` | Yes | Use optimized tropical-gemm kernels |
| `cuda` | No | Enable CUDA GPU support |

### Minimal Build

For a minimal build without optional dependencies:

```toml
[dependencies]
omeinsum = { version = "0.1", default-features = false }
```

### With CUDA

```toml
[dependencies]
omeinsum = { version = "0.1", features = ["cuda"] }
```

## Verification

Verify the installation:

```rust
use omeinsum::{Tensor, Cpu};

fn main() {
    let t = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0], &[3]);
    println!("omeinsum installed successfully!");
    println!("Tensor shape: {:?}", t.shape());
}
```
