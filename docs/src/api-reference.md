# API Reference

Full API documentation is available at:

**[https://tensor4all.github.io/omeinsum-rs/api/omeinsum/](https://tensor4all.github.io/omeinsum-rs/api/omeinsum/)**

## Quick Reference

### Main Types

| Type | Description |
|------|-------------|
| `Tensor<T, B>` | N-dimensional tensor with backend B |
| `Einsum<L>` | Einsum specification and executor |
| `Cpu` | CPU backend |

### Algebra Types

| Type | Addition | Multiplication |
|------|----------|----------------|
| `Standard<T>` | + | × |
| `MaxPlus<T>` | max | + |
| `MinPlus<T>` | min | + |
| `MaxMul<T>` | max | × |

### Key Functions

```rust
// Quick einsum
fn einsum<A, T, B>(tensors: &[&Tensor<T, B>], ixs: &[&[usize]], iy: &[usize]) -> Tensor<T, B>

// Einsum with gradient support
fn einsum_with_grad<A, T, B>(...) -> (Tensor<T, B>, EinsumGradient<T, B>)
```

### Tensor Methods

```rust
impl<T, B> Tensor<T, B> {
    // Creation
    fn from_data(data: &[T], shape: &[usize]) -> Self
    fn zeros(shape: &[usize]) -> Self
    fn ones(shape: &[usize]) -> Self

    // Properties
    fn shape(&self) -> &[usize]
    fn strides(&self) -> &[usize]
    fn ndim(&self) -> usize
    fn numel(&self) -> usize
    fn is_contiguous(&self) -> bool

    // Transformations
    fn permute(&self, order: &[usize]) -> Self
    fn reshape(&self, new_shape: &[usize]) -> Self
    fn contiguous(&self) -> Self

    // Operations
    fn gemm<A: Algebra>(&self, other: &Self) -> Self
    fn contract_binary<A>(&self, other: &Self, ia: &[usize], ib: &[usize], iy: &[usize]) -> Self

    // Data
    fn to_vec(&self) -> Vec<T>
}
```

### Einsum Methods

```rust
impl<L> Einsum<L> {
    fn new(ixs: Vec<Vec<L>>, iy: Vec<L>, size_dict: HashMap<L, usize>) -> Self
    fn code(&self) -> EinCode<L>
    fn optimize_greedy(&mut self) -> &mut Self
    fn optimize_treesa(&mut self) -> &mut Self
    fn is_optimized(&self) -> bool
    fn contraction_tree(&self) -> Option<&NestedEinsum<L>>
}

impl Einsum<usize> {
    fn execute<A, T, B>(&self, tensors: &[&Tensor<T, B>]) -> Tensor<T, B>
    fn execute_with_argmax<A, T, B>(&self, tensors: &[&Tensor<T, B>])
        -> (Tensor<T, B>, Vec<Tensor<u32, B>>)
}
```

## Building Documentation Locally

```bash
make docs-build   # Rust API docs
make docs-serve   # Serve at localhost:8000
```
