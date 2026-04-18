# Permutation Correctness Tests - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 19 non-trivial correctness tests for tensor permutation using a naive reference implementation.

**Architecture:** Single test file with reference implementation, helper function, and categorized test modules.

**Tech Stack:** Rust, omeinsum crate

---

## Task 1: Create test file with reference implementation

**Files:**
- Create: `tests/permute_correctness.rs`

**Step 1: Write the reference implementation and helper**

```rust
//! Correctness tests for tensor permutation.
//!
//! Uses a naive reference implementation to verify the optimized
//! permutation code produces correct results for realistic einsum shapes.

use omeinsum::{Cpu, Tensor};

/// Obviously-correct reference implementation for verification.
/// O(n) with explicit index computation - slow but verifiably correct.
fn naive_permute<T: Copy + Default>(data: &[T], shape: &[usize], perm: &[usize]) -> Vec<T> {
    let ndim = shape.len();
    if ndim == 0 {
        return data.to_vec();
    }

    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let numel: usize = shape.iter().product();

    // Compute strides for input (column-major)
    let mut in_strides = vec![1usize; ndim];
    for i in 1..ndim {
        in_strides[i] = in_strides[i - 1] * shape[i - 1];
    }

    // Compute strides for output (column-major)
    let mut out_strides = vec![1usize; ndim];
    for i in 1..ndim {
        out_strides[i] = out_strides[i - 1] * new_shape[i - 1];
    }

    let mut result = vec![T::default(); numel];

    for out_idx in 0..numel {
        // Convert linear index to output coordinates
        let mut out_coords = vec![0usize; ndim];
        let mut remaining = out_idx;
        for d in (0..ndim).rev() {
            out_coords[d] = remaining / out_strides[d];
            remaining %= out_strides[d];
        }

        // Map output coords to input coords via inverse permutation
        let mut in_idx = 0usize;
        for d in 0..ndim {
            let in_dim = perm[d];
            in_idx += out_coords[d] * in_strides[in_dim];
        }

        result[out_idx] = data[in_idx];
    }

    result
}

/// Run permutation through both implementations and compare.
fn verify_permute<T: Copy + Default + PartialEq + std::fmt::Debug>(
    data: &[T],
    shape: &[usize],
    perm: &[usize],
) {
    let expected = naive_permute(data, shape, perm);
    let tensor = Tensor::<T, Cpu>::from_data(data, shape);
    let actual = tensor.permute(perm).contiguous().to_vec();
    assert_eq!(actual, expected, "shape={:?}, perm={:?}", shape, perm);
}

/// Generate sequential test data.
fn seq_data(n: usize) -> Vec<i32> {
    (0..n as i32).collect()
}

fn seq_data_f64(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64).collect()
}
```

**Step 2: Verify file compiles**

Run: `cargo test --test permute_correctness --no-run`
Expected: Compiles without errors

**Step 3: Commit**

```bash
git add tests/permute_correctness.rs
git commit -m "test(permute): add reference implementation and helpers"
```

---

## Task 2: Add matrix tests (2D)

**Files:**
- Modify: `tests/permute_correctness.rs`

**Step 1: Add matrix test module**

```rust
mod matrix_tests {
    use super::*;

    #[test]
    fn test_transpose_100x50() {
        let shape = [100, 50];
        let data = seq_data(100 * 50);
        verify_permute(&data, &shape, &[1, 0]);
    }

    #[test]
    fn test_transpose_square_64() {
        let shape = [64, 64];
        let data = seq_data(64 * 64);
        verify_permute(&data, &shape, &[1, 0]);
    }

    #[test]
    fn test_transpose_tall_1000x3() {
        let shape = [1000, 3];
        let data = seq_data(1000 * 3);
        verify_permute(&data, &shape, &[1, 0]);
    }

    #[test]
    fn test_transpose_wide_3x1000() {
        let shape = [3, 1000];
        let data = seq_data(3 * 1000);
        verify_permute(&data, &shape, &[1, 0]);
    }
}
```

**Step 2: Run tests**

Run: `cargo test --test permute_correctness matrix_tests`
Expected: 4 tests pass

**Step 3: Commit**

```bash
git add tests/permute_correctness.rs
git commit -m "test(permute): add 2D matrix transpose tests"
```

---

## Task 3: Add batched tests (3D)

**Files:**
- Modify: `tests/permute_correctness.rs`

**Step 1: Add batched test module**

```rust
mod batched_tests {
    use super::*;

    #[test]
    fn test_batch_to_front() {
        let shape = [32, 64, 128];
        let data = seq_data(32 * 64 * 128);
        verify_permute(&data, &shape, &[2, 0, 1]);
    }

    #[test]
    fn test_batch_to_back() {
        let shape = [64, 128, 32];
        let data = seq_data(64 * 128 * 32);
        verify_permute(&data, &shape, &[1, 2, 0]);
    }

    #[test]
    fn test_swap_inner_dims() {
        let shape = [32, 64, 128];
        let data = seq_data(32 * 64 * 128);
        verify_permute(&data, &shape, &[0, 2, 1]);
    }

    #[test]
    fn test_identity_3d() {
        let shape = [32, 64, 128];
        let data = seq_data(32 * 64 * 128);
        verify_permute(&data, &shape, &[0, 1, 2]);
    }
}
```

**Step 2: Run tests**

Run: `cargo test --test permute_correctness batched_tests`
Expected: 4 tests pass

**Step 3: Commit**

```bash
git add tests/permute_correctness.rs
git commit -m "test(permute): add 3D batched operation tests"
```

---

## Task 4: Add tensor network tests (4-8D)

**Files:**
- Modify: `tests/permute_correctness.rs`

**Step 1: Add tensor network test module**

```rust
mod tensor_network_tests {
    use super::*;

    #[test]
    fn test_peps_contraction_4d() {
        let shape = [2, 3, 2, 3];
        let data = seq_data(2 * 3 * 2 * 3);
        verify_permute(&data, &shape, &[0, 2, 1, 3]);
    }

    #[test]
    fn test_mps_bond_reversal() {
        let shape = [4, 8, 4];
        let data = seq_data(4 * 8 * 4);
        verify_permute(&data, &shape, &[2, 1, 0]);
    }

    #[test]
    fn test_6leg_full_reverse() {
        let shape = [2, 3, 4, 5, 6, 7];
        let data = seq_data(2 * 3 * 4 * 5 * 6 * 7);
        verify_permute(&data, &shape, &[5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_5d_partial_reorder() {
        let shape = [2, 3, 4, 5, 6];
        let data = seq_data(2 * 3 * 4 * 5 * 6);
        verify_permute(&data, &shape, &[0, 1, 4, 3, 2]);
    }
}
```

**Step 2: Run tests**

Run: `cargo test --test permute_correctness tensor_network_tests`
Expected: 4 tests pass

**Step 3: Commit**

```bash
git add tests/permute_correctness.rs
git commit -m "test(permute): add 4-8D tensor network tests"
```

---

## Task 5: Add high-rank tests (10-15D)

**Files:**
- Modify: `tests/permute_correctness.rs`

**Step 1: Add high-rank test module**

```rust
mod high_rank_tests {
    use super::*;

    #[test]
    fn test_rank10_cyclic_shift() {
        let shape = [2; 10];
        let data = seq_data(1024); // 2^10
        let perm = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0];
        verify_permute(&data, &shape, &perm);
    }

    #[test]
    fn test_rank12_swap_halves() {
        let shape = [2; 12];
        let data = seq_data(4096); // 2^12
        let perm = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5];
        verify_permute(&data, &shape, &perm);
    }

    #[test]
    fn test_rank15_reverse() {
        let shape = [2; 15];
        let data = seq_data(32768); // 2^15
        let perm = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        verify_permute(&data, &shape, &perm);
    }
}
```

**Step 2: Run tests**

Run: `cargo test --test permute_correctness high_rank_tests`
Expected: 3 tests pass

**Step 3: Commit**

```bash
git add tests/permute_correctness.rs
git commit -m "test(permute): add 10-15D high-rank tests"
```

---

## Task 6: Add data type tests

**Files:**
- Modify: `tests/permute_correctness.rs`

**Step 1: Add data type test module**

```rust
mod dtype_tests {
    use super::*;

    #[test]
    fn test_f32_permute() {
        let shape = [10, 20, 30];
        let data: Vec<f32> = (0..6000).map(|i| i as f32).collect();
        verify_permute(&data, &shape, &[2, 0, 1]);
    }

    #[test]
    fn test_f64_permute() {
        let shape = [10, 20, 30];
        let data = seq_data_f64(6000);
        verify_permute(&data, &shape, &[2, 0, 1]);
    }

    #[test]
    fn test_i32_permute() {
        let shape = [10, 20, 30];
        let data = seq_data(6000);
        verify_permute(&data, &shape, &[2, 0, 1]);
    }

    #[test]
    fn test_i64_permute() {
        let shape = [10, 20, 30];
        let data: Vec<i64> = (0..6000).map(|i| i as i64).collect();
        verify_permute(&data, &shape, &[2, 0, 1]);
    }
}
```

**Step 2: Run all tests**

Run: `cargo test --test permute_correctness`
Expected: 19 tests pass

**Step 3: Commit**

```bash
git add tests/permute_correctness.rs
git commit -m "test(permute): add data type tests for f32, f64, i32, i64"
```

---

## Summary

| Task | Tests Added | Cumulative |
|------|-------------|------------|
| 1 | 0 (infrastructure) | 0 |
| 2 | 4 (matrix) | 4 |
| 3 | 4 (batched) | 8 |
| 4 | 4 (tensor network) | 12 |
| 5 | 3 (high-rank) | 15 |
| 6 | 4 (data types) | 19 |
