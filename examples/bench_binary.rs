//! Benchmark binary einsum with high-dimensional tensors (rank ~25, dim size 2)
//!
//! Run with: cargo run --release --example bench_binary

use omeinsum::algebra::Standard;
use omeinsum::{einsum, Cpu, Tensor};
use std::time::Instant;

fn main() {
    println!("=== Binary Einsum Benchmark (dim size = 2) ===\n");

    // Test cases: (name, rank_a, rank_b, num_contracted, num_batch)
    let test_cases = [
        // Simple cases
        ("matmul 10x10", 10, 10, 5, 0),
        ("batched matmul 8x8 batch=4", 8, 8, 4, 4),
        // High-dimensional cases (like tensor network contractions)
        ("high-D 12x12 contract=6", 12, 12, 6, 0),
        ("high-D 15x15 contract=7", 15, 15, 7, 0),
        ("high-D 18x18 contract=8", 18, 18, 8, 0),
        ("high-D 20x20 contract=9", 20, 20, 9, 0),
        // With batch dimensions
        ("high-D 12x12 contract=4 batch=4", 12, 12, 4, 4),
        ("high-D 15x15 contract=5 batch=5", 15, 15, 5, 5),
    ];

    for (name, rank_a, rank_b, num_contracted, num_batch) in test_cases {
        bench_binary_einsum(name, rank_a, rank_b, num_contracted, num_batch);
    }
}

fn bench_binary_einsum(
    name: &str,
    rank_a: usize,
    rank_b: usize,
    num_contracted: usize,
    num_batch: usize,
) {
    // Build index labels
    // A has: [left_a..., contracted..., batch...]
    // B has: [contracted..., right_b..., batch...]
    // C has: [left_a..., right_b..., batch...]

    let num_left_a = rank_a - num_contracted - num_batch;
    let num_right_b = rank_b - num_contracted - num_batch;

    // Assign index labels
    let mut next_idx = 0usize;

    // Left indices (only in A)
    let left_indices: Vec<usize> = (next_idx..next_idx + num_left_a).collect();
    next_idx += num_left_a;

    // Contracted indices (in A and B, not in C)
    let contracted_indices: Vec<usize> = (next_idx..next_idx + num_contracted).collect();
    next_idx += num_contracted;

    // Right indices (only in B)
    let right_indices: Vec<usize> = (next_idx..next_idx + num_right_b).collect();
    next_idx += num_right_b;

    // Batch indices (in A, B, and C)
    let batch_indices: Vec<usize> = (next_idx..next_idx + num_batch).collect();

    // Build index arrays
    let ixs_a: Vec<usize> = left_indices
        .iter()
        .chain(contracted_indices.iter())
        .chain(batch_indices.iter())
        .copied()
        .collect();

    let ixs_b: Vec<usize> = contracted_indices
        .iter()
        .chain(right_indices.iter())
        .chain(batch_indices.iter())
        .copied()
        .collect();

    let ixs_c: Vec<usize> = left_indices
        .iter()
        .chain(right_indices.iter())
        .chain(batch_indices.iter())
        .copied()
        .collect();

    // Create tensors with shape [2, 2, 2, ...]
    let shape_a: Vec<usize> = vec![2; rank_a];
    let shape_b: Vec<usize> = vec![2; rank_b];

    let numel_a: usize = shape_a.iter().product();
    let numel_b: usize = shape_b.iter().product();
    let numel_c: usize = 2usize.pow((num_left_a + num_right_b + num_batch) as u32);

    let data_a: Vec<f32> = (0..numel_a).map(|i| (i as f32) * 0.001).collect();
    let data_b: Vec<f32> = (0..numel_b).map(|i| (i as f32) * 0.001).collect();

    let tensor_a = Tensor::<f32, Cpu>::from_data(&data_a, &shape_a);
    let tensor_b = Tensor::<f32, Cpu>::from_data(&data_b, &shape_b);

    // Warm up
    for _ in 0..3 {
        let _ = einsum::<Standard<f32>, _, _>(
            &[&tensor_a, &tensor_b],
            &[&ixs_a[..], &ixs_b[..]],
            &ixs_c,
        );
    }

    // Benchmark
    let num_iters = 10;
    let start = Instant::now();
    for _ in 0..num_iters {
        let _ = einsum::<Standard<f32>, _, _>(
            &[&tensor_a, &tensor_b],
            &[&ixs_a[..], &ixs_b[..]],
            &ixs_c,
        );
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / num_iters as f64;

    // Compute theoretical sizes
    let left_size = 2usize.pow(num_left_a as u32);
    let right_size = 2usize.pow(num_right_b as u32);
    let contract_size = 2usize.pow(num_contracted as u32);
    let batch_size = 2usize.pow(num_batch as u32);

    // Memory bandwidth (read A + read B + write C)
    let bytes_moved = (numel_a + numel_b + numel_c) * 4;
    let bandwidth_gbs = bytes_moved as f64 / (avg_ms / 1000.0) / 1e9;

    println!("{}", name);
    println!("  A: rank={}, B: rank={}", rank_a, rank_b);
    println!(
        "  left={}, contract={}, right={}, batch={}",
        num_left_a, num_contracted, num_right_b, num_batch
    );
    println!(
        "  GEMM: [{}x{}] @ [{}x{}] x {} batches",
        left_size, contract_size, contract_size, right_size, batch_size
    );
    println!(
        "  Time: {:.3} ms, Bandwidth: {:.2} GB/s",
        avg_ms, bandwidth_gbs
    );
    println!();
}
