//! Investigate faer parallel GEMM overhead
use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};
use std::time::Instant;

fn main() {
    // Test different sizes with Seq vs Parallel
    let sizes = [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ];

    println!("Comparing Sequential vs Parallel GEMM:\n");
    println!(
        "{:>8} {:>8} {:>8} | {:>12} {:>8} | {:>12} {:>8} | {:>8}",
        "M", "K", "N", "Seq (µs)", "GFLOPS", "Par (µs)", "GFLOPS", "Ratio"
    );
    println!("{}", "-".repeat(90));

    for (m, k, n) in sizes {
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.001).collect();
        let mut c = vec![0.0f32; m * n];

        // Warm up
        for _ in 0..5 {
            gemm_seq(&a, m, k, &b, n, &mut c);
            gemm_par(&a, m, k, &b, n, &mut c);
        }

        // Benchmark sequential
        let num_iters = 50;
        let start = Instant::now();
        for _ in 0..num_iters {
            gemm_seq(&a, m, k, &b, n, &mut c);
        }
        let seq_us = start.elapsed().as_secs_f64() * 1e6 / num_iters as f64;

        // Benchmark parallel
        let start = Instant::now();
        for _ in 0..num_iters {
            gemm_par(&a, m, k, &b, n, &mut c);
        }
        let par_us = start.elapsed().as_secs_f64() * 1e6 / num_iters as f64;

        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let seq_gflops = flops / seq_us / 1e3;
        let par_gflops = flops / par_us / 1e3;
        let ratio = seq_us / par_us;

        println!(
            "{:>8} {:>8} {:>8} | {:>12.2} {:>8.1} | {:>12.2} {:>8.1} | {:>8.2}x",
            m, k, n, seq_us, seq_gflops, par_us, par_gflops, ratio
        );
    }

    println!("\nNote: Ratio > 1 means parallel is faster");
    println!("Thread pool overhead is the fixed cost of scheduling work to threads.");
    println!("For small matrices, this overhead dominates over compute time.");
}

fn gemm_seq(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    let a_mat = unsafe { MatRef::from_raw_parts(a.as_ptr(), m, k, 1, m as isize) };
    let b_mat = unsafe { MatRef::from_raw_parts(b.as_ptr(), k, n, 1, k as isize) };
    let mut c_mat = unsafe { MatMut::from_raw_parts_mut(c.as_mut_ptr(), m, n, 1, m as isize) };
    matmul(
        c_mat.as_mut(),
        Accum::Replace,
        a_mat,
        b_mat,
        1.0f32,
        Par::Seq,
    );
}

fn gemm_par(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    let a_mat = unsafe { MatRef::from_raw_parts(a.as_ptr(), m, k, 1, m as isize) };
    let b_mat = unsafe { MatRef::from_raw_parts(b.as_ptr(), k, n, 1, k as isize) };
    let mut c_mat = unsafe { MatMut::from_raw_parts_mut(c.as_mut_ptr(), m, n, 1, m as isize) };
    matmul(
        c_mat.as_mut(),
        Accum::Replace,
        a_mat,
        b_mat,
        1.0f32,
        Par::rayon(0),
    );
}
