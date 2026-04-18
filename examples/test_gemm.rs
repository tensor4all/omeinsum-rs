//! Test raw GEMM performance
use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};
use std::time::Instant;

fn main() {
    for (m, k, n) in [
        (32, 32, 32),
        (64, 64, 64),
        (64, 64, 64),
        (256, 128, 256),
        (1024, 256, 1024),
        (2048, 512, 2048),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.001).collect();
        let mut c = vec![0.0f32; m * n];

        // Warm up
        for _ in 0..10 {
            faer_gemm(&a, m, k, &b, n, &mut c);
        }

        let num_iters = 100;
        let start = Instant::now();
        for _ in 0..num_iters {
            faer_gemm(&a, m, k, &b, n, &mut c);
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_secs_f64() * 1e6 / num_iters as f64;

        let gflops = 2.0 * m as f64 * k as f64 * n as f64 / avg_us / 1e3;
        println!(
            "GEMM {}x{}x{}: {:.2} µs ({:.1} GFLOPS)",
            m, k, n, avg_us, gflops
        );
    }
}

const GEMM_PARALLEL_THRESHOLD: usize = 100_000_000;

fn faer_gemm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    let a_mat = unsafe { MatRef::from_raw_parts(a.as_ptr(), m, k, 1, m as isize) };
    let b_mat = unsafe { MatRef::from_raw_parts(b.as_ptr(), k, n, 1, k as isize) };
    let mut c_mat = unsafe { MatMut::from_raw_parts_mut(c.as_mut_ptr(), m, n, 1, m as isize) };

    // Use sequential for small matrices, parallel for large
    let flops = 2 * m * k * n;
    let par = if flops < GEMM_PARALLEL_THRESHOLD {
        Par::Seq
    } else {
        Par::rayon(0)
    };
    matmul(c_mat.as_mut(), Accum::Replace, a_mat, b_mat, 1.0f32, par);

    // Print once for debug
    static PRINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !PRINTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        println!(
            "Threshold: {} FLOPs, using {:?} for {}x{}x{} ({} FLOPs)",
            GEMM_PARALLEL_THRESHOLD,
            if flops < GEMM_PARALLEL_THRESHOLD {
                "Seq"
            } else {
                "Parallel"
            },
            m,
            k,
            n,
            flops
        );
    }
}
