//! CUDA tropical-contraction tests: GPU (`tropical-gemm-cuda`) vs CPU parity.
//!
//! These exercise the `cuda-tropical` paths and assert they produce the *same
//! values* as the reference CPU backend for max-plus / min-plus / max-mul over
//! f32 & f64:
//! - the forward path (`Cuda::contract` → `contract_tropical`);
//! - the argmax path (`Cuda::contract_with_argmax` → `contract_tropical_with_argmax`),
//!   compared result-and-argmax against the CPU; and
//! - end-to-end gradients (`einsum_with_grad` + `backward`) on a `Cuda` tensor.
//!
//! # Requirements
//! - An NVIDIA GPU visible to the process.
//! - Built with `--features cuda-tropical` (independent of cuTENSOR).
//!
//! Compiled only under `cuda-tropical`; on a host without a GPU the `Cuda::new()`
//! guard skips each test instead of failing.

#![cfg(feature = "cuda-tropical")]

use omeinsum::backend::{Backend, Storage};
use omeinsum::{
    einsum_with_grad, Algebra, BackendScalar, Cpu, Cuda, MaxMul, MaxPlus, MinPlus, Tensor,
};

/// Run the same contraction on CPU and the CUDA tropical path and assert the
/// downloaded GPU result matches the CPU result element-wise.
///
/// Skips (returns early) when no CUDA device is available so the suite is a
/// no-op on CPU-only CI rather than a failure.
#[allow(clippy::too_many_arguments)]
fn assert_tropical_match<A: Algebra>(
    label: &str,
    a: &[A::Scalar],
    shape_a: &[usize],
    strides_a: &[usize],
    modes_a: &[i32],
    b: &[A::Scalar],
    shape_b: &[usize],
    strides_b: &[usize],
    modes_b: &[i32],
    shape_c: &[usize],
    modes_c: &[i32],
) where
    A::Scalar: omeinsum::BackendScalar<Cpu> + omeinsum::BackendScalar<Cuda> + Into<f64>,
{
    let cuda = match Cuda::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[{label}] skipped: no CUDA device ({e})");
            return;
        }
    };
    let cpu = Cpu;

    // CPU reference.
    let cpu_a = cpu.from_slice(a);
    let cpu_b = cpu.from_slice(b);
    let cpu_c = cpu
        .contract::<A>(
            &cpu_a, shape_a, strides_a, modes_a, &cpu_b, shape_b, strides_b, modes_b, shape_c,
            modes_c,
        )
        .to_vec();

    // GPU path.
    let gpu_a = cuda.from_slice(a);
    let gpu_b = cuda.from_slice(b);
    let gpu_c = cuda
        .contract::<A>(
            &gpu_a, shape_a, strides_a, modes_a, &gpu_b, shape_b, strides_b, modes_b, shape_c,
            modes_c,
        )
        .to_vec();

    assert_eq!(
        gpu_c.len(),
        cpu_c.len(),
        "[{label}] result length mismatch (gpu {} vs cpu {})",
        gpu_c.len(),
        cpu_c.len()
    );
    for (i, (g, c)) in gpu_c.iter().zip(cpu_c.iter()).enumerate() {
        let (g, c): (f64, f64) = ((*g).into(), (*c).into());
        let tol = 1e-4 * (1.0 + c.abs());
        assert!(
            (g - c).abs() <= tol,
            "[{label}] element {i} mismatch: gpu={g} cpu={c} (tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// Plain matmul  "ik,kj->ij"  (i=0,k=1,j=2): A 2x3, B 3x4 -> C 2x4 (column-major)
// ---------------------------------------------------------------------------
fn matmul_a_f32() -> Vec<f32> {
    // 2x3 column-major
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
}
fn matmul_b_f32() -> Vec<f32> {
    // 3x4 column-major
    (1..=12).map(|x| x as f32 * 0.5).collect()
}

#[test]
fn maxplus_matmul_f32() {
    assert_tropical_match::<MaxPlus<f32>>(
        "maxplus_matmul_f32",
        &matmul_a_f32(),
        &[2, 3],
        &[1, 2],
        &[0, 1],
        &matmul_b_f32(),
        &[3, 4],
        &[1, 3],
        &[1, 2],
        &[2, 4],
        &[0, 2],
    );
}

#[test]
fn minplus_matmul_f64() {
    let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: Vec<f64> = (1..=12).map(|x| x as f64 * 0.5).collect();
    assert_tropical_match::<MinPlus<f64>>(
        "minplus_matmul_f64",
        &a,
        &[2, 3],
        &[1, 2],
        &[0, 1],
        &b,
        &[3, 4],
        &[1, 3],
        &[1, 2],
        &[2, 4],
        &[0, 2],
    );
}

#[test]
fn maxmul_matmul_f32() {
    // MaxMul: keep inputs positive (identity = 1, zero = 0).
    let a: Vec<f32> = vec![0.2, 0.5, 0.9, 0.1, 0.7, 0.3];
    let b: Vec<f32> = (1..=12).map(|x| x as f32 / 13.0).collect();
    assert_tropical_match::<MaxMul<f32>>(
        "maxmul_matmul_f32",
        &a,
        &[2, 3],
        &[1, 2],
        &[0, 1],
        &b,
        &[3, 4],
        &[1, 3],
        &[1, 2],
        &[2, 4],
        &[0, 2],
    );
}

// ---------------------------------------------------------------------------
// Transposed left operand  "ki,kj->ij"  (k=0,i=1,j=2): A stored [k,i] must
// permute to [i,k] before the GEMM.
// ---------------------------------------------------------------------------
#[test]
fn maxplus_transposed_left_f64() {
    let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2 column-major (k=3,i=2)
    let b: Vec<f64> = (1..=12).map(|x| x as f64 * 0.25).collect(); // 3x4 (k=3,j=4)
    assert_tropical_match::<MaxPlus<f64>>(
        "maxplus_transposed_left_f64",
        &a,
        &[3, 2],
        &[1, 3],
        &[0, 1],
        &b,
        &[3, 4],
        &[1, 3],
        &[0, 2],
        &[2, 4],
        &[1, 2],
    );
}

// ---------------------------------------------------------------------------
// Batched matmul  "bik,bkj->bij" (b=0,i=1,k=2,j=3): batch=2, A 2x2, B 2x3.
// Batch axis first in storage; output needs a permute back to [b,i,j].
// ---------------------------------------------------------------------------
#[test]
fn maxplus_batched_f32() {
    // batch=2, i=2, k=2  -> shape [2,2,2], column-major strides [1,2,4]
    let a: Vec<f32> = vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];
    // batch=2, k=2, j=3  -> shape [2,2,3], strides [1,2,4]
    let b: Vec<f32> = (1..=12).map(|x| x as f32 * 0.5).collect();
    assert_tropical_match::<MaxPlus<f32>>(
        "maxplus_batched_f32",
        &a,
        &[2, 2, 2],
        &[1, 2, 4],
        &[0, 1, 2],
        &b,
        &[2, 2, 3],
        &[1, 2, 4],
        &[0, 2, 3],
        &[2, 2, 3],
        &[0, 1, 3],
    );
}

// ---------------------------------------------------------------------------
// Trace (left) mode  "ikl,kj->ij" (i=0,k=1,l=2,j=3): l only in A and not in the
// output -> must be reduced via the semiring add before the GEMM.
// ---------------------------------------------------------------------------
#[test]
fn maxplus_left_trace_f64() {
    // A shape [i=2, k=3, l=2] column-major strides [1,2,6], 12 elements.
    let a: Vec<f64> = (1..=12).map(|x| x as f64 * 0.3).collect();
    // B shape [k=3, j=4] strides [1,3].
    let b: Vec<f64> = (1..=12).map(|x| x as f64 * 0.4).collect();
    assert_tropical_match::<MaxPlus<f64>>(
        "maxplus_left_trace_f64",
        &a,
        &[2, 3, 2],
        &[1, 2, 6],
        &[0, 1, 2],
        &b,
        &[3, 4],
        &[1, 3],
        &[1, 3],
        &[2, 4],
        &[0, 3],
    );
}

// ---------------------------------------------------------------------------
// Outer product  "i,j->ij" (no contracted index): exercises the k=1 GEMM path.
// ---------------------------------------------------------------------------
#[test]
fn maxplus_outer_product_f32() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![4.0, 5.0];
    assert_tropical_match::<MaxPlus<f32>>(
        "maxplus_outer_product_f32",
        &a,
        &[3],
        &[1],
        &[0],
        &b,
        &[2],
        &[1],
        &[1],
        &[3, 2],
        &[0, 1],
    );
}

// ---------------------------------------------------------------------------
// MinPlus batched  "bik,bkj->bij" (b=0,i=1,k=2,j=3): batch=2, A 2x2, B 2x3.
// Exercises the batched path for a second algebra.
// ---------------------------------------------------------------------------
#[test]
fn minplus_batched_f64() {
    let a: Vec<f64> = vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];
    let b: Vec<f64> = (1..=12).map(|x| x as f64 * 0.5).collect();
    assert_tropical_match::<MinPlus<f64>>(
        "minplus_batched_f64",
        &a,
        &[2, 2, 2],
        &[1, 2, 4],
        &[0, 1, 2],
        &b,
        &[2, 2, 3],
        &[1, 2, 4],
        &[0, 2, 3],
        &[2, 2, 3],
        &[0, 1, 3],
    );
}

// ---------------------------------------------------------------------------
// MaxMul left-trace  "ikl,kj->ij": trace reduction uses MaxMul's add (= max),
// so this verifies semiring-specific reduction for a non-MaxPlus algebra.
// ---------------------------------------------------------------------------
#[test]
fn maxmul_left_trace_f32() {
    // A shape [i=2, k=3, l=2] strides [1,2,6]; positive values for MaxMul.
    let a: Vec<f32> = (1..=12).map(|x| x as f32 / 13.0).collect();
    let b: Vec<f32> = (1..=12).map(|x| x as f32 / 11.0).collect(); // [k=3, j=4]
    assert_tropical_match::<MaxMul<f32>>(
        "maxmul_left_trace_f32",
        &a,
        &[2, 3, 2],
        &[1, 2, 6],
        &[0, 1, 2],
        &b,
        &[3, 4],
        &[1, 3],
        &[1, 3],
        &[2, 4],
        &[0, 3],
    );
}

// ---------------------------------------------------------------------------
// MaxMul outer product  "i,j->ij" f64.
// ---------------------------------------------------------------------------
#[test]
fn maxmul_outer_product_f64() {
    let a: Vec<f64> = vec![0.3, 0.7, 0.5];
    let b: Vec<f64> = vec![0.9, 0.2];
    assert_tropical_match::<MaxMul<f64>>(
        "maxmul_outer_product_f64",
        &a,
        &[3],
        &[1],
        &[0],
        &b,
        &[2],
        &[1],
        &[1],
        &[3, 2],
        &[0, 1],
    );
}

// ---------------------------------------------------------------------------
// Right-trace  "ik,kjl->ij" (i=0,k=1,j=2,l=3): l only in B, not in output ->
// must be reduced on the B side before the GEMM.
// ---------------------------------------------------------------------------
#[test]
fn maxplus_right_trace_f32() {
    let a: Vec<f32> = (1..=6).map(|x| x as f32 * 0.5).collect(); // [i=2, k=3] strides [1,2]
    let b: Vec<f32> = (1..=24).map(|x| x as f32 * 0.1).collect(); // [k=3, j=4, l=2] strides [1,3,12]
    assert_tropical_match::<MaxPlus<f32>>(
        "maxplus_right_trace_f32",
        &a,
        &[2, 3],
        &[1, 2],
        &[0, 1],
        &b,
        &[3, 4, 2],
        &[1, 3, 12],
        &[1, 2, 3],
        &[2, 4],
        &[0, 2],
    );
}

// ---------------------------------------------------------------------------
// Multi-mode contraction  "ijk,jkl->il" (i=0,j=1,k=2,l=3): contracts two shared
// modes (j,k), stressing product_of_dims over multi-mode groups.
// ---------------------------------------------------------------------------
#[test]
fn maxplus_multimode_contract_f64() {
    let a: Vec<f64> = (1..=12).map(|x| x as f64 * 0.3).collect(); // [i=2, j=3, k=2] strides [1,2,6]
    let b: Vec<f64> = (1..=24).map(|x| x as f64 * 0.2).collect(); // [j=3, k=2, l=4] strides [1,3,6]
    assert_tropical_match::<MaxPlus<f64>>(
        "maxplus_multimode_contract_f64",
        &a,
        &[2, 3, 2],
        &[1, 2, 6],
        &[0, 1, 2],
        &b,
        &[3, 2, 4],
        &[1, 3, 6],
        &[1, 2, 3],
        &[2, 4],
        &[0, 3],
    );
}

// ---------------------------------------------------------------------------
// Scalar output  "ij,ij->" (i=0,j=1): both modes contracted, output is 0-dim.
// Exercises the shape_c=[] permute/return path.
// ---------------------------------------------------------------------------
#[test]
fn maxplus_full_contraction_scalar_f32() {
    let a: Vec<f32> = (1..=6).map(|x| x as f32 * 0.5).collect(); // [i=2, j=3]
    let b: Vec<f32> = (1..=6).map(|x| x as f32 * 0.25).collect();
    assert_tropical_match::<MaxPlus<f32>>(
        "maxplus_full_contraction_scalar_f32",
        &a,
        &[2, 3],
        &[1, 2],
        &[0, 1],
        &b,
        &[2, 3],
        &[1, 2],
        &[0, 1],
        &[],
        &[],
    );
}

// ===========================================================================
// Argmax / backward parity (Phase 3): `Cuda::contract_with_argmax` vs CPU.
// ===========================================================================

/// Run `contract_with_argmax` on CPU and the CUDA tropical path and assert both
/// the result *and* the winner `k`-indices match.
///
/// The argmax is expected to be **exactly** equal: both paths linearize the
/// contracted modes in the same `classify_modes` order, and both kernels come
/// from the same `tropical-gemm` lineage. Inputs are chosen with unique winners
/// (irregular decimals) so there is no tie-break ambiguity to begin with.
///
/// Skips when no CUDA device is available.
#[allow(clippy::too_many_arguments)]
fn assert_tropical_argmax_match<A: Algebra<Index = u32>>(
    label: &str,
    a: &[A::Scalar],
    shape_a: &[usize],
    strides_a: &[usize],
    modes_a: &[i32],
    b: &[A::Scalar],
    shape_b: &[usize],
    strides_b: &[usize],
    modes_b: &[i32],
    shape_c: &[usize],
    modes_c: &[i32],
) where
    A::Scalar: BackendScalar<Cpu> + BackendScalar<Cuda> + Into<f64>,
{
    let cuda = match Cuda::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[{label}] skipped: no CUDA device ({e})");
            return;
        }
    };
    let cpu = Cpu;

    // CPU reference.
    let cpu_a = cpu.from_slice(a);
    let cpu_b = cpu.from_slice(b);
    let (cpu_c, cpu_argmax) = cpu.contract_with_argmax::<A>(
        &cpu_a, shape_a, strides_a, modes_a, &cpu_b, shape_b, strides_b, modes_b, shape_c, modes_c,
    );
    let cpu_c = Storage::to_vec(&cpu_c);
    let cpu_argmax = Storage::to_vec(&cpu_argmax);

    // GPU path.
    let gpu_a = cuda.from_slice(a);
    let gpu_b = cuda.from_slice(b);
    let (gpu_c, gpu_argmax) = cuda.contract_with_argmax::<A>(
        &gpu_a, shape_a, strides_a, modes_a, &gpu_b, shape_b, strides_b, modes_b, shape_c, modes_c,
    );
    let gpu_c = Storage::to_vec(&gpu_c);
    let gpu_argmax = Storage::to_vec(&gpu_argmax);

    assert_eq!(
        gpu_c.len(),
        cpu_c.len(),
        "[{label}] result length mismatch (gpu {} vs cpu {})",
        gpu_c.len(),
        cpu_c.len()
    );
    for (i, (g, c)) in gpu_c.iter().zip(cpu_c.iter()).enumerate() {
        let (g, c): (f64, f64) = ((*g).into(), (*c).into());
        let tol = 1e-4 * (1.0 + c.abs());
        assert!(
            (g - c).abs() <= tol,
            "[{label}] result element {i} mismatch: gpu={g} cpu={c} (tol {tol})"
        );
    }
    assert_eq!(
        gpu_argmax, cpu_argmax,
        "[{label}] argmax mismatch: gpu={gpu_argmax:?} cpu={cpu_argmax:?}"
    );
}

// Unique-winner operands (irregular decimals) for "ik,kj->ij": A 2x3, B 3x4.
fn argmax_a_f64() -> Vec<f64> {
    // 2x3 column-major: a[i,k]
    vec![0.13, 0.91, 0.42, 0.27, 0.68, 0.55]
}
fn argmax_b_f64() -> Vec<f64> {
    // 3x4 column-major: b[k,j]
    vec![
        0.31, 0.07, 0.88, 0.52, 0.19, 0.63, 0.44, 0.96, 0.11, 0.74, 0.28, 0.85,
    ]
}

#[test]
fn maxplus_argmax_matmul_f64() {
    assert_tropical_argmax_match::<MaxPlus<f64>>(
        "maxplus_argmax_matmul_f64",
        &argmax_a_f64(),
        &[2, 3],
        &[1, 2],
        &[0, 1],
        &argmax_b_f64(),
        &[3, 4],
        &[1, 3],
        &[1, 2],
        &[2, 4],
        &[0, 2],
    );
}

#[test]
fn minplus_argmax_matmul_f32() {
    let a: Vec<f32> = vec![0.13, 0.91, 0.42, 0.27, 0.68, 0.55];
    let b: Vec<f32> = vec![
        0.31, 0.07, 0.88, 0.52, 0.19, 0.63, 0.44, 0.96, 0.11, 0.74, 0.28, 0.85,
    ];
    assert_tropical_argmax_match::<MinPlus<f32>>(
        "minplus_argmax_matmul_f32",
        &a,
        &[2, 3],
        &[1, 2],
        &[0, 1],
        &b,
        &[3, 4],
        &[1, 3],
        &[1, 2],
        &[2, 4],
        &[0, 2],
    );
}

#[test]
fn maxmul_argmax_matmul_f64() {
    // Positive operands for MaxMul; irregular so the product winners are unique.
    let a: Vec<f64> = vec![0.13, 0.91, 0.42, 0.27, 0.68, 0.55];
    let b: Vec<f64> = vec![
        0.31, 0.07, 0.88, 0.52, 0.19, 0.63, 0.44, 0.96, 0.11, 0.74, 0.28, 0.85,
    ];
    assert_tropical_argmax_match::<MaxMul<f64>>(
        "maxmul_argmax_matmul_f64",
        &a,
        &[2, 3],
        &[1, 2],
        &[0, 1],
        &b,
        &[3, 4],
        &[1, 3],
        &[1, 2],
        &[2, 4],
        &[0, 2],
    );
}

#[test]
fn maxplus_argmax_transposed_left_f64() {
    // "ki,kj->ij": A stored [k,i] must permute to [i,k]; argmax must follow.
    let a: Vec<f64> = vec![0.13, 0.91, 0.42, 0.27, 0.68, 0.55]; // 3x2 (k=3,i=2)
    let b: Vec<f64> = vec![
        0.31, 0.07, 0.88, 0.52, 0.19, 0.63, 0.44, 0.96, 0.11, 0.74, 0.28, 0.85,
    ];
    assert_tropical_argmax_match::<MaxPlus<f64>>(
        "maxplus_argmax_transposed_left_f64",
        &a,
        &[3, 2],
        &[1, 3],
        &[0, 1],
        &b,
        &[3, 4],
        &[1, 3],
        &[0, 2],
        &[2, 4],
        &[1, 2],
    );
}

#[test]
fn maxplus_argmax_batched_f32() {
    // "bik,bkj->bij": batch=2, output permute back to [b,i,j] also permutes argmax.
    let a: Vec<f32> = vec![0.13, 0.91, 0.42, 0.27, 0.68, 0.55, 0.31, 0.84];
    let b: Vec<f32> = vec![
        0.07, 0.88, 0.52, 0.19, 0.63, 0.44, 0.96, 0.11, 0.74, 0.28, 0.85, 0.36,
    ];
    assert_tropical_argmax_match::<MaxPlus<f32>>(
        "maxplus_argmax_batched_f32",
        &a,
        &[2, 2, 2],
        &[1, 2, 4],
        &[0, 1, 2],
        &b,
        &[2, 2, 3],
        &[1, 2, 4],
        &[0, 2, 3],
        &[2, 2, 3],
        &[0, 1, 3],
    );
}

#[test]
fn maxplus_argmax_multimode_contract_f64() {
    // "ijk,jkl->il": two contracted modes (j,k); argmax linearizes both.
    let a: Vec<f64> = vec![
        0.13, 0.91, 0.42, 0.27, 0.68, 0.55, 0.31, 0.84, 0.06, 0.77, 0.49, 0.22,
    ]; // [i=2,j=3,k=2]
    let b: Vec<f64> = vec![
        0.07, 0.88, 0.52, 0.19, 0.63, 0.44, 0.96, 0.11, 0.74, 0.28, 0.85, 0.36, 0.61, 0.04, 0.93,
        0.47, 0.15, 0.79, 0.33, 0.58, 0.02, 0.71, 0.40, 0.66,
    ]; // [j=3,k=2,l=4]
    assert_tropical_argmax_match::<MaxPlus<f64>>(
        "maxplus_argmax_multimode_contract_f64",
        &a,
        &[2, 3, 2],
        &[1, 2, 6],
        &[0, 1, 2],
        &b,
        &[3, 2, 4],
        &[1, 3, 6],
        &[1, 2, 3],
        &[2, 4],
        &[0, 3],
    );
}

// ===========================================================================
// End-to-end gradient parity (Phase 3): `einsum_with_grad` on a Cuda tensor
// vs the same graph on Cpu, comparing the input gradients element-wise.
// ===========================================================================

/// Build `(result, grads)` for a binary tropical einsum on backend `B`, driving
/// the backward pass with an all-ones output gradient.
#[allow(clippy::too_many_arguments)]
fn tropical_forward_and_grads<A, B>(
    backend: B,
    a_data: &[f64],
    shape_a: &[usize],
    ia: &[usize],
    b_data: &[f64],
    shape_b: &[usize],
    ib: &[usize],
    iy: &[usize],
) -> (Vec<f64>, Vec<Vec<f64>>)
where
    A: Algebra<Scalar = f64, Index = u32>,
    B: Backend,
    f64: BackendScalar<B>,
{
    let a = Tensor::<f64, B>::from_data_with_backend(a_data, shape_a, backend.clone());
    let b = Tensor::<f64, B>::from_data_with_backend(b_data, shape_b, backend.clone());

    let (result, grad) = einsum_with_grad::<A, _, _>(&[&a, &b], &[ia, ib], iy);

    let out_shape = result.shape().to_vec();
    let out_numel: usize = out_shape.iter().product::<usize>().max(1);
    let grad_output =
        Tensor::<f64, B>::from_data_with_backend(&vec![1.0f64; out_numel], &out_shape, backend);

    let grads = grad.backward::<A>(&grad_output, &[&a, &b]);
    (result.to_vec(), grads.iter().map(|g| g.to_vec()).collect())
}

/// Assert that a tropical einsum's forward result and input gradients agree
/// between the CUDA backend and the CPU reference. Skips without a GPU.
#[allow(clippy::too_many_arguments)]
fn assert_tropical_gradient_parity<A>(
    label: &str,
    a_data: &[f64],
    shape_a: &[usize],
    ia: &[usize],
    b_data: &[f64],
    shape_b: &[usize],
    ib: &[usize],
    iy: &[usize],
) where
    A: Algebra<Scalar = f64, Index = u32>,
{
    let cuda = match Cuda::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[{label}] skipped: no CUDA device ({e})");
            return;
        }
    };

    let (cpu_result, cpu_grads) =
        tropical_forward_and_grads::<A, Cpu>(Cpu, a_data, shape_a, ia, b_data, shape_b, ib, iy);
    let (gpu_result, gpu_grads) =
        tropical_forward_and_grads::<A, Cuda>(cuda, a_data, shape_a, ia, b_data, shape_b, ib, iy);

    let cmp = |label: &str, what: &str, g: &[f64], c: &[f64]| {
        assert_eq!(
            g.len(),
            c.len(),
            "[{label}] {what} length mismatch (gpu {} vs cpu {})",
            g.len(),
            c.len()
        );
        for (i, (gv, cv)) in g.iter().zip(c.iter()).enumerate() {
            let tol = 1e-4 * (1.0 + cv.abs());
            assert!(
                (gv - cv).abs() <= tol,
                "[{label}] {what} element {i} mismatch: gpu={gv} cpu={cv} (tol {tol})"
            );
        }
    };

    cmp(label, "result", &gpu_result, &cpu_result);
    assert_eq!(
        gpu_grads.len(),
        cpu_grads.len(),
        "[{label}] gradient count mismatch"
    );
    for (idx, (g, c)) in gpu_grads.iter().zip(cpu_grads.iter()).enumerate() {
        cmp(label, &format!("grad[{idx}]"), g, c);
    }
}

#[test]
fn maxplus_gradient_parity_matmul_f64() {
    // "ik,kj->ij": gradients route through the argmax winners.
    assert_tropical_gradient_parity::<MaxPlus<f64>>(
        "maxplus_gradient_parity_matmul_f64",
        &argmax_a_f64(),
        &[2, 3],
        &[0, 1],
        &argmax_b_f64(),
        &[3, 4],
        &[1, 2],
        &[0, 2],
    );
}

#[test]
fn minplus_gradient_parity_matmul_f64() {
    assert_tropical_gradient_parity::<MinPlus<f64>>(
        "minplus_gradient_parity_matmul_f64",
        &argmax_a_f64(),
        &[2, 3],
        &[0, 1],
        &argmax_b_f64(),
        &[3, 4],
        &[1, 2],
        &[0, 2],
    );
}

#[test]
fn maxmul_gradient_parity_matmul_f64() {
    let a: Vec<f64> = vec![0.13, 0.91, 0.42, 0.27, 0.68, 0.55];
    let b: Vec<f64> = vec![
        0.31, 0.07, 0.88, 0.52, 0.19, 0.63, 0.44, 0.96, 0.11, 0.74, 0.28, 0.85,
    ];
    assert_tropical_gradient_parity::<MaxMul<f64>>(
        "maxmul_gradient_parity_matmul_f64",
        &a,
        &[2, 3],
        &[0, 1],
        &b,
        &[3, 4],
        &[1, 2],
        &[0, 2],
    );
}
