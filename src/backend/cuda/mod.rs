//! CUDA backend for GPU execution.
//!
//! This module provides the CUDA backend implementation using cudarc and cuTENSOR.
//!
//! # Requirements
//!
//! - **CUDA Toolkit**: 11.0 or later
//! - **cuTENSOR**: **Version 2.0 or later** (REQUIRED - version 1.x will NOT work)
//!
//! # cuTENSOR Version Mismatch
//!
//! If you see linker errors like:
//! ```text
//! undefined symbol: cutensorContract
//! undefined symbol: cutensorCreatePlan
//! undefined symbol: CUTENSOR_COMPUTE_DESC_32F
//! ```
//!
//! This means you have cuTENSOR 1.x installed. The API changed significantly
//! in cuTENSOR 2.0. You need to install cuTENSOR 2.0+:
//!
//! ```bash
//! # For CUDA 12:
//! conda install -c nvidia cutensor-cu12
//!
//! # Or download from NVIDIA:
//! # https://developer.nvidia.com/cutensor-downloads
//! ```
//!
//! Then set the library path:
//! ```bash
//! export CUTENSOR_PATH=/path/to/cutensor/lib
//! export LD_LIBRARY_PATH=$CUTENSOR_PATH:$LD_LIBRARY_PATH
//! ```

// The CUDA backend stores scalars behind a generic `T: Scalar` but dispatches the
// actual device work by `TypeId`, reinterpreting the (layout-identical) concrete
// type via `transmute` inside each type-checked branch. That pattern appears
// dozens of times here; the source/target types are evident from the enclosing
// `TypeId` guard and the `let` binding, so the explicit-annotation lint is just
// noise for this file.
#![allow(clippy::missing_transmute_annotations)]

#[cfg(feature = "cuda")]
mod cutensor;
mod storage;

pub use storage::CudaStorage;

use cudarc::driver::{CudaContext, CudaStream};
#[cfg(feature = "cuda")]
use cutensor::{contract, CacheKey, CutensorType, Handle, PlanCache, TensorDesc};
use num_complex::Complex;
use std::sync::Arc;
#[cfg(any(feature = "cuda", feature = "cuda-tropical"))]
use std::sync::Mutex;

use crate::algebra::{Algebra, Scalar};
use crate::backend::traits::{Backend, BackendScalar, Storage};

// ============================================================================
// CUDA-compatible complex number wrapper
// ============================================================================
//
// Due to Rust's orphan rule, we cannot implement cudarc traits for num_complex
// types directly. This generic newtype wrapper provides CUDA-compatible complex.

/// CUDA-compatible wrapper for complex numbers.
///
/// This type has the same memory layout as `num_complex::Complex<T>` and CUDA's
/// complex types, but can implement cudarc traits since it's a local type.
///
/// Use `CudaComplex<f32>` for single-precision and `CudaComplex<f64>` for double.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct CudaComplex<T>(pub Complex<T>);

impl<T> CudaComplex<T> {
    /// Create a new CudaComplex from real and imaginary parts.
    pub fn new(re: T, im: T) -> Self {
        CudaComplex(Complex::new(re, im))
    }

    /// Get the real part.
    pub fn re(&self) -> T
    where
        T: Clone,
    {
        self.0.re.clone()
    }

    /// Get the imaginary part.
    pub fn im(&self) -> T
    where
        T: Clone,
    {
        self.0.im.clone()
    }
}

// SAFETY: CudaComplex<T> is repr(transparent) over Complex<T>, which is repr(C)
// with two T fields. This is compatible with CUDA's complex types.
unsafe impl<T: cudarc::driver::DeviceRepr> cudarc::driver::DeviceRepr for CudaComplex<T> {}
// SAFETY: Zero-initialized CudaComplex<T> is valid if T is valid as zero bits.
unsafe impl<T: cudarc::driver::ValidAsZeroBits> cudarc::driver::ValidAsZeroBits for CudaComplex<T> {}

// Arithmetic for CudaComplex<f32>
impl std::ops::Add for CudaComplex<f32> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        CudaComplex(self.0 + rhs.0)
    }
}

impl std::ops::Mul for CudaComplex<f32> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        CudaComplex(self.0 * rhs.0)
    }
}

impl num_traits::Zero for CudaComplex<f32> {
    fn zero() -> Self {
        CudaComplex(Complex::new(0.0, 0.0))
    }
    fn is_zero(&self) -> bool {
        self.0.re == 0.0 && self.0.im == 0.0
    }
}

impl num_traits::One for CudaComplex<f32> {
    fn one() -> Self {
        CudaComplex(Complex::new(1.0, 0.0))
    }
}

impl std::ops::AddAssign for CudaComplex<f32> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

// Arithmetic for CudaComplex<f64>
impl std::ops::Add for CudaComplex<f64> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        CudaComplex(self.0 + rhs.0)
    }
}

impl std::ops::Mul for CudaComplex<f64> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        CudaComplex(self.0 * rhs.0)
    }
}

impl num_traits::Zero for CudaComplex<f64> {
    fn zero() -> Self {
        CudaComplex(Complex::new(0.0, 0.0))
    }
    fn is_zero(&self) -> bool {
        self.0.re == 0.0 && self.0.im == 0.0
    }
}

impl num_traits::One for CudaComplex<f64> {
    fn one() -> Self {
        CudaComplex(Complex::new(1.0, 0.0))
    }
}

impl std::ops::AddAssign for CudaComplex<f64> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

// SAFETY: CudaComplex<f32> is repr(transparent) over Complex<f32>,
// which is repr(C) with two f32 fields. This is a valid Pod type.
unsafe impl bytemuck::Zeroable for CudaComplex<f32> {}
unsafe impl bytemuck::Pod for CudaComplex<f32> {}

// SAFETY: CudaComplex<f64> is repr(transparent) over Complex<f64>,
// which is repr(C) with two f64 fields. This is a valid Pod type.
unsafe impl bytemuck::Zeroable for CudaComplex<f64> {}
unsafe impl bytemuck::Pod for CudaComplex<f64> {}

// Scalar implementations for CudaComplex
// This enables the high-level einsum API to work with complex numbers on GPU.
impl Scalar for CudaComplex<f32> {}
impl Scalar for CudaComplex<f64> {}

// CutensorType implementations
#[cfg(feature = "cuda")]
impl CutensorType for CudaComplex<f32> {
    const DATA: cutensor::sys::cutensorDataType_t = cutensor::sys::cutensorDataType_t::C_32F;
    fn compute_desc() -> cutensor::sys::cutensorComputeDescriptor_t {
        unsafe { cutensor::sys::CUTENSOR_COMPUTE_DESC_32F }
    }
}

#[cfg(feature = "cuda")]
impl CutensorType for CudaComplex<f64> {
    const DATA: cutensor::sys::cutensorDataType_t = cutensor::sys::cutensorDataType_t::C_64F;
    fn compute_desc() -> cutensor::sys::cutensorComputeDescriptor_t {
        unsafe { cutensor::sys::CUTENSOR_COMPUTE_DESC_64F }
    }
}

// Conversion traits
impl<T> From<Complex<T>> for CudaComplex<T> {
    fn from(c: Complex<T>) -> Self {
        CudaComplex(c)
    }
}

impl<T> From<CudaComplex<T>> for Complex<T> {
    fn from(c: CudaComplex<T>) -> Self {
        c.0
    }
}

/// CUDA backend for GPU tensor operations.
///
/// Wraps a CUDA device and provides methods for GPU memory management
/// and tensor contractions via cuTENSOR.
pub struct Cuda {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    //
    // The kernel/plan caches below are wrapped in `Arc` so that `Cuda::clone()`
    // *shares* them rather than resetting them. omeinsum's `Tensor<T, B>` owns its
    // backend and clones it on every operation (so each contraction node gets a
    // cloned backend); if clone produced empty caches, every node would rebuild
    // the `tropical-gemm-cuda` context and the permute module — each an NVRTC
    // `compile_ptx` recompile of the CUDA kernels — making compilation, not
    // compute, dominate the wall (~34 ms/node). Sharing via `Arc` makes the
    // NVRTC compile happen once per process. The inner `Mutex` still guards
    // concurrent access; each is held only long enough to hand out an `Arc`
    // clone, so contractions are not serialized on it.
    #[cfg(feature = "cuda")]
    handle: Arc<Mutex<Option<Handle>>>,
    #[cfg(feature = "cuda")]
    cache: Arc<Mutex<PlanCache>>,
    /// `tropical-gemm-cuda` context built once from the shared CUDA device and
    /// reused across contraction nodes (and across backend clones), instead of
    /// rebuilt (which reloads/recompiles the kernel module) on every tropical
    /// GEMM. Lazily initialized on first use.
    #[cfg(feature = "cuda-tropical")]
    tropical_ctx: Arc<Mutex<Option<Arc<tropical_gemm_cuda::CudaContext>>>>,
    /// NVRTC-compiled module holding the device strided-gather kernels
    /// (`gather32`/`gather64`) used for on-device operand canonicalization and
    /// output permutation. Compiled once and cached (shared across clones),
    /// never per contraction.
    #[cfg(feature = "cuda-tropical")]
    permute_module: Arc<Mutex<Option<Arc<cudarc::driver::CudaModule>>>>,
    /// Reusable device buffers for gather shape/stride metadata, keyed by tensor
    /// rank (`ndim`). Each `device_gather` used to `clone_htod` two *fresh*
    /// device buffers (shape + strides) per call — a large slice of the per-node
    /// allocation churn (2 allocs × 2–3 gathers/node). Here one (shape, strides)
    /// pair is allocated per distinct `ndim` on first use and re-uploaded via
    /// `memcpy_htod`, so steady-state metadata allocations drop to ~0. Shared
    /// across backend clones (like the other caches). The inner `Mutex` makes a
    /// gather's upload-through-launch a critical section so concurrent slices
    /// don't overwrite the shared buffer between each other's upload and kernel
    /// (they serialize on the single stream regardless), and the buffers persist
    /// in the map so a kernel reading them is never freed out from under it.
    ///
    /// One combined `2·ndim` buffer per rank — `[shape (ndim) ‖ strides (ndim)]` —
    /// so each gather uploads its metadata in a single `memcpy_htod` (one HtoD
    /// instead of two); the kernel reads `shape` and `strides` as two non-
    /// overlapping sub-views of the one buffer.
    #[cfg(feature = "cuda-tropical")]
    gather_meta: Arc<Mutex<std::collections::HashMap<usize, cudarc::driver::CudaSlice<i64>>>>,
}

/// Device strided-gather kernels: `out[o] = in[Σ coord_ax · src_strides[ax]]`,
/// where `o` ranges over the contiguous column-major output and its multi-index
/// is decoded against `new_shape`. One kernel per element width (32/64-bit);
/// `src_strides`/`new_shape` are passed in element units as `long long`. This is
/// the device counterpart of the host `materialize_strided` and covers operand
/// canonicalization (strided view → canonical layout) and the output permute,
/// for f32/f64 (via width) and the u32 argmax buffer (via `gather32`).
#[cfg(feature = "cuda-tropical")]
const PERMUTE_KERNEL_SRC: &str = r#"
extern "C" __global__ void gather32(unsigned int* out, const unsigned int* in,
                                    int ndim, const long long* new_shape,
                                    const long long* src_strides, long long numel) {
    long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= numel) return;
    long long rem = o, src = 0;
    for (int ax = 0; ax < ndim; ++ax) {
        long long c = rem % new_shape[ax];
        rem /= new_shape[ax];
        src += c * src_strides[ax];
    }
    out[o] = in[src];
}
extern "C" __global__ void gather64(unsigned long long* out, const unsigned long long* in,
                                    int ndim, const long long* new_shape,
                                    const long long* src_strides, long long numel) {
    long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= numel) return;
    long long rem = o, src = 0;
    for (int ax = 0; ax < ndim; ++ax) {
        long long c = rem % new_shape[ax];
        rem /= new_shape[ax];
        src += c * src_strides[ax];
    }
    out[o] = in[src];
}

// Batched gather: one launch performs `k` independent strided gathers (the
// "grouped/segmented batched" pattern used by cuBLAS grouped-GEMM / MAGMA
// vbatch — the standard way to kill per-launch overhead for many tiny ops).
// `out_ptrs[i]`/`in_ptrs[i]` are the i-th gather's device buffers (array of
// pointers, as in `cublas*gemmBatched`). `meta` packs, per gather and pointed at
// by `meta_off[i]`: `[ndim_i, new_shape_i[ndim_i], src_strides_i[ndim_i]]`.
// `prefix` is the (k+1)-entry prefix sum of output element counts; `total =
// prefix[k]`. A grid-stride loop over the flattened work range gives even load
// balance across gathers of very different sizes; a per-element binary search
// over `prefix` maps the flat index to its gather. Generic over element width
// (32/64-bit) exactly like the single-gather kernels above; works for arbitrary
// dimensions (no power-of-two assumption).
extern "C" __global__ void gather_batched32(unsigned long long* out_ptrs,
                                            const unsigned long long* in_ptrs,
                                            const long long* meta, const long long* meta_off,
                                            const long long* prefix, int k, long long total) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x; g < total; g += stride) {
        int lo = 0, hi = k;
        while (lo + 1 < hi) { int mid = (lo + hi) >> 1; if (prefix[mid] <= g) lo = mid; else hi = mid; }
        long long o = g - prefix[lo];
        const long long* m = meta + meta_off[lo];
        int ndim = (int)m[0];
        const long long* new_shape = m + 1;
        const long long* src_strides = m + 1 + ndim;
        long long rem = o, src = 0;
        for (int ax = 0; ax < ndim; ++ax) {
            long long c = rem % new_shape[ax];
            rem /= new_shape[ax];
            src += c * src_strides[ax];
        }
        unsigned int* out = (unsigned int*)out_ptrs[lo];
        const unsigned int* in = (const unsigned int*)in_ptrs[lo];
        out[o] = in[src];
    }
}
extern "C" __global__ void gather_batched64(unsigned long long* out_ptrs,
                                            const unsigned long long* in_ptrs,
                                            const long long* meta, const long long* meta_off,
                                            const long long* prefix, int k, long long total) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x; g < total; g += stride) {
        int lo = 0, hi = k;
        while (lo + 1 < hi) { int mid = (lo + hi) >> 1; if (prefix[mid] <= g) lo = mid; else hi = mid; }
        long long o = g - prefix[lo];
        const long long* m = meta + meta_off[lo];
        int ndim = (int)m[0];
        const long long* new_shape = m + 1;
        const long long* src_strides = m + 1 + ndim;
        long long rem = o, src = 0;
        for (int ax = 0; ax < ndim; ++ax) {
            long long c = rem % new_shape[ax];
            rem /= new_shape[ax];
            src += c * src_strides[ax];
        }
        unsigned long long* out = (unsigned long long*)out_ptrs[lo];
        const unsigned long long* in = (const unsigned long long*)in_ptrs[lo];
        out[o] = in[src];
    }
}
"#;

/// 1-D launch config for the elementwise gather kernels, sized for `numel`
/// output elements with cudarc's `for_num_elems` layout (block = 1024) but the
/// grid computed in 64-bit. `numel` is a `usize` and can exceed `u32::MAX` (a
/// 2^32-element gather occurs at sc-target 32); `numel as u32` would wrap — at
/// exactly 2^32 to 0, yielding a zero grid and `CUDA_ERROR_INVALID_VALUE`. The
/// gather kernels index globally in `long long`, so a wide grid is safe. The
/// grid is clamped to at least 1 block. Fails only past the physically
/// impossible ~2^41 elements (grid x-dim > u32::MAX).
#[cfg(feature = "cuda-tropical")]
fn gather_launch_config(numel: usize) -> cudarc::driver::LaunchConfig {
    const NUM_THREADS: u32 = 1024;
    let num_blocks = u32::try_from((numel as u64).div_ceil(NUM_THREADS as u64).max(1))
        .expect("gather grid dimension exceeds u32::MAX");
    cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

// SAFETY: Cuda is Send because all fields are Send.
// The Mutex ensures safe concurrent access to handle and cache.
unsafe impl Send for Cuda {}
// SAFETY: Cuda is Sync because all fields are protected by Mutex.
unsafe impl Sync for Cuda {}

impl Clone for Cuda {
    fn clone(&self) -> Self {
        // Share the same device/stream AND the same kernel/plan caches. Cloning
        // the `Arc`s (rather than reinitializing to empty) is what keeps the
        // NVRTC kernel compilation a once-per-process cost: every per-node
        // backend clone now reuses the already-compiled tropical-gemm context
        // and permute module instead of recompiling them.
        Self {
            ctx: self.ctx.clone(),
            stream: self.stream.clone(),
            #[cfg(feature = "cuda")]
            handle: self.handle.clone(),
            #[cfg(feature = "cuda")]
            cache: self.cache.clone(),
            #[cfg(feature = "cuda-tropical")]
            tropical_ctx: self.tropical_ctx.clone(),
            #[cfg(feature = "cuda-tropical")]
            permute_module: self.permute_module.clone(),
            #[cfg(feature = "cuda-tropical")]
            gather_meta: self.gather_meta.clone(),
        }
    }
}

impl Default for Cuda {
    /// Create a default CUDA backend on device 0.
    ///
    /// # Panics
    /// Panics if CUDA initialization fails (e.g., no GPU available).
    fn default() -> Self {
        Self::new().expect("Failed to initialize CUDA device. Is a GPU available?")
    }
}

impl Cuda {
    /// Create a new CUDA backend on the default device (device 0).
    pub fn new() -> Result<Self, CudaError> {
        Self::on_device(0)
    }

    /// Create a new CUDA backend on a specific device.
    ///
    /// # Arguments
    /// * `ordinal` - The device ordinal (0-indexed)
    pub fn on_device(ordinal: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(ordinal).map_err(|e| CudaError::Device(e.to_string()))?;
        let stream = ctx.default_stream();
        Ok(Self {
            ctx,
            stream,
            #[cfg(feature = "cuda")]
            handle: Arc::new(Mutex::new(None)),
            #[cfg(feature = "cuda")]
            cache: Arc::new(Mutex::new(PlanCache::new(64))),
            #[cfg(feature = "cuda-tropical")]
            tropical_ctx: Arc::new(Mutex::new(None)),
            #[cfg(feature = "cuda-tropical")]
            permute_module: Arc::new(Mutex::new(None)),
            #[cfg(feature = "cuda-tropical")]
            gather_meta: Arc::new(Mutex::new(std::collections::HashMap::new())),
        })
    }

    /// Get a reference to the CUDA context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Get a reference to the CUDA stream used for transfers and kernel launches.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Ensure the cuTENSOR handle is initialized and execute a function with it.
    ///
    /// This method acquires the handle lock and ensures the handle is initialized,
    /// then calls the provided function with access to the handle.
    #[cfg(feature = "cuda")]
    fn with_handle<R>(
        &self,
        f: impl FnOnce(&Handle) -> Result<R, CudaError>,
    ) -> Result<R, CudaError> {
        let mut h = self.handle.lock().unwrap();
        if h.is_none() {
            *h = Some(
                Handle::new(self.stream.clone())
                    .map_err(|e| CudaError::Cutensor(format!("{}", e)))?,
            );
        }
        f(h.as_ref().unwrap())
    }

    /// Return the cached `tropical-gemm-cuda` context, building it once from the
    /// shared CUDA device on first use. Rebuilding this per node reloads the
    /// kernel module, so it is cached here and an `Arc` clone is handed out
    /// (the lock is released before the GEMM runs, so concurrent contractions
    /// are not serialized on it).
    #[cfg(feature = "cuda-tropical")]
    fn tropical_context(&self) -> Arc<tropical_gemm_cuda::CudaContext> {
        let mut guard = self.tropical_ctx.lock().unwrap();
        if guard.is_none() {
            let tctx = tropical_gemm_cuda::CudaContext::from_device(self.ctx.clone())
                .expect("failed to build tropical-gemm-cuda context from the shared CUDA device");
            *guard = Some(Arc::new(tctx));
        }
        guard.as_ref().unwrap().clone()
    }

    /// Load a device gather kernel by name, compiling and caching the NVRTC
    /// module ([`PERMUTE_KERNEL_SRC`]) once on first use.
    #[cfg(feature = "cuda-tropical")]
    fn permute_function(&self, name: &str) -> cudarc::driver::CudaFunction {
        let mut guard = self.permute_module.lock().unwrap();
        if guard.is_none() {
            let ptx = cudarc::nvrtc::compile_ptx(PERMUTE_KERNEL_SRC)
                .expect("compile device permute kernels");
            let module = self
                .ctx
                .load_module(ptx)
                .expect("load device permute module");
            *guard = Some(module);
        }
        guard
            .as_ref()
            .unwrap()
            .load_function(name)
            .expect("load device gather function")
    }

    /// Device counterpart of `materialize_strided`: gather `input` (a column-major
    /// buffer with element strides `src_strides[ax]` along output axis `ax`) into a
    /// fresh contiguous column-major buffer of shape `new_shape`. Used for operand
    /// canonicalization and output permutation without leaving the GPU. The kernel
    /// is selected by element width (f32/u32 → `gather32`, f64 → `gather64`).
    #[cfg(feature = "cuda-tropical")]
    fn device_gather<T>(
        &self,
        input: &cudarc::driver::CudaSlice<T>,
        new_shape: &[usize],
        src_strides: &[usize],
    ) -> cudarc::driver::CudaSlice<T>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default + Clone,
    {
        use cudarc::driver::PushKernelArg;

        let numel: usize = new_shape.iter().product();
        // Diagnostic only (env-gated, zero cost when off): record the executed
        // gather's (new_shape, src_strides) to study whether fusing the gather into
        // the GEMM would keep coalesced loads (inner axis src-stride 1) — see B9.
        record_gather_stat(new_shape, src_strides, numel);
        // Uninitialized: the gather kernel writes out[o] for every o in 0..numel
        // (grid sized via `for_num_elems(numel)`, with an `o >= numel` guard), so
        // every element of the output is overwritten before any reader — zeroing
        // it would be wasted work (this is the per-node gather memset that, after
        // the batched-GEMM fix, became the dominant residual). The numel==0 guard
        // returns the max(1) placeholder unread.
        let mut out =
            unsafe { self.stream.alloc::<T>(numel.max(1)) }.expect("alloc device gather output");
        if numel == 0 {
            return out;
        }

        // One combined upload: [shape (ndim) ‖ strides (ndim)].
        let mut combined: Vec<i64> = Vec::with_capacity(2 * new_shape.len());
        combined.extend(new_shape.iter().map(|&x| x as i64));
        combined.extend(src_strides.iter().map(|&x| x as i64));
        let ndim_us = new_shape.len();
        let ndim = ndim_us as i32;
        let numel_i64 = numel as i64;

        let fname = match std::mem::size_of::<T>() {
            4 => "gather32",
            8 => "gather64",
            other => panic!("device_gather: unsupported element width {other} bytes"),
        };
        let func = self.permute_function(fname);
        // `numel` can exceed u32::MAX: a 2^32-element gather (sc-target 32) would
        // make `numel as u32` wrap to 0 -> grid dim 0 -> CUDA_ERROR_INVALID_VALUE.
        // Compute the 1-D grid in 64-bit (the kernel already does 64-bit global
        // indexing, see gather32/64), matching cudarc's block=1024 layout.
        let cfg = gather_launch_config(numel);

        // Reuse per-ndim metadata buffers: allocate one (shape, strides) pair per
        // distinct rank on first use, then re-upload into them with `memcpy_htod`
        // instead of `clone_htod`-allocating a fresh pair every gather. The lock
        // is held through the launch so a concurrent slice can't overwrite the
        // shared buffers between this gather's upload and its kernel (they
        // serialize on the single stream anyway); the buffers live in the map, so
        // the enqueued kernel never reads a freed allocation.
        let mut meta = self.gather_meta.lock().unwrap();
        if !meta.contains_key(&ndim_us) {
            let buf = self
                .stream
                .alloc_zeros::<i64>(2 * ndim_us)
                .expect("alloc gather metadata scratch");
            meta.insert(ndim_us, buf);
        }
        let d_combined = meta.get_mut(&ndim_us).unwrap();
        self.stream
            .memcpy_htod(&combined, d_combined)
            .expect("upload gather metadata");
        // Two non-overlapping sub-views of the one buffer: shape then strides.
        let d_shape = d_combined.slice(0..ndim_us);
        let d_strides = d_combined.slice(ndim_us..2 * ndim_us);

        let mut builder = self.stream.launch_builder(&func);
        builder
            .arg(&mut out)
            .arg(input)
            .arg(&ndim)
            .arg(&d_shape)
            .arg(&d_strides)
            .arg(&numel_i64);
        unsafe { builder.launch(cfg) }.expect("launch device gather kernel");
        drop(meta);
        out
    }

    /// Batched counterpart of [`device_gather`]: perform `reqs.len()` independent
    /// strided gathers in a **single** kernel launch, returning one fresh
    /// contiguous output buffer per request. This is the grouped/segmented
    /// batched pattern (cuBLAS grouped-GEMM / MAGMA vbatch): the operands are
    /// passed as an array of device pointers (like `cublas*gemmBatched`) and the
    /// flattened output work range is split by a prefix sum, so one launch
    /// replaces N per-node gather launches — the lever against the launch-bound
    /// dispatch cost. Each request is `(input, new_shape, src_strides)` with the
    /// same column-major strided-view semantics as [`device_gather`]; outputs are
    /// allocated uninitialised (the kernel writes every element). Works for
    /// arbitrary dimensions; element width (32/64-bit) selected by `T` as above.
    ///
    /// Kept (and unit-tested) as a primitive for a future large-`k` cross-node
    /// batching: at the per-node k=2 granularity it was a measured regression
    /// (its metadata uploads outweigh the single saved launch — B9), so the
    /// contraction path does not call it yet.
    #[cfg(feature = "cuda-tropical")]
    #[allow(dead_code)]
    fn device_gather_batched<T>(
        &self,
        reqs: &[(&cudarc::driver::CudaSlice<T>, &[usize], &[usize])],
    ) -> Vec<cudarc::driver::CudaSlice<T>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default + Clone,
    {
        use cudarc::driver::{DevicePtr, DevicePtrMut, PushKernelArg};

        let k = reqs.len();
        let numels: Vec<usize> = reqs.iter().map(|(_, sh, _)| sh.iter().product()).collect();
        let mut outs: Vec<cudarc::driver::CudaSlice<T>> = numels
            .iter()
            .map(|&n| {
                unsafe { self.stream.alloc::<T>(n.max(1)) }.expect("alloc batched gather output")
            })
            .collect();

        // Prefix sum of output element counts (k+1 entries); total = prefix[k].
        let mut prefix: Vec<i64> = Vec::with_capacity(k + 1);
        prefix.push(0);
        for &n in &numels {
            let last = *prefix.last().unwrap();
            prefix.push(last + n as i64);
        }
        let total = *prefix.last().unwrap();
        if total == 0 {
            return outs;
        }

        // Pack per-gather metadata [ndim, shape.., strides..] with an offset table.
        let mut meta: Vec<i64> = Vec::new();
        let mut meta_off: Vec<i64> = Vec::with_capacity(k);
        for (_, sh, ss) in reqs {
            meta_off.push(meta.len() as i64);
            meta.push(sh.len() as i64);
            meta.extend(sh.iter().map(|&x| x as i64));
            meta.extend(ss.iter().map(|&x| x as i64));
        }

        // Array-of-pointers: collect each operand's device address. The Record
        // guards (cheap event-record on drop, not a stream sync) are held until
        // after the launch is enqueued.
        let mut in_guards = Vec::with_capacity(k);
        let mut out_guards = Vec::with_capacity(k);
        let mut in_ptrs: Vec<u64> = Vec::with_capacity(k);
        let mut out_ptrs: Vec<u64> = Vec::with_capacity(k);
        for ((inp, _, _), out) in reqs.iter().zip(outs.iter_mut()) {
            let (ip, ig) = inp.device_ptr(&self.stream);
            in_ptrs.push(ip);
            in_guards.push(ig);
            let (op, og) = out.device_ptr_mut(&self.stream);
            out_ptrs.push(op);
            out_guards.push(og);
        }

        let d_in = self
            .stream
            .clone_htod(&in_ptrs)
            .expect("upload gather in ptrs");
        let d_out = self
            .stream
            .clone_htod(&out_ptrs)
            .expect("upload gather out ptrs");
        let d_meta = self.stream.clone_htod(&meta).expect("upload gather meta");
        let d_off = self
            .stream
            .clone_htod(&meta_off)
            .expect("upload gather meta_off");
        let d_prefix = self
            .stream
            .clone_htod(&prefix)
            .expect("upload gather prefix");

        let fname = match std::mem::size_of::<T>() {
            4 => "gather_batched32",
            8 => "gather_batched64",
            other => panic!("device_gather_batched: unsupported element width {other} bytes"),
        };
        let func = self.permute_function(fname);
        // See `gather_launch_config`: `total` can exceed u32::MAX at large scale,
        // and `total as u32` would wrap (the batched kernel grid-stride-loops, so
        // it only needs a non-zero grid, but compute it in 64-bit regardless).
        let cfg = gather_launch_config(total as usize);
        let k_i32 = k as i32;
        let mut builder = self.stream.launch_builder(&func);
        builder
            .arg(&d_out)
            .arg(&d_in)
            .arg(&d_meta)
            .arg(&d_off)
            .arg(&d_prefix)
            .arg(&k_i32)
            .arg(&total);
        unsafe { builder.launch(cfg) }.expect("launch batched gather kernel");
        drop(in_guards);
        drop(out_guards);
        outs
    }

    /// Perform a tensor contraction using cuTENSOR.
    ///
    /// Computes: C = A * B, where the contraction is specified by mode indices.
    ///
    /// # Arguments
    /// * `a` - Input tensor A storage
    /// * `shape_a` - Shape (extents) of tensor A
    /// * `strides_a` - Strides of tensor A
    /// * `modes_a` - Mode indices for tensor A
    /// * `b` - Input tensor B storage
    /// * `shape_b` - Shape (extents) of tensor B
    /// * `strides_b` - Strides of tensor B
    /// * `modes_b` - Mode indices for tensor B
    /// * `shape_c` - Shape (extents) of output tensor C
    /// * `strides_c` - Strides of output tensor C
    /// * `modes_c` - Mode indices for output tensor C
    ///
    /// # Returns
    /// * `Ok(CudaStorage<T>)` containing the contraction result
    /// * `Err(CudaError)` if the contraction fails
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn contract_cutensor<T>(
        &self,
        a: &CudaStorage<T>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &CudaStorage<T>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        strides_c: &[usize],
        modes_c: &[i32],
    ) -> Result<CudaStorage<T>, CudaError>
    where
        T: CutensorType
            + cudarc::driver::DeviceRepr
            + cudarc::driver::ValidAsZeroBits
            + num_traits::One
            + num_traits::Zero,
    {
        // Allocate output storage first (outside of locks)
        let len: usize = shape_c.iter().product();
        let mut c = self
            .stream
            .alloc_zeros::<T>(len)
            .map_err(|e| CudaError::Alloc(e.to_string()))?;

        // Build cache key
        let key = CacheKey {
            shapes: vec![shape_a.to_vec(), shape_b.to_vec(), shape_c.to_vec()],
            strides: vec![strides_a.to_vec(), strides_b.to_vec(), strides_c.to_vec()],
            modes: vec![modes_a.to_vec(), modes_b.to_vec(), modes_c.to_vec()],
            dtype: T::DATA as u32,
        };

        // Do all cuTENSOR operations with both locks held
        self.with_handle(|handle| {
            // Create tensor descriptors
            let desc_a = TensorDesc::new::<T>(handle, shape_a, strides_a)
                .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;
            let desc_b = TensorDesc::new::<T>(handle, shape_b, strides_b)
                .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;
            let desc_c = TensorDesc::new::<T>(handle, shape_c, strides_c)
                .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;

            // Get or create the execution plan from cache and execute contraction
            let mut cache = self.cache.lock().unwrap();
            let plan = cache
                .get_or_create::<T>(
                    handle, key, &desc_a, modes_a, &desc_b, modes_b, &desc_c, modes_c,
                )
                .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;

            // Execute the contraction
            contract::<T>(handle, plan, T::one(), a.slice(), b.slice(), &mut c)
                .map_err(|e| CudaError::Cutensor(format!("{}", e)))?;

            Ok(())
        })?;

        Ok(CudaStorage::new(c, self.stream.clone()))
    }

    /// Compute column-major strides for a given shape.
    // Used by the cuTENSOR path and (Phase 2) the tropical executor.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = 1;
        for &dim in shape {
            strides.push(stride);
            stride *= dim;
        }
        strides
    }

    /// Tropical contraction on GPU via `tropical-gemm-cuda`.
    ///
    /// cuTENSOR has no tropical semiring, so max-plus / min-plus / max-mul
    /// contractions are routed here. Reuses the backend-neutral
    /// [`crate::backend::contract_plan`] planner to reduce the contraction to a
    /// (batched) matmul, lays operands out in the canonical column-major
    /// `[left, contracted, batch]` / `[contracted, right, batch]` layout, then
    /// runs the tropical GEMM kernel on the **shared** device (one
    /// `CudaContext::from_device`, no duplicate driver context).
    ///
    /// Layout note: unlike the CPU `try_tropical_gemm` (which feeds column-major
    /// bytes to a *row-major* `tropical_matmul` and therefore swaps a↔b / m↔n),
    /// `tropical-gemm-cuda`'s `GpuMatrix` + `tropical_gemm_gpu` are
    /// **column-major** — the same order omeinsum uses — so operands are passed
    /// straight through with no swap.
    ///
    /// Operand canonicalization (gather of strided views and the canonical
    /// permutation) and the output permute run **on the device** via the
    /// [`Cuda::device_gather`] NVRTC kernel, so the whole contraction stays
    /// GPU-resident with no host roundtrip ([`Cuda::contract_tropical_device`]).
    /// The one exception is trace modes (repeated labels needing a semiring
    /// pre-reduction the gather kernel doesn't perform): those fall back to the
    /// host operand-prep path ([`Cuda::plan_tropical_operands`]).
    #[cfg(feature = "cuda-tropical")]
    #[allow(clippy::too_many_arguments)]
    fn contract_tropical<A: Algebra>(
        &self,
        a: &CudaStorage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &CudaStorage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> CudaStorage<A::Scalar>
    where
        A::Scalar: BackendScalar<Self>,
    {
        let plan = crate::backend::contract_plan::plan_contraction(
            modes_a, shape_a, modes_b, shape_b, modes_c,
        );
        if !plan.has_trace() {
            // Device-resident fast path: canonicalize operands and permute the
            // result entirely on the GPU (no host roundtrip).
            return self.contract_tropical_device::<A>(
                &plan, a, shape_a, strides_a, modes_a, b, shape_b, strides_b, modes_b, shape_c,
                modes_c,
            );
        }
        // Trace fallback: repeated-label modes need a semiring (max/min)
        // pre-reduction that the device gather kernel doesn't perform, so prep
        // operands on the host (download → reduce → permute → upload).
        let (plan, a_canon, b_canon) = self.plan_tropical_operands::<A>(
            a, shape_a, strides_a, modes_a, b, shape_b, strides_b, modes_b, modes_c,
        );
        let batch = plan.batch_size.max(1);
        let c_canon = self.run_tropical_gemm::<A>(
            &a_canon,
            &b_canon,
            batch,
            plan.left_size,
            plan.contract_size,
            plan.right_size,
        );
        let c_final = permute_tropical_output(c_canon, &plan, shape_c, modes_c);
        self.from_slice(&c_final)
    }

    /// Device-resident no-trace tropical contraction: gather each operand into
    /// its canonical column-major layout with [`Cuda::device_gather`], run the
    /// device GEMM core, and permute the result back to `modes_c` — all on the
    /// GPU, returning device storage with no host transfer. Caller guarantees
    /// `!plan.has_trace()`.
    #[cfg(feature = "cuda-tropical")]
    #[allow(clippy::too_many_arguments)]
    fn contract_tropical_device<A: Algebra>(
        &self,
        plan: &crate::backend::contract_plan::ContractionPlan,
        a: &CudaStorage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &CudaStorage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> CudaStorage<A::Scalar>
    where
        A::Scalar: BackendScalar<Self>,
    {
        use crate::algebra::{MaxMul, MaxPlus, MinPlus};
        use std::any::TypeId;
        use tropical_gemm::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus};

        let (a_new_shape, a_src_strides) =
            canonical_gather_args(plan.a_permutation(modes_a).as_slice(), shape_a, strides_a);
        let (b_new_shape, b_src_strides) =
            canonical_gather_args(plan.b_permutation(modes_b).as_slice(), shape_b, strides_b);
        let out_gather = output_gather_args(plan, shape_c, modes_c);
        let batch = plan.batch_size.max(1);
        let (left, contract, right) = (plan.left_size, plan.contract_size, plan.right_size);

        let tctx = self.tropical_context();
        let stream = &self.stream;

        macro_rules! dispatch {
            ($kernel:ty, $scalar:ty) => {{
                let a_slice: &cudarc::driver::CudaSlice<$scalar> =
                    unsafe { std::mem::transmute(a.slice()) };
                let b_slice: &cudarc::driver::CudaSlice<$scalar> =
                    unsafe { std::mem::transmute(b.slice()) };
                // Skip the operand gather when it would copy the input verbatim
                // (canonical layout already): one fewer launch + alloc per node.
                // NB: batching the A+B pair into one launch (device_gather_batched)
                // was a measured net regression at this k=2 granularity — its extra
                // metadata uploads outweigh the single saved launch (B9 Log). The
                // batched primitive is kept for a future large-k (cross-node) use.
                let a_numel: usize = a_new_shape.iter().product();
                let a_canon = if is_identity_gather(&a_new_shape, &a_src_strides)
                    && a_slice.len() == a_numel
                {
                    None
                } else {
                    Some(self.device_gather::<$scalar>(a_slice, &a_new_shape, &a_src_strides))
                };
                let b_numel: usize = b_new_shape.iter().product();
                let b_canon = if is_identity_gather(&b_new_shape, &b_src_strides)
                    && b_slice.len() == b_numel
                {
                    None
                } else {
                    Some(self.device_gather::<$scalar>(b_slice, &b_new_shape, &b_src_strides))
                };
                let c_canon = batched_tropical_gemm_dev::<$kernel>(
                    &tctx,
                    stream,
                    a_canon.as_ref().unwrap_or(a_slice),
                    b_canon.as_ref().unwrap_or(b_slice),
                    batch,
                    left,
                    contract,
                    right,
                );
                let c_final = match &out_gather {
                    None => c_canon,
                    Some((ns, ss)) => self.device_gather::<$scalar>(&c_canon, ns, ss),
                };
                let c_storage: cudarc::driver::CudaSlice<A::Scalar> =
                    unsafe { std::mem::transmute(c_final) };
                CudaStorage::new(c_storage, self.stream.clone())
            }};
        }

        if TypeId::of::<A>() == TypeId::of::<MaxPlus<f32>>() {
            dispatch!(TropicalMaxPlus<f32>, f32)
        } else if TypeId::of::<A>() == TypeId::of::<MaxPlus<f64>>() {
            dispatch!(TropicalMaxPlus<f64>, f64)
        } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f32>>() {
            dispatch!(TropicalMinPlus<f32>, f32)
        } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f64>>() {
            dispatch!(TropicalMinPlus<f64>, f64)
        } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f32>>() {
            dispatch!(TropicalMaxMul<f32>, f32)
        } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f64>>() {
            dispatch!(TropicalMaxMul<f64>, f64)
        } else {
            panic!(
                "CUDA tropical contraction is only implemented for MaxPlus/MinPlus/MaxMul \
                 over f32/f64; got algebra {:?}",
                std::any::type_name::<A>()
            );
        }
    }

    /// Tropical forward contraction with argmax tracking (winner `k`-index per
    /// output element), the GPU counterpart of the CPU
    /// [`contract::contract_with_argmax`](crate::backend::cpu::contract).
    ///
    /// Shares the exact operand preparation of [`Cuda::contract_tropical`] — the
    /// same trace reduction and the same canonical permutation (whose contracted
    /// mode order comes straight from `classify_modes`, identical to the CPU
    /// path) — so the emitted `argmax` linearizes the contracted modes in the
    /// order [`crate::einsum::backward`] expects when it decodes the winner via
    /// `linear_to_coords(k, contracted_shape)`. The argmax buffer is permuted
    /// alongside the result (it has the same `[left, right, batch]` shape).
    #[cfg(feature = "cuda-tropical")]
    #[allow(clippy::too_many_arguments)]
    fn contract_tropical_with_argmax<A: Algebra<Index = u32>>(
        &self,
        a: &CudaStorage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &CudaStorage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> (CudaStorage<A::Scalar>, CudaStorage<u32>)
    where
        A::Scalar: BackendScalar<Self>,
    {
        let plan = crate::backend::contract_plan::plan_contraction(
            modes_a, shape_a, modes_b, shape_b, modes_c,
        );
        if !plan.has_trace() {
            return self.contract_tropical_with_argmax_device::<A>(
                &plan, a, shape_a, strides_a, modes_a, b, shape_b, strides_b, modes_b, shape_c,
                modes_c,
            );
        }
        // Trace fallback (host operand prep); see `contract_tropical`.
        let (plan, a_canon, b_canon) = self.plan_tropical_operands::<A>(
            a, shape_a, strides_a, modes_a, b, shape_b, strides_b, modes_b, modes_c,
        );
        let batch = plan.batch_size.max(1);
        let (c_canon, argmax_canon) = self.run_tropical_gemm_with_argmax::<A>(
            &a_canon,
            &b_canon,
            batch,
            plan.left_size,
            plan.contract_size,
            plan.right_size,
        );
        let c_final = permute_tropical_output(c_canon, &plan, shape_c, modes_c);
        let argmax_final = permute_tropical_output(argmax_canon, &plan, shape_c, modes_c);
        (self.from_slice(&c_final), self.from_slice(&argmax_final))
    }

    /// Device-resident no-trace counterpart of [`Cuda::contract_tropical_device`]
    /// with argmax tracking. The argmax buffer is permuted by the **same**
    /// `output_perm` as the result (via `gather32`, since it is `u32`), so the two
    /// stay index-consistent for `backward`. Caller guarantees `!plan.has_trace()`.
    #[cfg(feature = "cuda-tropical")]
    #[allow(clippy::too_many_arguments)]
    fn contract_tropical_with_argmax_device<A: Algebra<Index = u32>>(
        &self,
        plan: &crate::backend::contract_plan::ContractionPlan,
        a: &CudaStorage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &CudaStorage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> (CudaStorage<A::Scalar>, CudaStorage<u32>)
    where
        A::Scalar: BackendScalar<Self>,
    {
        use crate::algebra::{MaxMul, MaxPlus, MinPlus};
        use std::any::TypeId;
        use tropical_gemm::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus};

        let (a_new_shape, a_src_strides) =
            canonical_gather_args(plan.a_permutation(modes_a).as_slice(), shape_a, strides_a);
        let (b_new_shape, b_src_strides) =
            canonical_gather_args(plan.b_permutation(modes_b).as_slice(), shape_b, strides_b);
        let out_gather = output_gather_args(plan, shape_c, modes_c);
        let batch = plan.batch_size.max(1);
        let (left, contract, right) = (plan.left_size, plan.contract_size, plan.right_size);

        let tctx = self.tropical_context();
        let stream = &self.stream;

        macro_rules! dispatch {
            ($kernel:ty, $scalar:ty) => {{
                let a_slice: &cudarc::driver::CudaSlice<$scalar> =
                    unsafe { std::mem::transmute(a.slice()) };
                let b_slice: &cudarc::driver::CudaSlice<$scalar> =
                    unsafe { std::mem::transmute(b.slice()) };
                let a_canon = self.device_gather::<$scalar>(a_slice, &a_new_shape, &a_src_strides);
                let b_canon = self.device_gather::<$scalar>(b_slice, &b_new_shape, &b_src_strides);
                let (c_canon, argmax_canon) = batched_tropical_gemm_dev_with_argmax::<$kernel>(
                    &tctx, stream, &a_canon, &b_canon, batch, left, contract, right,
                );
                let (c_final, argmax_final) = match &out_gather {
                    None => (c_canon, argmax_canon),
                    Some((ns, ss)) => (
                        self.device_gather::<$scalar>(&c_canon, ns, ss),
                        self.device_gather::<u32>(&argmax_canon, ns, ss),
                    ),
                };
                let c_storage: cudarc::driver::CudaSlice<A::Scalar> =
                    unsafe { std::mem::transmute(c_final) };
                (
                    CudaStorage::new(c_storage, self.stream.clone()),
                    CudaStorage::new(argmax_final, self.stream.clone()),
                )
            }};
        }

        if TypeId::of::<A>() == TypeId::of::<MaxPlus<f32>>() {
            dispatch!(TropicalMaxPlus<f32>, f32)
        } else if TypeId::of::<A>() == TypeId::of::<MaxPlus<f64>>() {
            dispatch!(TropicalMaxPlus<f64>, f64)
        } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f32>>() {
            dispatch!(TropicalMinPlus<f32>, f32)
        } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f64>>() {
            dispatch!(TropicalMinPlus<f64>, f64)
        } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f32>>() {
            dispatch!(TropicalMaxMul<f32>, f32)
        } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f64>>() {
            dispatch!(TropicalMaxMul<f64>, f64)
        } else {
            panic!(
                "CUDA tropical argmax contraction is only implemented for \
                 MaxPlus/MinPlus/MaxMul over f32/f64; got algebra {:?}",
                std::any::type_name::<A>()
            );
        }
    }

    /// Download both operands, reduce any trace modes, and permute each into the
    /// canonical column-major matmul layout (`A → [left, contracted, batch]`,
    /// `B → [contracted, right, batch]`). Returns the re-planned
    /// [`ContractionPlan`] (built on the trace-free operands) plus the two
    /// canonical host buffers. Shared by the tropical forward and argmax paths.
    #[cfg(feature = "cuda-tropical")]
    #[allow(clippy::too_many_arguments)]
    fn plan_tropical_operands<A: Algebra>(
        &self,
        a: &CudaStorage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &CudaStorage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        modes_c: &[i32],
    ) -> (
        crate::backend::contract_plan::ContractionPlan,
        Vec<A::Scalar>,
        Vec<A::Scalar>,
    ) {
        use crate::backend::contract_plan::{
            gather_contiguous, materialize_strided, plan_contraction, reduce_trace,
        };
        use crate::tensor::compute_contiguous_strides;

        // 1. Download operands and gather any strided view into contiguous
        //    column-major host buffers.
        let a_contig = gather_contiguous::<A::Scalar>(&a.to_vec(), shape_a, strides_a);
        let b_contig = gather_contiguous::<A::Scalar>(&b.to_vec(), shape_b, strides_b);

        // 2. Reduce trace modes (single-operand modes absent from the output).
        //    These cannot be handled by GEMM and must be summed via the semiring
        //    add *before* the matmul.
        let probe = plan_contraction(modes_a, shape_a, modes_b, shape_b, modes_c);
        let (a_data, a_shape, a_modes) = if probe.left_trace.is_empty() {
            (a_contig, shape_a.to_vec(), modes_a.to_vec())
        } else {
            reduce_trace::<A>(&a_contig, shape_a, modes_a, &probe.left_trace)
        };
        let (b_data, b_shape, b_modes) = if probe.right_trace.is_empty() {
            (b_contig, shape_b.to_vec(), modes_b.to_vec())
        } else {
            reduce_trace::<A>(&b_contig, shape_b, modes_b, &probe.right_trace)
        };

        // 3. Re-plan on the (now trace-free) reduced operands and permute each
        //    into the canonical matmul layout.
        let plan = plan_contraction(&a_modes, &a_shape, &b_modes, &b_shape, modes_c);
        let a_canon = materialize_strided::<A::Scalar>(
            &a_data,
            &a_shape,
            &compute_contiguous_strides(&a_shape),
            &plan.a_permutation(&a_modes),
        );
        let b_canon = materialize_strided::<A::Scalar>(
            &b_data,
            &b_shape,
            &compute_contiguous_strides(&b_shape),
            &plan.b_permutation(&b_modes),
        );
        (plan, a_canon, b_canon)
    }

    /// Dispatch a canonical column-major (batched) tropical GEMM to the concrete
    /// `tropical-gemm-cuda` kernel for `A`'s semiring × scalar, sharing this
    /// backend's CUDA device.
    ///
    /// `a` is `batch × (m × k)` and `b` is `batch × (k × n)`, both contiguous
    /// column-major; the returned buffer is `batch × (m × n)` in the same order.
    #[cfg(feature = "cuda-tropical")]
    fn run_tropical_gemm<A: Algebra>(
        &self,
        a: &[A::Scalar],
        b: &[A::Scalar],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<A::Scalar> {
        use crate::algebra::{MaxMul, MaxPlus, MinPlus};
        use std::any::TypeId;
        use tropical_gemm::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus};

        let tctx = self.tropical_context();
        let stream = &self.stream;

        // The omeinsum algebra types (`A::Scalar` = f32/f64) and the tropical-gemm
        // kernel scalar types share an identical `repr(transparent)` layout, so the
        // slice/Vec transmutes below are byte-identity reinterpretations (same as
        // the CPU `try_tropical_gemm`).
        macro_rules! dispatch {
            ($kernel:ty, $scalar:ty) => {{
                let a_s: &[$scalar] = unsafe { std::mem::transmute(a) };
                let b_s: &[$scalar] = unsafe { std::mem::transmute(b) };
                let out = batched_tropical_gemm::<$kernel>(&tctx, stream, a_s, b_s, batch, m, k, n);
                unsafe { std::mem::transmute::<Vec<$scalar>, Vec<A::Scalar>>(out) }
            }};
        }

        if TypeId::of::<A>() == TypeId::of::<MaxPlus<f32>>() {
            dispatch!(TropicalMaxPlus<f32>, f32)
        } else if TypeId::of::<A>() == TypeId::of::<MaxPlus<f64>>() {
            dispatch!(TropicalMaxPlus<f64>, f64)
        } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f32>>() {
            dispatch!(TropicalMinPlus<f32>, f32)
        } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f64>>() {
            dispatch!(TropicalMinPlus<f64>, f64)
        } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f32>>() {
            dispatch!(TropicalMaxMul<f32>, f32)
        } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f64>>() {
            dispatch!(TropicalMaxMul<f64>, f64)
        } else {
            panic!(
                "CUDA tropical contraction is only implemented for MaxPlus/MinPlus/MaxMul \
                 over f32/f64; got algebra {:?}",
                std::any::type_name::<A>()
            );
        }
    }

    /// Argmax-tracking counterpart of [`Cuda::run_tropical_gemm`]: dispatches a
    /// canonical column-major (batched) tropical GEMM to the concrete
    /// `tropical-gemm-cuda` *argmax* kernel for `A`'s semiring × scalar, returning
    /// both the result buffer and the winner `k`-index per output element (both
    /// `batch × (m × n)` column-major).
    #[cfg(feature = "cuda-tropical")]
    fn run_tropical_gemm_with_argmax<A: Algebra>(
        &self,
        a: &[A::Scalar],
        b: &[A::Scalar],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> (Vec<A::Scalar>, Vec<u32>) {
        use crate::algebra::{MaxMul, MaxPlus, MinPlus};
        use std::any::TypeId;
        use tropical_gemm::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus};

        let tctx = self.tropical_context();
        let stream = &self.stream;

        // Same `repr(transparent)` byte-identity transmutes as `run_tropical_gemm`;
        // the argmax buffer is `u32` on both sides, so it needs no reinterpretation.
        macro_rules! dispatch {
            ($kernel:ty, $scalar:ty) => {{
                let a_s: &[$scalar] = unsafe { std::mem::transmute(a) };
                let b_s: &[$scalar] = unsafe { std::mem::transmute(b) };
                let (out, argmax) = batched_tropical_gemm_with_argmax::<$kernel>(
                    &tctx, stream, a_s, b_s, batch, m, k, n,
                );
                (
                    unsafe { std::mem::transmute::<Vec<$scalar>, Vec<A::Scalar>>(out) },
                    argmax,
                )
            }};
        }

        if TypeId::of::<A>() == TypeId::of::<MaxPlus<f32>>() {
            dispatch!(TropicalMaxPlus<f32>, f32)
        } else if TypeId::of::<A>() == TypeId::of::<MaxPlus<f64>>() {
            dispatch!(TropicalMaxPlus<f64>, f64)
        } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f32>>() {
            dispatch!(TropicalMinPlus<f32>, f32)
        } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f64>>() {
            dispatch!(TropicalMinPlus<f64>, f64)
        } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f32>>() {
            dispatch!(TropicalMaxMul<f32>, f32)
        } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f64>>() {
            dispatch!(TropicalMaxMul<f64>, f64)
        } else {
            panic!(
                "CUDA tropical argmax contraction is only implemented for \
                 MaxPlus/MinPlus/MaxMul over f32/f64; got algebra {:?}",
                std::any::type_name::<A>()
            );
        }
    }
}

/// `(new_shape, src_strides)` for a [`Cuda::device_gather`] that applies `perm`
/// to a column-major operand of shape `shape` with element strides `strides`:
/// `new_shape[i] = shape[perm[i]]`, `src_strides[i] = strides[perm[i]]`. This is
/// the device-side counterpart of the host `materialize_strided`'s indexing.
#[cfg(feature = "cuda-tropical")]
/// Diagnostic (B9): when `MISO_GATHER_STATS` is set to a file path, accumulate a
/// histogram of every executed device gather's `(new_shape, src_strides)` and
/// rewrite the file (whole histogram, sorted) whenever a new pattern appears or
/// every 4096 calls — so the final state after a run is complete. Records
/// `inner_contig` = whether the innermost (fastest-varying, output-stride-1) axis
/// is also stride-1 in the *source* (`src_strides[0] == 1`), the make-or-break
/// condition for fusing the gather into the GEMM with coalesced loads. Zero cost
/// when the env var is unset (early return before any locking).
#[cfg(feature = "cuda-tropical")]
fn record_gather_stat(new_shape: &[usize], src_strides: &[usize], numel: usize) {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static STATS: OnceLock<Mutex<(HashMap<(Vec<usize>, Vec<usize>), (u64, u64)>, u64)>> =
        OnceLock::new();
    let path = match std::env::var("MISO_GATHER_STATS") {
        Ok(p) => p,
        Err(_) => return,
    };
    let cell = STATS.get_or_init(|| Mutex::new((HashMap::new(), 0)));
    let mut g = cell.lock().unwrap();
    let key = (new_shape.to_vec(), src_strides.to_vec());
    let is_new = !g.0.contains_key(&key);
    let e = g.0.entry(key).or_insert((0, 0));
    e.0 += 1;
    e.1 += numel as u64;
    g.1 += 1;
    if is_new || g.1 % 4096 == 0 {
        let mut lines: Vec<String> = g
            .0
            .iter()
            .map(|((ns, ss), (cnt, nm))| {
                let inner_contig = ss.first() == Some(&1);
                format!(
                    "count={cnt}\tnumel_sum={nm}\tinner_contig={inner_contig}\tndim={}\tnew_shape={ns:?}\tsrc_strides={ss:?}",
                    ns.len()
                )
            })
            .collect();
        lines.sort();
        let total_calls = g.1;
        let header = format!(
            "# total_gather_calls={total_calls} unique_patterns={}",
            g.0.len()
        );
        let _ = std::fs::write(&path, format!("{header}\n{}\n", lines.join("\n")));
    }
}

fn canonical_gather_args(
    perm: &[usize],
    shape: &[usize],
    strides: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let new_shape = perm.iter().map(|&p| shape[p]).collect();
    let src_strides = perm.iter().map(|&p| strides[p]).collect();
    (new_shape, src_strides)
}

/// True when a `device_gather(new_shape, src_strides)` would copy its input
/// verbatim — i.e. `src_strides` already equals the contiguous column-major
/// strides of `new_shape`, so `out[o] == in[o]` for every element. In that case
/// the gather is a pure no-op copy and the caller can feed the input slice
/// straight to the GEMM, skipping one kernel launch (and one output alloc) per
/// skipped operand. This is the common case for operands already laid out in
/// contraction-canonical order. The caller must additionally check that the
/// input length equals the gather's element count, since a no-op gather over a
/// *prefix* of a longer buffer still differs from passing the whole buffer.
#[cfg(feature = "cuda-tropical")]
fn is_identity_gather(new_shape: &[usize], src_strides: &[usize]) -> bool {
    use crate::tensor::compute_contiguous_strides;
    src_strides == compute_contiguous_strides(new_shape).as_slice()
}

/// `(new_shape, src_strides)` for the device output permute, or `None` when the
/// canonical `[left, right, batch]` order already equals `modes_c`. Matches
/// [`permute_tropical_output`]: the GEMM result is contiguous column-major over
/// `current_order = left ++ right ++ batch` modes, permuted by `output_perm`.
#[cfg(feature = "cuda-tropical")]
fn output_gather_args(
    plan: &crate::backend::contract_plan::ContractionPlan,
    shape_c: &[usize],
    modes_c: &[i32],
) -> Option<(Vec<usize>, Vec<usize>)> {
    use crate::backend::contract_plan::mode_position;
    use crate::tensor::compute_contiguous_strides;

    plan.output_perm.as_ref().map(|out_perm| {
        let current_order: Vec<i32> = plan
            .left_modes
            .iter()
            .chain(plan.right_modes.iter())
            .chain(plan.batch_modes.iter())
            .copied()
            .collect();
        let c_shape_current: Vec<usize> = current_order
            .iter()
            .map(|&m| shape_c[mode_position(modes_c, m)])
            .collect();
        let contig = compute_contiguous_strides(&c_shape_current);
        let new_shape = out_perm.iter().map(|&p| c_shape_current[p]).collect();
        let src_strides = out_perm.iter().map(|&p| contig[p]).collect();
        (new_shape, src_strides)
    })
}

/// Permute a GEMM output buffer (column-major `[left, right, batch]`) back to the
/// requested `modes_c` order. Used for both the result and the argmax buffers
/// (which share the output shape), so it is generic over the element type.
#[cfg(feature = "cuda-tropical")]
fn permute_tropical_output<T: Copy + Default>(
    canonical: Vec<T>,
    plan: &crate::backend::contract_plan::ContractionPlan,
    shape_c: &[usize],
    modes_c: &[i32],
) -> Vec<T> {
    use crate::backend::contract_plan::{materialize_strided, mode_position};
    use crate::tensor::compute_contiguous_strides;

    match &plan.output_perm {
        None => canonical,
        Some(out_perm) => {
            let current_order: Vec<i32> = plan
                .left_modes
                .iter()
                .chain(plan.right_modes.iter())
                .chain(plan.batch_modes.iter())
                .copied()
                .collect();
            let c_shape_current: Vec<usize> = current_order
                .iter()
                .map(|&m| shape_c[mode_position(modes_c, m)])
                .collect();
            materialize_strided::<T>(
                &canonical,
                &c_shape_current,
                &compute_contiguous_strides(&c_shape_current),
                out_perm,
            )
        }
    }
}

/// Device-resident core of the canonical column-major (batched) tropical GEMM:
/// operands are already on the GPU, and the result stays on the GPU. The whole
/// batch runs in a *single* strided-batched kernel launch
/// ([`CudaKernel::launch_gemm_batched`], `blockIdx.z` = batch element) directly
/// over the contiguous `a_dev`/`b_dev` buffers — no per-slice `clone_dtod`, no
/// per-slice allocation, no reassembly `memcpy_dtod`. The output is allocated
/// *uninitialized* (`alloc`, not `alloc_zeros`): the GEMM kernel fully writes
/// every `batch·m·n` element, so zeroing it would be wasted work (≈16 GB of
/// memset on the large KSG networks). `K` selects the concrete semiring ×
/// scalar kernel. Shared by the device contraction path and the host-buffer
/// wrapper below.
#[cfg(feature = "cuda-tropical")]
#[allow(clippy::too_many_arguments)]
fn batched_tropical_gemm_dev<K>(
    ctx: &tropical_gemm_cuda::CudaContext,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    a_dev: &cudarc::driver::CudaSlice<K::Scalar>,
    b_dev: &cudarc::driver::CudaSlice<K::Scalar>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> cudarc::driver::CudaSlice<K::Scalar>
where
    K: tropical_gemm_cuda::CudaKernel,
    K::Scalar: cudarc::driver::DeviceRepr + Default + Clone + cudarc::driver::ValidAsZeroBits,
{
    let c_stride = m * n;
    // Uninitialized: the batched kernel writes every element of C. Safe because
    // the launch is enqueued on the same stream before any reader (stream
    // ordering), and the only host read goes through a synchronizing download.
    let mut c_dev = unsafe { stream.alloc::<K::Scalar>(batch * c_stride) }
        .expect("alloc tropical GEMM batch C");
    K::launch_gemm_batched(ctx, a_dev, b_dev, &mut c_dev, batch, m, k, n)
        .expect("batched tropical GEMM kernel");
    c_dev
}

/// Argmax-tracking device-resident core (see [`batched_tropical_gemm_dev`]).
/// Returns the result and the argmax `k`-indices, both still on the GPU.
#[cfg(feature = "cuda-tropical")]
#[allow(clippy::too_many_arguments)]
fn batched_tropical_gemm_dev_with_argmax<K>(
    ctx: &tropical_gemm_cuda::CudaContext,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    a_dev: &cudarc::driver::CudaSlice<K::Scalar>,
    b_dev: &cudarc::driver::CudaSlice<K::Scalar>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> (
    cudarc::driver::CudaSlice<K::Scalar>,
    cudarc::driver::CudaSlice<u32>,
)
where
    K: tropical_gemm_cuda::CudaKernelWithArgmax,
    K::Scalar: cudarc::driver::DeviceRepr + Default + Clone + cudarc::driver::ValidAsZeroBits,
{
    use tropical_gemm_cuda::{tropical_gemm_gpu_with_argmax, GpuMatrix, GpuMatrixWithArgmax};

    let (a_stride, b_stride, c_stride) = (m * k, k * n, m * n);
    let mut c_dev = stream
        .alloc_zeros::<K::Scalar>(batch * c_stride)
        .expect("alloc tropical GEMM batch C");
    let mut argmax_dev = stream
        .alloc_zeros::<u32>(batch * c_stride)
        .expect("alloc tropical GEMM batch argmax");

    for i in 0..batch {
        let a_i = stream
            .clone_dtod(&a_dev.slice(i * a_stride..(i + 1) * a_stride))
            .expect("slice tropical GEMM operand A");
        let b_i = stream
            .clone_dtod(&b_dev.slice(i * b_stride..(i + 1) * b_stride))
            .expect("slice tropical GEMM operand B");
        let a_gpu =
            GpuMatrix::from_cuda_slice(ctx, a_i, m, k).expect("wrap tropical GEMM operand A");
        let b_gpu =
            GpuMatrix::from_cuda_slice(ctx, b_i, k, n).expect("wrap tropical GEMM operand B");
        let mut c_gpu = GpuMatrixWithArgmax::alloc(ctx, m, n).expect("alloc tropical GEMM output");
        tropical_gemm_gpu_with_argmax::<K>(ctx, &a_gpu, &b_gpu, &mut c_gpu)
            .expect("tropical GEMM argmax kernel");
        let (mat, arg) = c_gpu.into_parts();
        let c_i = mat.into_inner();
        let arg_i = arg.into_inner();
        let mut c_dst = c_dev
            .try_slice_mut(i * c_stride..(i + 1) * c_stride)
            .expect("slice tropical GEMM output");
        stream
            .memcpy_dtod(&c_i, &mut c_dst)
            .expect("assemble tropical GEMM output on device");
        let mut arg_dst = argmax_dev
            .try_slice_mut(i * c_stride..(i + 1) * c_stride)
            .expect("slice tropical GEMM argmax");
        stream
            .memcpy_dtod(&arg_i, &mut arg_dst)
            .expect("assemble tropical GEMM argmax on device");
    }
    (c_dev, argmax_dev)
}

/// Host-buffer wrapper over [`batched_tropical_gemm_dev`]: bulk-upload the batch
/// once, run the device core, bulk-download. Used by the trace fallback path,
/// whose operand prep still happens on the host.
#[cfg(feature = "cuda-tropical")]
#[allow(clippy::too_many_arguments)]
fn batched_tropical_gemm<K>(
    ctx: &tropical_gemm_cuda::CudaContext,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    a: &[K::Scalar],
    b: &[K::Scalar],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<K::Scalar>
where
    K: tropical_gemm_cuda::CudaKernel,
    K::Scalar: cudarc::driver::DeviceRepr + Default + Clone + cudarc::driver::ValidAsZeroBits,
{
    let a_dev = stream.clone_htod(a).expect("upload tropical GEMM batch A");
    let b_dev = stream.clone_htod(b).expect("upload tropical GEMM batch B");
    let c_dev = batched_tropical_gemm_dev::<K>(ctx, stream, &a_dev, &b_dev, batch, m, k, n);
    stream
        .clone_dtoh(&c_dev)
        .expect("download tropical GEMM batch C")
}

/// Host-buffer wrapper over [`batched_tropical_gemm_dev_with_argmax`].
#[cfg(feature = "cuda-tropical")]
#[allow(clippy::too_many_arguments)]
fn batched_tropical_gemm_with_argmax<K>(
    ctx: &tropical_gemm_cuda::CudaContext,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    a: &[K::Scalar],
    b: &[K::Scalar],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<K::Scalar>, Vec<u32>)
where
    K: tropical_gemm_cuda::CudaKernelWithArgmax,
    K::Scalar: cudarc::driver::DeviceRepr + Default + Clone + cudarc::driver::ValidAsZeroBits,
{
    let a_dev = stream.clone_htod(a).expect("upload tropical GEMM batch A");
    let b_dev = stream.clone_htod(b).expect("upload tropical GEMM batch B");
    let (c_dev, argmax_dev) =
        batched_tropical_gemm_dev_with_argmax::<K>(ctx, stream, &a_dev, &b_dev, batch, m, k, n);
    let out = stream
        .clone_dtoh(&c_dev)
        .expect("download tropical GEMM batch C");
    let argmax = stream
        .clone_dtoh(&argmax_dev)
        .expect("download tropical GEMM batch argmax");
    (out, argmax)
}

/// Errors that can occur during CUDA operations.
#[derive(Debug)]
pub enum CudaError {
    /// Error initializing or accessing the CUDA device.
    Device(String),
    /// Error allocating GPU memory.
    Alloc(String),
    /// Error in cuTENSOR operations.
    Cutensor(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::Device(msg) => write!(f, "CUDA device error: {}", msg),
            CudaError::Alloc(msg) => write!(f, "CUDA allocation error: {}", msg),
            CudaError::Cutensor(msg) => write!(f, "cuTENSOR error: {}", msg),
        }
    }
}

impl std::error::Error for CudaError {}

// ============================================================================
// Storage implementation for CudaStorage
// ============================================================================

// Note: Storage<T> requires Scalar, but actual CUDA operations need CudaScalar.
// We implement Storage<T> for all T: Scalar to satisfy Backend::Storage bounds,
// but use runtime type dispatch for the actual operations.

impl<T: Scalar> Storage<T> for CudaStorage<T> {
    fn len(&self) -> usize {
        self.slice().len()
    }

    fn get(&self, index: usize) -> T {
        let buf = self.to_vec();
        buf[index]
    }

    fn set(&mut self, index: usize, value: T) {
        use std::any::TypeId;
        // Download, modify, upload - slow but correct
        let mut buf = self.to_vec();
        buf[index] = value;
        // Re-upload via type dispatch
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let buf_f32: Vec<f32> = unsafe { std::mem::transmute(buf) };
            let new_slice = self
                .stream()
                .clone_htod(&buf_f32)
                .expect("Failed to upload");
            *self = CudaStorage::new(
                unsafe { std::mem::transmute(new_slice) },
                self.stream().clone(),
            );
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let buf_f64: Vec<f64> = unsafe { std::mem::transmute(buf) };
            let new_slice = self
                .stream()
                .clone_htod(&buf_f64)
                .expect("Failed to upload");
            *self = CudaStorage::new(
                unsafe { std::mem::transmute(new_slice) },
                self.stream().clone(),
            );
        } else {
            panic!(
                "CudaStorage::set not supported for type {:?}",
                std::any::type_name::<T>()
            );
        }
    }

    fn to_vec(&self) -> Vec<T> {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let slice_f32: &cudarc::driver::CudaSlice<f32> =
                unsafe { std::mem::transmute(self.slice()) };
            let result = self
                .stream()
                .clone_dtoh(slice_f32)
                .expect("Failed to download");
            unsafe { std::mem::transmute(result) }
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let slice_f64: &cudarc::driver::CudaSlice<f64> =
                unsafe { std::mem::transmute(self.slice()) };
            let result = self
                .stream()
                .clone_dtoh(slice_f64)
                .expect("Failed to download");
            unsafe { std::mem::transmute(result) }
        } else if TypeId::of::<T>() == TypeId::of::<u32>() {
            let slice_u32: &cudarc::driver::CudaSlice<u32> =
                unsafe { std::mem::transmute(self.slice()) };
            let result = self
                .stream()
                .clone_dtoh(slice_u32)
                .expect("Failed to download");
            unsafe { std::mem::transmute(result) }
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f32>>() {
            let slice_c32: &cudarc::driver::CudaSlice<CudaComplex<f32>> =
                unsafe { std::mem::transmute(self.slice()) };
            let result = self
                .stream()
                .clone_dtoh(slice_c32)
                .expect("Failed to download");
            unsafe { std::mem::transmute(result) }
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f64>>() {
            let slice_c64: &cudarc::driver::CudaSlice<CudaComplex<f64>> =
                unsafe { std::mem::transmute(self.slice()) };
            let result = self
                .stream()
                .clone_dtoh(slice_c64)
                .expect("Failed to download");
            unsafe { std::mem::transmute(result) }
        } else {
            panic!(
                "CudaStorage::to_vec not supported for type {:?}",
                std::any::type_name::<T>()
            );
        }
    }

    fn from_slice(_data: &[T]) -> Self {
        panic!("CudaStorage::from_slice requires device context. Use Cuda::from_slice instead.")
    }

    fn zeros(_len: usize) -> Self {
        panic!("CudaStorage::zeros requires device context. Use Cuda::alloc instead.")
    }
}

impl<T: Scalar> Clone for CudaStorage<T> {
    fn clone(&self) -> Self {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let data: Vec<f32> = unsafe { std::mem::transmute(self.to_vec()) };
            let new_slice = self.stream().clone_htod(&data).expect("Failed to clone");
            CudaStorage::new(
                unsafe { std::mem::transmute(new_slice) },
                self.stream().clone(),
            )
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let data: Vec<f64> = unsafe { std::mem::transmute(self.to_vec()) };
            let new_slice = self.stream().clone_htod(&data).expect("Failed to clone");
            CudaStorage::new(
                unsafe { std::mem::transmute(new_slice) },
                self.stream().clone(),
            )
        } else if TypeId::of::<T>() == TypeId::of::<u32>() {
            let data: Vec<u32> = unsafe { std::mem::transmute(self.to_vec()) };
            let new_slice = self.stream().clone_htod(&data).expect("Failed to clone");
            CudaStorage::new(
                unsafe { std::mem::transmute(new_slice) },
                self.stream().clone(),
            )
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f32>>() {
            let data: Vec<CudaComplex<f32>> = unsafe { std::mem::transmute(self.to_vec()) };
            let new_slice = self.stream().clone_htod(&data).expect("Failed to clone");
            CudaStorage::new(
                unsafe { std::mem::transmute(new_slice) },
                self.stream().clone(),
            )
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f64>>() {
            let data: Vec<CudaComplex<f64>> = unsafe { std::mem::transmute(self.to_vec()) };
            let new_slice = self.stream().clone_htod(&data).expect("Failed to clone");
            CudaStorage::new(
                unsafe { std::mem::transmute(new_slice) },
                self.stream().clone(),
            )
        } else {
            panic!(
                "CudaStorage::clone not supported for type {:?}",
                std::any::type_name::<T>()
            );
        }
    }
}

// ============================================================================
// Backend implementation for Cuda
// ============================================================================

impl Backend for Cuda {
    type Storage<T: Scalar> = CudaStorage<T>;

    fn name() -> &'static str {
        "cuda"
    }

    fn synchronize(&self) {
        self.stream
            .synchronize()
            .expect("Failed to synchronize CUDA device");
    }

    fn alloc<T: Scalar>(&self, len: usize) -> CudaStorage<T> {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let slice = self
                .stream
                .alloc_zeros::<f32>(len)
                .expect("Failed to allocate");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.stream.clone())
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let slice = self
                .stream
                .alloc_zeros::<f64>(len)
                .expect("Failed to allocate");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.stream.clone())
        } else if TypeId::of::<T>() == TypeId::of::<u32>() {
            let slice = self
                .stream
                .alloc_zeros::<u32>(len)
                .expect("Failed to allocate");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.stream.clone())
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f32>>() {
            let slice = self
                .stream
                .alloc_zeros::<CudaComplex<f32>>(len)
                .expect("Failed to allocate");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.stream.clone())
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f64>>() {
            let slice = self
                .stream
                .alloc_zeros::<CudaComplex<f64>>(len)
                .expect("Failed to allocate");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.stream.clone())
        } else {
            panic!(
                "CUDA alloc not supported for type {:?}",
                std::any::type_name::<T>()
            );
        }
    }

    fn from_slice<T: Scalar>(&self, data: &[T]) -> CudaStorage<T> {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let data_f32: &[f32] = unsafe { std::mem::transmute(data) };
            let slice = self.stream.clone_htod(data_f32).expect("Failed to copy");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.stream.clone())
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let data_f64: &[f64] = unsafe { std::mem::transmute(data) };
            let slice = self.stream.clone_htod(data_f64).expect("Failed to copy");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.stream.clone())
        } else if TypeId::of::<T>() == TypeId::of::<u32>() {
            let data_u32: &[u32] = unsafe { std::mem::transmute(data) };
            let slice = self.stream.clone_htod(data_u32).expect("Failed to copy");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.stream.clone())
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f32>>() {
            let data_c32: &[CudaComplex<f32>] = unsafe { std::mem::transmute(data) };
            let slice = self.stream.clone_htod(data_c32).expect("Failed to copy");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.stream.clone())
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f64>>() {
            let data_c64: &[CudaComplex<f64>] = unsafe { std::mem::transmute(data) };
            let slice = self.stream.clone_htod(data_c64).expect("Failed to copy");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.stream.clone())
        } else {
            panic!(
                "CUDA from_slice not supported for type {:?}",
                std::any::type_name::<T>()
            );
        }
    }

    fn copy_strided<T: Scalar>(
        &self,
        src: &CudaStorage<T>,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> CudaStorage<T> {
        // Download to CPU, copy with strides, upload back
        // Storage::to_vec and from_slice handle type dispatch
        let src_data = src.to_vec();
        let numel: usize = shape.iter().product();
        let mut dst_data = vec![T::default(); numel];

        // Iterate over all indices and copy
        let mut indices = vec![0usize; shape.len()];
        for dst_elem in dst_data.iter_mut() {
            // Compute source offset using strides
            let src_offset: usize = offset
                + indices
                    .iter()
                    .zip(strides.iter())
                    .map(|(i, s)| i * s)
                    .sum::<usize>();

            *dst_elem = src_data[src_offset];

            // Increment indices (column-major order)
            for dim in 0..shape.len() {
                indices[dim] += 1;
                if indices[dim] < shape[dim] {
                    break;
                }
                indices[dim] = 0;
            }
        }

        self.from_slice(&dst_data)
    }

    fn contract<A: Algebra>(
        &self,
        a: &CudaStorage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &CudaStorage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> CudaStorage<A::Scalar>
    where
        A::Scalar: BackendScalar<Self>,
    {
        // Tropical algebras (max-plus / min-plus / max-mul) have no cuTENSOR
        // semiring; route them to the tropical-gemm-cuda executor.
        //
        // Routing invariant: `needs_argmax()` is true for *exactly* the algebras
        // that have a tropical-gemm-cuda kernel (the idempotent semirings whose
        // backward pass needs argmax — MaxPlus/MinPlus/MaxMul). `contract_tropical`
        // → `run_tropical_gemm` therefore covers every algebra that reaches this
        // branch. If a future algebra returns `needs_argmax() == true` without a
        // matching GPU kernel, it would hit the `panic!` in `run_tropical_gemm`;
        // add its kernel (or a dedicated routing predicate) when that happens.
        #[cfg(feature = "cuda-tropical")]
        if A::needs_argmax() {
            return self.contract_tropical::<A>(
                a, shape_a, strides_a, modes_a, b, shape_b, strides_b, modes_b, shape_c, modes_c,
            );
        }

        // Standard algebras use cuTENSOR.
        #[cfg(feature = "cuda")]
        {
            // Compute output strides (column-major)
            let strides_c = Self::compute_strides(shape_c);

            // Dispatch based on scalar type using type ID
            use std::any::TypeId;

            if TypeId::of::<A::Scalar>() == TypeId::of::<f32>() {
                // SAFETY: We've verified the type is f32
                let a_f32: &CudaStorage<f32> = unsafe { std::mem::transmute(a) };
                let b_f32: &CudaStorage<f32> = unsafe { std::mem::transmute(b) };

                let result = self
                    .contract_cutensor(
                        a_f32, shape_a, strides_a, modes_a, b_f32, shape_b, strides_b, modes_b,
                        shape_c, &strides_c, modes_c,
                    )
                    .expect("cuTENSOR contraction failed");

                unsafe { std::mem::transmute(result) }
            } else if TypeId::of::<A::Scalar>() == TypeId::of::<f64>() {
                // SAFETY: We've verified the type is f64
                let a_f64: &CudaStorage<f64> = unsafe { std::mem::transmute(a) };
                let b_f64: &CudaStorage<f64> = unsafe { std::mem::transmute(b) };

                let result = self
                    .contract_cutensor(
                        a_f64, shape_a, strides_a, modes_a, b_f64, shape_b, strides_b, modes_b,
                        shape_c, &strides_c, modes_c,
                    )
                    .expect("cuTENSOR contraction failed");

                unsafe { std::mem::transmute(result) }
            } else if TypeId::of::<A::Scalar>() == TypeId::of::<CudaComplex<f32>>() {
                // SAFETY: We've verified the type is CudaComplex<f32>
                let a_c32: &CudaStorage<CudaComplex<f32>> = unsafe { std::mem::transmute(a) };
                let b_c32: &CudaStorage<CudaComplex<f32>> = unsafe { std::mem::transmute(b) };

                let result = self
                    .contract_cutensor(
                        a_c32, shape_a, strides_a, modes_a, b_c32, shape_b, strides_b, modes_b,
                        shape_c, &strides_c, modes_c,
                    )
                    .expect("cuTENSOR contraction failed");

                unsafe { std::mem::transmute(result) }
            } else if TypeId::of::<A::Scalar>() == TypeId::of::<CudaComplex<f64>>() {
                // SAFETY: We've verified the type is CudaComplex<f64>
                let a_c64: &CudaStorage<CudaComplex<f64>> = unsafe { std::mem::transmute(a) };
                let b_c64: &CudaStorage<CudaComplex<f64>> = unsafe { std::mem::transmute(b) };

                let result = self
                    .contract_cutensor(
                        a_c64, shape_a, strides_a, modes_a, b_c64, shape_b, strides_b, modes_b,
                        shape_c, &strides_c, modes_c,
                    )
                    .expect("cuTENSOR contraction failed");

                unsafe { std::mem::transmute(result) }
            } else {
                panic!(
                    "CUDA backend only supports f32, f64, CudaComplex<f32>, and \
                     CudaComplex<f64> for contractions. Got type: {:?}",
                    std::any::type_name::<A::Scalar>()
                );
            }
        }

        // No cuTENSOR available: only the tropical path (handled above) is supported.
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                a, shape_a, strides_a, modes_a, b, shape_b, strides_b, modes_b, shape_c, modes_c,
            );
            panic!(
                "Standard-algebra contractions on the CUDA backend require the `cuda` \
                 (cuTENSOR) feature; `cuda-tropical` only provides tropical contractions."
            );
        }
    }

    fn contract_with_argmax<A: Algebra<Index = u32>>(
        &self,
        a: &CudaStorage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &CudaStorage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> (CudaStorage<A::Scalar>, CudaStorage<u32>)
    where
        A::Scalar: BackendScalar<Self>,
    {
        // Argmax tracking is a tropical-only concern (`needs_argmax()` ⟺ a
        // tropical-gemm-cuda kernel exists); route it through the tropical-gemm
        // argmax executor. cuTENSOR provides no argmax, so standard algebras —
        // which never set `needs_argmax()` and so never reach this method — are
        // not supported here.
        #[cfg(feature = "cuda-tropical")]
        if A::needs_argmax() {
            return self.contract_tropical_with_argmax::<A>(
                a, shape_a, strides_a, modes_a, b, shape_b, strides_b, modes_b, shape_c, modes_c,
            );
        }

        let _ = (
            a, shape_a, strides_a, modes_a, b, shape_b, strides_b, modes_b, shape_c, modes_c,
        );
        panic!(
            "CUDA contract_with_argmax is only supported for tropical algebras \
             (MaxPlus/MinPlus/MaxMul) via the `cuda-tropical` feature; cuTENSOR \
             does not provide argmax tracking."
        );
    }
}

/// Correctness gate for the batched gather (B9): `device_gather_batched` must be
/// bit-exact with running `device_gather` once per request — the single-gather
/// path is the trusted reference. Covers mixed ndim and non-power-of-two dims
/// (so it can never silently special-case the dim-2 hypercube), for both element
/// widths (32/64-bit).
#[cfg(all(test, feature = "cuda-tropical"))]
mod batched_gather_tests {
    use super::*;

    fn colmajor_contig(shape: &[usize]) -> Vec<usize> {
        let mut s = vec![0usize; shape.len()];
        let mut acc = 1;
        for (i, &d) in shape.iter().enumerate() {
            s[i] = acc;
            acc *= d;
        }
        s
    }

    /// `(new_shape, src_strides)` for permuting a contiguous column-major tensor
    /// of `shape` by `perm` — same semantics `canonical_gather_args` produces.
    fn perm_gather(shape: &[usize], perm: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let c = colmajor_contig(shape);
        let new_shape = perm.iter().map(|&p| shape[p]).collect();
        let src_strides = perm.iter().map(|&p| c[p]).collect();
        (new_shape, src_strides)
    }

    // (shape, perm) cases: mixed ndim, incl. non-power-of-two extents.
    fn cases() -> Vec<(Vec<usize>, Vec<usize>)> {
        vec![
            (vec![2, 2, 2, 2], vec![3, 1, 0, 2]),
            (vec![3, 5, 2, 7], vec![2, 0, 3, 1]), // non-power-of-two
            (vec![4, 4], vec![1, 0]),
            (vec![2, 3, 4], vec![2, 1, 0]),
            (vec![6], vec![0]), // ndim == 1
        ]
    }

    #[test]
    fn batched_gather_matches_single_u32() {
        let cuda = Cuda::new().expect("init cuda");
        let stream = cuda.stream.clone();
        let cs = cases();
        let inputs: Vec<_> = cs
            .iter()
            .map(|(shape, _)| {
                let n: usize = shape.iter().product();
                let host: Vec<u32> = (0..n as u32).collect();
                stream.clone_htod(&host).expect("upload input")
            })
            .collect();
        let gathers: Vec<(Vec<usize>, Vec<usize>)> =
            cs.iter().map(|(s, p)| perm_gather(s, p)).collect();

        let single: Vec<Vec<u32>> = inputs
            .iter()
            .zip(gathers.iter())
            .map(|(inp, (ns, ss))| {
                let o = cuda.device_gather::<u32>(inp, ns, ss);
                stream.clone_dtoh(&o).expect("dtov single")
            })
            .collect();

        let reqs: Vec<(&_, &[usize], &[usize])> = inputs
            .iter()
            .zip(gathers.iter())
            .map(|(inp, (ns, ss))| (inp, ns.as_slice(), ss.as_slice()))
            .collect();
        let batched = cuda.device_gather_batched::<u32>(&reqs);

        assert_eq!(batched.len(), single.len());
        for (i, (b, s)) in batched.iter().zip(single.iter()).enumerate() {
            let bh = stream.clone_dtoh(b).expect("dtov batched");
            assert_eq!(&bh, s, "batched gather case {i} (u32) mismatch vs single");
        }
    }

    #[test]
    fn batched_gather_matches_single_u64() {
        let cuda = Cuda::new().expect("init cuda");
        let stream = cuda.stream.clone();
        let cs = cases();
        let inputs: Vec<_> = cs
            .iter()
            .map(|(shape, _)| {
                let n: usize = shape.iter().product();
                let host: Vec<u64> = (0..n as u64).collect();
                stream.clone_htod(&host).expect("upload input")
            })
            .collect();
        let gathers: Vec<(Vec<usize>, Vec<usize>)> =
            cs.iter().map(|(s, p)| perm_gather(s, p)).collect();

        let single: Vec<Vec<u64>> = inputs
            .iter()
            .zip(gathers.iter())
            .map(|(inp, (ns, ss))| {
                let o = cuda.device_gather::<u64>(inp, ns, ss);
                stream.clone_dtoh(&o).expect("dtov single")
            })
            .collect();

        let reqs: Vec<(&_, &[usize], &[usize])> = inputs
            .iter()
            .zip(gathers.iter())
            .map(|(inp, (ns, ss))| (inp, ns.as_slice(), ss.as_slice()))
            .collect();
        let batched = cuda.device_gather_batched::<u64>(&reqs);

        assert_eq!(batched.len(), single.len());
        for (i, (b, s)) in batched.iter().zip(single.iter()).enumerate() {
            let bh = stream.clone_dtoh(b).expect("dtov batched");
            assert_eq!(&bh, s, "batched gather case {i} (u64) mismatch vs single");
        }
    }
}
