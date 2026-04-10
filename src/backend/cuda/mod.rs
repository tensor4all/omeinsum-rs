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

mod cutensor;
mod storage;

pub use storage::CudaStorage;

use cudarc::driver::CudaDevice;
use cutensor::{contract, CacheKey, CutensorType, Handle, PlanCache, TensorDesc};
use num_complex::Complex;
use std::sync::{Arc, Mutex};

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
impl CutensorType for CudaComplex<f32> {
    const DATA: cutensor::sys::cutensorDataType_t = cutensor::sys::cutensorDataType_t::C_32F;
    fn compute_desc() -> cutensor::sys::cutensorComputeDescriptor_t {
        unsafe { cutensor::sys::CUTENSOR_COMPUTE_DESC_32F }
    }
}

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
    device: Arc<CudaDevice>,
    handle: Mutex<Option<Handle>>,
    cache: Mutex<PlanCache>,
}

// SAFETY: Cuda is Send because all fields are Send.
// The Mutex ensures safe concurrent access to handle and cache.
unsafe impl Send for Cuda {}
// SAFETY: Cuda is Sync because all fields are protected by Mutex.
unsafe impl Sync for Cuda {}

impl Clone for Cuda {
    fn clone(&self) -> Self {
        // Create a new Cuda instance sharing the same device
        // but with fresh handle and cache (lazy initialization)
        Self {
            device: self.device.clone(),
            handle: Mutex::new(None),
            cache: Mutex::new(PlanCache::new(64)),
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
        let device = CudaDevice::new(ordinal).map_err(|e| CudaError::Device(e.to_string()))?;
        Ok(Self {
            device,
            handle: Mutex::new(None),
            cache: Mutex::new(PlanCache::new(64)),
        })
    }

    /// Get a reference to the CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Ensure the cuTENSOR handle is initialized and execute a function with it.
    ///
    /// This method acquires the handle lock and ensures the handle is initialized,
    /// then calls the provided function with access to the handle.
    fn with_handle<R>(
        &self,
        f: impl FnOnce(&Handle) -> Result<R, CudaError>,
    ) -> Result<R, CudaError> {
        let mut h = self.handle.lock().unwrap();
        if h.is_none() {
            *h = Some(
                Handle::new(self.device.clone())
                    .map_err(|e| CudaError::Cutensor(format!("{}", e)))?,
            );
        }
        f(h.as_ref().unwrap())
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
            .device
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

        Ok(CudaStorage::new(c, self.device.clone()))
    }

    /// Compute column-major strides for a given shape.
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = 1;
        for &dim in shape {
            strides.push(stride);
            stride *= dim;
        }
        strides
    }
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
// Marker trait for CUDA-compatible scalar types
// ============================================================================

/// Marker trait for scalar types that can be used with CUDA.
///
/// This unifies the requirements of `Scalar` (for the type system) and
/// cudarc traits (for GPU operations). Types implementing this trait can
/// be used with `CudaStorage` and CUDA tensor operations.
pub trait CudaScalar:
    Scalar
    + cudarc::driver::DeviceRepr
    + cudarc::driver::ValidAsZeroBits
    + CutensorType
    + num_traits::One
    + num_traits::Zero
{
}

impl CudaScalar for f32 {}
impl CudaScalar for f64 {}

// ============================================================================
// Storage implementation for CudaStorage
// ============================================================================

// Note: Storage<T> requires Scalar, but actual CUDA operations need CudaScalar.
// We implement Storage<T> for all T: Scalar to satisfy Backend::Storage bounds,
// but use runtime type dispatch for the actual operations.

impl<T: Scalar> Storage<T> for CudaStorage<T> {
    fn len(&self) -> usize {
        use cudarc::driver::DeviceSlice;
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
                .device()
                .htod_sync_copy(&buf_f32)
                .expect("Failed to upload");
            *self = CudaStorage::new(
                unsafe { std::mem::transmute(new_slice) },
                self.device().clone(),
            );
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let buf_f64: Vec<f64> = unsafe { std::mem::transmute(buf) };
            let new_slice = self
                .device()
                .htod_sync_copy(&buf_f64)
                .expect("Failed to upload");
            *self = CudaStorage::new(
                unsafe { std::mem::transmute(new_slice) },
                self.device().clone(),
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
                .device()
                .dtoh_sync_copy(slice_f32)
                .expect("Failed to download");
            unsafe { std::mem::transmute(result) }
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let slice_f64: &cudarc::driver::CudaSlice<f64> =
                unsafe { std::mem::transmute(self.slice()) };
            let result = self
                .device()
                .dtoh_sync_copy(slice_f64)
                .expect("Failed to download");
            unsafe { std::mem::transmute(result) }
        } else if TypeId::of::<T>() == TypeId::of::<u32>() {
            let slice_u32: &cudarc::driver::CudaSlice<u32> =
                unsafe { std::mem::transmute(self.slice()) };
            let result = self
                .device()
                .dtoh_sync_copy(slice_u32)
                .expect("Failed to download");
            unsafe { std::mem::transmute(result) }
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f32>>() {
            let slice_c32: &cudarc::driver::CudaSlice<CudaComplex<f32>> =
                unsafe { std::mem::transmute(self.slice()) };
            let result = self
                .device()
                .dtoh_sync_copy(slice_c32)
                .expect("Failed to download");
            unsafe { std::mem::transmute(result) }
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f64>>() {
            let slice_c64: &cudarc::driver::CudaSlice<CudaComplex<f64>> =
                unsafe { std::mem::transmute(self.slice()) };
            let result = self
                .device()
                .dtoh_sync_copy(slice_c64)
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
            let new_slice = self
                .device()
                .htod_sync_copy(&data)
                .expect("Failed to clone");
            CudaStorage::new(
                unsafe { std::mem::transmute(new_slice) },
                self.device().clone(),
            )
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let data: Vec<f64> = unsafe { std::mem::transmute(self.to_vec()) };
            let new_slice = self
                .device()
                .htod_sync_copy(&data)
                .expect("Failed to clone");
            CudaStorage::new(
                unsafe { std::mem::transmute(new_slice) },
                self.device().clone(),
            )
        } else if TypeId::of::<T>() == TypeId::of::<u32>() {
            let data: Vec<u32> = unsafe { std::mem::transmute(self.to_vec()) };
            let new_slice = self
                .device()
                .htod_sync_copy(&data)
                .expect("Failed to clone");
            CudaStorage::new(
                unsafe { std::mem::transmute(new_slice) },
                self.device().clone(),
            )
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f32>>() {
            let data: Vec<CudaComplex<f32>> = unsafe { std::mem::transmute(self.to_vec()) };
            let new_slice = self
                .device()
                .htod_sync_copy(&data)
                .expect("Failed to clone");
            CudaStorage::new(
                unsafe { std::mem::transmute(new_slice) },
                self.device().clone(),
            )
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f64>>() {
            let data: Vec<CudaComplex<f64>> = unsafe { std::mem::transmute(self.to_vec()) };
            let new_slice = self
                .device()
                .htod_sync_copy(&data)
                .expect("Failed to clone");
            CudaStorage::new(
                unsafe { std::mem::transmute(new_slice) },
                self.device().clone(),
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
        self.device
            .synchronize()
            .expect("Failed to synchronize CUDA device");
    }

    fn alloc<T: Scalar>(&self, len: usize) -> CudaStorage<T> {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let slice = self
                .device
                .alloc_zeros::<f32>(len)
                .expect("Failed to allocate");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.device.clone())
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let slice = self
                .device
                .alloc_zeros::<f64>(len)
                .expect("Failed to allocate");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.device.clone())
        } else if TypeId::of::<T>() == TypeId::of::<u32>() {
            let slice = self
                .device
                .alloc_zeros::<u32>(len)
                .expect("Failed to allocate");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.device.clone())
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f32>>() {
            let slice = self
                .device
                .alloc_zeros::<CudaComplex<f32>>(len)
                .expect("Failed to allocate");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.device.clone())
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f64>>() {
            let slice = self
                .device
                .alloc_zeros::<CudaComplex<f64>>(len)
                .expect("Failed to allocate");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.device.clone())
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
            let slice = self
                .device
                .htod_sync_copy(data_f32)
                .expect("Failed to copy");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.device.clone())
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let data_f64: &[f64] = unsafe { std::mem::transmute(data) };
            let slice = self
                .device
                .htod_sync_copy(data_f64)
                .expect("Failed to copy");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.device.clone())
        } else if TypeId::of::<T>() == TypeId::of::<u32>() {
            let data_u32: &[u32] = unsafe { std::mem::transmute(data) };
            let slice = self
                .device
                .htod_sync_copy(data_u32)
                .expect("Failed to copy");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.device.clone())
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f32>>() {
            let data_c32: &[CudaComplex<f32>] = unsafe { std::mem::transmute(data) };
            let slice = self
                .device
                .htod_sync_copy(data_c32)
                .expect("Failed to copy");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.device.clone())
        } else if TypeId::of::<T>() == TypeId::of::<CudaComplex<f64>>() {
            let data_c64: &[CudaComplex<f64>] = unsafe { std::mem::transmute(data) };
            let slice = self
                .device
                .htod_sync_copy(data_c64)
                .expect("Failed to copy");
            CudaStorage::new(unsafe { std::mem::transmute(slice) }, self.device.clone())
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

    fn contract_with_argmax<A: Algebra<Index = u32>>(
        &self,
        _a: &CudaStorage<A::Scalar>,
        _shape_a: &[usize],
        _strides_a: &[usize],
        _modes_a: &[i32],
        _b: &CudaStorage<A::Scalar>,
        _shape_b: &[usize],
        _strides_b: &[usize],
        _modes_b: &[i32],
        _shape_c: &[usize],
        _modes_c: &[i32],
    ) -> (CudaStorage<A::Scalar>, CudaStorage<u32>)
    where
        A::Scalar: BackendScalar<Self>,
    {
        // cuTENSOR does not support argmax tracking.
        // This would require a custom CUDA kernel.
        panic!(
            "CUDA backend does not support contract_with_argmax. \
             cuTENSOR does not provide argmax tracking. \
             A custom kernel would be needed for tropical backpropagation on GPU."
        );
    }
}
