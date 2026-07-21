//! Plan caching and tensor contraction execution.
//!
//! This module provides a cache for cuTENSOR execution plans and the
//! contract function for executing tensor contractions.

use super::handle::*;
use super::sys::cutensorContract;
use super::{check, CutensorError};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use std::collections::HashMap;

/// Cache key for identifying unique tensor contraction configurations.
///
/// Two contractions with the same shapes, strides, modes, and data type
/// can reuse the same execution plan.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct CacheKey {
    /// Shapes of input tensors and output tensor.
    pub shapes: Vec<Vec<usize>>,
    /// Strides of input tensors and output tensor.
    pub strides: Vec<Vec<usize>>,
    /// Mode indices for input tensors and output tensor.
    pub modes: Vec<Vec<i32>>,
    /// Data type identifier (from cutensorDataType_t).
    pub dtype: u32,
}

/// A cache for cuTENSOR execution plans.
///
/// Execution plans are expensive to create, so caching them can significantly
/// improve performance when the same contraction pattern is used multiple times.
///
/// Uses an arbitrary eviction policy when the cache reaches capacity.
pub struct PlanCache {
    cache: HashMap<CacheKey, Plan>,
    capacity: usize,
}

impl PlanCache {
    /// Create a new plan cache with the given capacity.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of plans to store in the cache
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::new(),
            capacity,
        }
    }

    /// Get an existing plan from the cache or create a new one.
    ///
    /// If a plan for the given key already exists, returns a reference to it.
    /// Otherwise, creates a new plan, stores it in the cache, and returns
    /// a reference to the newly created plan.
    ///
    /// If the cache is at capacity, removes an arbitrary existing entry
    /// before inserting the new plan.
    ///
    /// # Arguments
    /// * `handle` - The cuTENSOR handle
    /// * `key` - The cache key identifying this contraction
    /// * `desc_a` - Tensor descriptor for input A
    /// * `modes_a` - Mode indices for tensor A
    /// * `desc_b` - Tensor descriptor for input B
    /// * `modes_b` - Mode indices for tensor B
    /// * `desc_c` - Tensor descriptor for output C
    /// * `modes_c` - Mode indices for tensor C
    ///
    /// # Type Parameters
    /// * `T` - The element type of the tensors
    ///
    /// # Returns
    /// * `Ok(&Plan)` on success
    /// * `Err(CutensorError)` if plan creation fails
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create<T: CutensorType>(
        &mut self,
        handle: &Handle,
        key: CacheKey,
        desc_a: &TensorDesc,
        modes_a: &[i32],
        desc_b: &TensorDesc,
        modes_b: &[i32],
        desc_c: &TensorDesc,
        modes_c: &[i32],
    ) -> Result<&Plan, CutensorError> {
        if !self.cache.contains_key(&key) {
            // Evict an entry if at capacity
            if self.capacity > 0 && self.cache.len() >= self.capacity {
                let k = self.cache.keys().next().cloned().unwrap();
                self.cache.remove(&k);
            }
            let plan = Plan::new::<T>(handle, desc_a, modes_a, desc_b, modes_b, desc_c, modes_c)?;
            self.cache.insert(key.clone(), plan);
        }
        Ok(self.cache.get(&key).unwrap())
    }
}

/// Execute a tensor contraction using a pre-compiled plan.
///
/// Computes: C = alpha * A * B
///
/// Note: This function always uses beta = 0, so the output tensor is
/// completely overwritten rather than accumulated into.
///
/// # Arguments
/// * `handle` - The cuTENSOR handle
/// * `plan` - The pre-compiled execution plan
/// * `alpha` - Scalar multiplier for the contraction result
/// * `a` - Input tensor A (device memory)
/// * `b` - Input tensor B (device memory)
/// * `c` - Output tensor C (device memory, will be overwritten)
///
/// # Type Parameters
/// * `T` - The element type of the tensors
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(CutensorError)` if the contraction fails
pub fn contract<T>(
    handle: &Handle,
    plan: &Plan,
    alpha: T,
    a: &CudaSlice<T>,
    b: &CudaSlice<T>,
    c: &mut CudaSlice<T>,
) -> Result<(), CutensorError>
where
    T: CutensorType + cudarc::driver::DeviceRepr + num_traits::Zero,
{
    let stream = handle.stream();

    // Allocate workspace if needed (cudarc 0.19: allocations live on the stream).
    let mut workspace = if plan.workspace_size > 0 {
        Some(
            stream
                .alloc_zeros::<u8>(plan.workspace_size as usize)
                .map_err(|e| CutensorError::Other(format!("Workspace allocation failed: {}", e)))?,
        )
    } else {
        None
    };

    // cudarc 0.19 device-pointer extraction: `device_ptr(stream)` returns the raw
    // `CUdeviceptr` plus a `SyncOnDrop` guard that orders the stream against this
    // access on drop. The guards (`_*_guard`) are bound for the whole unsafe call
    // so the pointers stay valid until cuTENSOR has been launched. `c` is the only
    // written buffer, so it uses the mutable accessor; the same pointer feeds both
    // the C (read, unused since beta=0) and D (write) arguments, as before.
    let (ws_ptr, _ws_guard) = match workspace.as_mut() {
        Some(w) => {
            let (p, g) = w.device_ptr_mut(stream);
            (p as *mut std::ffi::c_void, Some(g))
        }
        None => (std::ptr::null_mut(), None),
    };
    let (a_ptr, _a_guard) = a.device_ptr(stream);
    let (b_ptr, _b_guard) = b.device_ptr(stream);
    let (c_ptr, _c_guard) = c.device_ptr_mut(stream);

    // beta = 0: completely overwrite output rather than accumulating
    let beta = T::zero();

    check(unsafe {
        cutensorContract(
            handle.raw(),
            plan.raw(),
            &alpha as *const T as *const _,
            a_ptr as *const _,
            b_ptr as *const _,
            &beta as *const T as *const _,
            c_ptr as *const _,
            c_ptr as *mut _,
            ws_ptr,
            plan.workspace_size,
            // cuTENSOR's FFI types the stream as an opaque `void*` (cudaStream_t);
            // cast cudarc's `CUstream` handle to match. Runs on the handle's
            // stream rather than the default stream.
            stream.cu_stream() as *mut std::ffi::c_void,
        )
    })
}
