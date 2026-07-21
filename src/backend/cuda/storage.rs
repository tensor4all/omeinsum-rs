//! CUDA storage implementation for GPU memory management.

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, DriverError};
use std::sync::Arc;

/// GPU memory storage backed by CUDA.
///
/// Wraps cudarc's `CudaSlice` together with the stream that owns its allocation
/// and transfers (cudarc 0.19 moved memory ops from the device onto the stream).
pub struct CudaStorage<T> {
    slice: CudaSlice<T>,
    stream: Arc<CudaStream>,
}

impl<T> CudaStorage<T> {
    /// Create a new CudaStorage from a CudaSlice and the owning stream.
    pub fn new(slice: CudaSlice<T>, stream: Arc<CudaStream>) -> Self {
        Self { slice, stream }
    }

    /// Get a reference to the underlying CUDA slice.
    pub fn slice(&self) -> &CudaSlice<T> {
        &self.slice
    }

    /// Get a mutable reference to the underlying CUDA slice.
    pub fn slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }

    /// Get the stream that owns this storage.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Number of elements in storage.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Check if storage is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: DeviceRepr + Clone> CudaStorage<T> {
    /// Copy all data from GPU to a Vec on the host.
    ///
    /// # Errors
    ///
    /// Returns a `DriverError` if the CUDA device-to-host copy fails.
    pub fn to_vec(&self) -> Result<Vec<T>, DriverError> {
        let host = self.stream.clone_dtoh(&self.slice)?;
        // Ensure the (async) D2H copy lands before the caller reads `host`.
        // Replaces cudarc 0.12's `dtoh_sync_copy`.
        self.stream.synchronize()?;
        Ok(host)
    }
}

// SAFETY: CudaStorage<T> can be sent between threads because:
// - CudaSlice<T> internally uses a CUDA device pointer which is thread-safe
// - Arc<CudaStream> is Send
// The actual GPU memory is managed by the CUDA runtime which handles synchronization.
unsafe impl<T: Send> Send for CudaStorage<T> {}

// SAFETY: CudaStorage<T> can be shared between threads because:
// - CudaSlice<T> only provides immutable access through &self methods
// - Arc<CudaStream> is Sync
// - CUDA operations are synchronized by the runtime
unsafe impl<T: Sync> Sync for CudaStorage<T> {}
