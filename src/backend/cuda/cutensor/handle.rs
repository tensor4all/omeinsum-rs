//! Safe wrappers for cuTENSOR handles and descriptors.
//!
//! This module provides RAII wrappers for cuTENSOR resources, ensuring
//! proper cleanup via Drop implementations.

use super::sys::*;
use super::{check, CutensorError};
use crate::algebra::{Complex32, Complex64};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

/// Safe wrapper for a cuTENSOR handle.
///
/// Manages the lifetime of a cuTENSOR library handle and its associated CUDA device.
pub struct Handle {
    raw: cutensorHandle_t,
    device: Arc<CudaDevice>,
}

impl Handle {
    /// Create a new cuTENSOR handle for the given CUDA device.
    ///
    /// # Arguments
    /// * `device` - The CUDA device to use for cuTENSOR operations
    ///
    /// # Returns
    /// * `Ok(Handle)` on success
    /// * `Err(CutensorError)` if handle creation fails
    ///
    /// # Version Requirements
    /// cuTENSOR 2.0+ is required. Version 1.x uses a different API and will fail
    /// at link time with undefined symbol errors for `cutensorContract`, etc.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CutensorError> {
        let mut raw = std::ptr::null_mut();
        check(unsafe { cutensorCreate(&mut raw) })?;
        Ok(Self { raw, device })
    }

    /// Get the raw cuTENSOR handle pointer.
    pub fn raw(&self) -> cutensorHandle_t {
        self.raw
    }

    /// Get a reference to the associated CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { cutensorDestroy(self.raw) };
    }
}

/// Trait for types that can be used with cuTENSOR.
///
/// Maps Rust types to cuTENSOR data types and compute descriptors.
pub trait CutensorType: Copy {
    /// The cuTENSOR data type for this Rust type.
    const DATA: cutensorDataType_t;
    /// Get the cuTENSOR compute descriptor for operations on this type.
    fn compute_desc() -> cutensorComputeDescriptor_t;
}

impl CutensorType for f32 {
    const DATA: cutensorDataType_t = cutensorDataType_t::R_32F;
    fn compute_desc() -> cutensorComputeDescriptor_t {
        unsafe { super::sys::CUTENSOR_COMPUTE_DESC_32F }
    }
}

impl CutensorType for f64 {
    const DATA: cutensorDataType_t = cutensorDataType_t::R_64F;
    fn compute_desc() -> cutensorComputeDescriptor_t {
        unsafe { super::sys::CUTENSOR_COMPUTE_DESC_64F }
    }
}

impl CutensorType for Complex32 {
    const DATA: cutensorDataType_t = cutensorDataType_t::C_32F;
    fn compute_desc() -> cutensorComputeDescriptor_t {
        unsafe { super::sys::CUTENSOR_COMPUTE_DESC_32F }
    }
}

impl CutensorType for Complex64 {
    const DATA: cutensorDataType_t = cutensorDataType_t::C_64F;
    fn compute_desc() -> cutensorComputeDescriptor_t {
        unsafe { super::sys::CUTENSOR_COMPUTE_DESC_64F }
    }
}

/// Safe wrapper for a cuTENSOR tensor descriptor.
///
/// Describes the shape, strides, and data type of a tensor.
pub struct TensorDesc {
    raw: cutensorTensorDescriptor_t,
}

impl TensorDesc {
    /// Create a new tensor descriptor.
    ///
    /// # Arguments
    /// * `handle` - The cuTENSOR handle
    /// * `shape` - The shape (extents) of each dimension
    /// * `strides` - The strides for each dimension
    ///
    /// # Type Parameters
    /// * `T` - The element type of the tensor
    ///
    /// # Returns
    /// * `Ok(TensorDesc)` on success
    /// * `Err(CutensorError)` if descriptor creation fails
    pub fn new<T: CutensorType>(
        handle: &Handle,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self, CutensorError> {
        let extent: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
        let stride: Vec<i64> = strides.iter().map(|&s| s as i64).collect();
        let mut raw = std::ptr::null_mut();
        // Alignment of 128 bytes is recommended for best performance
        const ALIGNMENT: u32 = 128;
        check(unsafe {
            cutensorCreateTensorDescriptor(
                handle.raw(),
                &mut raw,
                shape.len() as u32,
                extent.as_ptr(),
                stride.as_ptr(),
                T::DATA,
                ALIGNMENT,
            )
        })?;
        Ok(Self { raw })
    }

    /// Get the raw cuTENSOR tensor descriptor pointer.
    pub fn raw(&self) -> cutensorTensorDescriptor_t {
        self.raw
    }
}

impl Drop for TensorDesc {
    fn drop(&mut self) {
        unsafe { cutensorDestroyTensorDescriptor(self.raw) };
    }
}

/// Safe wrapper for a cuTENSOR execution plan.
///
/// Contains a pre-compiled plan for executing a tensor contraction.
pub struct Plan {
    raw: cutensorPlan_t,
    /// The workspace size required by this plan (in bytes).
    pub workspace_size: u64,
}

impl Plan {
    /// Create a new execution plan for a tensor contraction.
    ///
    /// Creates a plan for computing: D = alpha * A * B + beta * C
    /// where the contraction is specified by the mode indices.
    ///
    /// # Arguments
    /// * `handle` - The cuTENSOR handle
    /// * `desc_a` - Tensor descriptor for input A
    /// * `modes_a` - Mode indices for tensor A
    /// * `desc_b` - Tensor descriptor for input B
    /// * `modes_b` - Mode indices for tensor B
    /// * `desc_c` - Tensor descriptor for output C (and D)
    /// * `modes_c` - Mode indices for tensor C (and D)
    ///
    /// # Type Parameters
    /// * `T` - The element type of the tensors
    ///
    /// # Returns
    /// * `Ok(Plan)` on success, containing the plan and required workspace size
    /// * `Err(CutensorError)` if plan creation fails
    pub fn new<T: CutensorType>(
        handle: &Handle,
        desc_a: &TensorDesc,
        modes_a: &[i32],
        desc_b: &TensorDesc,
        modes_b: &[i32],
        desc_c: &TensorDesc,
        modes_c: &[i32],
    ) -> Result<Self, CutensorError> {
        let compute = T::compute_desc();

        // Create operation descriptor
        let mut op = std::ptr::null_mut();
        check(unsafe {
            cutensorCreateContraction(
                handle.raw(),
                &mut op,
                desc_a.raw(),
                modes_a.as_ptr(),
                super::sys::cutensorOperator_t::IDENTITY,
                desc_b.raw(),
                modes_b.as_ptr(),
                super::sys::cutensorOperator_t::IDENTITY,
                desc_c.raw(),
                modes_c.as_ptr(),
                super::sys::cutensorOperator_t::IDENTITY,
                desc_c.raw(),
                modes_c.as_ptr(),
                compute,
            )
        })?;

        // Create plan preference
        let mut pref = std::ptr::null_mut();
        check(unsafe {
            cutensorCreatePlanPreference(
                handle.raw(),
                &mut pref,
                super::sys::cutensorAlgo_t::DEFAULT,
                super::sys::cutensorJitMode_t::NONE,
            )
        })?;

        // Estimate workspace
        let mut workspace_size = 0u64;
        check(unsafe {
            cutensorEstimateWorkspaceSize(
                handle.raw(),
                op,
                pref,
                cutensorWorksizePreference_t::DEFAULT,
                &mut workspace_size,
            )
        })?;

        // Create plan
        let mut raw = std::ptr::null_mut();
        check(unsafe { cutensorCreatePlan(handle.raw(), &mut raw, op, pref, workspace_size) })?;

        // Cleanup temporary objects
        unsafe {
            cutensorDestroyPlanPreference(pref);
            cutensorDestroyOperationDescriptor(op);
        }

        Ok(Self {
            raw,
            workspace_size,
        })
    }

    /// Get the raw cuTENSOR plan pointer.
    pub fn raw(&self) -> cutensorPlan_t {
        self.raw
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        unsafe { cutensorDestroyPlan(self.raw) };
    }
}
