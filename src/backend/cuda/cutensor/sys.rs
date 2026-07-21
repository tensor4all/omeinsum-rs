//! Raw FFI bindings to cuTENSOR library.
//!
//! This module provides low-level bindings to NVIDIA's cuTENSOR library
//! for high-performance tensor contractions on CUDA GPUs.
//!
//! # Version Requirements
//!
//! **cuTENSOR 2.0+ is REQUIRED.** The API changed significantly between
//! cuTENSOR 1.x and 2.x. This crate uses the cuTENSOR 2.x API with functions like:
//! - `cutensorCreate` (not `cutensorInit`)
//! - `cutensorContract` (not `cutensorContraction`)
//! - `cutensorCreatePlan` / `cutensorCreatePlanPreference`
//!
//! If you see linker errors about undefined symbols like `cutensorContract`,
//! `cutensorCreatePlan`, or `CUTENSOR_COMPUTE_DESC_32F`, you likely have
//! cuTENSOR 1.x installed instead of 2.x.
//!
//! ## Installation
//!
//! For CUDA 12:
//! ```bash
//! conda install -c nvidia cutensor-cu12
//! ```
//!
//! Or download from: <https://developer.nvidia.com/cutensor-downloads>
//!
//! Make sure to set `CUTENSOR_PATH` to the directory containing `libcutensor.so`.

#![allow(non_camel_case_types)]
// FFI enum variants (`SUCCESS`, `IDENTITY`, `DEFAULT`, `NONE`, …) mirror the
// cuTENSOR C API names verbatim; keep them as-is rather than renaming bindings.
#![allow(clippy::upper_case_acronyms)]

use std::ffi::c_void;

// Opaque handle types
pub type cutensorHandle_t = *mut c_void;
pub type cutensorTensorDescriptor_t = *mut c_void;
pub type cutensorOperationDescriptor_t = *mut c_void;
pub type cutensorPlanPreference_t = *mut c_void;
pub type cutensorPlan_t = *mut c_void;

/// Status codes returned by cuTENSOR functions.
/// Note: These variants are used in pattern matching but Rust's dead_code
/// analysis doesn't recognize that as "construction".
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum cutensorStatus_t {
    SUCCESS = 0,
    NOT_INITIALIZED = 1,
    ALLOC_FAILED = 3,
    INVALID_VALUE = 7,
    ARCH_MISMATCH = 8,
    NOT_SUPPORTED = 15,
}

/// Data types supported by cuTENSOR (cuTENSOR 2.x values).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum cutensorDataType_t {
    R_32F = 0,
    R_64F = 1,
    C_32F = 4,
    C_64F = 5,
}

/// Opaque compute descriptor type (cuTENSOR 2.x uses mutable pointers).
pub type cutensorComputeDescriptor_t = *mut c_void;

// Predefined compute descriptors from cuTENSOR 2.x
// These are global symbols exported by libcutensor.so
#[link(name = "cutensor")]
extern "C" {
    pub static CUTENSOR_COMPUTE_DESC_32F: cutensorComputeDescriptor_t;
    pub static CUTENSOR_COMPUTE_DESC_64F: cutensorComputeDescriptor_t;
}

/// Unary operators that can be applied to tensor elements.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum cutensorOperator_t {
    IDENTITY = 1,
}

/// Workspace size preference for operation planning (cuTENSOR 2.x values).
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum cutensorWorksizePreference_t {
    DEFAULT = 2,
}

/// Algorithm selection (cuTENSOR 2.x values).
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum cutensorAlgo_t {
    DEFAULT = -1,
}

/// JIT compilation mode.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum cutensorJitMode_t {
    NONE = 0,
}

#[link(name = "cutensor")]
extern "C" {
    /// Create a cuTENSOR handle.
    pub fn cutensorCreate(handle: *mut cutensorHandle_t) -> cutensorStatus_t;

    /// Destroy a cuTENSOR handle.
    pub fn cutensorDestroy(handle: cutensorHandle_t) -> cutensorStatus_t;

    /// Create a tensor descriptor.
    pub fn cutensorCreateTensorDescriptor(
        handle: cutensorHandle_t,
        desc: *mut cutensorTensorDescriptor_t,
        num_modes: u32,
        extent: *const i64,
        stride: *const i64,
        data_type: cutensorDataType_t,
        alignment: u32,
    ) -> cutensorStatus_t;

    /// Destroy a tensor descriptor.
    pub fn cutensorDestroyTensorDescriptor(desc: cutensorTensorDescriptor_t) -> cutensorStatus_t;

    /// Create a contraction operation descriptor.
    /// D = alpha * opA(A) * opB(B) + beta * opC(C)
    pub fn cutensorCreateContraction(
        handle: cutensorHandle_t,
        desc: *mut cutensorOperationDescriptor_t,
        desc_a: cutensorTensorDescriptor_t,
        modes_a: *const i32,
        op_a: cutensorOperator_t,
        desc_b: cutensorTensorDescriptor_t,
        modes_b: *const i32,
        op_b: cutensorOperator_t,
        desc_c: cutensorTensorDescriptor_t,
        modes_c: *const i32,
        op_c: cutensorOperator_t,
        desc_d: cutensorTensorDescriptor_t,
        modes_d: *const i32,
        compute: cutensorComputeDescriptor_t,
    ) -> cutensorStatus_t;

    /// Destroy an operation descriptor.
    pub fn cutensorDestroyOperationDescriptor(
        desc: cutensorOperationDescriptor_t,
    ) -> cutensorStatus_t;

    /// Create a plan preference object.
    pub fn cutensorCreatePlanPreference(
        handle: cutensorHandle_t,
        pref: *mut cutensorPlanPreference_t,
        algo: cutensorAlgo_t,
        jit: cutensorJitMode_t,
    ) -> cutensorStatus_t;

    /// Destroy a plan preference object.
    pub fn cutensorDestroyPlanPreference(pref: cutensorPlanPreference_t) -> cutensorStatus_t;

    /// Estimate the workspace size required for an operation.
    pub fn cutensorEstimateWorkspaceSize(
        handle: cutensorHandle_t,
        desc: cutensorOperationDescriptor_t,
        pref: cutensorPlanPreference_t,
        ws_pref: cutensorWorksizePreference_t,
        ws_size: *mut u64,
    ) -> cutensorStatus_t;

    /// Create an execution plan for a contraction.
    pub fn cutensorCreatePlan(
        handle: cutensorHandle_t,
        plan: *mut cutensorPlan_t,
        desc: cutensorOperationDescriptor_t,
        pref: cutensorPlanPreference_t,
        ws_size: u64,
    ) -> cutensorStatus_t;

    /// Destroy an execution plan.
    pub fn cutensorDestroyPlan(plan: cutensorPlan_t) -> cutensorStatus_t;

    /// Execute a tensor contraction.
    ///
    /// Computes: D = alpha * A * B + beta * C
    pub fn cutensorContract(
        handle: cutensorHandle_t,
        plan: cutensorPlan_t,
        alpha: *const c_void,
        a: *const c_void,
        b: *const c_void,
        beta: *const c_void,
        c: *const c_void,
        d: *mut c_void,
        workspace: *mut c_void,
        ws_size: u64,
        stream: *mut c_void,
    ) -> cutensorStatus_t;
}
