// borrowed from https://github.com/supranational/sppark/blob/main/rust/src/lib.rs
// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Raw FFI status struct returned by value from every `_halo2_*` extern fn.
/// Mirrors the C++ `RustError` return struct: `int code; char* str` (where
/// `str` is a `strdup`'d message owned by the C side, freed via this
/// type's `Drop` impl).
///
/// This is the wire-level FFI ABI type, **not** the Rust-side `Result`
/// error type. The canonical Rust-side error is
/// `openvm_cuda_common::error::CudaError`; `cuda::funcs` wrappers convert
/// at the boundary via `CudaError::new(status.code)`.
#[repr(C)]
#[derive(Debug)]
pub struct CudaStatus {
    pub code: i32,
    str: Option<core::ptr::NonNull<i8>>, // just strdup("string") from C/C++
}

// Compile-time FFI safety assertions.
// `CudaStatus` is returned by value from every halo2 CUDA FFI fn, so its
// layout must match the C++ return struct `int code; char* str;` on the
// target platform. `Option<NonNull<T>>` is niche-optimized to the same
// layout as the raw pointer (8 bytes on 64-bit), giving a 4-byte i32 +
// 4-byte padding + 8-byte pointer = 16 bytes total.
const _: () = assert!(
    std::mem::size_of::<CudaStatus>() == 16,
    "CudaStatus must be 16 bytes (i32 code + 4-byte padding + 8-byte niche-opt pointer) to match the C++ return struct"
);
const _: () = assert!(
    std::mem::align_of::<CudaStatus>() == 8,
    "CudaStatus alignment must be 8 to match the 8-byte pointer field in the C++ return struct"
);

impl Drop for CudaStatus {
    fn drop(&mut self) {
        extern "C" {
            fn free(str: Option<core::ptr::NonNull<i8>>);
        }
        unsafe { free(self.str) };
        self.str = None;
    }
}

impl From<CudaStatus> for String {
    fn from(status: CudaStatus) -> Self {
        let c_str = if let Some(ptr) = status.str {
            unsafe { std::ffi::CStr::from_ptr(ptr.as_ptr()) }
        } else {
            extern "C" {
                fn cudaGetErrorString(code: i32) -> *const i8;
            }
            unsafe { std::ffi::CStr::from_ptr(cudaGetErrorString(status.code)) }
        };
        String::from(c_str.to_str().unwrap_or("unintelligible"))
    }
}

/// Convert an `_halo2_*` extern FFI return into the canonical
/// `cuda_common::CudaError`. Preserves the C++-side `strdup`'d message via
/// `String::from(CudaStatus)` (which consumes the status and frees the C
/// buffer through its `Drop` impl); for null messages that conversion falls
/// back to `cudaGetErrorString`. The `name` field is looked up via the CUDA
/// driver from `code`, so the resulting `CudaError` carries the standard
/// CUDA error name alongside the C++-supplied message.
impl From<CudaStatus> for openvm_cuda_common::error::CudaError {
    fn from(status: CudaStatus) -> Self {
        let code = status.code;
        let message = String::from(status);
        let name = openvm_cuda_common::error::get_cuda_error_name(code);
        Self { code, name, message }
    }
}

impl From<CudaStatus> for Result<(), anyhow::Error> {
    fn from(err: CudaStatus) -> Self {
        if err.code == 0 {
            Ok(())
        } else {
            Err(anyhow::anyhow!("CUDA Error: {}", String::from(err)))
        }
    }
}
