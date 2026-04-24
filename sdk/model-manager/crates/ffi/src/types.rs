use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};

use model_manager_core::error::Error;

use crate::logging;

/// Error codes mirroring geniex_ErrorCode from geniex.h.
pub const GENIEX_SUCCESS: i32 = 0;
pub const GENIEX_ERROR_COMMON_UNKNOWN: i32 = -100000;
pub const GENIEX_ERROR_COMMON_INVALID_INPUT: i32 = -100001;
pub const GENIEX_ERROR_COMMON_NOT_INITIALIZED: i32 = -100007;
pub const GENIEX_ERROR_COMMON_FILE_NOT_FOUND: i32 = -100004;

/// C-compatible per-file progress entry. Must mirror `geniex_FileProgress`
/// in geniex_model.h.
#[repr(C)]
pub struct GeniexFileProgress {
    pub file_name: *const c_char,
    pub downloaded_bytes: i64,
    pub total_bytes: i64,
}

pub fn err_to_code(e: &Error) -> i32 {
    match e {
        Error::NotInitialized => GENIEX_ERROR_COMMON_NOT_INITIALIZED,
        Error::ModelNotFound(_) => GENIEX_ERROR_COMMON_FILE_NOT_FOUND,
        Error::QuantNotFound(_, _)
        | Error::QuantNotDownloaded(_, _)
        | Error::NoDownloadedQuant(_)
        | Error::InvalidModelName(_)
        | Error::InvalidFileName(_) => GENIEX_ERROR_COMMON_INVALID_INPUT,
        _ => GENIEX_ERROR_COMMON_UNKNOWN,
    }
}

/// Log `e` via the geniex log callback and return the corresponding code.
pub fn report(e: &Error) -> i32 {
    logging::error(&format!("{e}"));
    err_to_code(e)
}

/// Wrap an FFI entry point so panics can't cross the C boundary.
/// Any panic is logged and converted to `GENIEX_ERROR_COMMON_UNKNOWN`.
pub fn ffi_guard<F>(f: F) -> i32
where
    F: FnOnce() -> i32,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(code) => code,
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "panic in FFI boundary".to_string()
            };
            logging::error(&format!("panic caught at FFI boundary: {msg}"));
            GENIEX_ERROR_COMMON_UNKNOWN
        }
    }
}

/// Convert a raw C string pointer to a &str. Returns None if ptr is null or invalid UTF-8.
pub unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        None
    } else {
        CStr::from_ptr(ptr).to_str().ok()
    }
}

/// Allocate a CString from a Rust &str and return its raw pointer.
/// The caller frees it via `geniex_free` (or a dedicated `_free` helper).
pub fn str_to_cptr(s: &str) -> *mut c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

/// Free a CString pointer allocated by `str_to_cptr`.
pub unsafe fn free_cptr(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

// Silence unused warning for c_void import when building without features that
// use it; pull.rs re-imports c_void directly when it needs it.
#[allow(dead_code)]
pub(crate) type VoidPtr = *mut c_void;
