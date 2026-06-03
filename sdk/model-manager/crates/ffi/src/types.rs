use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};

use model_manager_core::error::Error;

use crate::logging;

thread_local! {
    /// Detailed message for the most recent failure on this thread, kept as a
    /// live CString so `geniex_model_last_error_message` can hand out a stable
    /// pointer valid until the next failing call.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: &str) {
    let c = CString::new(msg).unwrap_or_default();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(c));
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Return the calling thread's last recorded error message, or NULL.
/// Library-owned; valid until the next failing geniex_model_* call.
#[no_mangle]
pub extern "C" fn geniex_model_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr())
    })
}

/// Error codes mirroring geniex_ErrorCode from geniex.h.
pub const GENIEX_SUCCESS: i32 = 0;
pub const GENIEX_ERROR_COMMON_UNKNOWN: i32 = -100000;
pub const GENIEX_ERROR_COMMON_INVALID_INPUT: i32 = -100001;
pub const GENIEX_ERROR_COMMON_FILE_NOT_FOUND: i32 = -100004;
pub const GENIEX_ERROR_COMMON_NETWORK: i32 = -100005;
pub const GENIEX_ERROR_COMMON_CANCELLED: i32 = -100006;
pub const GENIEX_ERROR_COMMON_NOT_INITIALIZED: i32 = -100007;
pub const GENIEX_ERROR_COMMON_ALREADY_INITIALIZED: i32 = -100008;
pub const GENIEX_ERROR_COMMON_MANIFEST_PARSE: i32 = -100014;
pub const GENIEX_ERROR_COMMON_CHIPSET_UNAVAILABLE: i32 = -100015;
pub const GENIEX_ERROR_COMMON_MODEL_INVALID: i32 = -100203;

/// C-compatible per-file progress entry. Must mirror `geniex_FileProgress`
/// in geniex_model.h.
#[repr(C)]
pub struct GenieXFileProgress {
    pub file_name: *const c_char,
    pub downloaded_bytes: i64,
    pub total_bytes: i64,
}

pub fn err_to_code(e: &Error) -> i32 {
    match e {
        Error::NotInitialized => GENIEX_ERROR_COMMON_NOT_INITIALIZED,
        Error::AlreadyInitialized => GENIEX_ERROR_COMMON_ALREADY_INITIALIZED,
        Error::ModelNotFound(_) => GENIEX_ERROR_COMMON_FILE_NOT_FOUND,
        Error::QuantNotFound(_, _)
        | Error::QuantNotDownloaded(_, _)
        | Error::NoDownloadedQuant(_)
        | Error::InvalidModelName(_)
        | Error::InvalidFileName(_) => GENIEX_ERROR_COMMON_INVALID_INPUT,
        Error::HttpStatus { .. } | Error::HttpTimeout(_) | Error::Http(_) => {
            GENIEX_ERROR_COMMON_NETWORK
        }
        // A local geniex.json that fails to deserialize is a manifest-parse
        // failure, same category as a malformed remote index.
        Error::ManifestParse { .. } | Error::Json(_) => GENIEX_ERROR_COMMON_MANIFEST_PARSE,
        Error::ChipsetUnavailable { .. } => GENIEX_ERROR_COMMON_CHIPSET_UNAVAILABLE,
        Error::Cancelled => GENIEX_ERROR_COMMON_CANCELLED,
        // Inference failed because the source has no files we recognize as a
        // model — surface it as an invalid-model error, not a generic unknown.
        Error::ManifestInferenceFailed(_) => GENIEX_ERROR_COMMON_MODEL_INVALID,
        // Io / Hub are genuinely freeform; keep them as unknown.
        Error::Io(_) | Error::Hub(_) => GENIEX_ERROR_COMMON_UNKNOWN,
    }
}

/// Log `e` via the geniex log callback, stash its message as this thread's
/// last error, and return the corresponding code.
pub fn report(e: &Error) -> i32 {
    let msg = format!("{e}");
    logging::error(&msg);
    set_last_error(&msg);
    err_to_code(e)
}

/// Wrap an FFI entry point so panics can't cross the C boundary.
/// Any panic is logged and converted to `GENIEX_ERROR_COMMON_UNKNOWN`.
pub fn ffi_guard<F>(f: F) -> i32
where
    F: FnOnce() -> i32,
{
    // Clear any message left by a prior call on this thread so
    // geniex_model_last_error_message only ever reflects THIS call's outcome
    // (set again by report() / the panic arm below on failure).
    clear_last_error();
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
            set_last_error(&msg);
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
