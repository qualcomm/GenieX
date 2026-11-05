// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::error::Error;

use crate::ffi::logging;

thread_local! {
    /// Kept alive so `geniex_model_last_error_message` can hand out a stable pointer.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: &str) {
    let c = CString::new(msg).unwrap_or_default();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(c));
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

#[no_mangle]
pub extern "C" fn geniex_model_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr())
    })
}

/// Mirrors geniex_ErrorCode from geniex.h.
pub const GENIEX_SUCCESS: i32 = 0;
pub const GENIEX_ERROR_COMMON_UNKNOWN: i32 = -100000;
pub const GENIEX_ERROR_COMMON_INVALID_INPUT: i32 = -100001;
pub const GENIEX_ERROR_COMMON_FILE_NOT_FOUND: i32 = -100004;
pub const GENIEX_ERROR_COMMON_NETWORK: i32 = -100005;
pub const GENIEX_ERROR_COMMON_CANCELLED: i32 = -100006;
pub const GENIEX_ERROR_COMMON_NOT_INITIALIZED: i32 = -100007;
pub const GENIEX_ERROR_COMMON_ALREADY_INITIALIZED: i32 = -100008;
pub const GENIEX_ERROR_COMMON_AUTH: i32 = -100009;
pub const GENIEX_ERROR_COMMON_HUB_MODEL_NOT_FOUND: i32 = -100010;
pub const GENIEX_ERROR_COMMON_RATE_LIMITED: i32 = -100011;
pub const GENIEX_ERROR_COMMON_HUB_SERVER: i32 = -100012;
pub const GENIEX_ERROR_COMMON_MANIFEST_PARSE: i32 = -100014;
pub const GENIEX_ERROR_COMMON_CHIPSET_UNAVAILABLE: i32 = -100015;
pub const GENIEX_ERROR_COMMON_INSUFFICIENT_DISK_SPACE: i32 = -100017;
pub const GENIEX_ERROR_COMMON_MODEL_INVALID: i32 = -100203;

/// Must mirror `geniex_FileProgress` in geniex_model.h.
#[repr(C)]
pub struct GenieXFileProgress {
    pub file_name: *const c_char,
    pub downloaded_bytes: i64,
    pub total_bytes: i64,
}

pub fn err_to_code(e: &Error) -> i32 {
    match e {
        Error::ModelNotFound(_) => GENIEX_ERROR_COMMON_FILE_NOT_FOUND,
        Error::HubModelNotFound(_) => GENIEX_ERROR_COMMON_HUB_MODEL_NOT_FOUND,
        Error::QuantNotFound(_, _)
        | Error::QuantNotDownloaded(_, _)
        | Error::InvalidModelName(_)
        | Error::InvalidUrl { .. } => GENIEX_ERROR_COMMON_INVALID_INPUT,
        Error::HttpStatus { status, .. } if *status == 401 || *status == 403 => {
            GENIEX_ERROR_COMMON_AUTH
        }
        Error::HttpStatus { status, .. } if *status == 404 => {
            GENIEX_ERROR_COMMON_HUB_MODEL_NOT_FOUND
        }
        Error::HttpStatus { status, .. } if *status == 429 => GENIEX_ERROR_COMMON_RATE_LIMITED,
        Error::HttpStatus { status, .. } if *status >= 500 => GENIEX_ERROR_COMMON_HUB_SERVER,
        Error::HttpStatus { .. } | Error::HttpTimeout(_) | Error::Http(_) => {
            GENIEX_ERROR_COMMON_NETWORK
        }
        Error::ManifestParse { .. } | Error::Json(_) => GENIEX_ERROR_COMMON_MANIFEST_PARSE,
        Error::ChipsetUnavailable { .. } => GENIEX_ERROR_COMMON_CHIPSET_UNAVAILABLE,
        Error::InsufficientDiskSpace { .. } => GENIEX_ERROR_COMMON_INSUFFICIENT_DISK_SPACE,
        Error::Cancelled => GENIEX_ERROR_COMMON_CANCELLED,
        Error::ManifestInferenceFailed(_) => GENIEX_ERROR_COMMON_MODEL_INVALID,
        Error::Io(_) | Error::Hub(_) => GENIEX_ERROR_COMMON_UNKNOWN,
    }
}

pub fn report(e: &Error) -> i32 {
    report_msg(&e.to_string(), err_to_code(e))
}

pub fn report_msg(msg: &str, code: i32) -> i32 {
    logging::error(msg);
    set_last_error(msg);
    code
}

/// Keeps panics from unwinding across the C boundary; `Ok` and `Err` both carry the return code.
pub fn ffi_guard<F>(f: F) -> i32
where
    F: FnOnce() -> Result<i32, i32>,
{
    // The last error must reflect only this call.
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(code)) | Ok(Err(code)) => code,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&'static str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic in FFI boundary".to_string());
            logging::error(&format!("panic caught at FFI boundary: {msg}"));
            set_last_error(&msg);
            GENIEX_ERROR_COMMON_UNKNOWN
        }
    }
}

/// # Safety
/// `ptr` must be null or point to a NUL-terminated C string valid for
/// the lifetime `'a`.
pub unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Result<&'a str, i32> {
    if ptr.is_null() {
        return Err(GENIEX_ERROR_COMMON_INVALID_INPUT);
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map_err(|_| GENIEX_ERROR_COMMON_INVALID_INPUT)
}

/// Freed via `geniex_free` or a dedicated `_free` helper.
pub fn str_to_cptr(s: &str) -> *mut c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

pub unsafe fn free_cptr(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Every geniex_model_*_free must reclaim via [`from_c_array`] with the same `(T, len)`.
pub fn into_c_array<T>(mut v: Vec<T>) -> (*mut T, i32) {
    v.shrink_to_fit();
    let len = v.len() as i32;
    if v.is_empty() {
        return (std::ptr::null_mut(), 0);
    }
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    (ptr, len)
}

pub fn c_str_array<S: AsRef<str>>(items: &[S]) -> (*mut *mut c_char, i32) {
    into_c_array(items.iter().map(|s| str_to_cptr(s.as_ref())).collect())
}

/// # Safety
/// Same contract as [`from_c_array`].
pub unsafe fn free_c_str_array(ptr: *mut *mut c_char, count: i32) {
    for s in from_c_array(ptr, count).unwrap_or_default() {
        free_cptr(s);
    }
}

/// # Safety
/// Same contract as [`from_c_array`].
pub unsafe fn free_c_list<T>(ptr: *mut T, count: i32, mut free: impl FnMut(&mut T)) {
    for mut item in from_c_array(ptr, count).unwrap_or_default() {
        free(&mut item);
    }
}

pub fn out_mut<'a, T>(out: *mut T) -> Result<&'a mut T, i32> {
    unsafe { out.as_mut() }.ok_or(GENIEX_ERROR_COMMON_INVALID_INPUT)
}

pub trait OrReport<T> {
    fn or_report(self) -> Result<T, i32>;
}

impl<T> OrReport<T> for crate::error::Result<T> {
    fn or_report(self) -> Result<T, i32> {
        self.map_err(|e| report(&e))
    }
}

/// # Safety
/// `ptr` must either be null, or come from [`into_c_array`] with the same
/// `T` and matching `count`; the caller must guarantee it isn't freed twice.
pub unsafe fn from_c_array<T>(ptr: *mut T, count: i32) -> Option<Vec<T>> {
    if ptr.is_null() {
        return None;
    }
    let len = count as usize;
    Some(Vec::from_raw_parts(ptr, len, len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn err_to_code_cases() {
        let http = |status| Error::HttpStatus {
            url: "https://hub.example/x".to_string(),
            status,
        };
        let cases = [
            (http(401), GENIEX_ERROR_COMMON_AUTH),
            (http(403), GENIEX_ERROR_COMMON_AUTH),
            (http(404), GENIEX_ERROR_COMMON_HUB_MODEL_NOT_FOUND),
            (http(429), GENIEX_ERROR_COMMON_RATE_LIMITED),
            (http(500), GENIEX_ERROR_COMMON_HUB_SERVER),
            (http(503), GENIEX_ERROR_COMMON_HUB_SERVER),
            (http(400), GENIEX_ERROR_COMMON_NETWORK),
            (
                Error::HttpTimeout("dns".into()),
                GENIEX_ERROR_COMMON_NETWORK,
            ),
            (Error::Http("reset".into()), GENIEX_ERROR_COMMON_NETWORK),
            (
                Error::ModelNotFound("m".into()),
                GENIEX_ERROR_COMMON_FILE_NOT_FOUND,
            ),
            (
                Error::HubModelNotFound("m".into()),
                GENIEX_ERROR_COMMON_HUB_MODEL_NOT_FOUND,
            ),
            (Error::Hub("misc".into()), GENIEX_ERROR_COMMON_UNKNOWN),
        ];
        for (err, want) in cases {
            assert_eq!(err_to_code(&err), want, "{err:?}");
        }
    }
}
