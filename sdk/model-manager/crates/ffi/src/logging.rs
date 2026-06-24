//! Bridge to geniex's C log callback.
//!
//! On ELF targets we can reach the C++-side global `geniex_log` directly
//! via an `extern "C" { static ... }` declaration, because the linker
//! resolves the reference at shared-object link time.
//!
//! On Windows COFF that shortcut is not available: a static import from
//! a sibling object file requires `__declspec(dllimport)` metadata that
//! we cannot add to a Rust `extern static`, and lld-link rejects the
//! reference with "undefined symbol". To stay portable the FFI crate
//! calls into a thin C trampoline (defined in `sdk/src/logging_bridge.cpp`)
//! that forwards the message to whatever `geniex_set_log` has installed.

use std::ffi::CString;
use std::os::raw::{c_char, c_int};

#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)] // levels are mapped from core's sink via a fn pointer
pub enum GenieXLogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
}

extern "C" {
    // Defined in sdk/src/logging_bridge.cpp — a thin wrapper around the
    // geniex_log function pointer.
    fn geniex_model_log_emit(level: c_int, msg: *const c_char);
}

pub fn log(level: GenieXLogLevel, msg: &str) {
    unsafe {
        if let Ok(cmsg) = CString::new(msg) {
            geniex_model_log_emit(level as c_int, cmsg.as_ptr());
        }
    }
}

/// Route core's pluggable log sink into the C log callback. Core can't
/// depend on this crate (that would invert `ffi -> core`), so it exposes
/// a sink we install once at init time.
pub fn install_core_sink() {
    use model_manager_core::logging::Level;
    model_manager_core::logging::set_sink(|level, msg| {
        let mapped = match level {
            Level::Trace => GenieXLogLevel::Trace,
            Level::Debug => GenieXLogLevel::Debug,
            Level::Info => GenieXLogLevel::Info,
            Level::Warn => GenieXLogLevel::Warn,
            Level::Error => GenieXLogLevel::Error,
        };
        log(mapped, msg);
    });
}

#[inline]
#[allow(dead_code)]
pub fn trace(msg: &str) {
    log(GenieXLogLevel::Trace, msg);
}

#[inline]
#[allow(dead_code)]
pub fn debug(msg: &str) {
    log(GenieXLogLevel::Debug, msg);
}

#[inline]
#[allow(dead_code)]
pub fn info(msg: &str) {
    log(GenieXLogLevel::Info, msg);
}

#[inline]
pub fn warn(msg: &str) {
    log(GenieXLogLevel::Warn, msg);
}

#[inline]
pub fn error(msg: &str) {
    log(GenieXLogLevel::Error, msg);
}
