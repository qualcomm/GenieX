// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Bridge to geniex's C log callback through a C trampoline (`sdk/src/logging_bridge.cpp`):
//! on Windows COFF a Rust `extern static` can't carry the `dllimport` needed to
//! read `geniex_log` directly.

use std::ffi::CString;
use std::os::raw::{c_char, c_int};

#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)] // levels are mapped from the crate's sink via a fn pointer
pub enum GenieXLogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
}

extern "C" {
    fn geniex_model_log_emit(level: c_int, msg: *const c_char);
}

pub fn log(level: GenieXLogLevel, msg: &str) {
    unsafe {
        if let Ok(cmsg) = CString::new(msg) {
            geniex_model_log_emit(level as c_int, cmsg.as_ptr());
        }
    }
}

pub fn install_core_sink() {
    use crate::logging::Level;
    crate::logging::set_sink(|level, msg| {
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
pub fn debug(msg: &str) {
    log(GenieXLogLevel::Debug, msg);
}

#[inline]
pub fn warn(msg: &str) {
    log(GenieXLogLevel::Warn, msg);
}

#[inline]
pub fn error(msg: &str) {
    log(GenieXLogLevel::Error, msg);
}
