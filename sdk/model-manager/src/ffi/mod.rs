// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! The `geniex_model.h` C ABI: each entry point converts C inputs, calls the
//! Rust API, and writes C outputs.

mod chipset;
mod hub_models;
mod init;
mod logging;
mod mapping;
mod pull;
mod query;
mod store;
mod types;

// The SDK's C side provides this at link time; Rust test binaries link without it.
#[cfg(any(test, feature = "test-support"))]
#[no_mangle]
extern "C" fn geniex_model_log_emit(
    _level: std::os::raw::c_int,
    _msg: *const std::os::raw::c_char,
) {
}
