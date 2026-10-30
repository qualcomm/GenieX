// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

mod chipset;
mod hub_models;
mod init;
mod logging;
mod mapping;
mod pull;
mod query;
mod store;
mod types;

// The SDK's C side provides this at link time; tests link without it.
#[cfg(test)]
#[no_mangle]
extern "C" fn geniex_model_log_emit(
    _level: std::os::raw::c_int,
    _msg: *const std::os::raw::c_char,
) {
}
