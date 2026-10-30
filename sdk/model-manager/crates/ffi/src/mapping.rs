// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::os::raw::c_char;

use model_manager_core::model_ref::{is_docker_hub_reference, resolve_alias};

use crate::pull::GenieXHubSource;
use crate::types::*;

/// Must stay in sync with the `use_docker` decision in `extract_name_and_intent`.
#[no_mangle]
pub extern "C" fn geniex_model_resolve_hub(
    model_name: *const c_char,
    hub_in: GenieXHubSource,
    out_hub: *mut GenieXHubSource,
) -> i32 {
    ffi_guard(|| {
        let out = out_mut(out_hub)?;
        let name = unsafe { cstr_to_str(model_name) }?;
        *out = match hub_in {
            GenieXHubSource::Auto if is_docker_hub_reference(name) => GenieXHubSource::Docker,
            other => other,
        };
        Ok(GENIEX_SUCCESS)
    })
}

#[no_mangle]
pub extern "C" fn geniex_model_resolve_alias(
    alias: *const c_char,
    out_full_name: *mut *mut c_char,
) -> i32 {
    ffi_guard(|| {
        let out = out_mut(out_full_name)?;
        let alias_str = unsafe { cstr_to_str(alias) }?;
        let full = resolve_alias(alias_str).ok_or(GENIEX_ERROR_COMMON_INVALID_INPUT)?;
        *out = str_to_cptr(&full);
        Ok(GENIEX_SUCCESS)
    })
}
