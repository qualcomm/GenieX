// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::os::raw::c_char;

use crate::model_ref::{
    aihub_display_name_from_repo, canonicalize_model_name, is_docker_hub_reference, resolve_alias,
};

use crate::ffi::pull::GenieXHubSource;
use crate::ffi::types::*;

/// Must agree with the Docker and AI Hub routing in `Hub::resolve`.
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
            GenieXHubSource::Auto
                if aihub_display_name_from_repo(&canonicalize_model_name(name)).is_some() =>
            {
                GenieXHubSource::AiHub
            }
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
