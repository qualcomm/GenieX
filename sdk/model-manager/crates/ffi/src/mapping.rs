// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::os::raw::c_char;

use model_manager_core::mapping::{is_llmman_reference, resolve_alias};

use crate::pull::GenieXHubSource;
use crate::types::*;

/// Resolve the hub a pull/query would *actually* use for `model_name`, given
/// the caller's requested `hub_in`. Mirrors the `use_llmman` decision inside
/// `extract_name_and_intent`: an explicit `GENIEX_HUB_LLMMAN` stays llmman, and
/// `GENIEX_HUB_AUTO` becomes llmman when the name carries an OCI registry
/// prefix (`docker.io/`, `ghcr.io/`, `oci://`, …). Every other input is
/// returned unchanged.
///
/// Bindings call this to decide binding-side flow (e.g. skip the GGUF precision
/// picker for llmman, whose `:<tag>` is a registry reference, not a quant)
/// without re-implementing the prefix table the SDK owns. No network I/O.
#[no_mangle]
pub extern "C" fn geniex_model_resolve_hub(
    model_name: *const c_char,
    hub_in: GenieXHubSource,
    out_hub: *mut GenieXHubSource,
) -> i32 {
    ffi_guard(|| {
        if out_hub.is_null() {
            return Err(GENIEX_ERROR_COMMON_INVALID_INPUT);
        }
        let name = unsafe { cstr_to_str(model_name) }?;
        let resolved = match hub_in {
            GenieXHubSource::Auto if is_llmman_reference(name) => GenieXHubSource::Llmman,
            other => other,
        };
        unsafe { *out_hub = resolved };
        Ok(GENIEX_SUCCESS)
    })
}

#[no_mangle]
pub extern "C" fn geniex_model_resolve_alias(
    alias: *const c_char,
    out_full_name: *mut *mut c_char,
) -> i32 {
    ffi_guard(|| {
        if out_full_name.is_null() {
            return Err(GENIEX_ERROR_COMMON_INVALID_INPUT);
        }
        let alias_str = unsafe { cstr_to_str(alias) }?;
        let full = resolve_alias(alias_str).ok_or(GENIEX_ERROR_COMMON_INVALID_INPUT)?;
        unsafe { *out_full_name = str_to_cptr(&full) };
        Ok(GENIEX_SUCCESS)
    })
}
