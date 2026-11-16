// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::os::raw::c_char;

use crate::manifest::infer::ManifestHint;
use crate::pull::PullRequest;
use crate::query::query;

use crate::ffi::init::{get_store, runtime_handle};
use crate::ffi::pull::{extract_name_and_intent, GenieXModelPullInput};
use crate::ffi::store::{to_ffi_type, GenieXModelType};
use crate::ffi::types::*;

/// Mirrors `geniex_QuantCandidate`.
#[repr(C)]
pub struct GenieXQuantCandidate {
    pub quant: *mut c_char,
    pub size: i64,
}

/// Mirrors `geniex_ModelQueryOutput`.
#[repr(C)]
pub struct GenieXModelQueryOutput {
    pub model_name: *mut c_char,
    pub plugin_id: *mut c_char,
    pub model_type: GenieXModelType,
    pub candidates: *mut GenieXQuantCandidate,
    pub candidate_count: i32,
}

#[no_mangle]
pub extern "C" fn geniex_model_query(
    input: *const GenieXModelPullInput,
    out: *mut GenieXModelQueryOutput,
) -> i32 {
    ffi_guard(|| {
        let out = out_mut(out)?;
        if input.is_null() {
            return Err(GENIEX_ERROR_COMMON_INVALID_INPUT);
        }
        // Query ignores the pull-only quant / callback / model_type fields.
        let (model_name, intent) = unsafe { extract_name_and_intent(input, "geniex_model_query") }?;
        let req = PullRequest {
            model_name,
            intent,
            on_progress: None,
            hint: ManifestHint::default(),
        };
        let result = runtime_handle()
            .block_on(query(get_store()?, req))
            .or_report()?;

        let cands: Vec<GenieXQuantCandidate> = result
            .candidates
            .iter()
            .map(|c| GenieXQuantCandidate {
                quant: str_to_cptr(&c.quant),
                size: c.size,
            })
            .collect();
        let (candidates, candidate_count) = into_c_array(cands);
        *out = GenieXModelQueryOutput {
            model_name: str_to_cptr(&result.model_name),
            plugin_id: str_to_cptr(&result.plugin_id),
            model_type: to_ffi_type(result.model_type),
            candidates,
            candidate_count,
        };
        Ok(GENIEX_SUCCESS)
    })
}

#[no_mangle]
pub unsafe extern "C" fn geniex_model_query_free(out: *mut GenieXModelQueryOutput) {
    let Some(o) = out.as_mut() else { return };
    free_cptr(o.model_name);
    free_cptr(o.plugin_id);
    free_c_list(o.candidates, o.candidate_count, |c| free_cptr(c.quant));
    *o = std::mem::zeroed();
}
