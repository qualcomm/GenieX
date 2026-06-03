use std::os::raw::c_char;

use model_manager_core::manifest_builder::ManifestHint;
use model_manager_core::pull::PullRequest;
use model_manager_core::query::query_blocking;

use crate::init::{get_store, runtime_handle};
use crate::pull::{extract_name_and_intent, GenieXModelPullInput};
use crate::store::{to_ffi_type, GenieXModelType};
use crate::types::*;

/// One advertised quantization. Mirrors `geniex_QuantCandidate`.
#[repr(C)]
pub struct GenieXQuantCandidate {
    pub quant: *mut c_char,
    pub size: i64,
    pub is_default: bool,
}

/// Result of `geniex_model_query`. Mirrors `geniex_ModelQueryOutput`.
#[repr(C)]
pub struct GenieXModelQueryOutput {
    pub model_name: *mut c_char,
    pub plugin_id: *mut c_char,
    pub model_type: GenieXModelType,
    pub candidates: *mut GenieXQuantCandidate,
    pub candidate_count: i32,
}

impl GenieXModelQueryOutput {
    fn null() -> Self {
        Self {
            model_name: std::ptr::null_mut(),
            plugin_id: std::ptr::null_mut(),
            model_type: GenieXModelType::Llm,
            candidates: std::ptr::null_mut(),
            candidate_count: 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn geniex_model_query(
    input: *const GenieXModelPullInput,
    out: *mut GenieXModelQueryOutput,
) -> i32 {
    ffi_guard(|| {
        if input.is_null() || out.is_null() {
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        // Query reuses the pull input struct; the quant / callback / model_type
        // fields are simply ignored here.
        let (model_name, intent) =
            match unsafe { extract_name_and_intent(input, "geniex_model_query") } {
                Ok(v) => v,
                Err(c) => return c,
            };

        let store = match get_store() {
            Ok(s) => s,
            Err(c) => return c,
        };

        let req = PullRequest {
            model_name,
            intent,
            on_progress: None,
            hint: ManifestHint::default(),
        };

        let result = match query_blocking(&runtime_handle(), store, req) {
            Ok(r) => r,
            Err(e) => return report(&e),
        };

        let mut cands: Vec<GenieXQuantCandidate> = result
            .candidates
            .iter()
            .map(|c| GenieXQuantCandidate {
                quant: str_to_cptr(&c.quant),
                size: c.size,
                is_default: c.is_default,
            })
            .collect();
        cands.shrink_to_fit();
        let candidate_count = cands.len() as i32;
        let candidates = if cands.is_empty() {
            std::ptr::null_mut()
        } else {
            cands.as_mut_ptr()
        };
        std::mem::forget(cands);

        unsafe {
            (*out).model_name = str_to_cptr(&result.model_name);
            (*out).plugin_id = str_to_cptr(&result.plugin_id);
            (*out).model_type = to_ffi_type(result.model_type);
            (*out).candidates = candidates;
            (*out).candidate_count = candidate_count;
        }
        GENIEX_SUCCESS
    })
}

#[no_mangle]
pub unsafe extern "C" fn geniex_model_query_free(out: *mut GenieXModelQueryOutput) {
    if out.is_null() {
        return;
    }
    let o = &mut *out;
    free_cptr(o.model_name);
    free_cptr(o.plugin_id);
    if !o.candidates.is_null() {
        let slice = std::slice::from_raw_parts_mut(o.candidates, o.candidate_count as usize);
        for c in slice.iter_mut() {
            free_cptr(c.quant);
        }
        drop(Vec::from_raw_parts(
            o.candidates,
            o.candidate_count as usize,
            o.candidate_count as usize,
        ));
    }
    *out = GenieXModelQueryOutput::null();
}
