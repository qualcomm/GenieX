// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::os::raw::c_char;

use model_manager_core::config::StoreConfig;
use model_manager_core::source::ai_hub::{list_hub_models, AiHubConfig};

use crate::init::{get_store, runtime_handle};
use crate::store::{to_ffi_type, GenieXModelType};
use crate::types::*;

/* ---- geniex_model_list_hub ---- */

#[repr(C)]
pub struct GenieXHubModelInfo {
    /// Pullable model name, e.g. `qualcomm/Qwen3-4B`.
    pub name: *mut c_char,
    pub model_type: GenieXModelType,
    pub chipsets: *mut *mut c_char,
    pub chipset_count: i32,
}

#[repr(C)]
pub struct GenieXHubModelList {
    pub models: *mut GenieXHubModelInfo,
    pub count: i32,
}

/// List AI Hub models geniex can run. `chipset` is a NUL-terminated canonical
/// chipset id to filter by, or NULL to list every model. Detecting the host
/// chipset is the caller's job (see `geniex_model_detect_chipset`).
#[no_mangle]
pub extern "C" fn geniex_model_list_hub(
    chipset: *const c_char,
    out: *mut GenieXHubModelList,
) -> i32 {
    ffi_guard(|| {
        if out.is_null() {
            return Err(GENIEX_ERROR_COMMON_INVALID_INPUT);
        }
        let chipset = if chipset.is_null() {
            None
        } else {
            Some(unsafe { cstr_to_str(chipset)? })
        };
        let store = get_store()?;
        let models = runtime_handle()
            .block_on(async {
                let endpoint = StoreConfig::ai_hub_base_url();
                let cache_dir = store.config().ai_hub_cache_dir();
                let version = StoreConfig::ai_hub_version_override()
                    .unwrap_or_else(StoreConfig::ai_hub_version_fallback);
                let cfg = AiHubConfig::new(endpoint, version, String::new(), cache_dir, false);
                list_hub_models(&cfg, chipset).await
            })
            .map_err(|e| report(&e))?;

        let infos: Vec<GenieXHubModelInfo> = models
            .into_iter()
            .map(|m| {
                let chipsets_vec: Vec<*mut c_char> = m
                    .supported_chipsets
                    .iter()
                    .map(|c| str_to_cptr(c))
                    .collect();
                let (chipsets, chipset_count) = into_c_array(chipsets_vec);
                GenieXHubModelInfo {
                    name: str_to_cptr(&format!("qualcomm/{}", m.display_name)),
                    model_type: to_ffi_type(m.model_type),
                    chipsets,
                    chipset_count,
                }
            })
            .collect();
        let (models_ptr, count) = into_c_array(infos);
        unsafe {
            (*out).models = models_ptr;
            (*out).count = count;
        }
        Ok(GENIEX_SUCCESS)
    })
}

#[no_mangle]
pub unsafe extern "C" fn geniex_model_list_hub_free(out: *mut GenieXHubModelList) {
    if out.is_null() {
        return;
    }
    let o = &mut *out;
    if let Some(mut infos) = from_c_array(o.models, o.count) {
        for info in infos.iter_mut() {
            free_cptr(info.name);
            if let Some(chipsets) = from_c_array(info.chipsets, info.chipset_count) {
                for c in chipsets {
                    free_cptr(c);
                }
            }
        }
    }
    o.models = std::ptr::null_mut();
    o.count = 0;
}
