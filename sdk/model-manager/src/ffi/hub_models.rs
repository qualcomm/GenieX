// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::os::raw::c_char;

use crate::source::ai_hub::{list_hub_models, AiHubConfig};

use crate::ffi::init::{get_store, runtime_handle};
use crate::ffi::store::{to_ffi_type, GenieXModelType};
use crate::ffi::types::*;

#[repr(C)]
pub struct GenieXHubModelInfo {
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

#[no_mangle]
pub extern "C" fn geniex_model_list_hub(
    chipset: *const c_char,
    out: *mut GenieXHubModelList,
) -> i32 {
    ffi_guard(|| {
        let out = out_mut(out)?;
        // Invalid UTF-8 is an error here, unlike an absent filter.
        let chipset = if chipset.is_null() {
            None
        } else {
            Some(unsafe { cstr_to_str(chipset)? })
        };
        let cfg = AiHubConfig::from_store(get_store()?, String::new());
        let models = runtime_handle()
            .block_on(list_hub_models(&cfg, chipset))
            .or_report()?;

        let infos: Vec<GenieXHubModelInfo> = models
            .into_iter()
            .map(|m| {
                let (chipsets, chipset_count) = c_str_array(&m.supported_chipsets);
                GenieXHubModelInfo {
                    name: str_to_cptr(&format!("qualcomm/{}", m.display_name)),
                    model_type: to_ffi_type(m.model_type),
                    chipsets,
                    chipset_count,
                }
            })
            .collect();
        let (models, count) = into_c_array(infos);
        *out = GenieXHubModelList { models, count };
        Ok(GENIEX_SUCCESS)
    })
}

#[no_mangle]
pub unsafe extern "C" fn geniex_model_list_hub_free(out: *mut GenieXHubModelList) {
    let Some(o) = out.as_mut() else { return };
    free_c_list(o.models, o.count, |info| {
        free_cptr(info.name);
        free_c_str_array(info.chipsets, info.chipset_count);
    });
    *o = std::mem::zeroed();
}
