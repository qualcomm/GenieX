// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::os::raw::c_char;

use model_manager_core::source::ai_hub::{
    detect_host_chipset_reference, list_supported_chipsets, AiHubConfig,
};

use crate::init::{get_store, runtime_handle};
use crate::types::*;

#[repr(C)]
pub struct GenieXChipsetInfo {
    pub name: *mut c_char,
    pub aliases: *mut *mut c_char,
    pub alias_count: i32,
}

#[repr(C)]
pub struct GenieXChipsetList {
    pub chipsets: *mut GenieXChipsetInfo,
    pub count: i32,
}

#[no_mangle]
pub extern "C" fn geniex_model_list_chipsets(out: *mut GenieXChipsetList) -> i32 {
    ffi_guard(|| {
        let out = out_mut(out)?;
        let cfg = AiHubConfig::from_store(get_store()?, String::new());
        let chipsets = runtime_handle()
            .block_on(list_supported_chipsets(&cfg))
            .or_report()?;

        let infos: Vec<GenieXChipsetInfo> = chipsets
            .iter()
            .map(|c| {
                // The canonical id is demoted into the alias list.
                let mut aliases: Vec<&str> = Vec::with_capacity(c.aliases.len() + 1);
                if !c.aliases.contains(&c.name) {
                    aliases.push(&c.name);
                }
                aliases.extend(c.aliases.iter().map(String::as_str));
                let (aliases, alias_count) = c_str_array(&aliases);
                GenieXChipsetInfo {
                    name: str_to_cptr(c.display_name()),
                    aliases,
                    alias_count,
                }
            })
            .collect();
        let (chipsets, count) = into_c_array(infos);
        *out = GenieXChipsetList { chipsets, count };
        Ok(GENIEX_SUCCESS)
    })
}

#[no_mangle]
pub unsafe extern "C" fn geniex_model_list_chipsets_free(out: *mut GenieXChipsetList) {
    let Some(o) = out.as_mut() else { return };
    free_c_list(o.chipsets, o.count, |info| {
        free_cptr(info.name);
        free_c_str_array(info.aliases, info.alias_count);
    });
    *o = std::mem::zeroed();
}

#[no_mangle]
pub extern "C" fn geniex_model_detect_chipset(offline: i32, out_chipset: *mut *mut c_char) -> i32 {
    ffi_guard(|| {
        let out = out_mut(out_chipset)?;
        // offline: local probe only; else translate online to the display name.
        let detected = if offline != 0 {
            model_manager_core::host::detect_chipset()
        } else {
            let cfg = AiHubConfig::from_store(get_store()?, String::new());
            runtime_handle().block_on(detect_host_chipset_reference(&cfg))
        };
        *out = detected.map_or(std::ptr::null_mut(), |s| str_to_cptr(&s));
        Ok(GENIEX_SUCCESS)
    })
}
