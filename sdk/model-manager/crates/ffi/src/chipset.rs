// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::os::raw::c_char;

use model_manager_core::config::StoreConfig;
use model_manager_core::source::ai_hub::{
    detect_host_chipset_reference, list_supported_chipsets, AiHubConfig,
};

use crate::init::{get_store, runtime_handle};
use crate::types::*;

/* ---- geniex_model_list_chipsets ---- */

#[repr(C)]
pub struct GenieXChipsetInfo {
    pub name: *mut c_char,
    /// Heap array of alias strings; `alias_count` long.
    pub aliases: *mut *mut c_char,
    pub alias_count: i32,
}

#[repr(C)]
pub struct GenieXChipsetList {
    pub chipsets: *mut GenieXChipsetInfo,
    pub count: i32,
}

fn ai_hub_cfg_for_chipset_query(store: &model_manager_core::store::Store) -> AiHubConfig {
    AiHubConfig::new(
        StoreConfig::ai_hub_base_url(),
        StoreConfig::ai_hub_version(),
        String::new(),
        store.config().ai_hub_cache_dir(),
        false,
    )
}

#[no_mangle]
pub extern "C" fn geniex_model_list_chipsets(out: *mut GenieXChipsetList) -> i32 {
    ffi_guard(|| {
        if out.is_null() {
            return Err(GENIEX_ERROR_COMMON_INVALID_INPUT);
        }
        let cfg = ai_hub_cfg_for_chipset_query(get_store()?);
        let chipsets = runtime_handle()
            .block_on(list_supported_chipsets(&cfg))
            .map_err(|e| report(&e))?;

        let infos: Vec<GenieXChipsetInfo> = chipsets
            .iter()
            .map(|c| {
                // Surface reference_device as `name` and demote the canonical
                // chipset id into the alias list; fall back to the id when no
                // reference device exists.
                let display = if c.reference_device.is_empty() {
                    c.name.as_str()
                } else {
                    c.reference_device.as_str()
                };
                let mut alias_strs: Vec<&str> = Vec::with_capacity(c.aliases.len() + 1);
                if !c.aliases.iter().any(|a| a == &c.name) {
                    alias_strs.push(c.name.as_str());
                }
                alias_strs.extend(c.aliases.iter().map(String::as_str));
                let aliases_vec: Vec<*mut c_char> =
                    alias_strs.iter().map(|a| str_to_cptr(a)).collect();
                let (aliases, alias_count) = into_c_array(aliases_vec);
                GenieXChipsetInfo {
                    name: str_to_cptr(display),
                    aliases,
                    alias_count,
                }
            })
            .collect();
        let (chipsets_ptr, count) = into_c_array(infos);
        unsafe {
            (*out).chipsets = chipsets_ptr;
            (*out).count = count;
        }
        Ok(GENIEX_SUCCESS)
    })
}

#[no_mangle]
pub unsafe extern "C" fn geniex_model_list_chipsets_free(out: *mut GenieXChipsetList) {
    if out.is_null() {
        return;
    }
    let o = &mut *out;
    if let Some(mut infos) = from_c_array(o.chipsets, o.count) {
        for info in infos.iter_mut() {
            free_cptr(info.name);
            if let Some(aliases) = from_c_array(info.aliases, info.alias_count) {
                for a in aliases {
                    free_cptr(a);
                }
            }
        }
    }
    o.chipsets = std::ptr::null_mut();
    o.count = 0;
}

/* ---- geniex_model_detect_chipset ---- */

#[no_mangle]
pub extern "C" fn geniex_model_detect_chipset(out_chipset: *mut *mut c_char) -> i32 {
    ffi_guard(|| {
        if out_chipset.is_null() {
            return Err(GENIEX_ERROR_COMMON_INVALID_INPUT);
        }
        let cfg = ai_hub_cfg_for_chipset_query(get_store()?);
        let ptr = runtime_handle()
            .block_on(detect_host_chipset_reference(&cfg))
            .map(|s| str_to_cptr(&s))
            .unwrap_or(std::ptr::null_mut());
        unsafe { *out_chipset = ptr };
        Ok(GENIEX_SUCCESS)
    })
}
