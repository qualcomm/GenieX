use std::os::raw::c_char;

use model_manager_core::config::StoreConfig;
use model_manager_core::source::ai_hub::detect::detect_host_chipset;
use model_manager_core::source::ai_hub::{list_supported_chipsets, AiHubConfig};

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

#[no_mangle]
pub extern "C" fn geniex_model_list_chipsets(out: *mut GenieXChipsetList) -> i32 {
    ffi_guard(|| {
        if out.is_null() {
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        let store = match get_store() {
            Ok(s) => s,
            Err(c) => return c,
        };
        let cfg = AiHubConfig::new(
            StoreConfig::ai_hub_base_url(),
            StoreConfig::ai_hub_version(),
            String::new(),
            store.config().ai_hub_cache_dir(),
            false,
        );
        let chipsets = match runtime_handle().block_on(list_supported_chipsets(&cfg)) {
            Ok(c) => c,
            Err(e) => return report(&e),
        };

        let mut infos: Vec<GenieXChipsetInfo> = chipsets
            .iter()
            .map(|c| {
                // Surface the reference device (e.g. "Snapdragon X Elite
                // CRD") as `name`, demoting the canonical chipset id
                // ("qualcomm-snapdragon-x-elite") into the alias list so
                // callers can still resolve / display it. Fall back to the
                // canonical id when the bucket omits a reference device.
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
                let mut aliases: Vec<*mut c_char> =
                    alias_strs.iter().map(|a| str_to_cptr(a)).collect();
                aliases.shrink_to_fit();
                let alias_count = aliases.len() as i32;
                let aliases_ptr = if aliases.is_empty() {
                    std::ptr::null_mut()
                } else {
                    aliases.as_mut_ptr()
                };
                std::mem::forget(aliases);
                GenieXChipsetInfo {
                    name: str_to_cptr(display),
                    aliases: aliases_ptr,
                    alias_count,
                }
            })
            .collect();
        infos.shrink_to_fit();
        let count = infos.len() as i32;
        let chipsets_ptr = if infos.is_empty() {
            std::ptr::null_mut()
        } else {
            infos.as_mut_ptr()
        };
        std::mem::forget(infos);
        unsafe {
            (*out).chipsets = chipsets_ptr;
            (*out).count = count;
        }
        GENIEX_SUCCESS
    })
}

#[no_mangle]
pub unsafe extern "C" fn geniex_model_list_chipsets_free(out: *mut GenieXChipsetList) {
    if out.is_null() {
        return;
    }
    let o = &mut *out;
    if !o.chipsets.is_null() {
        let slice = std::slice::from_raw_parts_mut(o.chipsets, o.count as usize);
        for info in slice.iter_mut() {
            free_cptr(info.name);
            if !info.aliases.is_null() {
                let aliases =
                    std::slice::from_raw_parts_mut(info.aliases, info.alias_count as usize);
                for a in aliases.iter_mut() {
                    free_cptr(*a);
                }
                drop(Vec::from_raw_parts(
                    info.aliases,
                    info.alias_count as usize,
                    info.alias_count as usize,
                ));
            }
        }
        drop(Vec::from_raw_parts(
            o.chipsets,
            o.count as usize,
            o.count as usize,
        ));
    }
    o.chipsets = std::ptr::null_mut();
    o.count = 0;
}

/* ---- geniex_model_detect_chipset ---- */

#[no_mangle]
pub extern "C" fn geniex_model_detect_chipset(out_chipset: *mut *mut c_char) -> i32 {
    ffi_guard(|| {
        if out_chipset.is_null() {
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        let ptr = match detect_host_chipset() {
            Some(s) => str_to_cptr(&s),
            None => std::ptr::null_mut(),
        };
        unsafe {
            *out_chipset = ptr;
        }
        GENIEX_SUCCESS
    })
}
