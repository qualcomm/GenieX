// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::os::raw::c_char;

use model_manager_core::manifest::ModelType;

use crate::init::get_store;
use crate::types::*;

/// C-compatible model type enum (mirrors `geniex_ModelType` in geniex_model.h).
#[repr(C)]
pub enum GenieXModelType {
    Llm = 0,
    Vlm = 1,
}

pub(crate) fn to_ffi_type(t: ModelType) -> GenieXModelType {
    match t {
        ModelType::Llm => GenieXModelType::Llm,
        ModelType::Vlm => GenieXModelType::Vlm,
    }
}

/* ---- geniex_ModelPaths ---- */

#[repr(C)]
pub struct GenieXModelPaths {
    pub model_path: *mut c_char,
    pub mmproj_path: *mut c_char,
    pub tokenizer_path: *mut c_char,
    pub model_dir: *mut c_char,
    pub model_name: *mut c_char,
    pub plugin_id: *mut c_char,
    pub model_type: GenieXModelType,
}

impl GenieXModelPaths {
    fn null() -> Self {
        Self {
            model_path: std::ptr::null_mut(),
            mmproj_path: std::ptr::null_mut(),
            tokenizer_path: std::ptr::null_mut(),
            model_dir: std::ptr::null_mut(),
            model_name: std::ptr::null_mut(),
            plugin_id: std::ptr::null_mut(),
            model_type: GenieXModelType::Llm,
        }
    }
}

#[no_mangle]
pub extern "C" fn geniex_model_get_paths(
    model_name: *const c_char,
    out_paths: *mut GenieXModelPaths,
) -> i32 {
    ffi_guard(|| {
        if out_paths.is_null() {
            return Err(GENIEX_ERROR_COMMON_INVALID_INPUT);
        }
        let name = normalize_quant_suffix(unsafe { cstr_to_str(model_name) }?);
        let store = get_store()?;
        let (_, paths) = store.get_paths(&name).map_err(|e| report(&e))?;
        let opt_path = |p: Option<&std::path::PathBuf>| {
            p.map(|p| str_to_cptr(&p.to_string_lossy()))
                .unwrap_or(std::ptr::null_mut())
        };
        unsafe {
            (*out_paths).model_path = str_to_cptr(&paths.model_path.to_string_lossy());
            (*out_paths).model_dir = str_to_cptr(&paths.model_dir.to_string_lossy());
            (*out_paths).model_name = str_to_cptr(&paths.model_name);
            (*out_paths).plugin_id = str_to_cptr(&paths.plugin_id);
            (*out_paths).mmproj_path = opt_path(paths.mmproj_path.as_ref());
            (*out_paths).tokenizer_path = opt_path(paths.tokenizer_path.as_ref());
            (*out_paths).model_type = to_ffi_type(paths.model_type);
        }
        Ok(GENIEX_SUCCESS)
    })
}

#[no_mangle]
pub unsafe extern "C" fn geniex_model_paths_free(paths: *mut GenieXModelPaths) {
    if paths.is_null() {
        return;
    }
    let p = &mut *paths;
    free_cptr(p.model_path);
    free_cptr(p.mmproj_path);
    free_cptr(p.tokenizer_path);
    free_cptr(p.model_dir);
    free_cptr(p.model_name);
    free_cptr(p.plugin_id);
    *paths = GenieXModelPaths::null();
}

/* ---- geniex_model_remove / clean ---- */

#[no_mangle]
pub extern "C" fn geniex_model_remove(model_name: *const c_char) -> i32 {
    ffi_guard(|| {
        let name = normalize_quant_suffix(unsafe { cstr_to_str(model_name) }?);
        get_store()?.remove(&name).map_err(|e| report(&e))?;
        Ok(GENIEX_SUCCESS)
    })
}

#[no_mangle]
pub extern "C" fn geniex_model_clean(removed_count: *mut i32) -> i32 {
    ffi_guard(|| {
        let n = get_store()?.clean().map_err(|e| report(&e))?;
        if !removed_count.is_null() {
            unsafe { *removed_count = n };
        }
        Ok(GENIEX_SUCCESS)
    })
}

/* ---- geniex_model_get_type ---- */

#[no_mangle]
pub extern "C" fn geniex_model_get_type(
    model_name: *const c_char,
    out_type: *mut GenieXModelType,
) -> i32 {
    ffi_guard(|| {
        if out_type.is_null() {
            return Err(GENIEX_ERROR_COMMON_INVALID_INPUT);
        }
        let name = unsafe { cstr_to_str(model_name) }?;
        let t = get_store()?.get_model_type(name).map_err(|e| report(&e))?;
        unsafe { *out_type = to_ffi_type(t) };
        Ok(GENIEX_SUCCESS)
    })
}

/* ---- geniex_model_set_type ---- */

#[no_mangle]
pub extern "C" fn geniex_model_set_type(
    model_name: *const c_char,
    model_type: GenieXModelType,
) -> i32 {
    ffi_guard(|| {
        let name = unsafe { cstr_to_str(model_name) }?;
        let t = match model_type {
            GenieXModelType::Llm => ModelType::Llm,
            GenieXModelType::Vlm => ModelType::Vlm,
        };
        get_store()?
            .set_model_type(name, t)
            .map_err(|e| report(&e))?;
        Ok(GENIEX_SUCCESS)
    })
}

/* ---- geniex_model_list_detailed ---- */

#[repr(C)]
pub struct GenieXModelDetail {
    pub name: *mut c_char,
    pub model_name: *mut c_char,
    pub plugin_id: *mut c_char,
    pub model_type: GenieXModelType,
    pub total_size: i64,
    /// Heap array of downloaded quant names; `precision_count` long.
    pub precisions: *mut *mut c_char,
    pub precision_count: i32,
}

#[repr(C)]
pub struct GenieXModelListDetailedOutput {
    pub models: *mut GenieXModelDetail,
    pub count: i32,
}

#[no_mangle]
pub extern "C" fn geniex_model_list_detailed(output: *mut GenieXModelListDetailedOutput) -> i32 {
    ffi_guard(|| {
        if output.is_null() {
            return Err(GENIEX_ERROR_COMMON_INVALID_INPUT);
        }
        let manifests = get_store()?.list().map_err(|e| report(&e))?;
        let details: Vec<GenieXModelDetail> = manifests
            .iter()
            .map(|m| {
                let downloaded: Vec<&str> = m
                    .model_file
                    .iter()
                    .filter(|(_, fi)| fi.downloaded)
                    .map(|(q, _)| q.as_str())
                    .collect();
                let precs: Vec<*mut c_char> = downloaded.iter().map(|q| str_to_cptr(q)).collect();
                let (precisions, precision_count) = into_c_array(precs);
                GenieXModelDetail {
                    name: str_to_cptr(&m.name),
                    model_name: str_to_cptr(&m.model_name),
                    plugin_id: str_to_cptr(&m.plugin_id),
                    model_type: to_ffi_type(m.model_type.clone()),
                    total_size: m.total_size(),
                    precisions,
                    precision_count,
                }
            })
            .collect();
        let (models, count) = into_c_array(details);
        unsafe {
            (*output).models = models;
            (*output).count = count;
        }
        Ok(GENIEX_SUCCESS)
    })
}

#[no_mangle]
pub unsafe extern "C" fn geniex_model_list_detailed_free(
    output: *mut GenieXModelListDetailedOutput,
) {
    if output.is_null() {
        return;
    }
    let o = &mut *output;
    if let Some(mut details) = from_c_array(o.models, o.count) {
        for d in details.iter_mut() {
            free_cptr(d.name);
            free_cptr(d.model_name);
            free_cptr(d.plugin_id);
            if let Some(precs) = from_c_array(d.precisions, d.precision_count) {
                for p in precs {
                    free_cptr(p);
                }
            }
        }
    }
    o.models = std::ptr::null_mut();
    o.count = 0;
}
