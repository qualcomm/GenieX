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
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        let name = match unsafe { cstr_to_str(model_name) } {
            Some(s) => s,
            None => return GENIEX_ERROR_COMMON_INVALID_INPUT,
        };
        let store = match get_store() {
            Ok(s) => s,
            Err(c) => return c,
        };
        match store.get_paths(name) {
            Ok((_, paths)) => {
                unsafe {
                    (*out_paths).model_path = str_to_cptr(&paths.model_path.to_string_lossy());
                    (*out_paths).model_dir = str_to_cptr(&paths.model_dir.to_string_lossy());
                    (*out_paths).model_name = str_to_cptr(&paths.model_name);
                    (*out_paths).plugin_id = str_to_cptr(&paths.plugin_id);
                    (*out_paths).mmproj_path = paths
                        .mmproj_path
                        .as_ref()
                        .map(|p| str_to_cptr(&p.to_string_lossy()))
                        .unwrap_or(std::ptr::null_mut());
                    (*out_paths).tokenizer_path = paths
                        .tokenizer_path
                        .as_ref()
                        .map(|p| str_to_cptr(&p.to_string_lossy()))
                        .unwrap_or(std::ptr::null_mut());
                    (*out_paths).model_type = to_ffi_type(paths.model_type);
                }
                GENIEX_SUCCESS
            }
            Err(e) => report(&e),
        }
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
        let name = match unsafe { cstr_to_str(model_name) } {
            Some(s) => s,
            None => return GENIEX_ERROR_COMMON_INVALID_INPUT,
        };
        let store = match get_store() {
            Ok(s) => s,
            Err(c) => return c,
        };
        match store.remove(name) {
            Ok(()) => GENIEX_SUCCESS,
            Err(e) => report(&e),
        }
    })
}

#[no_mangle]
pub extern "C" fn geniex_model_clean(removed_count: *mut i32) -> i32 {
    ffi_guard(|| {
        let store = match get_store() {
            Ok(s) => s,
            Err(c) => return c,
        };
        match store.clean() {
            Ok(n) => {
                if !removed_count.is_null() {
                    unsafe {
                        *removed_count = n;
                    }
                }
                GENIEX_SUCCESS
            }
            Err(e) => report(&e),
        }
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
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        let name = match unsafe { cstr_to_str(model_name) } {
            Some(s) => s,
            None => return GENIEX_ERROR_COMMON_INVALID_INPUT,
        };
        let store = match get_store() {
            Ok(s) => s,
            Err(c) => return c,
        };
        match store.get_model_type(name) {
            Ok(t) => {
                unsafe {
                    *out_type = to_ffi_type(t);
                }
                GENIEX_SUCCESS
            }
            Err(e) => report(&e),
        }
    })
}

/* ---- geniex_model_set_type ---- */

#[no_mangle]
pub extern "C" fn geniex_model_set_type(
    model_name: *const c_char,
    model_type: GenieXModelType,
) -> i32 {
    ffi_guard(|| {
        let name = match unsafe { cstr_to_str(model_name) } {
            Some(s) => s,
            None => return GENIEX_ERROR_COMMON_INVALID_INPUT,
        };
        let store = match get_store() {
            Ok(s) => s,
            Err(c) => return c,
        };
        let t = match model_type {
            GenieXModelType::Llm => ModelType::Llm,
            GenieXModelType::Vlm => ModelType::Vlm,
        };
        match store.set_model_type(name, t) {
            Ok(()) => GENIEX_SUCCESS,
            Err(e) => report(&e),
        }
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
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        let store = match get_store() {
            Ok(s) => s,
            Err(c) => return c,
        };
        match store.list() {
            Ok(manifests) => {
                let mut details: Vec<GenieXModelDetail> = manifests
                    .iter()
                    .map(|m| {
                        // QAIRT manifests key model_file under "N/A" and carry the
                        // real precision (e.g. "W4A16") on the top-level field —
                        // surface that to bindings instead of the placeholder.
                        let downloaded_quants: Vec<&str> = m
                            .model_file
                            .iter()
                            .filter(|(_, fi)| fi.downloaded)
                            .map(|(q, _)| q.as_str())
                            .collect();
                        let use_top_level = !m.precision.is_empty()
                            && !downloaded_quants.is_empty()
                            && downloaded_quants.iter().all(|q| *q == "N/A");
                        let mut precs: Vec<*mut c_char> = if use_top_level {
                            vec![str_to_cptr(&m.precision)]
                        } else {
                            downloaded_quants.iter().map(|q| str_to_cptr(q)).collect()
                        };
                        precs.shrink_to_fit();
                        let precision_count = precs.len() as i32;
                        let precisions = if precs.is_empty() {
                            std::ptr::null_mut()
                        } else {
                            precs.as_mut_ptr()
                        };
                        std::mem::forget(precs);
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
                details.shrink_to_fit();
                let count = details.len() as i32;
                let models = if details.is_empty() {
                    std::ptr::null_mut()
                } else {
                    details.as_mut_ptr()
                };
                std::mem::forget(details);
                unsafe {
                    (*output).models = models;
                    (*output).count = count;
                }
                GENIEX_SUCCESS
            }
            Err(e) => report(&e),
        }
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
    if !o.models.is_null() {
        let slice = std::slice::from_raw_parts_mut(o.models, o.count as usize);
        for d in slice.iter_mut() {
            free_cptr(d.name);
            free_cptr(d.model_name);
            free_cptr(d.plugin_id);
            if !d.precisions.is_null() {
                let precs =
                    std::slice::from_raw_parts_mut(d.precisions, d.precision_count as usize);
                for p in precs.iter_mut() {
                    free_cptr(*p);
                }
                drop(Vec::from_raw_parts(
                    d.precisions,
                    d.precision_count as usize,
                    d.precision_count as usize,
                ));
            }
        }
        drop(Vec::from_raw_parts(
            o.models,
            o.count as usize,
            o.count as usize,
        ));
    }
    o.models = std::ptr::null_mut();
    o.count = 0;
}
