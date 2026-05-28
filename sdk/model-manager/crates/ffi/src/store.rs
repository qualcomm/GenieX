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

fn to_ffi_type(t: ModelType) -> GenieXModelType {
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
    pub device_id: *mut c_char,
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
            device_id: std::ptr::null_mut(),
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
                    (*out_paths).device_id = paths
                        .device_id
                        .as_deref()
                        .map(str_to_cptr)
                        .unwrap_or(std::ptr::null_mut());
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
    free_cptr(p.device_id);
    *paths = GenieXModelPaths::null();
}

/* ---- geniex_model_list ---- */

#[repr(C)]
pub struct GenieXModelListOutput {
    pub names: *mut *mut c_char,
    pub count: i32,
}

#[no_mangle]
pub extern "C" fn geniex_model_list(output: *mut GenieXModelListOutput) -> i32 {
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
                let mut ptrs: Vec<*mut c_char> =
                    manifests.iter().map(|m| str_to_cptr(&m.name)).collect();
                ptrs.shrink_to_fit();
                let count = ptrs.len() as i32;
                let names_ptr = if ptrs.is_empty() {
                    std::ptr::null_mut()
                } else {
                    ptrs.as_mut_ptr()
                };
                std::mem::forget(ptrs);
                unsafe {
                    (*output).names = names_ptr;
                    (*output).count = count;
                }
                GENIEX_SUCCESS
            }
            Err(e) => report(&e),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn geniex_model_list_free(output: *mut GenieXModelListOutput) {
    if output.is_null() {
        return;
    }
    let o = &mut *output;
    if !o.names.is_null() {
        let slice = std::slice::from_raw_parts_mut(o.names, o.count as usize);
        for ptr in slice.iter_mut() {
            free_cptr(*ptr);
        }
        drop(Vec::from_raw_parts(
            o.names,
            o.count as usize,
            o.count as usize,
        ));
    }
    o.names = std::ptr::null_mut();
    o.count = 0;
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
