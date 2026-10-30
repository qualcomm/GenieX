// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::os::raw::c_char;
use std::path::Path;

use model_manager_core::manifest::{ModelManifest, ModelType};
use model_manager_core::manifest_builder::quant_sort_key;

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

fn path_to_cptr(p: &Path) -> *mut c_char {
    str_to_cptr(&p.to_string_lossy())
}

#[no_mangle]
pub extern "C" fn geniex_model_get_paths(
    model_name: *const c_char,
    out_paths: *mut GenieXModelPaths,
) -> i32 {
    ffi_guard(|| {
        let out = out_mut(out_paths)?;
        let name = normalize_quant_suffix(unsafe { cstr_to_str(model_name) }?);
        let (_, paths) = get_store()?.get_paths(&name).or_report()?;
        let opt_path =
            |p: Option<&std::path::PathBuf>| p.map_or(std::ptr::null_mut(), |p| path_to_cptr(p));
        *out = GenieXModelPaths {
            model_path: path_to_cptr(&paths.model_path),
            mmproj_path: opt_path(paths.mmproj_path.as_ref()),
            tokenizer_path: opt_path(paths.tokenizer_path.as_ref()),
            model_dir: path_to_cptr(&paths.model_dir),
            model_name: str_to_cptr(&paths.model_name),
            plugin_id: str_to_cptr(&paths.plugin_id),
            model_type: to_ffi_type(paths.model_type),
        };
        Ok(GENIEX_SUCCESS)
    })
}

#[no_mangle]
pub unsafe extern "C" fn geniex_model_paths_free(paths: *mut GenieXModelPaths) {
    let Some(p) = paths.as_mut() else { return };
    for s in [
        p.model_path,
        p.mmproj_path,
        p.tokenizer_path,
        p.model_dir,
        p.model_name,
        p.plugin_id,
    ] {
        free_cptr(s);
    }
    *p = std::mem::zeroed();
}

/// Build the C view of one manifest: every heap member is owned by the caller
/// and reclaimed by [`free_detail`].
fn detail_from_manifest(m: &ModelManifest) -> GenieXModelDetail {
    // model_file is a HashMap: its iteration order is reshuffled on every parse.
    let mut downloaded: Vec<&str> = m
        .model_file
        .iter()
        .filter(|(_, fi)| fi.downloaded)
        .map(|(q, _)| q.as_str())
        .collect();
    downloaded.sort_by_key(|q| quant_sort_key(q));
    let (precisions, precision_count) = c_str_array(&downloaded);
    GenieXModelDetail {
        name: str_to_cptr(&m.name),
        model_name: str_to_cptr(&m.model_name),
        plugin_id: str_to_cptr(&m.plugin_id),
        model_type: to_ffi_type(m.model_type.clone()),
        total_size: m.total_size(),
        precisions,
        precision_count,
    }
}

/// # Safety
/// `d` must have been produced by [`detail_from_manifest`] and not freed yet.
unsafe fn free_detail(d: &mut GenieXModelDetail) {
    free_cptr(d.name);
    free_cptr(d.model_name);
    free_cptr(d.plugin_id);
    free_c_str_array(d.precisions, d.precision_count);
    *d = std::mem::zeroed();
}

/* ---- geniex_model_get_detailed ---- */

#[no_mangle]
pub extern "C" fn geniex_model_get_detailed(
    model_name: *const c_char,
    out: *mut GenieXModelDetail,
) -> i32 {
    ffi_guard(|| {
        let out = out_mut(out)?;
        let name = unsafe { cstr_to_str(model_name) }?;
        let manifest = get_store()?.resolve_detail_manifest(name).or_report()?;
        *out = detail_from_manifest(&manifest);
        Ok(GENIEX_SUCCESS)
    })
}

#[no_mangle]
pub unsafe extern "C" fn geniex_model_detail_free(detail: *mut GenieXModelDetail) {
    if let Some(d) = detail.as_mut() {
        free_detail(d);
    }
}

/* ---- geniex_model_remove / clean ---- */

#[no_mangle]
pub extern "C" fn geniex_model_remove(model_name: *const c_char) -> i32 {
    ffi_guard(|| {
        let name = normalize_quant_suffix(unsafe { cstr_to_str(model_name) }?);
        get_store()?.remove(&name).or_report()?;
        Ok(GENIEX_SUCCESS)
    })
}

#[no_mangle]
pub extern "C" fn geniex_model_clean(removed_count: *mut i32) -> i32 {
    ffi_guard(|| {
        let n = get_store()?.clean().or_report()?;
        if let Ok(out) = out_mut(removed_count) {
            *out = n;
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
        let out = out_mut(out_type)?;
        let name = unsafe { cstr_to_str(model_name) }?;
        *out = to_ffi_type(get_store()?.get_model_type(name).or_report()?);
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
        get_store()?.set_model_type(name, t).or_report()?;
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
        let out = out_mut(output)?;
        let manifests = get_store()?.list().or_report()?;
        let (models, count) = into_c_array(manifests.iter().map(detail_from_manifest).collect());
        *out = GenieXModelListDetailedOutput { models, count };
        Ok(GENIEX_SUCCESS)
    })
}

#[no_mangle]
pub unsafe extern "C" fn geniex_model_list_detailed_free(
    output: *mut GenieXModelListDetailedOutput,
) {
    let Some(o) = output.as_mut() else { return };
    free_c_list(o.models, o.count, |d| free_detail(d));
    *o = std::mem::zeroed();
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_manager_core::manifest::ModelFileInfo;
    use model_manager_core::test_support::{file_info, manifest};
    use std::ffi::CStr;

    /// Bindings hand this order to callers: stable, headed by the bare-name pick.
    #[test]
    fn detail_reports_precisions_in_store_priority_order() {
        let m = manifest(
            "Org/Repo",
            [
                ("IQ4_XS", file_info("m-IQ4_XS.gguf", 1)),
                ("Q8_0", file_info("m-Q8_0.gguf", 1)),
                ("Q4_K_M", file_info("m-Q4_K_M.gguf", 1)),
                (
                    "Q4_0",
                    ModelFileInfo {
                        downloaded: false,
                        ..file_info("m-Q4_0.gguf", 1)
                    },
                ),
            ],
        );
        let mut d = detail_from_manifest(&m);
        let got: Vec<String> = unsafe {
            std::slice::from_raw_parts(d.precisions, d.precision_count as usize)
                .iter()
                .map(|p| CStr::from_ptr(*p).to_string_lossy().into_owned())
                .collect()
        };
        assert_eq!(got, ["Q4_K_M", "Q8_0", "IQ4_XS"]);
        unsafe { free_detail(&mut d) };
    }
}
