// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::os::raw::{c_char, c_void};
use std::path::PathBuf;

use model_manager_core::config::StoreConfig;
use model_manager_core::executor::{FileProgress, ProgressCallback};
use model_manager_core::manifest::infer::ManifestHint;
use model_manager_core::manifest::ModelType;
use model_manager_core::pull::{pull, Hub, PullIntent, PullRequest, PullTarget};

use crate::init::{get_store, runtime_handle};
use crate::types::*;

/// Values MUST match `geniex_HubSource` in `include/geniex_model.h`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum GenieXHubSource {
    Auto = 0,
    HuggingFace = 1,
    ModelScope = 2,
    AiHub = 3,
    Volces = 4,
    Docker = 5,
    LocalFs = 127,
}

impl From<GenieXHubSource> for Hub {
    fn from(h: GenieXHubSource) -> Self {
        match h {
            GenieXHubSource::Auto => Hub::Auto,
            // Volces is a placeholder that pulls from HuggingFace.
            GenieXHubSource::HuggingFace | GenieXHubSource::Volces => Hub::HuggingFace,
            GenieXHubSource::ModelScope => Hub::ModelScope,
            GenieXHubSource::AiHub => Hub::AiHub,
            GenieXHubSource::Docker => Hub::DockerHub,
            GenieXHubSource::LocalFs => Hub::LocalFs,
        }
    }
}

pub type GenieXDownloadProgressCb =
    Option<unsafe extern "C" fn(*const GenieXFileProgress, i32, *mut c_void) -> bool>;

#[repr(C)]
pub struct GenieXModelPullInput {
    pub struct_size: u32,
    pub model_name: *const c_char,
    pub quant: *const c_char,
    pub hub: GenieXHubSource,
    pub local_path: *const c_char,
    pub hf_token: *const c_char,
    pub chipset: *const c_char,
    pub display_name: *const c_char,
    pub on_progress: GenieXDownloadProgressCb,
    pub user_data: *mut c_void,
    pub model_type: i32,
}

/// ABI gate. When fields are appended, add the previous size so older callers keep working.
const ACCEPTED_PULL_INPUT_SIZES: &[u32] = &[std::mem::size_of::<GenieXModelPullInput>() as u32];

/// An optional C string: NULL, invalid UTF-8 and empty all read as `None`.
unsafe fn opt_str(ptr: *const c_char) -> Option<String> {
    cstr_to_str(ptr)
        .ok()
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// # Safety
/// `input` must be non-null and point to a valid `GenieXModelPullInput`.
pub(crate) unsafe fn extract_name_and_intent(
    input: *const GenieXModelPullInput,
    fn_name: &str,
) -> Result<(String, PullIntent), i32> {
    // Read struct_size before any other field so a layout mismatch can't corrupt later reads.
    let struct_size = std::ptr::read(&(*input).struct_size);
    if !ACCEPTED_PULL_INPUT_SIZES.contains(&struct_size) {
        let msg = format!(
            "{fn_name}: unsupported struct_size {struct_size}; expected one of \
             {ACCEPTED_PULL_INPUT_SIZES:?} (recompile your binding against the \
             current geniex_model.h)",
        );
        return Err(report_msg(&msg, GENIEX_ERROR_COMMON_INVALID_INPUT));
    }
    let inp = &*input;
    let target = PullTarget {
        token: opt_str(inp.hf_token).or_else(StoreConfig::hf_token_from_env),
        chipset: opt_str(inp.chipset).unwrap_or_default(),
        display_name: opt_str(inp.display_name),
        local_path: opt_str(inp.local_path).map(PathBuf::from),
        // `quant` doubles as the Docker tag/digest.
        reference: opt_str(inp.quant).unwrap_or_default(),
    };
    Hub::from(inp.hub)
        .resolve(cstr_to_str(inp.model_name)?, target)
        .or_report()
}

struct CCallback {
    cb: unsafe extern "C" fn(*const GenieXFileProgress, i32, *mut c_void) -> bool,
    user_data: *mut c_void,
}
// SAFETY: the caller guarantees `user_data` may be used from the download threads.
unsafe impl Send for CCallback {}
unsafe impl Sync for CCallback {}

fn progress_callback(cc: CCallback) -> ProgressCallback {
    Box::new(move |files: &[FileProgress]| {
        let cc = &cc;
        let names: Vec<std::ffi::CString> = files
            .iter()
            .map(|f| std::ffi::CString::new(f.file_name.as_bytes()).unwrap_or_default())
            .collect();
        let entries: Vec<GenieXFileProgress> = files
            .iter()
            .zip(&names)
            .map(|(f, name)| GenieXFileProgress {
                file_name: name.as_ptr(),
                downloaded_bytes: f.downloaded_bytes,
                total_bytes: f.total_bytes,
            })
            .collect();
        unsafe { (cc.cb)(entries.as_ptr(), entries.len() as i32, cc.user_data) }
    })
}

#[no_mangle]
pub extern "C" fn geniex_model_pull(input: *const GenieXModelPullInput) -> i32 {
    ffi_guard(|| {
        if input.is_null() {
            return Err(GENIEX_ERROR_COMMON_INVALID_INPUT);
        }
        let (model_name, intent) = unsafe { extract_name_and_intent(input, "geniex_model_pull") }?;
        let inp = unsafe { &*input };
        let on_progress = inp.on_progress.map(|cb| {
            progress_callback(CCallback {
                cb,
                user_data: inp.user_data,
            })
        });
        let model_type = match inp.model_type {
            0 => Some(ModelType::Llm),
            1 => Some(ModelType::Vlm),
            _ => None,
        };
        let req = PullRequest {
            model_name,
            intent,
            on_progress,
            hint: ManifestHint {
                // Manifest keys are upper-case.
                quant: unsafe { opt_str(inp.quant) }.map(|q| q.to_ascii_uppercase()),
                model_type,
                ..ManifestHint::default()
            },
        };
        runtime_handle()
            .block_on(pull(get_store()?, req))
            .or_report()?;
        Ok(GENIEX_SUCCESS)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_size_mismatch_sets_last_error() {
        let mut input: GenieXModelPullInput = unsafe { std::mem::zeroed() };
        input.struct_size = 1;
        let code = ffi_guard(|| unsafe { extract_name_and_intent(&input, "pull") }.map(|_| 0));
        assert_eq!(code, GENIEX_ERROR_COMMON_INVALID_INPUT);
        let ptr = geniex_model_last_error_message();
        assert!(!ptr.is_null(), "last error not set");
        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert!(msg.to_string_lossy().contains("struct_size 1"), "{msg:?}");
    }
}
