// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::os::raw::{c_char, c_void};
use std::path::PathBuf;

use model_manager_core::config::StoreConfig;
use model_manager_core::manifest_builder::ManifestHint;
use model_manager_core::mapping::{
    aihub_display_name_from_repo, canonicalize_model_name, docker_hub_repo_from_name,
    is_docker_hub_reference, is_modelscope_reference,
};
use model_manager_core::pull::{pull, PullIntent, PullRequest};

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

/// Shared by pull and query so hub resolution can't drift. `model_name` must be canonicalised.
pub(crate) fn build_pull_intent(
    hub: &GenieXHubSource,
    model_name: &str,
    hf_token: Option<String>,
    chipset: String,
    explicit_display_name: Option<String>,
    local_path: Option<PathBuf>,
) -> Result<PullIntent, i32> {
    let intent = match hub {
        GenieXHubSource::Auto => {
            // Bare names were canonicalised to "qualcomm/<name>", so they route to AI Hub too.
            if let Some(repo) = aihub_display_name_from_repo(model_name) {
                PullIntent::AiHub {
                    display_name: explicit_display_name.unwrap_or_else(|| repo.to_string()),
                    chipset,
                }
            } else {
                PullIntent::HuggingFace {
                    repo: model_name.to_string(),
                    token: hf_token,
                }
            }
        }
        GenieXHubSource::HuggingFace => PullIntent::HuggingFace {
            repo: model_name.to_string(),
            token: hf_token,
        },
        GenieXHubSource::ModelScope => PullIntent::ModelScope {
            repo: model_name.to_string(),
            token: hf_token,
        },
        GenieXHubSource::LocalFs => {
            let path = local_path.ok_or(GENIEX_ERROR_COMMON_INVALID_INPUT)?;
            PullIntent::LocalFs { source_dir: path }
        }
        GenieXHubSource::AiHub => {
            let display_name = explicit_display_name
                .or_else(|| aihub_display_name_from_repo(model_name).map(str::to_string))
                .ok_or(GENIEX_ERROR_COMMON_INVALID_INPUT)?;
            PullIntent::AiHub {
                display_name,
                chipset,
            }
        }
        // Volces falls back to HuggingFace; Docker is routed in extract_name_and_intent.
        _ => PullIntent::HuggingFace {
            repo: model_name.to_string(),
            token: hf_token,
        },
    };
    Ok(intent)
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
    let raw_model_name = cstr_to_str(inp.model_name)?;

    // Route Docker on the raw string; canonicalisation discards the hub prefix.
    let use_docker = matches!(inp.hub, GenieXHubSource::Docker)
        || (matches!(inp.hub, GenieXHubSource::Auto) && is_docker_hub_reference(raw_model_name));
    if use_docker {
        let repo = canonicalize_model_name(&docker_hub_repo_from_name(raw_model_name));
        // `quant` doubles as the Docker tag/digest.
        let reference = cstr_to_str(inp.quant)
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or("latest")
            .to_string();
        return Ok((repo.clone(), PullIntent::DockerHub { repo, reference }));
    }

    // Likewise, a pasted ModelScope URL is only detectable before canonicalisation.
    let hub = if matches!(inp.hub, GenieXHubSource::Auto) && is_modelscope_reference(raw_model_name)
    {
        GenieXHubSource::ModelScope
    } else {
        inp.hub
    };

    let model_name = canonicalize_model_name(raw_model_name);

    let hf_token = cstr_to_str(inp.hf_token)
        .ok()
        .map(str::to_string)
        .or_else(StoreConfig::hf_token_from_env);
    let chipset = cstr_to_str(inp.chipset).unwrap_or("").to_string();
    let explicit_display_name = cstr_to_str(inp.display_name)
        .ok()
        .map(str::to_string)
        .filter(|s| !s.is_empty());
    let local_path = cstr_to_str(inp.local_path).ok().map(PathBuf::from);

    let intent = build_pull_intent(
        &hub,
        &model_name,
        hf_token,
        chipset,
        explicit_display_name,
        local_path,
    )?;
    Ok((model_name, intent))
}

#[no_mangle]
pub extern "C" fn geniex_model_pull(input: *const GenieXModelPullInput) -> i32 {
    ffi_guard(|| {
        if input.is_null() {
            return Err(GENIEX_ERROR_COMMON_INVALID_INPUT);
        }

        let (model_name, intent) = unsafe { extract_name_and_intent(input, "geniex_model_pull") }?;
        let inp = unsafe { &*input };

        struct CCallback {
            cb: unsafe extern "C" fn(*const GenieXFileProgress, i32, *mut c_void) -> bool,
            user_data: *mut c_void,
        }
        unsafe impl Send for CCallback {}
        unsafe impl Sync for CCallback {}

        let progress_cb: Option<model_manager_core::executor::ProgressCallback> =
            inp.on_progress.map(|cb| {
                let cc = std::sync::Arc::new(CCallback {
                    cb,
                    user_data: inp.user_data,
                });
                Box::new(
                    move |files: &[model_manager_core::executor::FileProgress]| -> bool {
                        let cstrings: Vec<std::ffi::CString> = files
                            .iter()
                            .map(|f| {
                                std::ffi::CString::new(f.file_name.as_bytes()).unwrap_or_default()
                            })
                            .collect();
                        let ffi_entries: Vec<GenieXFileProgress> = files
                            .iter()
                            .zip(cstrings.iter())
                            .map(|(f, cs)| GenieXFileProgress {
                                file_name: cs.as_ptr(),
                                downloaded_bytes: f.downloaded_bytes,
                                total_bytes: f.total_bytes,
                            })
                            .collect();
                        let result = unsafe {
                            (cc.cb)(ffi_entries.as_ptr(), ffi_entries.len() as i32, cc.user_data)
                        };
                        let _ = cstrings;
                        result
                    },
                ) as model_manager_core::executor::ProgressCallback
            });

        // Manifest keys from `extract_quant` are upper-case.
        let quant = unsafe { cstr_to_str(inp.quant) }
            .ok()
            .filter(|s| !s.is_empty())
            .map(str::to_ascii_uppercase);
        let model_type = match inp.model_type {
            0 => Some(model_manager_core::manifest::ModelType::Llm),
            1 => Some(model_manager_core::manifest::ModelType::Vlm),
            _ => None,
        };
        let req = PullRequest {
            model_name,
            intent,
            on_progress: progress_cb,
            hint: ManifestHint {
                quant,
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
    fn build_pull_intent_routes_hubs() {
        use GenieXHubSource::{Auto, HuggingFace, ModelScope};
        let cases = [
            (ModelScope, "Qwen/Qwen3-0.6B-GGUF", "modelscope"),
            (HuggingFace, "ggml-org/Qwen3-1.7B-GGUF", "hf"),
            (Auto, "qualcomm/Qwen3-4B", "aihub"),
            (Auto, "ggml-org/Qwen3-1.7B-GGUF", "hf"),
        ];
        for (hub, name, want) in cases {
            let intent = build_pull_intent(&hub, name, None, String::new(), None, None).unwrap();
            let got = match &intent {
                PullIntent::ModelScope { repo, token } => {
                    assert_eq!((repo.as_str(), token), (name, &None));
                    "modelscope"
                }
                PullIntent::HuggingFace { .. } => "hf",
                PullIntent::AiHub { .. } => "aihub",
                _ => "other",
            };
            assert_eq!(got, want, "{name}");
        }
    }

    #[test]
    fn modelscope_forward_keeps_token() {
        let intent = build_pull_intent(
            &GenieXHubSource::ModelScope,
            "Qwen/Qwen3-0.6B-GGUF",
            Some("tok".to_string()),
            String::new(),
            None,
            None,
        )
        .unwrap();
        assert!(matches!(intent, PullIntent::ModelScope { token: Some(t), .. } if t == "tok"));
    }

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
