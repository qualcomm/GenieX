use std::os::raw::{c_char, c_void};
use std::path::PathBuf;

use model_manager_core::config::StoreConfig;
use model_manager_core::manifest_builder::ManifestHint;
use model_manager_core::mapping::{aihub_display_name_from_repo, canonicalize_model_name};
use model_manager_core::pull::{pull_blocking, PullIntent, PullRequest};

use crate::init::{get_store, runtime_handle};
use crate::types::*;

/// C-compatible hub source enum. Values MUST match `geniex_HubSource` in
/// `include/geniex_model.h`.
#[repr(C)]
#[allow(dead_code)]
pub enum GeniexHubSource {
    Auto = 0,
    HuggingFace = 1,
    ModelScope = 2,
    AiHub = 3,
    Volces = 4,
    /// Local filesystem — intentionally 127, not 5, to keep "not a real
    /// hub" visually separated from the network hub IDs above.
    LocalFs = 127,
}

/// Progress callback: invoked with an array of per-file `GeniexFileProgress`
/// entries; return `false` to cancel. The pointer is only valid during the
/// call — callbacks must not retain it.
pub type GeniexDownloadProgressCb =
    Option<unsafe extern "C" fn(*const GeniexFileProgress, i32, *mut c_void) -> bool>;

#[repr(C)]
pub struct GeniexModelPullInput {
    /// Must equal `size_of::<GeniexModelPullInput>()`. See the C header
    /// doc on `geniex_ModelPullInput.struct_size` — this is the ABI
    /// version gate: callers compiled against an older header won't
    /// match any recognized size and are rejected before any field is
    /// dereferenced.
    pub struct_size: u32,
    pub model_name: *const c_char,
    pub quant: *const c_char,
    pub hub: GeniexHubSource,
    pub local_path: *const c_char,
    /// HuggingFace bearer token (NULL = fall back to GENIEX_HFTOKEN env,
    /// then anonymous). Only meaningful when `hub == GENIEX_HUB_HUGGINGFACE`.
    pub hf_token: *const c_char,
    /// Target chipset for AI Hub pulls. Required when
    /// `hub == GENIEX_HUB_AIHUB`; ignored otherwise. Matched against the
    /// `name` / `aliases` fields of `platform.json`.
    pub chipset: *const c_char,
    /// AI Hub `display_name` of the model to download. Required when
    /// `hub == GENIEX_HUB_AIHUB`; ignored otherwise. `model_name` still
    /// names the on-disk directory ("org/repo" shape), mirroring the
    /// Go CLI's `storedName` / `displayName` split.
    pub display_name: *const c_char,
    pub on_progress: GeniexDownloadProgressCb,
    pub user_data: *mut c_void,
}

/// Struct sizes the Rust FFI knows how to read. The only entry today
/// is the current layout; if we add fields in a forward-compatible
/// way, append the *previous* size here so older callers still work.
/// A caller that passes a size not in this list is rejected up front.
const ACCEPTED_PULL_INPUT_SIZES: &[u32] = &[std::mem::size_of::<GeniexModelPullInput>() as u32];

#[no_mangle]
pub extern "C" fn geniex_model_pull(input: *const GeniexModelPullInput) -> i32 {
    ffi_guard(|| {
        if input.is_null() {
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        // ABI gate: `struct_size` has to be set to `sizeof(struct)` at
        // the caller's compile time. It's read before we touch any other
        // field, so an ABI mismatch can't corrupt downstream reads.
        // See ACCEPTED_PULL_INPUT_SIZES doc.
        let struct_size = unsafe { std::ptr::read(&(*input).struct_size) };
        if !ACCEPTED_PULL_INPUT_SIZES.contains(&struct_size) {
            crate::logging::error(&format!(
                "geniex_model_pull: unsupported struct_size {}; expected one of {:?} \
                 (recompile your binding against the current geniex_model.h)",
                struct_size, ACCEPTED_PULL_INPUT_SIZES,
            ));
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        let inp = unsafe { &*input };

        let raw_model_name = match unsafe { cstr_to_str(inp.model_name) } {
            Some(s) => s.to_string(),
            None => return GENIEX_ERROR_COMMON_INVALID_INPUT,
        };
        // Bare names (no '/') are treated as AI Hub model ids and stored
        // under `aihub/<name>`; anything with '/' is passed through.
        let model_name = canonicalize_model_name(&raw_model_name);

        // Explicit token wins; env var is the fallback; anonymous otherwise.
        let hf_token = unsafe { cstr_to_str(inp.hf_token) }
            .map(str::to_string)
            .or_else(StoreConfig::hf_token_from_env);

        // `chipset` / `display_name` are only meaningful for AI Hub but
        // we read them once up front so both the explicit-AiHub branch
        // and the Auto-→-AiHub auto-detect path can share the values.
        let chipset = unsafe { cstr_to_str(inp.chipset) }
            .unwrap_or("")
            .to_string();
        let explicit_display_name = unsafe { cstr_to_str(inp.display_name) }
            .map(str::to_string)
            .filter(|s| !s.is_empty());

        let intent = match inp.hub {
            GeniexHubSource::Auto => {
                // "qualcomm/*", "qai-hub-models/*", "aihub/*" and bare names
                // (which canonicalize_model_name above rewrote to "aihub/<name>")
                // all route to AI Hub without the caller setting hub=AIHUB.
                // The derived display_name is the repo after the slash;
                // callers may still override via inp.display_name.
                if let Some(repo) = aihub_display_name_from_repo(&model_name) {
                    PullIntent::AiHub {
                        display_name: explicit_display_name.unwrap_or_else(|| repo.to_string()),
                        chipset,
                    }
                } else {
                    PullIntent::HuggingFace {
                        repo: model_name.clone(),
                        token: hf_token,
                    }
                }
            }
            GeniexHubSource::HuggingFace => PullIntent::HuggingFace {
                repo: model_name.clone(),
                token: hf_token,
            },
            GeniexHubSource::LocalFs => {
                let path = match unsafe { cstr_to_str(inp.local_path) } {
                    Some(s) => PathBuf::from(s),
                    None => return GENIEX_ERROR_COMMON_INVALID_INPUT,
                };
                PullIntent::LocalFs { source_dir: path }
            }
            GeniexHubSource::AiHub => {
                // chipset NULL or empty → SDK auto-detects (currently
                // Windows-on-Snapdragon only). Non-empty: caller override.
                let display_name = match explicit_display_name {
                    Some(s) => s,
                    None => return GENIEX_ERROR_COMMON_INVALID_INPUT,
                };
                PullIntent::AiHub {
                    display_name,
                    chipset,
                }
            }
            // ModelScope / Volces remain placeholders — fall back to HuggingFace.
            _ => PullIntent::HuggingFace {
                repo: model_name.clone(),
                token: hf_token,
            },
        };

        // Build a Rust closure that re-marshals Rust FileProgress → C array
        // and invokes the caller's function pointer.
        struct CCallback {
            cb: unsafe extern "C" fn(*const GeniexFileProgress, i32, *mut c_void) -> bool,
            user_data: *mut c_void,
        }
        unsafe impl Send for CCallback {}
        unsafe impl Sync for CCallback {}

        let progress_cb: Option<model_manager_core::executor::ProgressCallback> = if let Some(cb) =
            inp.on_progress
        {
            let cc = std::sync::Arc::new(CCallback {
                cb,
                user_data: inp.user_data,
            });
            Some(Box::new(
                move |files: &[model_manager_core::executor::FileProgress]| -> bool {
                    let cstrings: Vec<std::ffi::CString> = files
                        .iter()
                        .map(|f| std::ffi::CString::new(f.file_name.as_bytes()).unwrap_or_default())
                        .collect();
                    let ffi_entries: Vec<GeniexFileProgress> = files
                        .iter()
                        .zip(cstrings.iter())
                        .map(|(f, cs)| GeniexFileProgress {
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
            ))
        } else {
            None
        };

        let store = match get_store() {
            Ok(s) => s,
            Err(c) => return c,
        };

        // Thread `quant` into the manifest hint so `pull` only fetches
        // the requested quantization instead of every GGUF in the repo.
        let quant = unsafe { cstr_to_str(inp.quant) }.map(str::to_string);
        let hint = ManifestHint {
            quant,
            ..ManifestHint::default()
        };

        let req = PullRequest {
            model_name,
            intent,
            on_progress: progress_cb,
            hint,
        };

        match pull_blocking(&runtime_handle(), store, req) {
            Ok(()) => GENIEX_SUCCESS,
            Err(e) => report(&e),
        }
    })
}
