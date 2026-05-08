use std::os::raw::{c_char, c_void};
use std::path::PathBuf;

use model_manager_core::config::StoreConfig;
use model_manager_core::hub::HubSource;
use model_manager_core::manifest_builder::ManifestHint;
use model_manager_core::pull::{pull_blocking, PullRequest};

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
    S3 = 3,
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
    pub model_name: *const c_char,
    pub quant: *const c_char,
    pub hub: GeniexHubSource,
    pub local_path: *const c_char,
    /// HuggingFace bearer token (NULL = fall back to GENIEX_HFTOKEN env,
    /// then anonymous). Only meaningful when `hub == GENIEX_HUB_HUGGINGFACE`.
    pub hf_token: *const c_char,
    /// Target chipset for AI Hub pulls. Required when
    /// `hub == GENIEX_HUB_S3`; ignored otherwise. Matched against the
    /// `name` / `aliases` fields of `platform.json`.
    pub chipset: *const c_char,
    /// AI Hub `display_name` of the model to download. Required when
    /// `hub == GENIEX_HUB_S3`; ignored otherwise. `model_name` still
    /// names the on-disk directory ("org/repo" shape), mirroring the
    /// Go CLI's `storedName` / `displayName` split.
    pub display_name: *const c_char,
    pub on_progress: GeniexDownloadProgressCb,
    pub user_data: *mut c_void,
}

#[no_mangle]
pub extern "C" fn geniex_model_pull(input: *const GeniexModelPullInput) -> i32 {
    ffi_guard(|| {
        if input.is_null() {
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        let inp = unsafe { &*input };

        let model_name = match unsafe { cstr_to_str(inp.model_name) } {
            Some(s) => s.to_string(),
            None => return GENIEX_ERROR_COMMON_INVALID_INPUT,
        };

        let hub = match inp.hub {
            GeniexHubSource::HuggingFace | GeniexHubSource::Auto => HubSource::HuggingFace,
            GeniexHubSource::LocalFs => {
                let path = match unsafe { cstr_to_str(inp.local_path) } {
                    Some(s) => PathBuf::from(s),
                    None => return GENIEX_ERROR_COMMON_INVALID_INPUT,
                };
                HubSource::LocalFs(path)
            }
            GeniexHubSource::S3 => {
                // chipset NULL or empty → SDK auto-detects (currently
                // Windows-on-Snapdragon only). Non-empty: caller override.
                let chipset = unsafe { cstr_to_str(inp.chipset) }
                    .unwrap_or("")
                    .to_string();
                let display_name = match unsafe { cstr_to_str(inp.display_name) } {
                    Some(s) if !s.is_empty() => s.to_string(),
                    _ => return GENIEX_ERROR_COMMON_INVALID_INPUT,
                };
                HubSource::S3 {
                    display_name,
                    chipset,
                }
            }
            // ModelScope / Volces remain placeholders — fall back to HuggingFace.
            _ => HubSource::HuggingFace,
        };

        // Build a Rust closure that re-marshals Rust FileProgress → C array
        // and invokes the caller's function pointer.
        //
        // Neither `*mut c_void` nor fn pointers with *mut c_void implement
        // Send/Sync automatically; we wrap them in a struct that asserts
        // it's fine for our single-threaded blocking call path.
        struct CCallback {
            cb: unsafe extern "C" fn(*const GeniexFileProgress, i32, *mut c_void) -> bool,
            user_data: *mut c_void,
        }
        unsafe impl Send for CCallback {}
        unsafe impl Sync for CCallback {}

        let progress_cb: Option<model_manager_core::hub::ProgressCallback> = if let Some(cb) =
            inp.on_progress
        {
            // Wrap in Arc so the closure captures the asserted
            // Send+Sync wrapper as a single unit. Rust 2021's
            // disjoint captures would otherwise split the fields
            // and see the bare `*mut c_void`, which is !Sync.
            let cc = std::sync::Arc::new(CCallback {
                cb,
                user_data: inp.user_data,
            });
            Some(Box::new(
                move |files: &[model_manager_core::hub::FileProgress]| -> bool {
                    // CStrings must live for the duration of the callback.
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
                    // cstrings drop here — after the callback returns.
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

        // Explicit token wins; env var is the fallback; anonymous otherwise.
        let hf_token = unsafe { cstr_to_str(inp.hf_token) }
            .map(str::to_string)
            .or_else(StoreConfig::hf_token_from_env);

        let req = PullRequest {
            model_name,
            hub,
            hf_token,
            on_progress: progress_cb,
            hint,
        };

        match pull_blocking(&runtime_handle(), store, req) {
            Ok(()) => GENIEX_SUCCESS,
            Err(e) => report(&e),
        }
    })
}
