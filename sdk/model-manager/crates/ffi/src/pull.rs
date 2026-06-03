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
pub enum GenieXHubSource {
    Auto = 0,
    HuggingFace = 1,
    ModelScope = 2,
    AiHub = 3,
    Volces = 4,
    /// Local filesystem — intentionally 127, not 5, to keep "not a real
    /// hub" visually separated from the network hub IDs above.
    LocalFs = 127,
}

/// Progress callback: invoked with an array of per-file `GenieXFileProgress`
/// entries; return `false` to cancel. The pointer is only valid during the
/// call — callbacks must not retain it.
pub type GenieXDownloadProgressCb =
    Option<unsafe extern "C" fn(*const GenieXFileProgress, i32, *mut c_void) -> bool>;

#[repr(C)]
pub struct GenieXModelPullInput {
    /// Must equal `size_of::<GenieXModelPullInput>()`. See the C header
    /// doc on `geniex_ModelPullInput.struct_size` — this is the ABI
    /// version gate: callers compiled against an older header won't
    /// match any recognized size and are rejected before any field is
    /// dereferenced.
    pub struct_size: u32,
    pub model_name: *const c_char,
    pub quant: *const c_char,
    pub hub: GenieXHubSource,
    pub local_path: *const c_char,
    /// HuggingFace bearer token (NULL = fall back to GENIEX_HFTOKEN env,
    /// then anonymous). Only meaningful when `hub == GENIEX_HUB_HUGGINGFACE`.
    pub hf_token: *const c_char,
    /// Target chipset for AI Hub pulls. Required when
    /// `hub == GENIEX_HUB_AIHUB`; ignored otherwise. Matched against the
    /// `name` / `aliases` fields of `platform.json`.
    pub chipset: *const c_char,
    /// AI Hub `display_name` of the model to download. Used when
    /// `hub == GENIEX_HUB_AIHUB` or `hub == GENIEX_HUB_AUTO` resolves
    /// to AI Hub. If NULL and `model_name` is in the form
    /// `qualcomm/<repo>` or `qai-hub-models/<repo>`, `<repo>` is used
    /// as the display_name; otherwise the caller must set this.
    /// `model_name` still names the on-disk directory ("org/repo"
    /// shape), mirroring the Go CLI's `storedName` / `displayName` split.
    pub display_name: *const c_char,
    pub on_progress: GenieXDownloadProgressCb,
    pub user_data: *mut c_void,
    /// Optional model-type override (-1 = auto-detect, 0 = LLM, 1 = VLM).
    /// Written into the manifest as the pull publishes, so callers that know
    /// the type avoid a separate geniex_model_set_type round-trip.
    pub model_type: i32,
}

/// Struct sizes the Rust FFI knows how to read. The only entry today
/// is the current layout; if we add fields in a forward-compatible
/// way, append the *previous* size here so older callers still work.
/// A caller that passes a size not in this list is rejected up front.
const ACCEPTED_PULL_INPUT_SIZES: &[u32] = &[std::mem::size_of::<GenieXModelPullInput>() as u32];

/// Route a hub selector + already-extracted parameters to a [`PullIntent`].
/// Shared by `geniex_model_pull` and `geniex_model_query` so their hub
/// resolution can't drift. `model_name` must already be canonicalised.
/// Returns a negative error code when a required input is missing.
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
            // "qualcomm/*" and "qai-hub-models/*" names, plus bare names
            // (which canonicalize_model_name rewrote to "qualcomm/<name>"),
            // route to AI Hub without the caller setting hub=AIHUB. The
            // derived display_name is the repo after the slash; callers may
            // still override via explicit_display_name.
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
        GenieXHubSource::LocalFs => {
            let path = local_path.ok_or(GENIEX_ERROR_COMMON_INVALID_INPUT)?;
            PullIntent::LocalFs { source_dir: path }
        }
        GenieXHubSource::AiHub => {
            // chipset NULL or empty → SDK auto-detects (currently
            // Windows-on-Snapdragon only). display_name: explicit value
            // wins; otherwise derive from any AI Hub org repo.
            let display_name = explicit_display_name
                .or_else(|| aihub_display_name_from_repo(model_name).map(str::to_string))
                .ok_or(GENIEX_ERROR_COMMON_INVALID_INPUT)?;
            PullIntent::AiHub {
                display_name,
                chipset,
            }
        }
        // ModelScope / Volces remain placeholders — fall back to HuggingFace.
        _ => PullIntent::HuggingFace {
            repo: model_name.to_string(),
            token: hf_token,
        },
    };
    Ok(intent)
}

/// Validate the ABI gate and extract the canonical model name + PullIntent
/// shared by `geniex_model_pull` and `geniex_model_query` (both take a
/// `GenieXModelPullInput`; query just ignores the quant / callback / model_type
/// fields). `fn_name` only labels the struct_size error message.
///
/// # Safety
/// `input` must be non-null and point to a valid `GenieXModelPullInput`.
pub(crate) unsafe fn extract_name_and_intent(
    input: *const GenieXModelPullInput,
    fn_name: &str,
) -> Result<(String, PullIntent), i32> {
    // ABI gate: read struct_size before any other field so a layout mismatch
    // can't corrupt downstream reads.
    let struct_size = std::ptr::read(&(*input).struct_size);
    if !ACCEPTED_PULL_INPUT_SIZES.contains(&struct_size) {
        crate::logging::error(&format!(
            "{fn_name}: unsupported struct_size {struct_size}; expected one of \
             {ACCEPTED_PULL_INPUT_SIZES:?} (recompile your binding against the \
             current geniex_model.h)",
        ));
        return Err(GENIEX_ERROR_COMMON_INVALID_INPUT);
    }

    let inp = &*input;

    let raw_model_name = cstr_to_str(inp.model_name).ok_or(GENIEX_ERROR_COMMON_INVALID_INPUT)?;
    // Bare names (no '/') are treated as AI Hub model ids and stored under
    // `qualcomm/<name>`; anything with '/' is passed through.
    let model_name = canonicalize_model_name(raw_model_name);

    // Explicit token wins; env var is the fallback; anonymous otherwise.
    let hf_token = cstr_to_str(inp.hf_token)
        .map(str::to_string)
        .or_else(StoreConfig::hf_token_from_env);
    // chipset / display_name are only meaningful for AI Hub; read up front so
    // both the explicit-AiHub and Auto-→-AiHub paths share them.
    let chipset = cstr_to_str(inp.chipset).unwrap_or("").to_string();
    let explicit_display_name = cstr_to_str(inp.display_name)
        .map(str::to_string)
        .filter(|s| !s.is_empty());
    let local_path = cstr_to_str(inp.local_path).map(PathBuf::from);

    let intent = build_pull_intent(
        &inp.hub,
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
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        let (model_name, intent) =
            match unsafe { extract_name_and_intent(input, "geniex_model_pull") } {
                Ok(v) => v,
                Err(c) => return c,
            };
        let inp = unsafe { &*input };

        // Build a Rust closure that re-marshals Rust FileProgress → C array
        // and invokes the caller's function pointer.
        struct CCallback {
            cb: unsafe extern "C" fn(*const GenieXFileProgress, i32, *mut c_void) -> bool,
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
        // -1 (GENIEX_MODEL_TYPE_AUTO) leaves detection to the inferer; 0/1 force
        // the type so the manifest is written correctly in one shot.
        let model_type = match inp.model_type {
            0 => Some(model_manager_core::manifest::ModelType::Llm),
            1 => Some(model_manager_core::manifest::ModelType::Vlm),
            _ => None,
        };
        let hint = ManifestHint {
            quant,
            model_type,
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
