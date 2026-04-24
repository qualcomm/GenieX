use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::OnceLock;

use model_manager_core::{config::StoreConfig, store::Store};

use crate::logging;
use crate::types::*;

static STORE: OnceLock<Store> = OnceLock::new();

/// Access the global store; returns an Err if not yet initialized.
pub(crate) fn get_store() -> Result<&'static Store, i32> {
    STORE.get().ok_or(GENIEX_ERROR_COMMON_NOT_INITIALIZED)
}

/// Initialize the model manager.
///
/// `data_dir` precedence: argument → `GENIEX_DATADIR` env → `~/.cache/geniex`.
///
/// Calling this function twice in the same process is a programmer
/// error; a warning is logged via `geniex_set_log` and
/// `GENIEX_ERROR_COMMON_INVALID_INPUT` is returned. The first successful
/// call is authoritative.
///
/// HuggingFace tokens are supplied per-pull (see
/// `geniex_ModelPullInput.hf_token`), not at init time.
#[no_mangle]
pub extern "C" fn geniex_model_init(data_dir: *const c_char) -> i32 {
    ffi_guard(|| {
        if STORE.get().is_some() {
            logging::warn(
                "geniex_model_init called after the model manager was already initialized; \
                 ignoring the new arguments",
            );
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        let data_dir_override = unsafe { cstr_to_str(data_dir).map(PathBuf::from) };

        let mut cfg = StoreConfig::from_env();
        if let Some(dir) = data_dir_override {
            cfg.data_dir = dir;
        }

        let store = match Store::new(cfg) {
            Ok(s) => s,
            Err(e) => return report(&e),
        };

        let _ = STORE.set(store);
        logging::info("geniex model manager initialized");
        GENIEX_SUCCESS
    })
}

/// Deinitialize the model manager. No-op; the global Store is never freed,
/// but the OnceLock guarantees no background threads were ever spawned.
#[no_mangle]
pub extern "C" fn geniex_model_deinit() -> i32 {
    GENIEX_SUCCESS
}
