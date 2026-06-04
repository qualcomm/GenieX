use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use model_manager_core::{config::StoreConfig, store::Store};
use tokio::runtime::{Builder, Handle, Runtime};

use crate::logging;
use crate::types::*;

static STORE: OnceLock<Store> = OnceLock::new();
static RUNTIME: OnceLock<Runtime> = OnceLock::new();
/// Serializes `geniex_model_init` callers so a TOCTOU between
/// `STORE.get().is_some()` and `STORE.set(...)` can't leave two threads
/// racing to initialize the same global.
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// Access the global store; returns an Err if not yet initialized.
pub(crate) fn get_store() -> Result<&'static Store, i32> {
    STORE.get().ok_or(GENIEX_ERROR_COMMON_NOT_INITIALIZED)
}

/// Access the process-wide tokio runtime. Lazily built on first use so
/// crates that only call sync APIs (like `geniex_model_list`) don't pay
/// for a reactor. Panics only if the tokio runtime itself fails to
/// spawn, which is a fatal OS-level error — the FFI guard converts the
/// panic back to `GENIEX_ERROR_COMMON_UNKNOWN`.
pub(crate) fn runtime_handle() -> Handle {
    RUNTIME
        .get_or_init(|| {
            let worker_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .min(8);
            Builder::new_multi_thread()
                .worker_threads(worker_threads)
                .enable_all()
                .thread_name("geniex-mm")
                .build()
                .expect("build model-manager runtime")
        })
        .handle()
        .clone()
}

/// Initialize the model manager.
///
/// `data_dir` precedence: argument → `GENIEX_DATADIR` env → `~/.cache/geniex`.
///
/// Re-initializing is a programmer error: a warning is logged and
/// `GENIEX_ERROR_COMMON_ALREADY_INITIALIZED` is returned. The first
/// successful call is authoritative. Concurrent callers serialize on an
/// internal mutex so exactly one call observes the uninitialized state.
///
/// HuggingFace tokens are supplied per-pull (see
/// `geniex_ModelPullInput.hf_token`), not at init time.
#[no_mangle]
pub extern "C" fn geniex_model_init(data_dir: *const c_char) -> i32 {
    ffi_guard(|| {
        let _guard = match INIT_LOCK.lock() {
            Ok(g) => g,
            // Mutex is poisoned only if a previous init panicked. Treat
            // as "try again" — the panic path already logged via ffi_guard.
            Err(poisoned) => poisoned.into_inner(),
        };

        if STORE.get().is_some() {
            logging::warn(
                "geniex_model_init called after the model manager was already initialized; \
                 call geniex_model_deinit first",
            );
            return GENIEX_ERROR_COMMON_ALREADY_INITIALIZED;
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

        // Holding INIT_LOCK means no other thread is between the get()
        // check and set() call, so this always succeeds.
        let _ = STORE.set(store);
        logging::debug("geniex model manager initialized");
        GENIEX_SUCCESS
    })
}

/// Deinitialize the model manager. No-op; the global Store is never freed,
/// but the OnceLock guarantees no background threads were ever spawned.
#[no_mangle]
pub extern "C" fn geniex_model_deinit() -> i32 {
    GENIEX_SUCCESS
}
