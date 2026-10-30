// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use model_manager_core::{config::StoreConfig, store::Store};
use tokio::runtime::{Builder, Handle, Runtime};

use crate::logging;
use crate::types::*;

static STORE: OnceLock<Store> = OnceLock::new();
static RUNTIME: OnceLock<Runtime> = OnceLock::new();
/// Closes the TOCTOU between `STORE.get()` and `STORE.set()`.
static INIT_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn get_store() -> Result<&'static Store, i32> {
    STORE.get().ok_or(GENIEX_ERROR_COMMON_NOT_INITIALIZED)
}

/// Lazy so sync-only callers don't pay for a reactor.
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

#[no_mangle]
pub extern "C" fn geniex_model_init(data_dir: *const c_char) -> i32 {
    ffi_guard(|| {
        logging::install_core_sink();

        // Poisoned means a previous init panicked; retry.
        let _guard = INIT_LOCK.lock().unwrap_or_else(|p| p.into_inner());

        if STORE.get().is_some() {
            logging::warn(
                "geniex_model_init called after the model manager was already initialized; \
                 call geniex_model_deinit first",
            );
            return Err(GENIEX_ERROR_COMMON_ALREADY_INITIALIZED);
        }

        let mut cfg = StoreConfig::from_env();
        if let Ok(s) = unsafe { cstr_to_str(data_dir) } {
            cfg.data_dir = PathBuf::from(s);
        }

        let store = Store::new(cfg).or_report()?;
        let _ = STORE.set(store);
        logging::debug("geniex model manager initialized");
        Ok(GENIEX_SUCCESS)
    })
}

/// No-op: the global Store is never freed.
#[no_mangle]
pub extern "C" fn geniex_model_deinit() -> i32 {
    GENIEX_SUCCESS
}
