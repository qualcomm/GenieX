//! Orchestrate a model download using the [`crate::source::ModelSource`]
//! + [`crate::executor::Executor`] pair.
//!
//! Three-step contract:
//!
//! ```text
//! ┌───────────────────┐   ┌───────────────────────────┐   ┌──────────────────┐
//! │  1. Plan          │   │  2. Fetch bytes           │   │  3. Publish      │
//! │                   │   │                           │   │                  │
//! │  source.plan() →  │──▶│  executor.run(plan,       │──▶│  write manifest  │
//! │   (manifest,      │   │    dest, on_progress)     │   │  atomically;     │
//! │    Vec<FileSpec>) │   │                           │   │  drop markers    │
//! └───────────────────┘   └───────────────────────────┘   └──────────────────┘
//! ```
//!
//! Layout during/after a pull of `org/repo`:
//!
//! ```text
//! models/
//! └── org/repo/
//!     ├── .lock                 # cross-process exclusive lock
//!     ├── .inflight/            # present only during a pull (sentinel)
//!     │   └── geniex.json       # staged manifest; renamed into place on success
//!     ├── model.gguf            # downloaded file
//!     ├── model.gguf.progress   # resume marker
//!     └── geniex.json           # published only when every file is complete
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::config::StoreConfig;
use crate::error::Result;
use crate::executor::{Executor, ProgressCallback};
use crate::manifest_builder::ManifestHint;
use crate::resume;
use crate::source::ai_hub::{AiHubConfig, AiHubSource};
use crate::source::hf::HfSource;
use crate::source::localfs::LocalFsSource;
use crate::source::ModelSource;
use crate::store::{Store, INFLIGHT_DIR, MANIFEST_FILE};
use crate::transport::{HttpTransport, ReqwestTransport};
use crate::validation::validate_model_name;

pub use crate::resume::PROGRESS_SUFFIX;

pub struct PullRequest {
    /// "org/repo" (already resolved from any short alias by the caller).
    pub model_name: String,
    pub intent: PullIntent,
    pub on_progress: Option<ProgressCallback>,
    pub hint: ManifestHint,
}

/// User intent, decoupled from the concrete [`ModelSource`] that will
/// execute it. Each variant carries exactly the parameters that hub
/// needs — no placeholders, no "unused for this hub" fields.
pub enum PullIntent {
    HuggingFace {
        /// "org/repo" on HF — usually identical to `PullRequest.model_name`
        /// but the caller may have canonicalised.
        repo: String,
        /// Bearer token; `None` means anonymous (with inevitable rate
        /// limits). Intentionally per-pull rather than stored in the
        /// Store, so callers can rotate credentials.
        token: Option<String>,
    },
    LocalFs {
        source_dir: PathBuf,
    },
    AiHub {
        display_name: String,
        chipset: String,
    },
}

/// Download a model, writing its manifest only after every file is
/// present.
///
/// Async-native; sync callers should use [`pull_blocking`] (which
/// drives this on the FFI runtime).
pub async fn pull(store: &Store, req: PullRequest) -> Result<()> {
    validate_model_name(&req.model_name)?;

    let transport: Arc<dyn HttpTransport> = Arc::new(ReqwestTransport::new()?);
    let source: Box<dyn ModelSource> = build_source(&req, store, transport.clone())?;
    pull_with_source(
        store,
        &req.model_name,
        source,
        transport,
        req.on_progress.as_ref(),
    )
    .await
}

/// Drive a pre-built [`ModelSource`] through the plan → fetch →
/// publish pipeline. Exposed so tests can point at a custom source
/// (e.g. a mock endpoint) without going through [`PullIntent`].
pub async fn pull_with_source(
    store: &Store,
    model_name: &str,
    source: Box<dyn ModelSource>,
    transport: Arc<dyn HttpTransport>,
    on_progress: Option<&ProgressCallback>,
) -> Result<()> {
    validate_model_name(model_name)?;
    let model_name_owned = model_name.to_string();
    store
        .with_model_lock_async(model_name, || async move {
            let dest_dir = store.model_file_path(&model_name_owned, "")?;
            fs::create_dir_all(&dest_dir)?;
            let inflight_dir = dest_dir.join(INFLIGHT_DIR);
            fs::create_dir_all(&inflight_dir)?;

            let mut plan = source.plan().await?;
            plan.manifest.name = model_name_owned.clone();

            let pending = resume::filter_pending(&plan.files, &dest_dir);
            if !pending.is_empty() {
                let executor = Executor::new(transport.clone(), 4);
                executor.run(&pending, &dest_dir, on_progress).await?;
            }

            let staged = inflight_dir.join(MANIFEST_FILE);
            fs::write(&staged, serde_json::to_string(&plan.manifest)?)?;
            let final_path = dest_dir.join(MANIFEST_FILE);
            fs::rename(&staged, &final_path)?;

            // Once the manifest is published, per-file markers are
            // redundant; the Go CLI does the same cleanup after a
            // successful pull, so cache layouts stay interchangeable.
            for f in &plan.files {
                drop_marker(&dest_dir, &f.name);
            }

            let _ = fs::remove_dir_all(&inflight_dir);

            Ok(())
        })
        .await
}

/// Sync entry point for callers without a runtime. Drives [`pull`] to
/// completion on a caller-supplied tokio handle — the FFI crate owns
/// one process-global runtime and calls this from sync
/// `geniex_model_pull`.
///
/// **Do not call from inside a tokio context** — it panics
/// (`Handle::block_on from within a runtime`). Async callers must
/// use [`pull`].
pub fn pull_blocking(
    handle: &tokio::runtime::Handle,
    store: &Store,
    req: PullRequest,
) -> Result<()> {
    handle.block_on(pull(store, req))
}

fn build_source(
    req: &PullRequest,
    store: &Store,
    transport: Arc<dyn HttpTransport>,
) -> Result<Box<dyn ModelSource>> {
    match &req.intent {
        PullIntent::HuggingFace { repo, token } => {
            let src = HfSource::with_endpoint_and_transport(
                repo.clone(),
                crate::source::hf::DEFAULT_HF_ENDPOINT,
                token.clone(),
                transport,
                req.hint.clone(),
            )?;
            Ok(Box::new(src))
        }
        PullIntent::LocalFs { source_dir } => Ok(Box::new(LocalFsSource::new(
            source_dir.clone(),
            req.model_name.clone(),
            req.hint.clone(),
        ))),
        PullIntent::AiHub {
            display_name,
            chipset,
        } => {
            let cfg = AiHubConfig::new(
                StoreConfig::ai_hub_base_url(),
                StoreConfig::ai_hub_version(),
                chipset.clone(),
                store.config().ai_hub_cache_dir(),
                false,
            );
            let src = AiHubSource::with_transport(
                display_name.clone(),
                req.model_name.clone(),
                cfg,
                transport,
            );
            Ok(Box::new(src))
        }
    }
}

fn drop_marker(dest_dir: &Path, file_name: &str) {
    if file_name.is_empty() {
        return;
    }
    let marker = dest_dir.join(format!("{file_name}{PROGRESS_SUFFIX}"));
    let _ = fs::remove_file(&marker);
}
