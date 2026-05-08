//! Orchestrates a model download: resolve manifest, fetch files with
//! per-file atomicity + resume markers, then atomically publish the manifest.
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
//!     ├── model.gguf.progress   # resume marker; content = [0x01] (whole-file done)
//!     └── geniex.json           # published only when every file is complete
//! ```
//!
//! The `.inflight/` sentinel makes [`crate::store::Store::list`] skip
//! incomplete pulls, so a crashed or killed process never exposes a
//! "half-downloaded" model.
//!
//! `.progress` marker format matches the Go CLI
//! (`cli/internal/model_hub/model_hub.go`): a byte array of length
//! `ceil(size / chunk_size)` where each `0x01` byte signals a completed
//! chunk. As of the concurrent-download refactor, the engine writes and
//! flips these bytes chunk-by-chunk — so a killed pull resumes at
//! chunk granularity, and a pull started by the Go CLI can be finished
//! by the Rust engine (and vice versa).

use std::fs;
use std::path::Path;

use crate::config::StoreConfig;
use crate::error::Result;
use crate::hub::s3::{pull_ai_hub, S3Config};
use crate::hub::{hf::HfHub, localfs::LocalFsHub, HubSource, ModelHub, ProgressCallback};
use crate::manifest::ModelManifest;
use crate::manifest_builder::{infer_manifest_from_names, ManifestHint};
use crate::store::{Store, INFLIGHT_DIR, MANIFEST_FILE};
use crate::validation::validate_model_name;

pub const PROGRESS_SUFFIX: &str = ".progress";

pub struct PullRequest {
    /// "org/repo" (already resolved from any short alias by the caller).
    pub model_name: String,
    pub hub: HubSource,
    /// HuggingFace bearer token for this pull. `None` means anonymous;
    /// rate limits will apply. This is intentionally per-pull rather than
    /// stored in the Store so callers can rotate credentials.
    pub hf_token: Option<String>,
    pub on_progress: Option<ProgressCallback>,
    pub hint: ManifestHint,
}

/// Download a model, writing its manifest only after all files are present.
///
/// Async-native; sync callers should use [`pull_blocking`] (which runs
/// this on the FFI runtime) rather than building their own.
pub async fn pull(store: &Store, req: PullRequest) -> Result<()> {
    validate_model_name(&req.model_name)?;

    match &req.hub {
        HubSource::HuggingFace => {
            let hub = HfHub::new(req.hf_token.clone())?;
            store
                .with_model_lock_async(&req.model_name, || pull_locked(store, &hub, &req))
                .await
        }
        HubSource::LocalFs(path) => {
            let hub = LocalFsHub::new(path.clone());
            store
                .with_model_lock_async(&req.model_name, || pull_locked(store, &hub, &req))
                .await
        }
        HubSource::S3 {
            display_name,
            chipset,
        } => {
            let cfg = S3Config::new(
                StoreConfig::ai_hub_base_url(),
                StoreConfig::ai_hub_version(),
                chipset.clone(),
                store.config().ai_hub_cache_dir(),
                false,
            );
            store
                .with_model_lock_async(&req.model_name, || {
                    pull_ai_hub(
                        store,
                        &req.model_name,
                        display_name,
                        cfg,
                        req.on_progress.as_ref(),
                    )
                })
                .await
        }
    }
}

/// Sync entry point for callers that don't have a runtime. Drives
/// [`pull`] to completion on a caller-supplied tokio handle — the FFI
/// crate owns one process-global runtime and calls this from sync
/// `geniex_model_pull`. **Do not call from inside a tokio context** —
/// it panics (`Handle::block_on from within a runtime`). Async callers
/// must use [`pull`].
pub fn pull_blocking(handle: &tokio::runtime::Handle, store: &Store, req: PullRequest) -> Result<()> {
    handle.block_on(pull(store, req))
}

async fn pull_locked(store: &Store, hub: &dyn ModelHub, req: &PullRequest) -> Result<()> {
    let dest_dir = store.model_file_path(&req.model_name, "")?;
    fs::create_dir_all(&dest_dir)?;

    let inflight_dir = dest_dir.join(INFLIGHT_DIR);
    fs::create_dir_all(&inflight_dir)?;

    // 1. Resolve manifest: prefer an explicit geniex.json from the hub,
    //    otherwise infer one from the remote file listing.
    let (remote_files, hub_manifest) = hub.list_files(&req.model_name).await?;
    let mut manifest: ModelManifest = match hub_manifest {
        Some(m) => m,
        None => {
            let names: Vec<String> = remote_files
                .iter()
                .filter(|f| f.name != "geniex.json")
                .map(|f| f.name.clone())
                .collect();
            let mut sizes = std::collections::HashMap::new();
            for f in &remote_files {
                sizes.insert(f.name.clone(), f.size);
            }
            infer_manifest_from_names(&req.model_name, &names, &sizes, req.hint.clone())?
        }
    };

    // Make sure the manifest's `Name` matches what the user asked for; a
    // mismatch would cause `Store::list` / `get_manifest` to misfile it.
    manifest.name = req.model_name.clone();

    // 2. Build the list of files to hand to the hub. With chunk-granular
    //    resume the engine itself decides what bytes to request, but we
    //    still filter out files that are already fully marked done —
    //    saves a HEAD per file on restart.
    let files = files_to_download(&manifest, &dest_dir);

    // 3. Fetch. The hub takes ownership of .progress bitmap management;
    //    on success every marker byte is already 0x01.
    hub.download(&req.model_name, &files, &dest_dir, req.on_progress.as_ref())
        .await?;

    // 4. Persist the manifest (staged then atomic rename).
    let staged = inflight_dir.join(MANIFEST_FILE);
    fs::write(&staged, serde_json::to_string(&manifest)?)?;
    let final_path = dest_dir.join(MANIFEST_FILE);
    fs::rename(&staged, &final_path)?;

    // 5. On a clean success, drop the per-file .progress markers — the
    //    published manifest is now the source of truth that the store
    //    considers this model complete. Matches the Go CLI's behavior
    //    at model_hub.go's `for _, p := range markerPaths { os.Remove(p) }`.
    for f in manifest.model_file.values() {
        drop_marker(&dest_dir, &f.name);
    }
    drop_marker(&dest_dir, &manifest.mmproj_file.name);
    drop_marker(&dest_dir, &manifest.tokenizer_file.name);
    for f in &manifest.extra_files {
        drop_marker(&dest_dir, &f.name);
    }

    // 6. Clear the in-flight sentinel.
    let _ = fs::remove_dir_all(&inflight_dir);

    Ok(())
}

fn drop_marker(dest_dir: &Path, file_name: &str) {
    if file_name.is_empty() {
        return;
    }
    let marker = dest_dir.join(format!("{file_name}{PROGRESS_SUFFIX}"));
    let _ = fs::remove_file(&marker);
}

/// Decide which files listed in the manifest still need fetching. A file
/// is considered "fully done" only when its `.progress` marker exists
/// and every byte is `0x01` — partial bitmaps mean we need to resume.
/// The engine does the actual chunk-level decision; this filter just
/// lets us skip the per-file HEAD round-trip on restart.
fn files_to_download(manifest: &ModelManifest, dest_dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();

    let mut push_if_needed = |name: &str| {
        if name.is_empty() {
            return;
        }
        // Already-published file: manifest exists in dest_dir but
        // we got a second pull attempt. The engine's bitmap check
        // will be authoritative; we still push so the engine
        // reopens, confirms, and moves on (a no-op).
        let marker = dest_dir.join(format!("{}{}", name, PROGRESS_SUFFIX));
        let output = dest_dir.join(name);
        if !marker.exists() && output.exists() {
            // Legacy published state: file present, no marker.
            // Previous pulls left this shape. Treat as done.
            return;
        }
        if let Ok(data) = fs::read(&marker) {
            if !data.is_empty() && data.iter().all(|b| *b == 0x01) {
                return;
            }
        }
        out.push(name.to_string());
    };

    for f in manifest.model_file.values() {
        if f.downloaded {
            push_if_needed(&f.name);
        }
    }
    if manifest.mmproj_file.downloaded {
        push_if_needed(&manifest.mmproj_file.name);
    }
    if manifest.tokenizer_file.downloaded {
        push_if_needed(&manifest.tokenizer_file.name);
    }
    for f in &manifest.extra_files {
        if f.downloaded {
            push_if_needed(&f.name);
        }
    }

    out
}
