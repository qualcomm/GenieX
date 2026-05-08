pub mod hf;
pub mod hf_metadata;
pub mod localfs;
pub mod metadata;
pub mod s3;

pub use hf_metadata::HfMetadata;
pub use metadata::{FileSource, HubContext, HubMetadata};
pub use s3::S3Config;

use std::path::Path;

use async_trait::async_trait;

use crate::error::Result;
use crate::manifest::ModelManifest;

/// Per-file download progress. `total_bytes == -1` means the total is unknown.
#[derive(Debug, Clone)]
pub struct FileProgress {
    pub file_name: String,
    pub downloaded_bytes: i64,
    pub total_bytes: i64,
}

/// Called periodically during a pull. The slice contains one entry per file
/// currently being tracked. Returning `false` signals the hub to cancel.
///
/// The slice is borrowed — callbacks must not retain it. The `Send + Sync`
/// bounds let the engine invoke the callback from a dedicated dispatcher
/// task while workers update counters from other threads.
pub type ProgressCallback = Box<dyn Fn(&[FileProgress]) -> bool + Send + Sync>;

pub enum HubSource {
    HuggingFace,
    LocalFs(std::path::PathBuf),
    /// AI Hub qairt runtime assets served out of the public S3 bucket.
    ///
    /// * `display_name` — the AI Hub manifest key; matches against the
    ///   `display_name` field of each `ReleaseManifest.models` entry.
    /// * `chipset` — user-configured device (e.g. `"SM8650"`), matched
    ///   against `platform.json` chipsets/aliases before download.
    ///
    /// The on-disk directory is named after `PullRequest.model_name`
    /// (the usual "org/repo" form), mirroring the Go CLI split between
    /// `storedName` and `displayName`.
    S3 {
        display_name: String,
        chipset: String,
    },
}

/// A hub that knows how to list a repo's files and fetch them into a
/// destination directory. The trait is `async` so composing hubs inside
/// already-async callers (services, web handlers, async Python bindings)
/// never has to block a worker thread with `block_on()`.
///
/// Sync callers should go through the FFI crate (`crates/ffi`), which
/// owns a single process-global tokio runtime.
#[async_trait]
pub trait ModelHub: Send + Sync {
    /// Fetch the remote file list for a repo, along with a manifest if the
    /// repo ships a `geniex.json` at its root.
    ///
    /// Hubs that have no concept of "remote listing" (e.g. `LocalFsHub`)
    /// should return files discovered under their source directory.
    async fn list_files(
        &self,
        repo_id: &str,
    ) -> Result<(Vec<RemoteFile>, Option<ModelManifest>)>;

    /// Download the named files to `dest_dir`.
    ///
    /// `on_progress` is invoked after each file completes (and periodically
    /// during download for hubs that can report byte-level progress).
    async fn download(
        &self,
        repo_id: &str,
        files: &[String],
        dest_dir: &Path,
        on_progress: Option<&ProgressCallback>,
    ) -> Result<()>;
}

/// A file entry returned by `list_files`.
#[derive(Debug, Clone)]
pub struct RemoteFile {
    pub name: String,
    /// Size in bytes; -1 if unknown.
    pub size: i64,
}
