pub mod hf;
pub mod hf_metadata;
pub mod localfs;
pub mod metadata;

pub use hf_metadata::HfMetadata;
pub use metadata::{FileSource, HubContext, HubMetadata};

use std::path::Path;

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
}

pub trait ModelHub {
    /// Fetch the remote file list for a repo, along with a manifest if the
    /// repo ships a `geniex.json` at its root.
    ///
    /// Hubs that have no concept of "remote listing" (e.g. `LocalFsHub`)
    /// should return files discovered under their source directory.
    fn list_files(&self, repo_id: &str) -> Result<(Vec<RemoteFile>, Option<ModelManifest>)>;

    /// Download the named files to `dest_dir`.
    ///
    /// `on_progress` is invoked after each file completes (and periodically
    /// during download for hubs that can report byte-level progress).
    fn download(
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
