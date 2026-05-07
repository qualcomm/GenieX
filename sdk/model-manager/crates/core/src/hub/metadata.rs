//! Hub-specific metadata adapters.
//!
//! `HubMetadata` is everything a hub knows that isn't the raw HTTP
//! fetch: how to list files in a repo, how to turn file names into
//! concrete download URLs, what auth header to send. Paired with any
//! `HttpTransport` and the shared download engine, a new hub (ModelScope,
//! S3, on-prem artifact server, ...) only needs to impl this trait.

use std::sync::Arc;

use async_trait::async_trait;
use url::Url;

use crate::error::Result;
use crate::manifest::ModelManifest;
use crate::transport::HttpTransport;

use super::RemoteFile;

/// A single concrete file to download: URL + expected total size + auth.
///
/// `size` is `Some` only when the hub's listing call already returned it;
/// otherwise the engine does a HEAD to discover the size.
#[derive(Debug, Clone)]
pub struct FileSource {
    pub name: String,
    pub size: Option<u64>,
    pub url: Url,
    pub auth: Option<String>,
}

#[async_trait]
pub trait HubMetadata: Send + Sync {
    /// Return the full file listing for `repo`, plus a parsed
    /// `geniex.json` manifest if the hub ships one at the repo root.
    async fn list_files(&self, repo: &str) -> Result<(Vec<RemoteFile>, Option<ModelManifest>)>;

    /// Build download coordinates for the given file names. Order of the
    /// returned vector matches the input.
    async fn resolve(&self, repo: &str, files: &[String]) -> Result<Vec<FileSource>>;

    /// Default file-level concurrency the hub is comfortable with. HF
    /// bumps this up with a token; unauthenticated it stays at 1 to
    /// avoid getting rate-limited.
    fn default_file_concurrency(&self) -> usize {
        4
    }
}

/// Convenience bundle of `(metadata, transport)` that the engine takes.
#[derive(Clone)]
pub struct HubContext {
    pub metadata: Arc<dyn HubMetadata>,
    pub transport: Arc<dyn HttpTransport>,
}

impl HubContext {
    pub fn new(metadata: Arc<dyn HubMetadata>, transport: Arc<dyn HttpTransport>) -> Self {
        Self {
            metadata,
            transport,
        }
    }
}
