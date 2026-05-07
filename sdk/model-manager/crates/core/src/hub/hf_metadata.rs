//! HuggingFace implementation of [`HubMetadata`].
//!
//! Listing goes through the public `/api/models/{repo}` JSON endpoint,
//! which returns the same `siblings` array hf-hub reads. Downloads point
//! at the stock `https://huggingface.co/{repo}/resolve/main/{file}` URL —
//! byte-identical to what the Go CLI uses
//! (`cli/internal/model_hub/model_hub_hf.go`), which means a cache
//! directory filled by either agent is interchangeable.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use url::Url;

use crate::error::{Error, Result};
use crate::manifest::ModelManifest;
use crate::transport::HttpTransport;

use super::metadata::{FileSource, HubMetadata};
use super::RemoteFile;

pub const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
const MANIFEST_FILE: &str = "geniex.json";
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

pub struct HfMetadata {
    endpoint: Url,
    token: Option<String>,
    transport: Arc<dyn HttpTransport>,
}

impl HfMetadata {
    pub fn new(token: Option<String>, transport: Arc<dyn HttpTransport>) -> Result<Self> {
        Self::with_endpoint(DEFAULT_HF_ENDPOINT, token, transport)
    }

    pub fn with_endpoint(
        endpoint: &str,
        token: Option<String>,
        transport: Arc<dyn HttpTransport>,
    ) -> Result<Self> {
        let endpoint = Url::parse(endpoint)
            .map_err(|e| Error::Hub(format!("invalid HF endpoint {endpoint}: {e}")))?;
        Ok(Self {
            endpoint,
            token,
            transport,
        })
    }

    fn api_model_url(&self, repo: &str) -> Result<Url> {
        self.endpoint
            .join(&format!("api/models/{repo}"))
            .map_err(|e| Error::Hub(format!("join api url for {repo}: {e}")))
    }

    fn resolve_file_url(&self, repo: &str, file_name: &str) -> Result<Url> {
        self.endpoint
            .join(&format!("{repo}/resolve/main/{file_name}"))
            .map_err(|e| Error::Hub(format!("join resolve url for {repo}/{file_name}: {e}")))
    }

    /// Fetch the raw bytes of `{repo}/resolve/main/{file_name}` via HEAD+GET.
    /// Used by `list_files` to pull a small manifest file when present.
    async fn fetch_small_file(&self, url: &Url, limit: u64) -> Result<Vec<u8>> {
        let head = self.transport.head(url, self.token.as_deref()).await?;
        if head.size > limit {
            return Err(Error::Hub(format!(
                "file at {url} is {} bytes, exceeds {limit} byte cap",
                head.size
            )));
        }
        let mut buf: Vec<u8> = Vec::with_capacity(head.size as usize);
        self.transport
            .get_range(url, self.token.as_deref(), 0, head.size, &mut buf)
            .await?;
        Ok(buf)
    }
}

#[derive(Debug, Deserialize)]
struct ApiModelResponse {
    #[serde(default)]
    siblings: Vec<ApiSibling>,
}

#[derive(Debug, Deserialize)]
struct ApiSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

#[async_trait]
impl HubMetadata for HfMetadata {
    async fn list_files(&self, repo: &str) -> Result<(Vec<RemoteFile>, Option<ModelManifest>)> {
        // 1. Hit the JSON API for the full sibling listing. The API's
        //    Content-Length is reliable, so we reuse the small-file path.
        let api_url = self.api_model_url(repo)?;
        let body = self.fetch_small_file(&api_url, MAX_MANIFEST_BYTES).await?;
        let parsed: ApiModelResponse = serde_json::from_slice(&body)?;

        let files: Vec<RemoteFile> = parsed
            .siblings
            .into_iter()
            .map(|s| RemoteFile {
                name: s.rfilename,
                size: s.size.map(|n| n as i64).unwrap_or(-1),
            })
            .collect();

        // 2. If a geniex.json ships with the repo, pull + parse it; any
        //    failure here is logged-ish (via Error) and surfaced to the
        //    caller, matching the old HfHub behavior of warning-then-
        //    inferring, but we prefer explicit Err so the orchestrator
        //    can decide. For parity with the old code we swallow parse
        //    errors by returning None — the orchestrator then falls back
        //    to manifest inference.
        let manifest = if files.iter().any(|f| f.name == MANIFEST_FILE) {
            let url = self.resolve_file_url(repo, MANIFEST_FILE)?;
            match self.fetch_small_file(&url, MAX_MANIFEST_BYTES).await {
                Ok(bytes) => serde_json::from_slice(&bytes).ok(),
                Err(_) => None,
            }
        } else {
            None
        };

        Ok((files, manifest))
    }

    async fn resolve(&self, repo: &str, files: &[String]) -> Result<Vec<FileSource>> {
        let mut out = Vec::with_capacity(files.len());
        for name in files {
            let url = self.resolve_file_url(repo, name)?;
            out.push(FileSource {
                name: name.clone(),
                size: None,
                url,
                auth: self.token.clone(),
            });
        }
        Ok(out)
    }

    fn default_file_concurrency(&self) -> usize {
        // Mirror the Go CLI knob: token unlocks real parallelism, anon
        // stays serial to stay under the free-tier rate limit.
        if self.token.is_some() {
            8
        } else {
            1
        }
    }
}
