//! HTTP transport layer for the download engine.
//!
//! `HttpTransport` is the minimum a hub needs from the network: HEAD to
//! discover size + `Accept-Ranges`, and a byte-range GET that streams into
//! an `AsyncWrite`. Keeping it this thin means a new transport (proxied
//! reqwest, a local MITM, a mocked test harness) is ~150 lines to add.
//!
//! The default [`ReqwestTransport`] honors `HTTP_PROXY` / `HTTPS_PROXY` /
//! `NO_PROXY` env vars (reqwest default), which is the whole point of
//! moving off hf-hub's ureq backend.

pub mod reqwest_impl;

use async_trait::async_trait;
use tokio::io::AsyncWrite;
use url::Url;

use crate::error::Result;

pub use reqwest_impl::{ReqwestTransport, TransportConfig};

#[derive(Debug, Clone)]
pub struct HeadInfo {
    pub size: u64,
    pub accepts_ranges: bool,
    pub etag: Option<String>,
}

#[async_trait]
pub trait HttpTransport: Send + Sync {
    async fn head(&self, url: &Url, auth: Option<&str>) -> Result<HeadInfo>;

    /// Stream bytes `[offset, offset+len)` from `url` into `sink`. Must
    /// write exactly `len` bytes; short reads or EOF mid-stream are
    /// surfaced as [`crate::error::Error::Http`] so the engine can retry.
    async fn get_range(
        &self,
        url: &Url,
        auth: Option<&str>,
        offset: u64,
        len: u64,
        sink: &mut (dyn AsyncWrite + Unpin + Send),
    ) -> Result<()>;
}
