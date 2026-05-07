//! `reqwest`-backed `HttpTransport` — rustls, proxy-aware, retrying.
//!
//! reqwest's default client honors `HTTP_PROXY` / `HTTPS_PROXY` /
//! `NO_PROXY` / `ALL_PROXY` env vars out of the box, which solves the
//! proxy-blind behavior we had under hf-hub+ureq. rustls keeps us off any
//! system OpenSSL dependency so the static `libgeniex_model.a` stays
//! portable.

use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use reqwest::header::{ACCEPT_RANGES, AUTHORIZATION, CONTENT_LENGTH, ETAG, RANGE};
use reqwest::{Client, StatusCode};
use tokio::io::{AsyncWrite, AsyncWriteExt};
use url::Url;

use crate::error::{Error, Result};

use super::{HeadInfo, HttpTransport};

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_RETRIES: u32 = 3;
const DEFAULT_RETRY_BACKOFF: Duration = Duration::from_secs(1);
const USER_AGENT: &str = concat!("geniex-model-manager/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Clone, Default)]
pub struct TransportConfig {
    pub connect_timeout: Option<Duration>,
    pub read_timeout: Option<Duration>,
    pub retries: Option<u32>,
    pub retry_backoff: Option<Duration>,
    /// Explicit proxy override. When `None`, reqwest reads
    /// `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` / `ALL_PROXY` from the
    /// environment automatically — which is the normal case. Setting
    /// this short-circuits env discovery, used mainly for tests and for
    /// users who want a per-process proxy distinct from the shell env.
    pub proxy_override: Option<String>,
}

pub struct ReqwestTransport {
    client: Client,
    retries: u32,
    retry_backoff: Duration,
}

impl ReqwestTransport {
    pub fn new() -> Result<Self> {
        Self::with_config(TransportConfig::default())
    }

    pub fn with_config(cfg: TransportConfig) -> Result<Self> {
        let connect = cfg.connect_timeout.unwrap_or(DEFAULT_CONNECT_TIMEOUT);
        let read = cfg.read_timeout.unwrap_or(DEFAULT_READ_TIMEOUT);
        let retries = cfg.retries.unwrap_or(DEFAULT_RETRIES);
        let retry_backoff = cfg.retry_backoff.unwrap_or(DEFAULT_RETRY_BACKOFF);

        let mut builder = Client::builder()
            .user_agent(USER_AGENT)
            .connect_timeout(connect)
            .read_timeout(read)
            .redirect(reqwest::redirect::Policy::limited(10));
        if let Some(p) = cfg.proxy_override {
            let proxy =
                reqwest::Proxy::all(&p).map_err(|e| Error::Http(format!("proxy {p}: {e}")))?;
            builder = builder.proxy(proxy);
        }
        let client = builder
            .build()
            .map_err(|e| Error::Http(format!("build reqwest client: {e}")))?;
        Ok(Self {
            client,
            retries,
            retry_backoff,
        })
    }

    fn is_transient(status: StatusCode) -> bool {
        status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS
    }
}

#[async_trait]
impl HttpTransport for ReqwestTransport {
    async fn head(&self, url: &Url, auth: Option<&str>) -> Result<HeadInfo> {
        let mut attempt = 0u32;
        loop {
            let mut req = self.client.head(url.clone());
            if let Some(tok) = auth {
                req = req.header(AUTHORIZATION, format!("Bearer {tok}"));
            }
            let resp = match req.send().await {
                Ok(r) => r,
                Err(e) if attempt < self.retries => {
                    attempt += 1;
                    tokio::time::sleep(self.retry_backoff).await;
                    tracing_log(format!("head retry {attempt}: {e}"));
                    continue;
                }
                Err(e) => return Err(Error::Http(format!("HEAD {url}: {e}"))),
            };

            let status = resp.status();
            if Self::is_transient(status) && attempt < self.retries {
                attempt += 1;
                tokio::time::sleep(self.retry_backoff).await;
                tracing_log(format!("head retry {attempt}: status {status}"));
                continue;
            }
            if !status.is_success() {
                return Err(Error::Http(format!("HEAD {url}: status {status}")));
            }

            let headers = resp.headers();
            let size = headers
                .get(CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .ok_or_else(|| Error::Http(format!("HEAD {url}: missing Content-Length")))?;
            let accepts_ranges = headers
                .get(ACCEPT_RANGES)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.eq_ignore_ascii_case("bytes"))
                .unwrap_or(false);
            let etag = headers
                .get(ETAG)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());

            return Ok(HeadInfo {
                size,
                accepts_ranges,
                etag,
            });
        }
    }

    async fn get_range(
        &self,
        url: &Url,
        auth: Option<&str>,
        offset: u64,
        len: u64,
        sink: &mut (dyn AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }

        let end_inclusive = offset + len - 1;
        let range = format!("bytes={}-{}", offset, end_inclusive);

        let mut attempt = 0u32;
        loop {
            let mut req = self.client.get(url.clone()).header(RANGE, range.clone());
            if let Some(tok) = auth {
                req = req.header(AUTHORIZATION, format!("Bearer {tok}"));
            }

            let resp = match req.send().await {
                Ok(r) => r,
                Err(e) if attempt < self.retries => {
                    attempt += 1;
                    tokio::time::sleep(self.retry_backoff).await;
                    tracing_log(format!("get_range retry {attempt}: {e}"));
                    continue;
                }
                Err(e) => {
                    return Err(Error::Http(format!("GET {url} {range}: {e}")));
                }
            };

            let status = resp.status();
            if Self::is_transient(status) && attempt < self.retries {
                attempt += 1;
                tokio::time::sleep(self.retry_backoff).await;
                tracing_log(format!("get_range retry {attempt}: status {status}"));
                continue;
            }
            // 200 is accepted only if the server returned the full body and
            // we asked for offset==0; otherwise we need 206 Partial Content.
            let ok =
                status == StatusCode::PARTIAL_CONTENT || (status == StatusCode::OK && offset == 0);
            if !ok {
                return Err(Error::Http(format!("GET {url} {range}: status {status}")));
            }

            let mut stream = resp.bytes_stream();
            let mut written: u64 = 0;
            while let Some(chunk) = stream.next().await {
                let bytes: Bytes = match chunk {
                    Ok(b) => b,
                    Err(e) if attempt < self.retries => {
                        attempt += 1;
                        tokio::time::sleep(self.retry_backoff).await;
                        tracing_log(format!("stream retry {attempt}: {e}"));
                        // Restart the whole range — we can't rewind the
                        // sink, but the engine's unit of retry is a chunk,
                        // so a full restart here is fine as long as the
                        // caller hasn't advertised progress yet. Caller
                        // writes to a pre-allocated region via seek, so
                        // overwriting from 0 on retry is safe; we just
                        // need the sink at the same start position. The
                        // engine passes us a freshly-seeked handle per
                        // attempt, so we simply return an error and let
                        // it retry.
                        return Err(Error::Http(format!("stream {url} {range}: {e}")));
                    }
                    Err(e) => return Err(Error::Http(format!("stream {url} {range}: {e}"))),
                };
                sink.write_all(&bytes)
                    .await
                    .map_err(|e| Error::Http(format!("write sink: {e}")))?;
                written += bytes.len() as u64;
            }
            sink.flush()
                .await
                .map_err(|e| Error::Http(format!("flush sink: {e}")))?;

            if written != len {
                if attempt < self.retries {
                    attempt += 1;
                    tokio::time::sleep(self.retry_backoff).await;
                    tracing_log(format!(
                        "short read retry {attempt}: got {written} / expected {len}"
                    ));
                    continue;
                }
                return Err(Error::Http(format!(
                    "short read {url} {range}: got {written} / expected {len}"
                )));
            }
            return Ok(());
        }
    }
}

fn tracing_log(msg: String) {
    // Keep dependency footprint minimal — route to stderr. Upstream FFI
    // log bridge can grab stderr if needed; using eprintln! matches what
    // the Go CLI does via slog before slog was wired.
    eprintln!("[geniex-model-manager] {msg}");
}
