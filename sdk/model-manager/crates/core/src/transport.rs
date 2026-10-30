// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! HTTP transport: HEAD for size plus a streaming byte-range GET. rustls
//! keeps the static library free of OpenSSL; certs are checked against the
//! OS trust store so corporate MITM proxy CAs work.

use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_LENGTH, RANGE};
use reqwest::{Client, StatusCode};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWrite, AsyncWriteExt};
use url::Url;

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct HeadInfo {
    pub size: u64,
}

#[async_trait]
pub trait HttpTransport: Send + Sync {
    async fn head(&self, url: &Url, auth: Option<&str>) -> Result<HeadInfo>;

    /// Writes exactly `len` bytes; stream errors are retried, sink errors are not.
    async fn get_range(
        &self,
        url: &Url,
        auth: Option<&str>,
        offset: u64,
        len: u64,
        sink: &mut (dyn AsyncWrite + Unpin + Send),
    ) -> Result<()>;
}

pub(crate) async fn fetch_capped(
    transport: &dyn HttpTransport,
    url: &Url,
    auth: Option<&str>,
    cap: u64,
) -> Result<Vec<u8>> {
    let size = transport.head(url, auth).await?.size;
    if size > cap {
        return Err(Error::Hub(format!(
            "{url} is {size} bytes, exceeds {cap} byte cap"
        )));
    }
    let mut buf = Vec::with_capacity(size as usize);
    transport.get_range(url, auth, 0, size, &mut buf).await?;
    Ok(buf)
}

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_RETRIES: u32 = 3;
const DEFAULT_RETRY_BACKOFF: Duration = Duration::from_secs(1);
pub(crate) const USER_AGENT: &str = concat!("geniex-model-manager/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Clone, Default)]
pub struct TransportConfig {
    pub connect_timeout: Option<Duration>,
    pub read_timeout: Option<Duration>,
    pub retries: Option<u32>,
    pub retry_backoff: Option<Duration>,
    /// Bypasses reqwest's `*_PROXY` env discovery when set.
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

        let mut builder = client_builder(connect, read)?;
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

fn client_builder(connect: Duration, read: Duration) -> Result<reqwest::ClientBuilder> {
    Ok(Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(connect)
        .read_timeout(read)
        .redirect(reqwest::redirect::Policy::limited(10))
        .use_preconfigured_tls(build_tls_config()?))
}

pub(crate) fn default_client() -> Result<Client> {
    client_builder(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)?
        .build()
        .map_err(|e| Error::Http(format!("build reqwest client: {e}")))
}

/// On Android `rustls-platform-verifier` needs JNI init the binding never
/// does and would panic on the first handshake, so use `webpki-roots` there.
pub(crate) fn build_tls_config() -> Result<rustls::ClientConfig> {
    // The ring provider must be installed before any rustls config is built.
    static PROVIDER_INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    PROVIDER_INIT.get_or_init(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
    let provider = rustls::crypto::CryptoProvider::get_default()
        .ok_or_else(|| Error::Http("rustls crypto provider not installed".into()))?
        .clone();

    let builder = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| Error::Http(format!("rustls protocol versions: {e}")))?;

    #[cfg(not(target_os = "android"))]
    let config = {
        use rustls_platform_verifier::BuilderVerifierExt;
        builder.with_platform_verifier().with_no_client_auth()
    };
    #[cfg(target_os = "android")]
    let config = {
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        builder.with_root_certificates(roots).with_no_client_auth()
    };
    Ok(config)
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
                    crate::logging::warn(&format!("head retry {attempt}: {e}"));
                    continue;
                }
                Err(e) => return Err(Error::HttpTimeout(format!("HEAD {url}: {e}"))),
            };

            let status = resp.status();
            if Self::is_transient(status) && attempt < self.retries {
                attempt += 1;
                tokio::time::sleep(self.retry_backoff).await;
                crate::logging::warn(&format!("head retry {attempt}: status {status}"));
                continue;
            }
            if !status.is_success() {
                return Err(Error::HttpStatus {
                    url: url.to_string(),
                    status: status.as_u16(),
                });
            }

            let size = resp
                .headers()
                .get(CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .ok_or_else(|| Error::Http(format!("HEAD {url}: missing Content-Length")))?;
            return Ok(HeadInfo { size });
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
        let mut written = 0u64;
        let mut attempt = 0u32;
        'requests: loop {
            let request_offset = offset + written;
            let range = format!("bytes={request_offset}-{end_inclusive}");
            let mut req = self.client.get(url.clone()).header(RANGE, range.clone());
            if let Some(tok) = auth {
                req = req.header(AUTHORIZATION, format!("Bearer {tok}"));
            }

            let mut resp = match req.send().await {
                Ok(r) => r,
                Err(e) if attempt < self.retries => {
                    attempt += 1;
                    tokio::time::sleep(self.retry_backoff).await;
                    crate::logging::warn(&format!("get_range retry {attempt}: {e}"));
                    continue;
                }
                Err(e) => {
                    return Err(Error::HttpTimeout(format!("GET {url} {range}: {e}")));
                }
            };

            let status = resp.status();
            if Self::is_transient(status) && attempt < self.retries {
                attempt += 1;
                tokio::time::sleep(self.retry_backoff).await;
                crate::logging::warn(&format!("get_range retry {attempt}: status {status}"));
                continue;
            }
            // A 200 (whole body) is only valid when the range starts at zero.
            let ok = status == StatusCode::PARTIAL_CONTENT
                || (status == StatusCode::OK && request_offset == 0);
            if !ok {
                return Err(Error::HttpStatus {
                    url: url.to_string(),
                    status: status.as_u16(),
                });
            }

            loop {
                let chunk = match resp.chunk().await {
                    Ok(chunk) => chunk,
                    Err(e) if written < len && attempt < self.retries => {
                        attempt += 1;
                        tokio::time::sleep(self.retry_backoff).await;
                        crate::logging::warn(&format!("get_range stream retry {attempt}: {e}"));
                        continue 'requests;
                    }
                    Err(e) => {
                        return Err(Error::HttpTimeout(format!("stream {url} {range}: {e}")));
                    }
                };
                let Some(bytes) = chunk else { break };
                let remaining = len - written;
                if bytes.len() as u64 > remaining {
                    return Err(Error::Http(format!(
                        "GET {url} {range}: response chunk exceeds remaining {remaining} bytes"
                    )));
                }
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
                    crate::logging::warn(&format!(
                        "short read retry {attempt}: got {written} / expected {len}"
                    ));
                    continue;
                }
                return Err(Error::Http(format!(
                    "short read {url} bytes={offset}-{end_inclusive}: got {written} / expected {len}"
                )));
            }
            return Ok(());
        }
    }
}

/// [`HttpTransport`] over one local file (the `Url` is ignored), so local zips reuse the remote parser.
#[derive(Debug)]
pub struct LocalFileTransport {
    path: PathBuf,
    len: u64,
}

impl LocalFileTransport {
    pub fn open(path: &Path) -> Result<Self> {
        let meta = std::fs::metadata(path).map_err(|e| {
            Error::Hub(format!(
                "local archive {} is not accessible: {e}",
                path.display()
            ))
        })?;
        if !meta.is_file() {
            return Err(Error::Hub(format!(
                "local archive {} is not a regular file",
                path.display()
            )));
        }
        Ok(Self {
            path: path.to_path_buf(),
            len: meta.len(),
        })
    }
}

#[async_trait]
impl HttpTransport for LocalFileTransport {
    async fn head(&self, _url: &Url, _auth: Option<&str>) -> Result<HeadInfo> {
        Ok(HeadInfo { size: self.len })
    }

    async fn get_range(
        &self,
        _url: &Url,
        _auth: Option<&str>,
        offset: u64,
        len: u64,
        sink: &mut (dyn AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::Hub(format!("range {offset}+{len} overflows u64")))?;
        if end > self.len {
            return Err(Error::Hub(format!(
                "local range {offset}+{len} exceeds archive size {}",
                self.len
            )));
        }

        let mut file = tokio::fs::File::open(&self.path).await?;
        file.seek(std::io::SeekFrom::Start(offset)).await?;
        let mut remaining = len;
        let mut buf = vec![0u8; len.min(1 << 20) as usize];
        while remaining > 0 {
            let want = remaining.min(buf.len() as u64) as usize;
            let n = file.read(&mut buf[..want]).await?;
            if n == 0 {
                return Err(Error::Hub(format!(
                    "local archive {}: unexpected EOF at offset {}",
                    self.path.display(),
                    offset + (len - remaining)
                )));
            }
            sink.write_all(&buf[..n]).await?;
            remaining -= n as u64;
        }
        sink.flush().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn local_transport_serves_ranges_and_rejects_oob() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.bin");
        std::fs::write(&p, b"abcdef").unwrap();
        let t = LocalFileTransport::open(&p).unwrap();
        let dummy = Url::parse("file:///dummy").unwrap();

        let info = t.head(&dummy, None).await.unwrap();
        assert_eq!(info.size, 6);

        let mut buf = Vec::new();
        t.get_range(&dummy, None, 2, 4, &mut buf).await.unwrap();
        assert_eq!(buf, b"cdef");

        let mut buf = Vec::new();
        let err = t
            .get_range(&dummy, None, 5, 10, &mut buf)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");
    }

    #[test]
    fn local_transport_open_rejects_missing() {
        let err = LocalFileTransport::open(Path::new("/nonexistent/zzz.zip")).unwrap_err();
        assert!(err.to_string().contains("not accessible"), "{err}");
    }
}
