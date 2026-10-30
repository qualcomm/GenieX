// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! AI Hub release index JSONs, fetched with a plain mtime + TTL disk cache.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use url::Url;

use crate::config;
use crate::error::{parse_manifest, Error, Result};
use crate::transport::{fetch_capped, HttpTransport};

use super::dto::{PlatformInfo, ReleaseManifest};
use super::AiHubConfig;

/// AIHM-maintained mirror of the current release; falls back to pinned.
const LATEST_DIR: &str = "latest";
const MAX_INDEX_BYTES: u64 = 8 * 1024 * 1024;

/// Short so new releases propagate.
const INDEX_TTL: Duration = Duration::from_secs(60 * 60);

pub(super) async fn platform(
    cfg: &AiHubConfig,
    transport: &Arc<dyn HttpTransport>,
) -> Result<PlatformInfo> {
    let bytes = release_index("platform.json", cfg, transport).await?;
    parse_manifest("platform.json", &bytes)
}

pub(super) async fn release_manifest(
    cfg: &AiHubConfig,
    transport: &Arc<dyn HttpTransport>,
) -> Result<ReleaseManifest> {
    let bytes = release_index("manifest.json", cfg, transport).await?;
    parse_manifest("manifest.json", &bytes)
}

async fn release_index(
    filename: &str,
    cfg: &AiHubConfig,
    transport: &Arc<dyn HttpTransport>,
) -> Result<Vec<u8>> {
    let cache_path = cfg.cache_dir.join(filename);
    if !cfg.skip_cache {
        if let Some(bytes) = read_cache_fresh(&cache_path, INDEX_TTL) {
            return Ok(bytes);
        }
    }
    if config::ai_hub_version_override().is_none() {
        let latest_url = format!("{}/releases/{LATEST_DIR}/{filename}", cfg.endpoint);
        match fetch(&latest_url, transport).await {
            Ok(bytes) => {
                if !cfg.skip_cache {
                    write_cache(&cache_path, &bytes);
                }
                return Ok(bytes);
            }
            Err(e) => {
                crate::logging::warn(&format!(
                    "aihub {filename} at {latest_url} unavailable ({e}); \
                     falling back to releases/{}",
                    cfg.version
                ));
            }
        }
    }
    let fallback_url = format!("{}/releases/{}/{filename}", cfg.endpoint, cfg.version);
    let bytes = fetch(&fallback_url, transport).await?;
    if !cfg.skip_cache {
        write_cache(&cache_path, &bytes);
    }
    Ok(bytes)
}

pub(super) async fn fetch_cached(
    url: &str,
    cache_path: &Path,
    ttl: Duration,
    skip_cache: bool,
    transport: &Arc<dyn HttpTransport>,
) -> Result<Vec<u8>> {
    if !skip_cache {
        if let Some(bytes) = read_cache_fresh(cache_path, ttl) {
            return Ok(bytes);
        }
    }
    let bytes = fetch(url, transport).await?;
    if !skip_cache {
        write_cache(cache_path, &bytes);
    }
    Ok(bytes)
}

pub(super) async fn fetch(url: &str, transport: &Arc<dyn HttpTransport>) -> Result<Vec<u8>> {
    let parsed = Url::parse(url).map_err(|e| Error::invalid_url(format!("url {url:?}"), e))?;
    fetch_capped(transport.as_ref(), &parsed, None, MAX_INDEX_BYTES).await
}

fn read_cache_fresh(path: &Path, ttl: Duration) -> Option<Vec<u8>> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?;
    if SystemTime::now().duration_since(mtime).ok()? > ttl {
        return None;
    }
    std::fs::read(path).ok()
}

fn write_cache(path: &Path, data: &[u8]) {
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            crate::logging::warn(&format!("aihub cache mkdir {}: {e}", parent.display()));
            return;
        }
    }
    if let Err(e) = std::fs::write(path, data) {
        crate::logging::warn(&format!("aihub cache write {}: {e}", path.display()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_respects_presence_and_ttl() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(read_cache_fresh(&tmp.path().join("nope.json"), INDEX_TTL).is_none());
        let p = tmp.path().join("x.json");
        write_cache(&p, br#"{"version":"0.58.0"}"#);
        assert!(read_cache_fresh(&p, INDEX_TTL).is_some());
        assert!(read_cache_fresh(&p, Duration::ZERO).is_none());
    }
}
