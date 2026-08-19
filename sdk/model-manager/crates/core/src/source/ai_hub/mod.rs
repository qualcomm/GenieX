// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Qualcomm AI Hub [`ModelSource`].
//!
//! Resolves the asset URL for a `(display_name, chipset)` pair via the
//! public S3 protojson chain, then reads the remote archive's ZIP64
//! central directory over HTTP Range reads to produce a complete
//! [`Plan`] — manifest + one per-entry [`BytesSource`] — without
//! downloading the multi-GB payload.

pub mod detect;
pub mod dto;
pub mod remote_zip;
pub mod selector;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use url::Url;

use crate::config::StoreConfig;
use crate::error::{parse_manifest, Error, Result};
use crate::manifest::{ModelFileInfo, ModelManifest, ModelType};
use crate::transport::{HttpTransport, ReqwestTransport};

use self::dto::{
    ChipsetInfo, InfoJson, ManifestModelEntry, ModelReleaseAssets, PlatformInfo, ReleaseManifest,
};
use self::remote_zip::{fetch_central_directory, Method, ZipEntry};
// `is_qairt_runtime` is shared with the download selector so the listing
// filter and the asset picker cannot drift apart.
use self::selector::{is_qairt_runtime, match_asset, UnavailableChipset};

use super::{basename, BytesSource, FileSpec, ModelSource, Plan};

const MANIFEST_FILENAME: &str = "manifest.json";
const PLATFORM_FILENAME: &str = "platform.json";
const LATEST_VERSION_FILENAME: &str = "latest.txt";
const MAX_INDEX_BYTES: u64 = 8 * 1024 * 1024;

/// TTL for the on-disk index cache (matches Go CLI's 24h). Also used for
/// the `latest.txt` pointer cache — AIHM releases roughly monthly, so a
/// separate, shorter TTL for the pointer wasn't worth the extra constant.
const CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);

#[derive(Debug, Clone)]
pub struct AiHubConfig {
    pub endpoint: String,
    pub version: String,
    /// Empty → auto-detect via [`detect::detect_host_chipset`].
    pub chipset: String,
    pub cache_dir: PathBuf,
    pub skip_cache: bool,
}

impl AiHubConfig {
    pub fn new(
        endpoint: String,
        version: String,
        chipset: String,
        cache_dir: PathBuf,
        skip_cache: bool,
    ) -> Self {
        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            version: version.trim_matches('/').to_string(),
            chipset,
            cache_dir,
            skip_cache,
        }
    }
}

/// Fetch and parse `platform.json` (the chipset catalogue) for `cfg`,
/// honouring the on-disk 24h cache. Shared by `AiHubSource::plan` and
/// [`list_supported_chipsets`].
async fn fetch_platform_info(
    cfg: &AiHubConfig,
    transport: &Arc<dyn HttpTransport>,
) -> Result<PlatformInfo> {
    let platform_url = format!(
        "{}/releases/{}/{PLATFORM_FILENAME}",
        cfg.endpoint, cfg.version
    );
    let platform_cache = cfg.cache_dir.join(PLATFORM_FILENAME);
    let platform_bytes = fetch_with_cache(
        &platform_url,
        &platform_cache,
        &cfg.version,
        cfg.skip_cache,
        transport,
    )
    .await?;
    parse_manifest("platform.json", &platform_bytes)
}

/// Resolve which AIHM release the AI Hub source should use.
///
/// `GENIEX_AIHUBVERSION` ([`StoreConfig::ai_hub_version_override`]) wins
/// outright when set — no network call. Otherwise fetch
/// `{endpoint}/releases/latest.txt`, the pointer AIHM's release pipeline
/// writes last once every model's assets are uploaded
/// (qcom-ai-hub/ai-hub-models-internal#4284), honouring the same 24h
/// on-disk [`CACHE_TTL`] as manifest.json/platform.json. Any failure
/// along the way (network, missing pointer, malformed
/// content) falls back to [`StoreConfig::ai_hub_version_fallback`] so this
/// lookup never blocks a pull.
pub async fn resolve_ai_hub_version(endpoint: &str, cache_dir: &Path) -> String {
    if let Some(pinned) = StoreConfig::ai_hub_version_override() {
        return pinned;
    }

    // `latest.txt` content *is* the version, so there's no embedded
    // version field to cross-check the way manifest/platform JSON does
    // — the mtime + TTL check from `read_cache_fresh` is all we need.
    let cache_path = cache_dir.join(LATEST_VERSION_FILENAME);
    if let Some(bytes) = read_cache_fresh(&cache_path, CACHE_TTL) {
        if let Some(v) = parse_latest_version(&bytes) {
            return v;
        }
    }

    // The three fallback arms below share one user-facing message — the
    // exact failure mode (transport init / network / malformed body) is
    // developer-facing detail that stays in `logging::warn`.
    let user_msg = |fallback: &str| {
        format!("AI Hub version discovery failed, using pinned fallback {fallback}")
    };

    let transport: Arc<dyn HttpTransport> = match ReqwestTransport::new() {
        Ok(t) => Arc::new(t),
        Err(e) => {
            let fallback = StoreConfig::ai_hub_version_fallback();
            crate::logging::warn(&format!("aihub latest.txt transport init: {e}"));
            crate::user_warning::emit(&user_msg(&fallback));
            return fallback;
        }
    };

    let url = format!("{endpoint}/releases/{LATEST_VERSION_FILENAME}");
    match fetch_direct(&url, &transport).await {
        Ok(bytes) => match parse_latest_version(&bytes) {
            Some(v) => {
                write_cache(&cache_path, &bytes);
                v
            }
            None => {
                let fallback = StoreConfig::ai_hub_version_fallback();
                crate::logging::warn(&format!("aihub latest.txt at {url} is empty/malformed"));
                crate::user_warning::emit(&user_msg(&fallback));
                fallback
            }
        },
        Err(e) => {
            let fallback = StoreConfig::ai_hub_version_fallback();
            crate::logging::warn(&format!("aihub latest.txt fetch from {url}: {e}"));
            crate::user_warning::emit(&user_msg(&fallback));
            fallback
        }
    }
}


/// Normalize `latest.txt`'s bare `0.61.0` body into the `v0.61.0` form the
/// `releases/{version}/...` path segment expects. `None` when the body is
/// empty, non-UTF-8, or doesn't look like a version (an HTML error page
/// served with 200 from a proxy, a stray changelog, etc.) — the caller
/// then falls back to the pinned default rather than blindly embedding
/// garbage in the release URL.
fn parse_latest_version(bytes: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(bytes)
        .ok()?
        .trim()
        .trim_start_matches('v');
    if !is_version_like(text) {
        return None;
    }
    Some(format!("v{text}"))
}

/// At least two dot-separated numeric segments (`MAJOR.MINOR`, typically
/// `MAJOR.MINOR.PATCH`), optionally followed by a `-prerelease` or
/// `+build` suffix from SemVer's identifier alphabet (`[A-Za-z0-9.-]`).
/// Rejects HTML, changelogs, multi-line content, and other free-form text
/// so a bad `latest.txt` doesn't end up spliced into a release URL.
fn is_version_like(s: &str) -> bool {
    let (main, suffix) = s
        .find(['-', '+'])
        .map(|i| (&s[..i], Some(&s[i + 1..])))
        .unwrap_or((s, None));

    let mut segments = 0usize;
    for part in main.split('.') {
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
        segments += 1;
    }
    if segments < 2 {
        return false;
    }

    if let Some(suffix) = suffix {
        if suffix.is_empty()
            || !suffix
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-')
        {
            return false;
        }
    }
    true
}

/// The `os.ostype` values this build runs on; `None` means "don't filter".
fn host_ostypes() -> Option<&'static [&'static str]> {
    #[cfg(target_os = "windows")]
    {
        Some(&["OPERATING_SYSTEM_TYPE_WINDOWS"])
    }
    #[cfg(target_os = "android")]
    {
        Some(&["OPERATING_SYSTEM_TYPE_ANDROID"])
    }
    #[cfg(target_os = "linux")]
    {
        Some(&[
            "OPERATING_SYSTEM_TYPE_LINUX",
            "OPERATING_SYSTEM_TYPE_QC_LINUX",
        ])
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "android")))]
    {
        None
    }
}

/// Drop chipsets that don't run on any `allowed` OS. A chipset with no
/// device entry is kept, so a payload lacking `devices` stays unfiltered.
fn filter_chipsets_for_host(plat: &PlatformInfo, allowed: &[&str]) -> Vec<ChipsetInfo> {
    plat.chipsets
        .iter()
        .filter(|c| {
            let mut has_os = false;
            for d in &plat.devices {
                if d.chipset != c.name {
                    continue;
                }
                has_os = true;
                if allowed.contains(&d.os.ostype.as_str()) {
                    return true;
                }
            }
            !has_os
        })
        .cloned()
        .collect()
}

/// List the chipsets AI Hub supports on the host OS, sorted by display name
/// (reference device, else canonical id) for a stable FFI order.
pub async fn list_supported_chipsets(cfg: &AiHubConfig) -> Result<Vec<ChipsetInfo>> {
    let transport: Arc<dyn HttpTransport> = Arc::new(ReqwestTransport::new()?);
    let plat = fetch_platform_info(cfg, &transport).await?;
    let mut chipsets = match host_ostypes() {
        Some(allowed) => filter_chipsets_for_host(&plat, allowed),
        None => plat.chipsets,
    };
    chipsets.sort_by(|a, b| {
        let an = if a.reference_device.is_empty() {
            &a.name
        } else {
            &a.reference_device
        };
        let bn = if b.reference_device.is_empty() {
            &b.name
        } else {
            &b.reference_device
        };
        an.to_ascii_lowercase().cmp(&bn.to_ascii_lowercase())
    });
    Ok(chipsets)
}

/// One AI Hub model geniex can run.
#[derive(Debug, Clone)]
pub struct HubModel {
    pub display_name: String,
    pub id: String,
    pub model_type: ModelType,
    pub supported_chipsets: Vec<String>,
}

/// List AI Hub models with a QAIRT asset, sorted by name.
/// `chipset` restricts to models compatible with it (any alias / display name
/// from `platform.json`); `None` lists every one. Host detection is the
/// caller's job — pass the chipset you want to match.
pub async fn list_hub_models(cfg: &AiHubConfig, chipset: Option<&str>) -> Result<Vec<HubModel>> {
    let transport: Arc<dyn HttpTransport> = Arc::new(ReqwestTransport::new()?);

    // `supported_chipsets` holds canonical ids, so map the caller's chipset
    // (which may be an alias or reference-device name) through platform.json.
    let canonical = match chipset {
        Some(chip) => {
            let plat = fetch_platform_info(cfg, &transport).await?;
            Some(selector::resolve_chipset(&plat, chip)?)
        }
        None => None,
    };

    let manifest_url = format!(
        "{}/releases/{}/{MANIFEST_FILENAME}",
        cfg.endpoint, cfg.version
    );
    let manifest_cache = cfg.cache_dir.join(MANIFEST_FILENAME);
    let manifest_bytes = fetch_with_cache(
        &manifest_url,
        &manifest_cache,
        &cfg.version,
        cfg.skip_cache,
        &transport,
    )
    .await?;
    let manifest: ReleaseManifest = parse_manifest("manifest.json", &manifest_bytes)?;

    Ok(select_hub_models(manifest, canonical.as_deref()))
}

/// Pure filter/sort behind [`list_hub_models`], split out for testing.
fn select_hub_models(manifest: ReleaseManifest, device_chipset: Option<&str>) -> Vec<HubModel> {
    let mut models: Vec<HubModel> = manifest
        .models
        .into_iter()
        .filter(|m| m.supported_runtimes.iter().any(|r| is_qairt_runtime(r)))
        .filter(|m| match device_chipset {
            Some(chip) => m.supported_chipsets.iter().any(|c| c == chip),
            None => true,
        })
        .map(|m| HubModel {
            // MODEL_TAG_VLM marks VLMs even when domain is generic GENERATIVE_AI.
            model_type: if m.tags.iter().any(|t| t == "MODEL_TAG_VLM") {
                ModelType::Vlm
            } else {
                ModelType::Llm
            },
            display_name: m.display_name,
            id: m.id,
            supported_chipsets: m.supported_chipsets,
        })
        .collect();

    models.sort_by(|a, b| {
        a.display_name
            .to_ascii_lowercase()
            .cmp(&b.display_name.to_ascii_lowercase())
    });
    models
}

/// Detect the host chipset and resolve it to the reference device name AI
/// Hub displays (e.g. `"Snapdragon X Elite CRD"`), the same name surfaced by
/// [`list_supported_chipsets`]. Best-effort: returns `None` when the host
/// cannot be probed, and falls back to the raw detected id when the chipset
/// catalogue is unavailable or has no entry for it.
pub async fn detect_host_chipset_reference(cfg: &AiHubConfig) -> Option<String> {
    let raw = detect::detect_host_chipset()?;
    let transport: Arc<dyn HttpTransport> = match ReqwestTransport::new() {
        Ok(t) => Arc::new(t),
        Err(_) => return Some(raw),
    };
    match fetch_platform_info(cfg, &transport).await {
        Ok(plat) => Some(selector::resolve_chipset_display(&plat, &raw).unwrap_or(raw)),
        Err(_) => Some(raw),
    }
}

pub struct AiHubSource {
    display_name: String,
    model_name: String,
    cfg: AiHubConfig,
    transport: Arc<dyn HttpTransport>,
}

impl AiHubSource {
    pub fn new(display_name: String, model_name: String, cfg: AiHubConfig) -> Result<Self> {
        let transport: Arc<dyn HttpTransport> = Arc::new(ReqwestTransport::new()?);
        Ok(Self {
            display_name,
            model_name,
            cfg,
            transport,
        })
    }

    pub fn with_transport(
        display_name: String,
        model_name: String,
        cfg: AiHubConfig,
        transport: Arc<dyn HttpTransport>,
    ) -> Self {
        Self {
            display_name,
            model_name,
            cfg,
            transport,
        }
    }

    /// Fetch and parse the per-model `info.json`. Returns `None` on any
    /// failure (missing URL, fetch error, malformed JSON), logging a
    /// warning first — modality classification tolerates absence.
    async fn fetch_info_json(&self, entry: &ManifestModelEntry, version: &str) -> Option<InfoJson> {
        if entry.manifest_urls.info.is_empty() {
            return None;
        }
        let cache_path = self
            .cfg
            .cache_dir
            .join("info")
            .join(format!("{}.json", entry.id));
        let bytes = fetch_with_cache(
            &entry.manifest_urls.info,
            &cache_path,
            version,
            self.cfg.skip_cache,
            &self.transport,
        )
        .await
        .inspect_err(|e| {
            crate::logging::warn(&format!("aihub info.json fetch for {}: {e}", entry.id));
        })
        .ok()?;
        serde_json::from_slice::<InfoJson>(&bytes)
            .inspect_err(|e| {
                crate::logging::warn(&format!("aihub info.json parse for {}: {e}", entry.id));
            })
            .ok()
    }
}

#[async_trait]
impl ModelSource for AiHubSource {
    async fn plan(&self) -> Result<Plan> {
        let endpoint = self.cfg.endpoint.as_str();
        let version = self.cfg.version.as_str();

        let manifest_url = format!("{endpoint}/releases/{version}/{MANIFEST_FILENAME}");
        let manifest_cache = self.cfg.cache_dir.join(MANIFEST_FILENAME);
        let manifest_bytes = fetch_with_cache(
            &manifest_url,
            &manifest_cache,
            version,
            self.cfg.skip_cache,
            &self.transport,
        )
        .await?;
        let release_manifest: ReleaseManifest = parse_manifest("manifest.json", &manifest_bytes)?;

        // Match by display_name first, then fall back to the snake_case
        // `id`, so callers can use either "Llama-v3.2-3B-Instruct" or
        // "llama_v3_2_3b_instruct".
        let entry = release_manifest
            .models
            .iter()
            .find(|m| m.display_name == self.display_name || m.id == self.display_name)
            .ok_or_else(|| Error::HubModelNotFound(self.display_name.clone()))?;

        if !selector::is_domain_supported(&entry.domain) {
            return Err(Error::Hub(format!(
                "AI Hub model {:?} has unsupported domain {:?}",
                self.display_name, entry.domain
            )));
        }

        let release_assets_url = &entry.manifest_urls.release_assets;
        if release_assets_url.is_empty() {
            return Err(Error::Hub(format!(
                "No pre-compiled assets available for {:?} due to licensing \
                 restrictions. Please use the qai-hub-models Python package to \
                 manually export the model. See export instructions here: \
                 https://github.com/qualcomm/ai-hub-apps/blob/main/tutorials/llm_on_genie/export.md",
                self.display_name
            )));
        }

        // release-assets.json is per-model with a URL that rotates each
        // release, so it's fetched uncached.
        let release_assets_bytes = fetch_direct(release_assets_url, &self.transport).await?;
        let release_assets: ModelReleaseAssets =
            parse_manifest("release-assets.json", &release_assets_bytes)?;

        let platform = fetch_platform_info(&self.cfg, &self.transport).await?;

        let chipset: String = if self.cfg.chipset.is_empty() {
            detect::detect_host_chipset().ok_or_else(|| {
                Error::Hub(
                    "chipset not provided and host auto-detect is not supported on this platform"
                        .to_string(),
                )
            })?
        } else {
            self.cfg.chipset.clone()
        };

        let asset = match match_asset(&release_assets, &platform, &chipset) {
            Ok(a) => a,
            Err(UnavailableChipset {
                requested,
                available,
            }) => {
                return Err(Error::ChipsetUnavailable {
                    requested,
                    available,
                });
            }
        };

        let download_url = Url::parse(&asset.download_url).map_err(|e| {
            Error::invalid_url(format!("asset download_url {:?}", asset.download_url), e)
        })?;

        // Only the ZIP64 footer + central directory are fetched here;
        // per-entry payloads stay remote until the executor requests
        // them.
        let raw_entries = fetch_central_directory(&self.transport, &download_url).await?;
        let flat_entries = prepare_flat_entries(&raw_entries)?;

        // Lex-first `.bin` matches the Go CLI's `ExtractFlat`.
        let (mut model_file, extra_files) = super::split_entrypoint_and_extras(
            &flat_entries,
            || "no .bin shard in archive".to_string(),
            |e| e.uncompressed_size as i64,
        )?;

        let precision_label = asset
            .precision
            .strip_prefix("PRECISION_")
            .unwrap_or(&asset.precision)
            .to_string();

        // Key by real precision so /v1/models ids round-trip to /v1/chat/completions (#1242).
        if !precision_label.is_empty() {
            if let Some(entry) = model_file.remove("N/A") {
                model_file.insert(precision_label.clone(), entry);
            }
        }

        // `domain` alone can't distinguish Qwen2.5-VL from text-only LLMs
        // (both report MODEL_DOMAIN_GENERATIVE_AI), so we also read the
        // per-model info.json. Failure is non-fatal: `classify_ai_hub`
        // falls back to the domain-only signal.
        let info = self.fetch_info_json(entry, version).await;
        let model_type = classify_ai_hub(info.as_ref(), entry);
        let manifest = ModelManifest {
            name: self.model_name.clone(),
            model_name: if entry.id.is_empty() {
                self.display_name.clone()
            } else {
                entry.id.clone()
            },
            model_type,
            plugin_id: "qairt".to_string(),
            precision: String::new(),
            model_file,
            mmproj_file: ModelFileInfo::default(),
            tokenizer_file: ModelFileInfo::default(),
            extra_files,
        };

        let files: Vec<FileSpec> = flat_entries
            .into_iter()
            .map(|(name, e)| {
                let bytes = match e.method {
                    Method::Stored => BytesSource::HttpRange {
                        url: download_url.clone(),
                        auth: None,
                        offset: e.payload_offset,
                        len: e.compressed_size,
                    },
                    Method::Deflate => BytesSource::HttpDeflate {
                        url: download_url.clone(),
                        auth: None,
                        offset: e.payload_offset,
                        compressed_len: e.compressed_size,
                    },
                };
                FileSpec {
                    name,
                    size: e.uncompressed_size,
                    bytes,
                }
            })
            .collect();

        Ok(Plan { manifest, files })
    }
}

/// Reduce the raw central-directory entries to a flat list of
/// `(basename, entry)` pairs, skipping directories + macOS AppleDouble
/// metadata and rejecting basename collisions.
pub(crate) fn prepare_flat_entries(raw: &[ZipEntry]) -> Result<Vec<(String, ZipEntry)>> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<(String, ZipEntry)> = Vec::new();
    for e in raw {
        if e.is_dir {
            continue;
        }
        if is_macos_metadata(&e.name) {
            continue;
        }
        let base = basename(&e.name);
        if base.is_empty() || base == "." || base == ".." {
            continue;
        }
        if !seen.insert(base.clone()) {
            return Err(Error::Hub(format!(
                "duplicate basename {base:?} in archive (from entry {:?})",
                e.name
            )));
        }
        out.push((base, e.clone()));
    }
    Ok(out)
}

fn is_macos_metadata(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    if normalized.starts_with("__MACOSX/") || normalized.contains("/__MACOSX/") {
        return true;
    }
    basename(&normalized).starts_with("._")
}

async fn fetch_with_cache(
    url: &str,
    cache_path: &Path,
    version: &str,
    skip_cache: bool,
    transport: &Arc<dyn HttpTransport>,
) -> Result<Vec<u8>> {
    if !skip_cache {
        if let Some(bytes) = read_cache(cache_path, version, CACHE_TTL) {
            return Ok(bytes);
        }
    }
    let bytes = fetch_direct(url, transport).await?;
    if !skip_cache {
        write_cache(cache_path, &bytes);
    }
    Ok(bytes)
}

/// Return cached bytes when the file's mtime is still within `ttl`. Pure
/// filesystem check — no content parsing, no version stamp comparison —
/// so it's the right fit for `latest.txt`, whose body *is* the version
/// and carries no self-identifying field to cross-check. Also used as
/// the freshness precheck inside [`read_cache`] before it validates the
/// embedded version stamp.
fn read_cache_fresh(path: &Path, ttl: Duration) -> Option<Vec<u8>> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?;
    if SystemTime::now().duration_since(mtime).ok()? > ttl {
        return None;
    }
    std::fs::read(path).ok()
}

/// Return cached bytes if the file is within `ttl` and stamped with the
/// requested `version`, so a release bump invalidates stale caches before
/// the TTL elapses. Any I/O or parse error is a cache miss.
fn read_cache(path: &Path, version: &str, ttl: Duration) -> Option<Vec<u8>> {
    let bytes = read_cache_fresh(path, ttl)?;
    // manifest.json stamps `version`; platform.json / info.json use
    // `aihm_version`. Both omit the leading `v` the config carries.
    let doc: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let cached = doc.get("version").or_else(|| doc.get("aihm_version"))?;
    (cached.as_str()? == version.trim_start_matches('v')).then_some(bytes)
}

/// Best-effort cache write. Failures are logged and swallowed — the
/// caller always gets the freshly fetched bytes regardless.
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

async fn fetch_direct(url: &str, transport: &Arc<dyn HttpTransport>) -> Result<Vec<u8>> {
    let parsed = Url::parse(url).map_err(|e| Error::invalid_url(format!("url {url:?}"), e))?;
    let head = transport.head(&parsed, None).await?;
    if head.size > MAX_INDEX_BYTES {
        return Err(Error::Hub(format!(
            "index at {url} is {} bytes, exceeds {MAX_INDEX_BYTES}-byte cap",
            head.size
        )));
    }
    let mut buf: Vec<u8> = Vec::with_capacity(head.size as usize);
    transport
        .get_range(&parsed, None, 0, head.size, &mut buf)
        .await?;
    Ok(buf)
}

/// Lowercase substrings in `info.description` / `info.headline` that
/// mark an AI Hub model as vision-language. Kept together so the set
/// is easy to audit when AI Hub updates its copy.
const AI_HUB_VLM_KEYWORDS: &[&str] = &[
    "vision-language",
    "vision language",
    "multimodal",
    "image-text",
    "image and text",
    "images and text",
    "process both images",
    "visual question answering",
    "understand images",
    "understanding images",
    "vlm",
];

/// Modality classifier driven by the `metadata.json` shipped *inside*
/// an AI Hub archive. Mirrors the Go CLI's `detectModelTypeFromDir`:
/// only `genie.supports_vision` is consulted today. Returns `None` when
/// the bytes don't parse or the field is absent so the caller can fall
/// back to a default (LLM) — matching the Go behaviour where an absent
/// or unparseable file degrades to LLM rather than aborting the pull.
pub(crate) fn classify_from_metadata_json(bytes: &[u8]) -> Option<ModelType> {
    #[derive(serde::Deserialize)]
    struct Outer {
        #[serde(default)]
        genie: Option<GenieSection>,
    }
    #[derive(serde::Deserialize)]
    struct GenieSection {
        #[serde(default)]
        supports_vision: Option<bool>,
    }
    let outer: Outer = serde_json::from_slice(bytes).ok()?;
    let supports_vision = outer.genie?.supports_vision?;
    Some(if supports_vision {
        ModelType::Vlm
    } else {
        ModelType::Llm
    })
}

/// Modality classifier for the AI Hub source. `domain == MULTIMODAL`
/// is retained as a positive signal; for `GENERATIVE_AI` models (which
/// include Qwen2.5-VL), we keyword-match the info.json description +
/// headline. Defaults to LLM when no positive signal is present.
fn classify_ai_hub(info: Option<&InfoJson>, entry: &ManifestModelEntry) -> ModelType {
    if entry.domain == "MODEL_DOMAIN_MULTIMODAL" {
        return ModelType::Vlm;
    }
    if let Some(info) = info {
        let haystack = format!("{} {}", info.description, info.headline).to_lowercase();
        if AI_HUB_VLM_KEYWORDS.iter().any(|kw| haystack.contains(kw)) {
            return ModelType::Vlm;
        }
    }
    ModelType::Llm
}

#[cfg(test)]
mod tests {
    use self::dto::ManifestUrls;
    use super::*;

    fn entry(id: &str, domain: &str) -> ManifestModelEntry {
        ManifestModelEntry {
            id: id.to_string(),
            display_name: id.to_string(),
            domain: domain.to_string(),
            manifest_urls: ManifestUrls::default(),
            supported_runtimes: Vec::new(),
            supported_chipsets: Vec::new(),
            tags: Vec::new(),
        }
    }

    fn info(description: &str, headline: &str) -> InfoJson {
        InfoJson {
            domain: "MODEL_DOMAIN_GENERATIVE_AI".to_string(),
            headline: headline.to_string(),
            description: description.to_string(),
            tags: Vec::new(),
        }
    }

    #[test]
    fn classify_domain_multimodal_is_vlm() {
        // #12
        let e = entry("qwen2_5_vl_7b_instruct", "MODEL_DOMAIN_MULTIMODAL");
        assert_eq!(classify_ai_hub(None, &e), ModelType::Vlm);
    }

    #[test]
    fn classify_qwen25_vl_generative_ai_info_description() {
        // #13 regression-blocker: live manifest reports GENERATIVE_AI
        // for Qwen2.5-VL. Description must rescue it.
        let e = entry("qwen2_5_vl_7b_instruct", "MODEL_DOMAIN_GENERATIVE_AI");
        let i = info(
            "Qwen2.5-VL-7B-Instruct is a multimodal vision-language model that processes both images and text",
            "",
        );
        assert_eq!(classify_ai_hub(Some(&i), &e), ModelType::Vlm);
    }

    #[test]
    fn classify_llama_generative_ai_stays_llm() {
        // #14
        let e = entry("llama_v3_8b_instruct", "MODEL_DOMAIN_GENERATIVE_AI");
        let i = info(
            "Llama 3 is a state-of-the-art large language model",
            "State-of-the-art large language model",
        );
        assert_eq!(classify_ai_hub(Some(&i), &e), ModelType::Llm);
    }

    #[test]
    fn classify_headline_only_hit() {
        // #15 — headline alone carries the VLM signal.
        let e = entry("some_vlm", "MODEL_DOMAIN_GENERATIVE_AI");
        let i = info("", "Visual question answering on mobile");
        assert_eq!(classify_ai_hub(Some(&i), &e), ModelType::Vlm);
    }

    #[test]
    fn classify_info_missing_defaults_llm() {
        // #16 — fetch failed; fall back to domain-only (GENERATIVE_AI
        // isn't a VLM signal) → LLM.
        let e = entry("mystery_model", "MODEL_DOMAIN_GENERATIVE_AI");
        assert_eq!(classify_ai_hub(None, &e), ModelType::Llm);
    }

    fn hub_entry(
        id: &str,
        runtimes: &[&str],
        chipsets: &[&str],
        tags: &[&str],
    ) -> ManifestModelEntry {
        ManifestModelEntry {
            id: id.to_string(),
            display_name: id.to_string(),
            domain: "MODEL_DOMAIN_GENERATIVE_AI".to_string(),
            manifest_urls: ManifestUrls::default(),
            supported_runtimes: runtimes.iter().map(|s| s.to_string()).collect(),
            supported_chipsets: chipsets.iter().map(|s| s.to_string()).collect(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn select_keeps_only_qairt_models() {
        let manifest = ReleaseManifest {
            platform_url: String::new(),
            models: vec![
                hub_entry("qairt_model", &["RUNTIME_GENIEX_QAIRT"], &["chipA"], &[]),
                hub_entry(
                    "llamacpp_only",
                    &["RUNTIME_GENIEX_LLAMACPP"],
                    &["chipA"],
                    &[],
                ),
                hub_entry(
                    "cv_model",
                    &["RUNTIME_TFLITE", "RUNTIME_ONNX"],
                    &["chipA"],
                    &[],
                ),
            ],
        };
        let out = select_hub_models(manifest, None);
        let ids: Vec<&str> = out.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["qairt_model"]);
    }

    /// `Phi-3.5-Mini-Instruct` (bucket v0.60.0) declares only the legacy
    /// `RUNTIME_GENIE` enum, so filtering on `RUNTIME_GENIEX_QAIRT` alone
    /// silently hid it from `list` even though its assets are installable.
    #[test]
    fn select_keeps_legacy_genie_only_models() {
        let manifest = ReleaseManifest {
            platform_url: String::new(),
            models: vec![hub_entry(
                "phi_3_5_mini_instruct",
                &["RUNTIME_QNN_CONTEXT_BINARY", "RUNTIME_GENIE"],
                &["qualcomm-snapdragon-x-elite"],
                &[],
            )],
        };
        let out = select_hub_models(manifest, None);
        let ids: Vec<&str> = out.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["phi_3_5_mini_instruct"]);
    }

    #[test]
    fn select_filters_by_device_chipset() {
        let manifest = ReleaseManifest {
            platform_url: String::new(),
            models: vec![
                hub_entry(
                    "on_device",
                    &["RUNTIME_GENIEX_QAIRT"],
                    &["chipA", "chipB"],
                    &[],
                ),
                hub_entry("off_device", &["RUNTIME_GENIEX_QAIRT"], &["chipB"], &[]),
            ],
        };
        let out = select_hub_models(manifest, Some("chipA"));
        let ids: Vec<&str> = out.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["on_device"]);
    }

    #[test]
    fn select_unfiltered_when_no_device() {
        let manifest = ReleaseManifest {
            platform_url: String::new(),
            models: vec![
                hub_entry("b_model", &["RUNTIME_GENIEX_QAIRT"], &["chipA"], &[]),
                hub_entry("a_model", &["RUNTIME_GENIEX_QAIRT"], &["chipB"], &[]),
            ],
        };
        // None → no device filter; also asserts case-insensitive name sort.
        let out = select_hub_models(manifest, None);
        let ids: Vec<&str> = out.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["a_model", "b_model"]);
    }

    #[test]
    fn select_classifies_vlm_from_tag() {
        let manifest = ReleaseManifest {
            platform_url: String::new(),
            models: vec![
                hub_entry(
                    "vl_model",
                    &["RUNTIME_GENIEX_QAIRT"],
                    &["chipA"],
                    &["MODEL_TAG_VLM"],
                ),
                hub_entry(
                    "text_model",
                    &["RUNTIME_GENIEX_QAIRT"],
                    &["chipA"],
                    &["MODEL_TAG_LLM"],
                ),
            ],
        };
        let out = select_hub_models(manifest, None);
        assert_eq!(out[0].id, "text_model"); // sorted: text_ before vl_
        assert_eq!(out[0].model_type, ModelType::Llm);
        assert_eq!(out[1].model_type, ModelType::Vlm);
    }

    #[test]
    fn rejects_basename_collision() {
        let raw = vec![
            ZipEntry {
                name: "a/model.bin".to_string(),
                method: Method::Stored,
                payload_offset: 0,
                compressed_size: 1,
                uncompressed_size: 1,
                is_dir: false,
            },
            ZipEntry {
                name: "b/model.bin".to_string(),
                method: Method::Stored,
                payload_offset: 100,
                compressed_size: 1,
                uncompressed_size: 1,
                is_dir: false,
            },
        ];
        assert!(prepare_flat_entries(&raw).is_err());
    }

    #[test]
    fn skips_macos_metadata() {
        let raw = vec![
            ZipEntry {
                name: "dir/model.bin".to_string(),
                method: Method::Stored,
                payload_offset: 0,
                compressed_size: 2,
                uncompressed_size: 2,
                is_dir: false,
            },
            ZipEntry {
                name: "__MACOSX/dir/._model.bin".to_string(),
                method: Method::Stored,
                payload_offset: 100,
                compressed_size: 4,
                uncompressed_size: 4,
                is_dir: false,
            },
        ];
        let out = prepare_flat_entries(&raw).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "model.bin");
    }

    #[test]
    fn cache_miss_on_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(read_cache(&tmp.path().join("nope.json"), "0.58.0", CACHE_TTL).is_none());
    }

    #[test]
    fn cache_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("x.json");
        write_cache(&p, br#"{"version":"0.58.0"}"#);
        // Requested version is passed with the leading `v`, as cfg holds it.
        assert!(read_cache(&p, "v0.58.0", CACHE_TTL).is_some());
        assert!(read_cache(&p, "v0.58.0", Duration::ZERO).is_none());
    }

    #[test]
    fn cache_miss_on_version_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = tmp.path().join("manifest.json");
        write_cache(&manifest, br#"{"version":"0.52.0"}"#);
        assert!(read_cache(&manifest, "v0.58.0", CACHE_TTL).is_none());
        assert!(read_cache(&manifest, "v0.52.0", CACHE_TTL).is_some());

        // platform.json / info.json use the `aihm_version` key instead.
        let platform = tmp.path().join("platform.json");
        write_cache(&platform, br#"{"aihm_version":"0.52.0"}"#);
        assert!(read_cache(&platform, "v0.58.0", CACHE_TTL).is_none());
        assert!(read_cache(&platform, "v0.52.0", CACHE_TTL).is_some());
    }

    fn chipset(name: &str) -> ChipsetInfo {
        ChipsetInfo {
            name: name.to_string(),
            reference_device: String::new(),
            aliases: Vec::new(),
        }
    }

    fn device(chipset: &str, ostype: &str) -> dto::DeviceInfo {
        dto::DeviceInfo {
            chipset: chipset.to_string(),
            os: dto::DeviceOs {
                ostype: ostype.to_string(),
            },
        }
    }

    fn sample_platform() -> PlatformInfo {
        PlatformInfo {
            chipsets: vec![
                chipset("qualcomm-snapdragon-x-elite"),
                chipset("qualcomm-qcs9075"),
                chipset("qualcomm-qcs8275"),
                chipset("qualcomm-snapdragon-8gen3"),
            ],
            devices: vec![
                device(
                    "qualcomm-snapdragon-x-elite",
                    "OPERATING_SYSTEM_TYPE_WINDOWS",
                ),
                device("qualcomm-qcs9075", "OPERATING_SYSTEM_TYPE_QC_LINUX"),
                device("qualcomm-qcs8275", "OPERATING_SYSTEM_TYPE_LINUX"),
                device("qualcomm-snapdragon-8gen3", "OPERATING_SYSTEM_TYPE_ANDROID"),
            ],
        }
    }

    fn names(chipsets: &[ChipsetInfo]) -> Vec<&str> {
        chipsets.iter().map(|c| c.name.as_str()).collect()
    }

    #[test]
    fn filter_keeps_only_linux_chipsets() {
        let out = filter_chipsets_for_host(
            &sample_platform(),
            &[
                "OPERATING_SYSTEM_TYPE_LINUX",
                "OPERATING_SYSTEM_TYPE_QC_LINUX",
            ],
        );
        assert_eq!(names(&out), vec!["qualcomm-qcs9075", "qualcomm-qcs8275"]);
    }

    #[test]
    fn filter_keeps_only_windows_chipset() {
        let out = filter_chipsets_for_host(&sample_platform(), &["OPERATING_SYSTEM_TYPE_WINDOWS"]);
        assert_eq!(names(&out), vec!["qualcomm-snapdragon-x-elite"]);
    }

    #[test]
    fn filter_keeps_only_android_chipset() {
        let out = filter_chipsets_for_host(&sample_platform(), &["OPERATING_SYSTEM_TYPE_ANDROID"]);
        assert_eq!(names(&out), vec!["qualcomm-snapdragon-8gen3"]);
    }

    #[test]
    fn filter_keeps_chipset_without_device_entry() {
        let plat = PlatformInfo {
            chipsets: vec![
                chipset("qualcomm-qcs9075"),
                chipset("qualcomm-snapdragon-8gen3"),
            ],
            devices: Vec::new(),
        };
        let out = filter_chipsets_for_host(&plat, &["OPERATING_SYSTEM_TYPE_WINDOWS"]);
        assert_eq!(
            names(&out),
            vec!["qualcomm-qcs9075", "qualcomm-snapdragon-8gen3"]
        );
    }

    #[test]
    fn filter_keeps_multi_os_chipset_when_any_matches() {
        let plat = PlatformInfo {
            chipsets: vec![chipset("dual")],
            devices: vec![
                device("dual", "OPERATING_SYSTEM_TYPE_ANDROID"),
                device("dual", "OPERATING_SYSTEM_TYPE_LINUX"),
            ],
        };
        assert_eq!(
            names(&filter_chipsets_for_host(
                &plat,
                &["OPERATING_SYSTEM_TYPE_LINUX"]
            )),
            vec!["dual"]
        );
    }

    #[test]
    fn parse_latest_version_accepts_common_shapes() {
        assert_eq!(parse_latest_version(b"0.61.0"), Some("v0.61.0".into()));
        assert_eq!(parse_latest_version(b"v0.61.0"), Some("v0.61.0".into()));
        assert_eq!(parse_latest_version(b"0.61.0\n"), Some("v0.61.0".into()));
        assert_eq!(parse_latest_version(b"  0.61.0  "), Some("v0.61.0".into()));
        // Two-segment and four-segment both look like versions.
        assert_eq!(parse_latest_version(b"1.0"), Some("v1.0".into()));
        assert_eq!(parse_latest_version(b"1.2.3.4"), Some("v1.2.3.4".into()));
        // SemVer pre-release / build suffixes.
        assert_eq!(
            parse_latest_version(b"0.61.0-rc1"),
            Some("v0.61.0-rc1".into())
        );
        assert_eq!(
            parse_latest_version(b"0.61.0+build.5"),
            Some("v0.61.0+build.5".into())
        );
    }

    #[test]
    fn parse_latest_version_rejects_non_versions() {
        // Empty / whitespace-only.
        assert_eq!(parse_latest_version(b""), None);
        assert_eq!(parse_latest_version(b"   \n"), None);
        // Non-UTF-8.
        assert_eq!(parse_latest_version(&[0xff, 0xfe, 0xfd]), None);
        // HTML error page served as 200 by a proxy.
        assert_eq!(
            parse_latest_version(b"<!doctype html><html>404 not found</html>"),
            None
        );
        // Free-form text (e.g. a stray changelog uploaded by mistake).
        assert_eq!(parse_latest_version(b"Release notes for 0.61.0"), None);
        // Single segment isn't enough to look like a version.
        assert_eq!(parse_latest_version(b"1"), None);
        // Trailing dot / empty segment.
        assert_eq!(parse_latest_version(b"1.0."), None);
        // Multi-line — after trim, still has an internal newline.
        assert_eq!(parse_latest_version(b"0.61.0\nextra"), None);
    }
}
