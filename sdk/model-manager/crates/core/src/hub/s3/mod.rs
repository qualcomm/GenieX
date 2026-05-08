//! AI Hub qairt runtime pull over the public `qaihub-public-assets` S3
//! bucket.
//!
//! Mirrors the Go CLI flow in `cli/internal/model_hub/aihub/`:
//!
//! 1. Fetch `manifest.json` from the pinned release (`<base>/releases/<ver>/manifest.json`).
//! 2. Look up the requested model by `display_name`, follow its
//!    `manifestUrls.releaseAssets` URL.
//! 3. Fetch `platform.json` to resolve chipset aliases, then pick the
//!    asset matching `(canonical_chipset, RUNTIME_GENIE)` (see
//!    [`selector::match_asset`]).
//! 4. Download the asset zip via the shared [`Engine`] (chunking +
//!    `.progress` resume for free).
//! 5. Flat-extract into the model directory and synthesise a
//!    `geniex.json` keyed by the user-supplied `model_name`.
//!
//! The produced directory layout is byte-identical to what
//! `Store.PullZipAsset` in the Go CLI writes, so a cache populated by
//! either agent is interchangeable.

pub mod cache;
pub mod detect;
pub mod extract;
pub mod manifest;
pub mod selector;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use url::Url;

use crate::config::MIN_SDK_VERSION;
use crate::download::{Engine, EngineConfig};
use crate::error::{Error, Result};
use crate::hub::metadata::{FileSource, HubContext, HubMetadata};
use crate::hub::{ProgressCallback, RemoteFile};
use crate::manifest::{ModelFileInfo, ModelManifest, ModelType};
use crate::store::{Store, INFLIGHT_DIR, MANIFEST_FILE};
use crate::transport::{HttpTransport, ReqwestTransport};

use self::manifest::{ModelReleaseAssets, PlatformInfo, ReleaseManifest};
use self::selector::{match_asset, UnavailableChipset};

const MANIFEST_FILENAME: &str = "manifest.json";
const PLATFORM_FILENAME: &str = "platform.json";
const MAX_INDEX_BYTES: u64 = 8 * 1024 * 1024;

/// Caller-supplied knobs. `endpoint` + `version` map to the Go CLI's
/// `AIHubBaseURL` / `AIHubVersion`.
///
/// Construct via [`S3Config::new`] to normalize `endpoint` (no trailing
/// `/`) and `version` (no surrounding `/`) once at construction time —
/// the pull hot path then formats URLs directly without per-call trim.
#[derive(Debug, Clone)]
pub struct S3Config {
    pub endpoint: String,
    pub version: String,
    /// Device name matched against `platform.json` aliases. An empty
    /// string triggers [`detect::detect_host_chipset`] as a fallback —
    /// currently only supported on Windows-on-Snapdragon hosts.
    pub chipset: String,
    pub cache_dir: PathBuf,
    pub skip_cache: bool,
}

impl S3Config {
    /// Normalize `endpoint` / `version` once so the download path can
    /// skip repeated `trim_end_matches('/')` calls.
    pub fn new(
        endpoint: impl Into<String>,
        version: impl Into<String>,
        chipset: impl Into<String>,
        cache_dir: PathBuf,
        skip_cache: bool,
    ) -> Self {
        let endpoint: String = endpoint.into().trim_end_matches('/').to_string();
        let version: String = version.into().trim_matches('/').to_string();
        Self {
            endpoint,
            version,
            chipset: chipset.into(),
            cache_dir,
            skip_cache,
        }
    }
}

/// Run the AI Hub pull for `model_name` (the "org/repo" the user typed,
/// used as the on-disk model directory). `display_name` selects which
/// entry in the AI Hub manifest to download — usually equal to
/// `model_name` or a short alias resolved by the caller.
///
/// Sync callers should go through [`crate::pull::pull_blocking`] which
/// drives this on the FFI crate's shared runtime.
pub async fn pull_ai_hub(
    store: &Store,
    model_name: &str,
    display_name: &str,
    cfg: S3Config,
    on_progress: Option<&ProgressCallback>,
) -> Result<()> {
    let transport: Arc<dyn HttpTransport> = Arc::new(ReqwestTransport::new()?);
    pull_ai_hub_inner(store, model_name, display_name, cfg, transport, on_progress).await
}

/// Same as [`pull_ai_hub`] but takes an explicit transport. Exposed for
/// integration tests that want to point at a mock server without going
/// through the default `ReqwestTransport` factory.
pub async fn pull_ai_hub_with_transport(
    store: &Store,
    model_name: &str,
    display_name: &str,
    cfg: S3Config,
    transport: Arc<dyn HttpTransport>,
    on_progress: Option<&ProgressCallback>,
) -> Result<()> {
    pull_ai_hub_inner(store, model_name, display_name, cfg, transport, on_progress).await
}

async fn pull_ai_hub_inner(
    store: &Store,
    model_name: &str,
    display_name: &str,
    cfg: S3Config,
    transport: Arc<dyn HttpTransport>,
    on_progress: Option<&ProgressCallback>,
) -> Result<()> {
    // `S3Config::new` normalizes these; callers who build the struct by
    // field literal still get correct URLs because the extra calls below
    // used to trim defensively. See S3Config::new doc.
    let endpoint = cfg.endpoint.as_str();
    let version = cfg.version.as_str();

    // 1. manifest.json + lookup
    let manifest_url = format!("{endpoint}/releases/{version}/{MANIFEST_FILENAME}");
    let manifest_cache = cfg.cache_dir.join(MANIFEST_FILENAME);
    let manifest_bytes =
        fetch_with_cache(&manifest_url, &manifest_cache, cfg.skip_cache, &transport).await?;
    let release_manifest: ReleaseManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|source| Error::ManifestParse {
            what: "manifest.json",
            source,
        })?;

    let entry = release_manifest
        .models
        .iter()
        .find(|m| m.display_name == display_name)
        .ok_or_else(|| {
            Error::Hub(format!(
                "model {display_name:?} not found in AI Hub manifest"
            ))
        })?;

    if !selector::is_domain_supported(&entry.domain) {
        return Err(Error::Hub(format!(
            "AI Hub model {display_name:?} has unsupported domain {:?}",
            entry.domain
        )));
    }

    let release_assets_url = &entry.manifest_urls.release_assets;
    if release_assets_url.is_empty() {
        return Err(Error::Hub(format!(
            "AI Hub model {display_name:?} has no release_assets URL"
        )));
    }

    // 2. release-assets.json (direct fetch — per-model, URL changes each release).
    let release_assets_bytes = fetch_direct(release_assets_url, &transport).await?;
    let release_assets: ModelReleaseAssets = serde_json::from_slice(&release_assets_bytes)
        .map_err(|source| Error::ManifestParse {
            what: "release-assets.json",
            source,
        })?;

    // 3. platform.json (shared across all models, very cacheable).
    let platform_url = format!("{endpoint}/releases/{version}/{PLATFORM_FILENAME}");
    let platform_cache = cfg.cache_dir.join(PLATFORM_FILENAME);
    let platform_bytes =
        fetch_with_cache(&platform_url, &platform_cache, cfg.skip_cache, &transport).await?;
    let platform: PlatformInfo =
        serde_json::from_slice(&platform_bytes).map_err(|source| Error::ManifestParse {
            what: "platform.json",
            source,
        })?;

    // 4. Resolve chipset. Empty string means "auto-detect from host".
    let chipset: String = if cfg.chipset.is_empty() {
        detect::detect_host_chipset().ok_or_else(|| {
            Error::Hub(
                "chipset not provided and host auto-detect is not supported on this platform"
                    .to_string(),
            )
        })?
    } else {
        cfg.chipset.clone()
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
        Error::Hub(format!(
            "invalid asset download_url {:?}: {e}",
            asset.download_url
        ))
    })?;

    // 4. Download the zip through the shared engine. `HubContext` needs a
    //    `HubMetadata` — for a single-file download we plug in a minimal
    //    shim so `EngineConfig::resolve` + `Engine::run` work unchanged.
    let zip_basename = format!(
        "{}.zip",
        model_name.rsplit('/').next().unwrap_or(model_name)
    );
    let file_source = FileSource {
        name: zip_basename.clone(),
        size: None, // Engine will HEAD to discover.
        url: download_url,
        auth: None,
    };

    let dest_dir = store.config().model_dir(model_name);
    std::fs::create_dir_all(&dest_dir)?;
    let inflight_dir = dest_dir.join(INFLIGHT_DIR);
    std::fs::create_dir_all(&inflight_dir)?;

    let ctx = HubContext::new(Arc::new(SingleFileMetadata), transport.clone());
    let engine = Engine::with_config(&ctx, EngineConfig::resolve(&ctx));
    engine
        .run(vec![file_source], &dest_dir, on_progress)
        .await?;

    // 5. Extract + synthesise manifest + clean up.
    let zip_path = dest_dir.join(&zip_basename);
    let result = extract::extract_flat(&zip_path, &dest_dir)?;
    if let Err(e) = std::fs::remove_file(&zip_path) {
        eprintln!(
            "[model-manager] aihub failed to remove zip after extract {}: {e}",
            zip_path.display()
        );
    }
    let marker = PathBuf::from(format!(
        "{}{}",
        zip_path.display(),
        crate::pull::PROGRESS_SUFFIX
    ));
    let _ = std::fs::remove_file(&marker);

    let precision_label = asset
        .precision
        .strip_prefix("PRECISION_")
        .unwrap_or(&asset.precision)
        .to_string();
    let mut model_file: HashMap<String, ModelFileInfo> = HashMap::new();
    model_file.insert(
        "N/A".to_string(),
        ModelFileInfo {
            name: result.entrypoint_basename.clone(),
            downloaded: true,
            size: result.total_size as i64,
        },
    );
    let extra_files: Vec<ModelFileInfo> = result
        .files
        .iter()
        .filter(|f| f.name != result.entrypoint_basename)
        .map(|f| ModelFileInfo {
            name: f.name.clone(),
            downloaded: true,
            size: f.size as i64,
        })
        .collect();

    let model_type = infer_model_type(&entry.domain);
    let synthesised = ModelManifest {
        name: model_name.to_string(),
        model_name: if entry.id.is_empty() {
            display_name.to_string()
        } else {
            entry.id.clone()
        },
        model_type,
        plugin_id: "qairt".to_string(),
        device_id: asset.chipset.clone().unwrap_or_default(),
        min_sdk_version: MIN_SDK_VERSION.to_string(),
        precision: precision_label,
        model_file,
        mmproj_file: ModelFileInfo::default(),
        tokenizer_file: ModelFileInfo::default(),
        extra_files,
    };

    let staged = inflight_dir.join(MANIFEST_FILE);
    std::fs::write(&staged, serde_json::to_string(&synthesised)?)?;
    let final_path = dest_dir.join(MANIFEST_FILE);
    std::fs::rename(&staged, &final_path)?;
    let _ = std::fs::remove_dir_all(&inflight_dir);

    Ok(())
}

fn infer_model_type(domain: &str) -> ModelType {
    match domain {
        "MODEL_DOMAIN_MULTIMODAL" => ModelType::Vlm,
        _ => ModelType::Llm,
    }
}

async fn fetch_with_cache(
    url: &str,
    cache_path: &Path,
    skip_cache: bool,
    transport: &Arc<dyn HttpTransport>,
) -> Result<Vec<u8>> {
    if !skip_cache {
        if let Some(bytes) = cache::read_if_fresh(cache_path, cache::DEFAULT_TTL) {
            return Ok(bytes);
        }
    }
    let bytes = fetch_direct(url, transport).await?;
    if !skip_cache {
        cache::write(cache_path, &bytes);
    }
    Ok(bytes)
}

async fn fetch_direct(url: &str, transport: &Arc<dyn HttpTransport>) -> Result<Vec<u8>> {
    let parsed = Url::parse(url).map_err(|e| Error::Hub(format!("invalid url {url:?}: {e}")))?;
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

/// Placeholder `HubMetadata` for the single-zip download case. The
/// engine only calls `default_file_concurrency`; `list_files` and
/// `resolve` are unreachable from this path but must be present.
struct SingleFileMetadata;

#[async_trait::async_trait]
impl HubMetadata for SingleFileMetadata {
    async fn list_files(&self, _repo: &str) -> Result<(Vec<RemoteFile>, Option<ModelManifest>)> {
        Err(Error::Hub(
            "SingleFileMetadata: list_files not supported".to_string(),
        ))
    }

    async fn resolve(&self, _repo: &str, _files: &[String]) -> Result<Vec<FileSource>> {
        Err(Error::Hub(
            "SingleFileMetadata: resolve not supported".to_string(),
        ))
    }

    fn default_file_concurrency(&self) -> usize {
        1
    }
}
