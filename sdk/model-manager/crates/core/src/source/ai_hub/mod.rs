//! Qualcomm AI Hub [`ModelSource`].
//!
//! Resolves the asset URL for a `(display_name, chipset)` pair via the
//! public S3 protojson chain, then reads the remote archive's ZIP64
//! central directory over HTTP Range reads to produce a complete
//! [`Plan`] — manifest + one per-entry [`BytesSource`] — without
//! downloading the multi-GB payload.

pub mod detect;
pub mod local_transport;
pub mod manifest;
pub mod remote_zip;
pub mod selector;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use url::Url;

use crate::error::{Error, Result};
use crate::manifest::{ModelFileInfo, ModelManifest, ModelType};
use crate::transport::{HttpTransport, ReqwestTransport};

use self::manifest::{
    InfoJson, ManifestModelEntry, ModelReleaseAssets, PlatformInfo, ReleaseManifest,
};
use self::remote_zip::{fetch_central_directory, Method, ZipEntry};
use self::selector::{match_asset, UnavailableChipset};

use super::{BytesSource, FileSpec, ModelSource, Plan};

const MANIFEST_FILENAME: &str = "manifest.json";
const PLATFORM_FILENAME: &str = "platform.json";
const MAX_INDEX_BYTES: u64 = 8 * 1024 * 1024;

/// TTL for the on-disk index cache (matches Go CLI's 24h).
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
            self.cfg.skip_cache,
            &self.transport,
        )
        .await?;
        let release_manifest: ReleaseManifest =
            serde_json::from_slice(&manifest_bytes).map_err(|source| Error::ManifestParse {
                what: "manifest.json",
                source,
            })?;

        // Match by display_name first, then fall back to the snake_case
        // `id`, so callers can use either "Llama-v3.2-3B-Instruct" or
        // "llama_v3_2_3b_instruct".
        let entry = release_manifest
            .models
            .iter()
            .find(|m| m.display_name == self.display_name || m.id == self.display_name)
            .ok_or_else(|| {
                Error::Hub(format!(
                    "model {:?} not found in AI Hub manifest",
                    self.display_name
                ))
            })?;

        if !selector::is_domain_supported(&entry.domain) {
            return Err(Error::Hub(format!(
                "AI Hub model {:?} has unsupported domain {:?}",
                self.display_name, entry.domain
            )));
        }

        let release_assets_url = &entry.manifest_urls.release_assets;
        if release_assets_url.is_empty() {
            return Err(Error::Hub(format!(
                "AI Hub model {:?} has no release_assets URL",
                self.display_name
            )));
        }

        // release-assets.json is per-model with a URL that rotates each
        // release, so it's fetched uncached.
        let release_assets_bytes = fetch_direct(release_assets_url, &self.transport).await?;
        let release_assets: ModelReleaseAssets = serde_json::from_slice(&release_assets_bytes)
            .map_err(|source| Error::ManifestParse {
                what: "release-assets.json",
                source,
            })?;

        let platform_url = format!("{endpoint}/releases/{version}/{PLATFORM_FILENAME}");
        let platform_cache = self.cfg.cache_dir.join(PLATFORM_FILENAME);
        let platform_bytes = fetch_with_cache(
            &platform_url,
            &platform_cache,
            self.cfg.skip_cache,
            &self.transport,
        )
        .await?;
        let platform: PlatformInfo =
            serde_json::from_slice(&platform_bytes).map_err(|source| Error::ManifestParse {
                what: "platform.json",
                source,
            })?;

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
            Error::Hub(format!(
                "invalid asset download_url {:?}: {e}",
                asset.download_url
            ))
        })?;

        // Only the ZIP64 footer + central directory are fetched here;
        // per-entry payloads stay remote until the executor requests
        // them.
        let raw_entries = fetch_central_directory(&self.transport, &download_url).await?;
        let flat_entries = prepare_flat_entries(&raw_entries)?;

        // Lex-first `.bin` matches the Go CLI's `ExtractFlat`, so a
        // cache populated by either agent is interchangeable.
        let entrypoint_name = flat_entries
            .iter()
            .find(|(name, _)| name.to_ascii_lowercase().ends_with(".bin"))
            .map(|(name, _)| name.clone())
            .ok_or_else(|| Error::Hub("no .bin shard in archive".to_string()))?;

        let precision_label = asset
            .precision
            .strip_prefix("PRECISION_")
            .unwrap_or(&asset.precision)
            .to_string();

        let total_uncompressed: u64 = flat_entries.iter().map(|(_, e)| e.uncompressed_size).sum();

        let mut model_file: HashMap<String, ModelFileInfo> = HashMap::new();
        model_file.insert(
            "N/A".to_string(),
            ModelFileInfo {
                name: entrypoint_name.clone(),
                downloaded: true,
                size: total_uncompressed as i64,
            },
        );
        let extra_files: Vec<ModelFileInfo> = flat_entries
            .iter()
            .filter(|(name, _)| name != &entrypoint_name)
            .map(|(name, e)| ModelFileInfo {
                name: name.clone(),
                downloaded: true,
                size: e.uncompressed_size as i64,
            })
            .collect();

        // `domain` alone cannot distinguish Qwen2.5-VL from text-only
        // LLMs (both report MODEL_DOMAIN_GENERATIVE_AI), so we also
        // read the per-model info.json. Fetch failure is non-fatal:
        // `classify_ai_hub` falls back to the domain-only signal and
        // defaults to LLM if even that is absent.
        let info: Option<InfoJson> = if entry.manifest_urls.info.is_empty() {
            None
        } else {
            let cache_path = self
                .cfg
                .cache_dir
                .join("info")
                .join(format!("{version}-{}.json", entry.id));
            match fetch_with_cache(
                &entry.manifest_urls.info,
                &cache_path,
                self.cfg.skip_cache,
                &self.transport,
            )
            .await
            {
                Ok(bytes) => match serde_json::from_slice::<InfoJson>(&bytes) {
                    Ok(info) => Some(info),
                    Err(e) => {
                        eprintln!(
                            "[model-manager] aihub info.json parse for {}: {e}",
                            entry.id
                        );
                        None
                    }
                },
                Err(e) => {
                    eprintln!(
                        "[model-manager] aihub info.json fetch for {}: {e}",
                        entry.id
                    );
                    None
                }
            }
        };
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
            precision: precision_label,
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

fn basename(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or("").to_string()
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
    skip_cache: bool,
    transport: &Arc<dyn HttpTransport>,
) -> Result<Vec<u8>> {
    if !skip_cache {
        if let Some(bytes) = read_cache(cache_path, CACHE_TTL) {
            return Ok(bytes);
        }
    }
    let bytes = fetch_direct(url, transport).await?;
    if !skip_cache {
        write_cache(cache_path, &bytes);
    }
    Ok(bytes)
}

/// Return cached bytes if the file was modified within `ttl`. Any I/O
/// error is treated as a cache miss.
fn read_cache(path: &Path, ttl: Duration) -> Option<Vec<u8>> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?;
    let age = SystemTime::now().duration_since(mtime).ok()?;
    if age > ttl {
        return None;
    }
    std::fs::read(path).ok()
}

/// Best-effort cache write. Failures are logged and swallowed — the
/// caller always gets the freshly fetched bytes regardless.
fn write_cache(path: &Path, data: &[u8]) {
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!(
                "[model-manager] aihub cache mkdir {}: {e}",
                parent.display()
            );
            return;
        }
    }
    if let Err(e) = std::fs::write(path, data) {
        eprintln!("[model-manager] aihub cache write {}: {e}", path.display());
    }
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
    use self::manifest::ManifestUrls;
    use super::*;

    fn entry(id: &str, domain: &str) -> ManifestModelEntry {
        ManifestModelEntry {
            id: id.to_string(),
            display_name: id.to_string(),
            domain: domain.to_string(),
            manifest_urls: ManifestUrls::default(),
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
        assert!(read_cache(&tmp.path().join("nope.json"), CACHE_TTL).is_none());
    }

    #[test]
    fn cache_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("x.json");
        write_cache(&p, b"hello");
        assert_eq!(read_cache(&p, CACHE_TTL).unwrap(), b"hello");
        assert!(read_cache(&p, Duration::ZERO).is_none());
    }
}
