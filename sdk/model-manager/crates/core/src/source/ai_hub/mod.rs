// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Qualcomm AI Hub [`ModelSource`]: plans per-entry Range reads from the
//! remote archive's central directory without downloading the payload.

mod catalog;
pub mod dto;
mod index;
pub mod selector;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use url::Url;

use crate::config::StoreConfig;
use crate::error::{parse_manifest, Error, Result};
use crate::host;
use crate::manifest::ModelType;
use crate::transport::{HttpTransport, ReqwestTransport};

use self::dto::{InfoJson, ManifestModelEntry, ModelReleaseAssets};
use self::selector::match_asset;
use super::zip::{fetch_central_directory, flatten, zip_files, Archive};
use super::{qairt_plan, ModelSource, Plan};

pub use self::catalog::{
    detect_host_chipset_reference, list_hub_models, list_supported_chipsets, HubModel,
};

/// info.json is immutable per release.
const INFO_TTL: Duration = Duration::from_secs(24 * 60 * 60);

#[derive(Debug, Clone)]
pub struct AiHubConfig {
    pub endpoint: String,
    /// Fallback for `latest/`, or the sole target when `GENIEX_AIHUBVERSION` is set.
    pub version: String,
    /// Empty → auto-detect via [`host::detect_chipset`].
    pub chipset: String,
    pub cache_dir: PathBuf,
    pub skip_cache: bool,
}

impl AiHubConfig {
    pub fn from_store(store: &crate::store::Store, chipset: String) -> Self {
        let version = StoreConfig::ai_hub_version_override()
            .unwrap_or_else(StoreConfig::ai_hub_version_fallback);
        Self::new(
            StoreConfig::ai_hub_base_url(),
            version,
            chipset,
            store.config().ai_hub_cache_dir(),
            false,
        )
    }

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
        Ok(Self::with_transport(
            display_name,
            model_name,
            cfg,
            transport,
        ))
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

    async fn fetch_info_json(&self, entry: &ManifestModelEntry) -> Option<InfoJson> {
        if entry.manifest_urls.info.is_empty() {
            return None;
        }
        let cache_path = self
            .cfg
            .cache_dir
            .join("info")
            .join(format!("{}.json", entry.id));
        let bytes = index::fetch_cached(
            &entry.manifest_urls.info,
            &cache_path,
            INFO_TTL,
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
        let release_manifest = index::release_manifest(&self.cfg, &self.transport).await?;
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

        // Uncached: the per-model URL rotates each release.
        let release_assets_bytes = index::fetch(release_assets_url, &self.transport).await?;
        let release_assets: ModelReleaseAssets =
            parse_manifest("release-assets.json", &release_assets_bytes)?;

        let platform = index::platform(&self.cfg, &self.transport).await?;

        let chipset: String = if self.cfg.chipset.is_empty() {
            host::detect_chipset().ok_or_else(|| {
                Error::Hub(
                    "chipset not provided and host auto-detect is not supported on this platform"
                        .to_string(),
                )
            })?
        } else {
            self.cfg.chipset.clone()
        };

        let asset = match_asset(&release_assets, &platform, &chipset)?;
        let download_url = Url::parse(&asset.download_url).map_err(|e| {
            Error::invalid_url(format!("asset download_url {:?}", asset.download_url), e)
        })?;

        let raw_entries = fetch_central_directory(&self.transport, &download_url).await?;
        let files = zip_files(flatten(&raw_entries)?, &Archive::Remote(download_url));
        let precision = asset
            .precision
            .strip_prefix("PRECISION_")
            .unwrap_or(&asset.precision);
        let info = self.fetch_info_json(entry).await;
        let model_name = if entry.id.is_empty() {
            &self.display_name
        } else {
            &entry.id
        };
        qairt_plan(
            &self.model_name,
            model_name,
            classify_ai_hub(info.as_ref(), entry),
            precision,
            files,
            || "no .bin shard in archive".to_string(),
        )
    }
}

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

/// `genie.supports_vision` and `precision` from an archive's `metadata.json`.
pub(crate) fn parse_archive_metadata(bytes: &[u8]) -> (Option<ModelType>, String) {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return (None, String::new());
    };
    let model_type = v["genie"]["supports_vision"].as_bool().map(|vision| {
        if vision {
            ModelType::Vlm
        } else {
            ModelType::Llm
        }
    });
    let precision = v["precision"].as_str().unwrap_or_default().to_string();
    (model_type, precision)
}

/// `GENERATIVE_AI` also covers VLMs such as Qwen2.5-VL, so info.json text breaks the tie.
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

    #[test]
    fn classify_ai_hub_cases() {
        let info = |description: &str, headline: &str| InfoJson {
            domain: "MODEL_DOMAIN_GENERATIVE_AI".to_string(),
            headline: headline.to_string(),
            description: description.to_string(),
            tags: Vec::new(),
        };
        let cases = [
            ("MODEL_DOMAIN_MULTIMODAL", None, ModelType::Vlm),
            // The live manifest reports GENERATIVE_AI for Qwen2.5-VL; the description rescues it.
            (
                "MODEL_DOMAIN_GENERATIVE_AI",
                Some(info("Qwen2.5-VL-7B-Instruct is a multimodal vision-language model that processes both images and text", "")),
                ModelType::Vlm,
            ),
            (
                "MODEL_DOMAIN_GENERATIVE_AI",
                Some(info("Llama 3 is a state-of-the-art large language model", "State-of-the-art large language model")),
                ModelType::Llm,
            ),
            ("MODEL_DOMAIN_GENERATIVE_AI", Some(info("", "Visual question answering on mobile")), ModelType::Vlm),
            ("MODEL_DOMAIN_GENERATIVE_AI", None, ModelType::Llm),
        ];
        for (domain, info, want) in cases {
            let e = ManifestModelEntry {
                id: "m".to_string(),
                display_name: "m".to_string(),
                domain: domain.to_string(),
                manifest_urls: ManifestUrls::default(),
                supported_runtimes: Vec::new(),
                supported_chipsets: Vec::new(),
                tags: Vec::new(),
            };
            assert_eq!(
                classify_ai_hub(info.as_ref(), &e),
                want,
                "{domain} {info:?}"
            );
        }
    }
}
