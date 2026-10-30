// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! What AI Hub offers: supported chipsets and runnable models.

use std::sync::Arc;

use crate::error::Result;
use crate::host;
use crate::manifest::ModelType;
use crate::transport::{HttpTransport, ReqwestTransport};

use super::dto::{ChipsetInfo, PlatformInfo, ReleaseManifest};
use super::selector::{self, is_qairt_runtime};
use super::{index, AiHubConfig};

/// `None` means "don't filter".
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

/// A chipset with no device entry is kept, so a payload lacking `devices` stays unfiltered.
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

pub async fn list_supported_chipsets(cfg: &AiHubConfig) -> Result<Vec<ChipsetInfo>> {
    let transport: Arc<dyn HttpTransport> = Arc::new(ReqwestTransport::new()?);
    let plat = index::platform(cfg, &transport).await?;
    let mut chipsets = match host_ostypes() {
        Some(allowed) => filter_chipsets_for_host(&plat, allowed),
        None => plat.chipsets,
    };
    chipsets.sort_by_key(|c| c.display_name().to_ascii_lowercase());
    Ok(chipsets)
}

#[derive(Debug, Clone)]
pub struct HubModel {
    pub display_name: String,
    pub id: String,
    pub model_type: ModelType,
    pub supported_chipsets: Vec<String>,
}

/// `chipset` may be any alias or display name from `platform.json`; it is not auto-detected.
pub async fn list_hub_models(cfg: &AiHubConfig, chipset: Option<&str>) -> Result<Vec<HubModel>> {
    let transport: Arc<dyn HttpTransport> = Arc::new(ReqwestTransport::new()?);

    // `supported_chipsets` holds canonical ids.
    let canonical = match chipset {
        Some(chip) => {
            let plat = index::platform(cfg, &transport).await?;
            Some(selector::resolve_chipset(&plat, chip)?)
        }
        None => None,
    };
    let manifest = index::release_manifest(cfg, &transport).await?;
    Ok(select_hub_models(manifest, canonical.as_deref()))
}

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

/// Reference device name (e.g. `"Snapdragon X Elite CRD"`), else the raw detected id.
pub async fn detect_host_chipset_reference(cfg: &AiHubConfig) -> Option<String> {
    let raw = host::detect_chipset()?;
    let transport: Arc<dyn HttpTransport> = match ReqwestTransport::new() {
        Ok(t) => Arc::new(t),
        Err(_) => return Some(raw),
    };
    match index::platform(cfg, &transport).await {
        Ok(plat) => Some(selector::resolve_chipset_display(&plat, &raw).unwrap_or(raw)),
        Err(_) => Some(raw),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::ai_hub::dto::{self, ManifestModelEntry, ManifestUrls};

    const QAIRT: &str = "RUNTIME_GENIEX_QAIRT";

    fn hub_entry(
        id: &str,
        runtimes: &[&str],
        chipsets: &[&str],
        tags: &[&str],
    ) -> ManifestModelEntry {
        let strings = |v: &[&str]| v.iter().map(|s| s.to_string()).collect();
        ManifestModelEntry {
            id: id.to_string(),
            display_name: id.to_string(),
            domain: "MODEL_DOMAIN_GENERATIVE_AI".to_string(),
            manifest_urls: ManifestUrls::default(),
            supported_runtimes: strings(runtimes),
            supported_chipsets: strings(chipsets),
            tags: strings(tags),
        }
    }

    fn select(models: Vec<ManifestModelEntry>, device: Option<&str>) -> Vec<(String, ModelType)> {
        let manifest = ReleaseManifest {
            platform_url: String::new(),
            models,
        };
        select_hub_models(manifest, device)
            .into_iter()
            .map(|m| (m.id, m.model_type))
            .collect()
    }

    fn ids(selected: &[(String, ModelType)]) -> Vec<&str> {
        selected.iter().map(|(id, _)| id.as_str()).collect()
    }

    #[test]
    fn select_keeps_qairt_and_legacy_genie_models() {
        // Phi-3.5-Mini-Instruct (v0.60.0) declares only the legacy RUNTIME_GENIE.
        let out = select(
            vec![
                hub_entry("qairt_model", &[QAIRT], &["chipA"], &[]),
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
                hub_entry(
                    "phi_3_5_mini_instruct",
                    &["RUNTIME_QNN_CONTEXT_BINARY", "RUNTIME_GENIE"],
                    &["x"],
                    &[],
                ),
            ],
            None,
        );
        assert_eq!(ids(&out), ["phi_3_5_mini_instruct", "qairt_model"]);
    }

    #[test]
    fn select_filters_by_device_and_sorts_by_name() {
        let models = || {
            vec![
                hub_entry("b_model", &[QAIRT], &["chipA", "chipB"], &[]),
                hub_entry("a_model", &[QAIRT], &["chipB"], &[]),
            ]
        };
        assert_eq!(ids(&select(models(), Some("chipA"))), ["b_model"]);
        assert_eq!(ids(&select(models(), None)), ["a_model", "b_model"]);
    }

    #[test]
    fn select_classifies_vlm_from_tag() {
        let out = select(
            vec![
                hub_entry("vl_model", &[QAIRT], &["chipA"], &["MODEL_TAG_VLM"]),
                hub_entry("text_model", &[QAIRT], &["chipA"], &["MODEL_TAG_LLM"]),
            ],
            None,
        );
        assert_eq!(
            out,
            [
                ("text_model".to_string(), ModelType::Llm),
                ("vl_model".to_string(), ModelType::Vlm)
            ]
        );
    }

    fn platform(chipsets: &[&str], devices: &[(&str, &str)]) -> PlatformInfo {
        PlatformInfo {
            chipsets: chipsets
                .iter()
                .map(|name| ChipsetInfo {
                    name: name.to_string(),
                    reference_device: String::new(),
                    aliases: Vec::new(),
                })
                .collect(),
            devices: devices
                .iter()
                .map(|(chipset, ostype)| dto::DeviceInfo {
                    chipset: chipset.to_string(),
                    os: dto::DeviceOs {
                        ostype: ostype.to_string(),
                    },
                })
                .collect(),
        }
    }

    #[test]
    fn filter_chipsets_for_host_cases() {
        const WIN: &str = "OPERATING_SYSTEM_TYPE_WINDOWS";
        const LINUX: &str = "OPERATING_SYSTEM_TYPE_LINUX";
        const QC_LINUX: &str = "OPERATING_SYSTEM_TYPE_QC_LINUX";
        const ANDROID: &str = "OPERATING_SYSTEM_TYPE_ANDROID";
        let (x_elite, qcs9075, qcs8275, gen3) = (
            "qualcomm-snapdragon-x-elite",
            "qualcomm-qcs9075",
            "qualcomm-qcs8275",
            "qualcomm-snapdragon-8gen3",
        );
        let sample = platform(
            &[x_elite, qcs9075, qcs8275, gen3],
            &[
                (x_elite, WIN),
                (qcs9075, QC_LINUX),
                (qcs8275, LINUX),
                (gen3, ANDROID),
            ],
        );
        let no_devices = platform(&[qcs9075, gen3], &[]);
        let dual = platform(&["dual"], &[("dual", ANDROID), ("dual", LINUX)]);
        let cases: [(&PlatformInfo, &[&str], &[&str]); 5] = [
            (&sample, &[LINUX, QC_LINUX], &[qcs9075, qcs8275]),
            (&sample, &[WIN], &[x_elite]),
            (&sample, &[ANDROID], &[gen3]),
            (&no_devices, &[WIN], &[qcs9075, gen3]),
            (&dual, &[LINUX], &["dual"]),
        ];
        for (plat, host, want) in cases {
            let out = filter_chipsets_for_host(plat, host);
            let got: Vec<&str> = out.iter().map(|c| c.name.as_str()).collect();
            assert_eq!(got, want, "host {host:?}");
        }
    }
}
