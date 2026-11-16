// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Minimal serde types for the AI Hub protojson payloads, which use proto
//! (snake_case) field names.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ReleaseManifest {
    #[serde(default)]
    pub platform_url: String,
    #[serde(default)]
    pub models: Vec<ManifestModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ManifestModelEntry {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub domain: String,
    #[serde(default)]
    pub manifest_urls: ManifestUrls,
    #[serde(default)]
    pub supported_runtimes: Vec<String>,
    #[serde(default)]
    pub supported_chipsets: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ManifestUrls {
    #[serde(default)]
    pub release_assets: String,
    #[serde(default)]
    pub info: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct InfoJson {
    #[serde(default)]
    pub domain: String,
    #[serde(default)]
    pub headline: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelReleaseAssets {
    #[serde(default)]
    pub model_id: String,
    #[serde(default)]
    pub assets: Vec<AssetDetails>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AssetDetails {
    /// `null` for non-Genie assets (ONNX, TFLite, QNN_DLC).
    #[serde(default)]
    pub chipset: Option<String>,
    #[serde(default)]
    pub runtime: String,
    #[serde(default)]
    pub precision: String,
    #[serde(default)]
    pub download_url: String,
    #[serde(default)]
    pub uncompressed_size: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PlatformInfo {
    #[serde(default)]
    pub chipsets: Vec<ChipsetInfo>,
    #[serde(default)]
    pub devices: Vec<DeviceInfo>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DeviceInfo {
    #[serde(default)]
    pub chipset: String,
    #[serde(default)]
    pub os: DeviceOs,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DeviceOs {
    #[serde(default)]
    pub ostype: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChipsetInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub reference_device: String,
    #[serde(default)]
    pub aliases: Vec<String>,
}

impl ChipsetInfo {
    pub fn display_name(&self) -> &str {
        if self.reference_device.is_empty() {
            &self.name
        } else {
            &self.reference_device
        }
    }
}
