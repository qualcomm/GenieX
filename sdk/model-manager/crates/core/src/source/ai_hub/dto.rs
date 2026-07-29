// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Minimal serde types for the Qualcomm AI Hub protojson payloads.
//!
//! The Go CLI parses these via `protojson.Unmarshal` of the full proto
//! schema under `cli/internal/qaihm/`. We only need a handful of fields
//! to pick and download the right asset, so we model them directly as
//! serde structs — avoiding a `prost` + `protoc` dependency in the Rust
//! build.
//!
//! Field naming: the live public bucket emits protojson with the
//! `emit_use_proto_names=true` option, so names are snake_case
//! (`display_name`, `manifest_urls`, `release_assets`). We rely on
//! `#[serde(rename_all = "snake_case")]` rather than per-field renames.

use serde::Deserialize;

/// Top-level `manifest.json`: catalogue of models with pointers to their
/// per-model `release-assets.json`.
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
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ManifestUrls {
    #[serde(default)]
    pub release_assets: String,
    /// Per-model `info.json` pointer. Live manifests populate it for
    /// every entry; kept optional so older cached manifests still parse.
    #[serde(default)]
    pub info: String,
}

/// Per-model `info.json`: verbose metadata used to distinguish VLM from
/// text-only LLM when `domain` alone is insufficient. Manifest-level
/// `domain` for Qwen2.5-VL, Llama-v3, and text-only LLMs is all
/// `MODEL_DOMAIN_GENERATIVE_AI`; the free-text `description` /
/// `headline` fields are what actually differ.
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

/// Per-model `release-assets.json`: one entry per (chipset, runtime,
/// precision) triple the model was published for.
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
    /// Non-Genie assets (ONNX, TFLite, QNN_DLC) ship with `chipset: null`.
    /// Our selector only looks at Genie entries so it's safe to default
    /// to empty, but keep the rename honest.
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

/// `platform.json`: chipset catalogue with aliases used to canonicalize
/// the user-supplied chipset string.
#[derive(Debug, Clone, Deserialize)]
pub struct PlatformInfo {
    #[serde(default)]
    pub chipsets: Vec<ChipsetInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChipsetInfo {
    #[serde(default)]
    pub name: String,
    /// Reference device AI Hub / Workbench shows for this chipset, e.g.
    /// "Snapdragon X Elite CRD". Empty when the bucket omits it.
    #[serde(default)]
    pub reference_device: String,
    #[serde(default)]
    pub aliases: Vec<String>,
}
