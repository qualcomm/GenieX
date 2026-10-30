// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelType {
    #[serde(rename = "llm")]
    Llm,
    #[serde(rename = "vlm")]
    Vlm,
}

// JSON field names match Go's sonic.Marshal output (PascalCase struct field names).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelFileInfo {
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Downloaded")]
    pub downloaded: bool,
    #[serde(rename = "Size")]
    pub size: i64,
}

/// Go's `sonic.Marshal` writes nil slices/maps (and old optional structs) as
/// `null`; `#[serde(default)]` only covers absent keys.
fn null_as_default<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Default + Deserialize<'de>,
    D: Deserializer<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// `geniex.json`. Legacy `DeviceId` / `MinSDKVersion` keys are ignored on read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "ModelName")]
    pub model_name: String,
    #[serde(rename = "ModelType")]
    pub model_type: ModelType,
    #[serde(rename = "PluginId")]
    pub plugin_id: String,
    /// Read-only compat for old QAIRT manifests; new ones key `ModelFile` by precision.
    #[serde(
        rename = "Precision",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub precision: String,
    #[serde(rename = "ModelFile", default, deserialize_with = "null_as_default")]
    pub model_file: HashMap<String, ModelFileInfo>,
    #[serde(rename = "MMProjFile", default, deserialize_with = "null_as_default")]
    pub mmproj_file: ModelFileInfo,
    #[serde(
        rename = "TokenizerFile",
        default,
        deserialize_with = "null_as_default"
    )]
    pub tokenizer_file: ModelFileInfo,
    #[serde(rename = "ExtraFiles", default, deserialize_with = "null_as_default")]
    pub extra_files: Vec<ModelFileInfo>,
}

impl ModelManifest {
    pub fn downloaded_files(&self) -> impl Iterator<Item = &ModelFileInfo> {
        self.model_file
            .values()
            .chain([&self.mmproj_file, &self.tokenizer_file])
            .chain(&self.extra_files)
            .filter(|f| f.downloaded)
    }

    pub fn total_size(&self) -> i64 {
        self.downloaded_files().map(|f| f.size).sum()
    }

    /// Fold in what an earlier pull of the same model left on disk.
    pub(crate) fn merge_existing(&mut self, existing: ModelManifest) {
        for (quant, info) in existing.model_file {
            self.model_file.entry(quant).or_insert(info);
        }
        let known: std::collections::HashSet<String> =
            self.extra_files.iter().map(|f| f.name.clone()).collect();
        self.extra_files.extend(
            existing
                .extra_files
                .into_iter()
                .filter(|f| !known.contains(&f.name)),
        );
        if self.mmproj_file.name.is_empty() {
            self.mmproj_file = existing.mmproj_file;
        }
        if self.tokenizer_file.name.is_empty() {
            self.tokenizer_file = existing.tokenizer_file;
        }
    }
}
