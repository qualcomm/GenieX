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

/// Accept JSON `null` as the field's `Default` value.
///
/// The Go CLI writes manifests with `sonic.Marshal`, which emits nil
/// slices/maps as `null` and, in older versions, emitted unset optional
/// structs (e.g. `MMProjFile`) as `null` too. Plain `#[serde(default)]`
/// only kicks in when the key is absent, so without this helper those
/// manifests fail with `invalid type: null, expected ...` and the whole
/// model gets skipped as corrupted. `list_models()` in Python surfaced
/// the resulting gap vs the Go `geniex list`.
fn null_as_default<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Default + Deserialize<'de>,
    D: Deserializer<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// On-disk manifest written next to a cached model as `geniex.json`.
///
/// Historical `DeviceId` and `MinSDKVersion` keys are accepted (serde
/// silently drops unknown JSON fields on deserialize) but no longer
/// serialised — qairt / llama_cpp plugins don't read them and AI Hub
/// hub already tracks the chipset out-of-band. `QairtVersion` (below) is
/// the current, purposeful version stamp, sourced from AI Hub asset
/// metadata rather than a placeholder.
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
    #[serde(
        rename = "Precision",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub precision: String,
    /// QAIRT version the model's assets were exported against, captured at
    /// pull time from AI Hub's `release-assets.json` (`tool_versions.qairt`).
    /// Lets a later run detect artifacts left stale by a GenieX update whose
    /// bundled QAIRT no longer matches. Empty for non-AI-Hub / llama.cpp models.
    #[serde(
        rename = "QairtVersion",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub qairt_version: String,
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
    pub fn total_size(&self) -> i64 {
        let mut total = 0i64;
        for f in self.model_file.values() {
            if f.downloaded {
                total += f.size;
            }
        }
        if self.mmproj_file.downloaded {
            total += self.mmproj_file.size;
        }
        if self.tokenizer_file.downloaded {
            total += self.tokenizer_file.size;
        }
        for f in &self.extra_files {
            if f.downloaded {
                total += f.size;
            }
        }
        total
    }
}

#[derive(Debug, Clone)]
pub struct DownloadInfo {
    pub total_downloaded: i64,
    pub total_size: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ModelManifest {
        ModelManifest {
            name: "org/repo".into(),
            model_name: "repo".into(),
            model_type: ModelType::Llm,
            plugin_id: "qairt".into(),
            precision: "W4A16".into(),
            qairt_version: "2.45.0.260326154327".into(),
            model_file: HashMap::new(),
            mmproj_file: ModelFileInfo::default(),
            tokenizer_file: ModelFileInfo::default(),
            extra_files: Vec::new(),
        }
    }

    #[test]
    fn serializes_qairt_version_key() {
        let json = serde_json::to_string(&sample()).unwrap();
        assert!(
            json.contains(r#""QairtVersion":"2.45.0.260326154327""#),
            "missing QairtVersion key: {json}"
        );
    }

    #[test]
    fn empty_qairt_version_is_omitted() {
        // Non-AI-Hub models leave the stamp empty; the key must not appear.
        let mut m = sample();
        m.qairt_version = String::new();
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains("QairtVersion"), "unexpected key: {json}");
    }

    #[test]
    fn legacy_manifest_without_qairt_version_deserializes() {
        // geniex.json written before this field must still load.
        let legacy =
            r#"{"Name":"org/repo","ModelName":"repo","ModelType":"llm","PluginId":"qairt"}"#;
        let m: ModelManifest = serde_json::from_str(legacy).unwrap();
        assert!(m.qairt_version.is_empty());
    }
}
