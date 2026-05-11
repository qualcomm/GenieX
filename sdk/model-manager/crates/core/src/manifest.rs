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
/// hub already tracks the chipset out-of-band.
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
