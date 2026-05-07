use serde::{Deserialize, Serialize};
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
    #[serde(rename = "DeviceId")]
    pub device_id: String,
    #[serde(rename = "MinSDKVersion", default)]
    pub min_sdk_version: String,
    #[serde(rename = "ModelFile", default)]
    pub model_file: HashMap<String, ModelFileInfo>,
    #[serde(rename = "MMProjFile", default)]
    pub mmproj_file: ModelFileInfo,
    #[serde(rename = "TokenizerFile", default)]
    pub tokenizer_file: ModelFileInfo,
    #[serde(rename = "ExtraFiles", default)]
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
