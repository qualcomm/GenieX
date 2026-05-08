//! Local filesystem [`ModelSource`].
//!
//! Given a `source_dir`, reads its `geniex.json` if present or
//! synthesises one by scanning the directory. Emits one `Local`
//! [`BytesSource`] per file the manifest actually uses.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;

use crate::error::Result;
use crate::manifest::ModelManifest;
use crate::manifest_builder::{infer_manifest_from_names, ManifestHint};

use super::{BytesSource, FileSpec, ModelSource, Plan};

const MANIFEST_FILE: &str = "geniex.json";

pub struct LocalFsSource {
    source_dir: PathBuf,
    model_name: String,
    hint: ManifestHint,
}

impl LocalFsSource {
    pub fn new(source_dir: PathBuf, model_name: String, hint: ManifestHint) -> Self {
        Self {
            source_dir,
            model_name,
            hint,
        }
    }
}

#[async_trait]
impl ModelSource for LocalFsSource {
    async fn plan(&self) -> Result<Plan> {
        let mut file_names: Vec<String> = Vec::new();
        let mut sizes: HashMap<String, i64> = HashMap::new();
        for entry in std::fs::read_dir(&self.source_dir)?.flatten() {
            let ft = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if !ft.is_file() {
                continue;
            }
            if let Some(name) = entry.file_name().to_str().map(str::to_string) {
                if name == MANIFEST_FILE {
                    continue;
                }
                let size = std::fs::metadata(entry.path())
                    .map(|m| m.len() as i64)
                    .unwrap_or(-1);
                sizes.insert(name.clone(), size);
                file_names.push(name);
            }
        }

        let manifest_path = self.source_dir.join(MANIFEST_FILE);
        let mut manifest: ModelManifest = if manifest_path.exists() {
            let data = std::fs::read_to_string(&manifest_path)?;
            serde_json::from_str(&data)?
        } else {
            infer_manifest_from_names(&self.model_name, &file_names, &sizes, self.hint.clone())?
        };
        manifest.name = self.model_name.clone();

        let mut files: Vec<FileSpec> = Vec::new();
        let mut push = |name: &str| {
            if name.is_empty() {
                return;
            }
            let path = self.source_dir.join(name);
            let size = sizes.get(name).copied().unwrap_or(-1).max(0) as u64;
            files.push(FileSpec {
                name: name.to_string(),
                size,
                bytes: BytesSource::Local { path },
            });
        };
        for f in manifest.model_file.values() {
            if f.downloaded {
                push(&f.name);
            }
        }
        if manifest.mmproj_file.downloaded {
            push(&manifest.mmproj_file.name);
        }
        if manifest.tokenizer_file.downloaded {
            push(&manifest.tokenizer_file.name);
        }
        for f in &manifest.extra_files {
            if f.downloaded {
                push(&f.name);
            }
        }

        Ok(Plan { manifest, files })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[tokio::test]
    async fn prefers_shipped_manifest_over_inference() {
        let tmp = tempfile::tempdir().unwrap();
        let src_dir = tmp.path().to_path_buf();
        fs::write(src_dir.join("model-Q4_K_M.gguf"), b"fake").unwrap();
        fs::write(
            src_dir.join("geniex.json"),
            r#"{
              "Name":"Org/Repo",
              "ModelName":"tiny",
              "ModelType":"llm",
              "PluginId":"llama_cpp",
              "ModelFile":{"Q4_K_M":{"Name":"model-Q4_K_M.gguf","Downloaded":true,"Size":4}},
              "MMProjFile":{"Name":"","Downloaded":false,"Size":0},
              "TokenizerFile":{"Name":"","Downloaded":false,"Size":0},
              "ExtraFiles":[]
            }"#,
        )
        .unwrap();
        let src = LocalFsSource::new(src_dir, "Org/Repo".to_string(), ManifestHint::default());
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.model_name, "tiny");
        assert_eq!(plan.files.len(), 1);
        match &plan.files[0].bytes {
            BytesSource::Local { path } => assert!(path.ends_with("model-Q4_K_M.gguf")),
            _ => panic!("LocalFs should emit BytesSource::Local"),
        }
    }

    #[tokio::test]
    async fn infers_manifest_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let src_dir = tmp.path().to_path_buf();
        fs::write(src_dir.join("model-Q4_K_M.gguf"), b"x").unwrap();
        let src = LocalFsSource::new(
            src_dir,
            "Org/Repo-GGUF".to_string(),
            ManifestHint::default(),
        );
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.name, "Org/Repo-GGUF");
        assert!(plan.manifest.model_file.contains_key("Q4_K_M"));
    }
}
