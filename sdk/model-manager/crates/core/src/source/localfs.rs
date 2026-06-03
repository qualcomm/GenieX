//! Local filesystem [`ModelSource`].
//!
//! Given a `source_dir` (or a local archive file), reads its layout and
//! produces a [`Plan`] whose [`BytesSource`]s point at on-disk bytes.
//! Three layouts are recognised by [`local_kind::detect`]:
//!
//! - HF GGUF directory — existing path: read shipped `geniex.json` if
//!   present, else infer via [`infer_manifest_from_names`]. Files emit
//!   as [`BytesSource::Local`].
//! - AI Hub extracted directory — `metadata.json` + `.bin` shards.
//!   Plugin id is forced to `qairt`, modality comes from
//!   [`classify_from_metadata_json`], lex-first `.bin` is the
//!   entrypoint. Mirrors what the remote AI Hub source produces.
//! - AI Hub local `.zip` — feed the existing
//!   [`fetch_central_directory`] parser through a [`LocalFileTransport`]
//!   adapter, then emit [`BytesSource::LocalRange`] for STORED entries
//!   and [`BytesSource::LocalDeflate`] for DEFLATE entries.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use url::Url;

use crate::error::{Error, Result};
use crate::manifest::{ModelFileInfo, ModelManifest, ModelType};
use crate::manifest_builder::{infer_manifest_from_names, ManifestHint};
use crate::transport::HttpTransport;

use super::ai_hub::local_transport::LocalFileTransport;
use super::ai_hub::remote_zip::{fetch_central_directory, Method};
use super::ai_hub::{classify_from_metadata_json, prepare_flat_entries};
use super::local_kind::{detect, LocalKind};
use super::{BytesSource, FileSpec, ModelSource, Plan};

const MANIFEST_FILE: &str = "geniex.json";
const CONFIG_FILE: &str = "config.json";
const AIHUB_METADATA_FILE: &str = "metadata.json";
const QAIRT_PLUGIN_ID: &str = "qairt";

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
        match detect(&self.source_dir)? {
            LocalKind::HfGguf => self.plan_hf_gguf(),
            LocalKind::AiHubExtracted => self.plan_ai_hub_extracted(),
            LocalKind::AiHubZip => self.plan_ai_hub_zip().await,
        }
    }
}

impl LocalFsSource {
    /// HF / GGUF directory — the original `LocalFsSource` behaviour.
    fn plan_hf_gguf(&self) -> Result<Plan> {
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
            let mut infer_hint = self.hint.clone();
            if file_names.iter().any(|n| n == CONFIG_FILE) {
                infer_hint.config_json_bytes =
                    std::fs::read(self.source_dir.join(CONFIG_FILE)).ok();
            }
            infer_manifest_from_names(&self.model_name, &file_names, &sizes, infer_hint)?
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

    /// Already-extracted AI Hub asset on disk. Layout matches what
    /// `aihub.ExtractFlat` produces: a flat directory containing one or
    /// more `.bin` shards, a `metadata.json`, and assorted siblings.
    fn plan_ai_hub_extracted(&self) -> Result<Plan> {
        let mut entries: Vec<(String, u64)> = Vec::new();
        for entry in std::fs::read_dir(&self.source_dir)?.flatten() {
            let ft = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if !ft.is_file() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if name == MANIFEST_FILE {
                continue;
            }
            let size = std::fs::metadata(entry.path())
                .map(|m| m.len())
                .unwrap_or(0);
            entries.push((name, size));
        }
        if entries.is_empty() {
            return Err(Error::Hub(format!(
                "AI Hub directory {} is empty",
                self.source_dir.display()
            )));
        }
        // Lex-first `.bin` mirrors the remote AiHub puller and the Go
        // CLI's ExtractFlat — a model populated by either path is then
        // interchangeable on disk.
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let entrypoint = entries
            .iter()
            .find(|(n, _)| n.to_ascii_lowercase().ends_with(".bin"))
            .map(|(n, _)| n.clone())
            .ok_or_else(|| {
                Error::Hub(format!(
                    "AI Hub directory {} has no .bin shard",
                    self.source_dir.display()
                ))
            })?;

        let total_size: u64 = entries.iter().map(|(_, s)| *s).sum();
        let mut model_file: HashMap<String, ModelFileInfo> = HashMap::new();
        model_file.insert(
            "N/A".to_string(),
            ModelFileInfo {
                name: entrypoint.clone(),
                downloaded: true,
                size: total_size as i64,
            },
        );
        let extra_files: Vec<ModelFileInfo> = entries
            .iter()
            .filter(|(n, _)| n != &entrypoint)
            .map(|(n, s)| ModelFileInfo {
                name: n.clone(),
                downloaded: true,
                size: *s as i64,
            })
            .collect();

        let model_type = std::fs::read(self.source_dir.join(AIHUB_METADATA_FILE))
            .ok()
            .as_deref()
            .and_then(classify_from_metadata_json)
            .or_else(|| self.hint.model_type.clone())
            .unwrap_or(ModelType::Llm);

        let manifest = ModelManifest {
            name: self.model_name.clone(),
            model_name: derive_model_name(&self.model_name),
            model_type,
            plugin_id: QAIRT_PLUGIN_ID.to_string(),
            precision: String::new(),
            model_file,
            mmproj_file: ModelFileInfo::default(),
            tokenizer_file: ModelFileInfo::default(),
            extra_files,
        };

        let files: Vec<FileSpec> = entries
            .into_iter()
            .map(|(name, size)| {
                let path = self.source_dir.join(&name);
                FileSpec {
                    name,
                    size,
                    bytes: BytesSource::Local { path },
                }
            })
            .collect();

        Ok(Plan { manifest, files })
    }

    /// AI Hub `.zip` archive on disk. Reuses the trait-based ZIP64
    /// parser from the remote source via a [`LocalFileTransport`].
    /// STORED entries emit [`BytesSource::LocalRange`]; DEFLATE entries
    /// emit [`BytesSource::LocalDeflate`].
    async fn plan_ai_hub_zip(&self) -> Result<Plan> {
        let zip_path = self.source_dir.clone();
        let transport: Arc<dyn HttpTransport> = Arc::new(LocalFileTransport::open(&zip_path)?);
        // The URL is unused by LocalFileTransport but the parser's
        // signature requires one; pick a stable dummy that round-trips
        // through `Url::parse`.
        let dummy = Url::parse("file:///local-archive").expect("valid dummy url");
        let raw = fetch_central_directory(&transport, &dummy).await?;
        let flat = prepare_flat_entries(&raw)?;
        if flat.is_empty() {
            return Err(Error::Hub(format!(
                "AI Hub archive {} contains no usable entries",
                zip_path.display()
            )));
        }

        // Modality: read metadata.json out of the zip without extracting.
        // Method::Stored → straight range read; Method::Deflate → range
        // read + flate2.
        let model_type = read_metadata_modality(&flat, &transport, &dummy)
            .await
            .or_else(|| self.hint.model_type.clone())
            .unwrap_or(ModelType::Llm);

        let entrypoint = flat
            .iter()
            .find(|(name, _)| name.to_ascii_lowercase().ends_with(".bin"))
            .map(|(name, _)| name.clone())
            .ok_or_else(|| {
                Error::Hub(format!(
                    "AI Hub archive {} has no .bin shard",
                    zip_path.display()
                ))
            })?;

        let total_uncompressed: u64 = flat.iter().map(|(_, e)| e.uncompressed_size).sum();

        let mut model_file: HashMap<String, ModelFileInfo> = HashMap::new();
        model_file.insert(
            "N/A".to_string(),
            ModelFileInfo {
                name: entrypoint.clone(),
                downloaded: true,
                size: total_uncompressed as i64,
            },
        );
        let extra_files: Vec<ModelFileInfo> = flat
            .iter()
            .filter(|(name, _)| name != &entrypoint)
            .map(|(name, e)| ModelFileInfo {
                name: name.clone(),
                downloaded: true,
                size: e.uncompressed_size as i64,
            })
            .collect();

        let manifest = ModelManifest {
            name: self.model_name.clone(),
            model_name: derive_model_name(&self.model_name),
            model_type,
            plugin_id: QAIRT_PLUGIN_ID.to_string(),
            precision: String::new(),
            model_file,
            mmproj_file: ModelFileInfo::default(),
            tokenizer_file: ModelFileInfo::default(),
            extra_files,
        };

        let files: Vec<FileSpec> = flat
            .into_iter()
            .map(|(name, e)| {
                let bytes = match e.method {
                    Method::Stored => BytesSource::LocalRange {
                        path: zip_path.clone(),
                        offset: e.payload_offset,
                        len: e.compressed_size,
                    },
                    Method::Deflate => BytesSource::LocalDeflate {
                        path: zip_path.clone(),
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

/// Slurp `metadata.json` out of a flat zip listing — STORED entries are
/// a direct range read; DEFLATE entries inflate the buffered slice in
/// memory. Returns `None` for any failure: missing entry, parse error,
/// absent `genie.supports_vision`. The caller falls back to LLM, the
/// same default as the Go CLI's `detectModelTypeFromDir`.
async fn read_metadata_modality(
    flat: &[(String, super::ai_hub::remote_zip::ZipEntry)],
    transport: &Arc<dyn HttpTransport>,
    url: &Url,
) -> Option<ModelType> {
    let (_, entry) = flat.iter().find(|(name, _)| name == AIHUB_METADATA_FILE)?;
    let mut compressed: Vec<u8> = Vec::with_capacity(entry.compressed_size as usize);
    transport
        .get_range(
            url,
            None,
            entry.payload_offset,
            entry.compressed_size,
            &mut compressed,
        )
        .await
        .ok()?;
    let bytes: Vec<u8> = match entry.method {
        Method::Stored => compressed,
        Method::Deflate => {
            use flate2::write::DeflateDecoder;
            use std::io::Write as _;
            let mut buf: Vec<u8> = Vec::with_capacity(entry.uncompressed_size as usize);
            let mut dec = DeflateDecoder::new(&mut buf);
            dec.write_all(&compressed).ok()?;
            dec.finish().ok()?;
            buf
        }
    };
    classify_from_metadata_json(&bytes)
}

/// Last path component of `name` (the `org/repo` string). Mirrors
/// what `infer_manifest_from_names` does so the on-disk `model_name`
/// stays consistent across local-vs-remote pulls of the same model.
fn derive_model_name(name: &str) -> String {
    name.rsplit('/').next().unwrap_or(name).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write as IoWrite;

    // ---------------- HF GGUF (existing behaviour) ----------------

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
    async fn config_json_beats_stray_mmproj_file() {
        let tmp = tempfile::tempdir().unwrap();
        let src_dir = tmp.path().to_path_buf();
        fs::write(src_dir.join("model-Q4_K_M.gguf"), b"fake-weights").unwrap();
        fs::write(src_dir.join("mmproj-x.gguf"), b"stray").unwrap();
        fs::write(
            src_dir.join("config.json"),
            r#"{"architectures":["LlamaForCausalLM"]}"#,
        )
        .unwrap();
        let src = LocalFsSource::new(src_dir, "Org/LLM".to_string(), ManifestHint::default());
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.model_type, ModelType::Llm);
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

    // ---------------- AI Hub extracted directory ----------------

    fn write_aihub_dir(dir: &std::path::Path, supports_vision: Option<bool>) {
        if let Some(v) = supports_vision {
            let body = format!(r#"{{"genie":{{"supports_vision":{v}}}}}"#);
            fs::write(dir.join("metadata.json"), body).unwrap();
        }
        fs::write(dir.join("weights_part_1.bin"), b"abc").unwrap();
        fs::write(dir.join("weights_part_2.bin"), b"defgh").unwrap();
        fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    }

    #[tokio::test]
    async fn aihub_extracted_uses_qairt_and_lex_first_bin() {
        let tmp = tempfile::tempdir().unwrap();
        write_aihub_dir(tmp.path(), Some(false));
        let src = LocalFsSource::new(
            tmp.path().to_path_buf(),
            "qualcomm/llama".to_string(),
            ManifestHint::default(),
        );
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.plugin_id, QAIRT_PLUGIN_ID);
        let entry = plan.manifest.model_file.get("N/A").expect("entrypoint");
        assert_eq!(entry.name, "weights_part_1.bin");
        let extras: Vec<&str> = plan
            .manifest
            .extra_files
            .iter()
            .map(|f| f.name.as_str())
            .collect();
        assert!(extras.contains(&"weights_part_2.bin"));
        assert!(extras.contains(&"tokenizer.json"));
        // metadata.json rides along as an extra — same shape the remote
        // AI Hub puller produces so a cache populated by either path is
        // byte-comparable on disk.
        assert!(extras.contains(&"metadata.json"));
        for f in &plan.files {
            match &f.bytes {
                BytesSource::Local { .. } => {}
                _ => panic!("expected Local, got {:?}", f.bytes),
            }
        }
    }

    #[tokio::test]
    async fn aihub_extracted_supports_vision_true_yields_vlm() {
        let tmp = tempfile::tempdir().unwrap();
        write_aihub_dir(tmp.path(), Some(true));
        let src = LocalFsSource::new(
            tmp.path().to_path_buf(),
            "qualcomm/qwen-vl".to_string(),
            ManifestHint::default(),
        );
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.model_type, ModelType::Vlm);
    }

    #[tokio::test]
    async fn aihub_extracted_metadata_field_absent_falls_back_to_llm() {
        let tmp = tempfile::tempdir().unwrap();
        // metadata.json present but without genie.supports_vision.
        fs::write(tmp.path().join("metadata.json"), r#"{"name":"x"}"#).unwrap();
        fs::write(tmp.path().join("model.bin"), b"x").unwrap();
        let src = LocalFsSource::new(
            tmp.path().to_path_buf(),
            "qualcomm/foo".to_string(),
            ManifestHint::default(),
        );
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.model_type, ModelType::Llm);
    }

    #[tokio::test]
    async fn aihub_extracted_unparseable_metadata_falls_back_to_llm() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("metadata.json"), b"not json").unwrap();
        fs::write(tmp.path().join("model.bin"), b"x").unwrap();
        let src = LocalFsSource::new(
            tmp.path().to_path_buf(),
            "qualcomm/foo".to_string(),
            ManifestHint::default(),
        );
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.model_type, ModelType::Llm);
    }

    // ---------------- AI Hub local zip ----------------

    fn build_zip(entries: &[(&str, &[u8])], compressed: bool) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zw = zip::ZipWriter::new(cursor);
            let method = if compressed {
                zip::CompressionMethod::Deflated
            } else {
                zip::CompressionMethod::Stored
            };
            let opts: zip::write::SimpleFileOptions =
                zip::write::SimpleFileOptions::default().compression_method(method);
            for (name, data) in entries {
                zw.start_file(*name, opts).unwrap();
                zw.write_all(data).unwrap();
            }
            zw.finish().unwrap();
        }
        buf
    }

    #[tokio::test]
    async fn aihub_zip_stored_emits_local_range() {
        let tmp = tempfile::tempdir().unwrap();
        let zip_path = tmp.path().join("model.zip");
        let body = build_zip(
            &[
                (
                    "metadata.json",
                    br#"{"genie":{"supports_vision":false}}"#.as_slice(),
                ),
                ("shard_a.bin", b"hello-shard-a"),
                ("shard_b.bin", b"hello-shard-b"),
            ],
            false,
        );
        fs::write(&zip_path, &body).unwrap();
        let src = LocalFsSource::new(
            zip_path.clone(),
            "qualcomm/foo".to_string(),
            ManifestHint::default(),
        );
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.plugin_id, QAIRT_PLUGIN_ID);
        assert_eq!(plan.manifest.model_type, ModelType::Llm);
        let entry = plan.manifest.model_file.get("N/A").expect("entrypoint");
        assert_eq!(entry.name, "shard_a.bin");
        for f in &plan.files {
            match &f.bytes {
                BytesSource::LocalRange { path, .. } => assert_eq!(path, &zip_path),
                BytesSource::Local { .. }
                | BytesSource::LocalDeflate { .. }
                | BytesSource::Http { .. }
                | BytesSource::HttpRange { .. }
                | BytesSource::HttpDeflate { .. } => {
                    panic!("expected LocalRange for STORED, got {:?}", f.bytes)
                }
            }
        }
    }

    #[tokio::test]
    async fn aihub_zip_deflate_emits_local_deflate_and_reads_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let zip_path = tmp.path().join("model.zip");
        // Bigger payload so DEFLATE actually compresses (small inputs
        // can round-trip as STORED depending on the encoder).
        let big = vec![b'A'; 8192];
        let body = build_zip(
            &[
                (
                    "metadata.json",
                    br#"{"genie":{"supports_vision":true}}"#.as_slice(),
                ),
                ("weights.bin", &big),
            ],
            true,
        );
        fs::write(&zip_path, &body).unwrap();
        let src = LocalFsSource::new(
            zip_path.clone(),
            "qualcomm/qwen-vl".to_string(),
            ManifestHint::default(),
        );
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.model_type, ModelType::Vlm);
        let weights = plan
            .files
            .iter()
            .find(|f| f.name == "weights.bin")
            .expect("weights present");
        match &weights.bytes {
            BytesSource::LocalDeflate { path, .. } => assert_eq!(path, &zip_path),
            other => panic!("expected LocalDeflate, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn aihub_zip_no_bin_returns_helpful_error() {
        let tmp = tempfile::tempdir().unwrap();
        let zip_path = tmp.path().join("model.zip");
        let body = build_zip(
            &[("metadata.json", b"{}".as_slice()), ("README.md", b"hi")],
            false,
        );
        fs::write(&zip_path, &body).unwrap();
        let src = LocalFsSource::new(
            zip_path,
            "qualcomm/foo".to_string(),
            ManifestHint::default(),
        );
        let err = src.plan().await.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains(".bin"), "msg: {msg}");
    }

    // ---------------- Detect-level errors ----------------

    #[tokio::test]
    async fn safetensors_only_dir_returns_unsupported() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), b"{}").unwrap();
        fs::write(tmp.path().join("model.safetensors"), b"x").unwrap();
        let src = LocalFsSource::new(
            tmp.path().to_path_buf(),
            "Org/Repo".to_string(),
            ManifestHint::default(),
        );
        let err = src.plan().await.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("safetensors"), "msg: {msg}");
    }
}
