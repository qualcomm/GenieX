// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Local filesystem [`ModelSource`].
//!
//! Given a `source_dir` (or a local archive file), reads its layout and
//! produces a [`Plan`] whose [`BytesSource`]s point at on-disk bytes.
//! Three layouts are recognised by [`detect_local_kind`]:
//!
//! - HF GGUF directory
//! - AI Hub extracted directory (`metadata.json` + `.bin` shards)
//! - AI Hub local `.zip`

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use url::Url;

use crate::error::{Error, Result};
use crate::gguf;
use crate::manifest::{ModelFileInfo, ModelManifest, ModelType};
use crate::manifest_builder::{infer_manifest_from_names, untagged_gguf_names, ManifestHint};
use crate::transport::{HttpTransport, LocalFileTransport};

use super::ai_hub::remote_zip::{fetch_central_directory, Method};
use super::ai_hub::{
    classify_from_metadata_json, precision_from_metadata_json, prepare_flat_entries,
};
use super::{BytesSource, FileSpec, ModelSource, Plan};

const MANIFEST_FILE: &str = "geniex.json";
const CONFIG_FILE: &str = "config.json";
const AIHUB_METADATA_FILE: &str = "metadata.json";
const QAIRT_PLUGIN_ID: &str = "qairt";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalKind {
    HfGguf,
    AiHubExtracted,
    AiHubZip,
}

/// Inspect `path` and decide which loader to dispatch to. Returns
/// [`Error::Hub`] with an actionable message when nothing matches.
fn detect_local_kind(path: &Path) -> Result<LocalKind> {
    let meta = fs::metadata(path).map_err(|e| {
        Error::Hub(format!(
            "local path {} is not accessible: {e}",
            path.display()
        ))
    })?;

    if meta.is_file() {
        if has_extension(path, "zip") {
            return Ok(LocalKind::AiHubZip);
        }
        return Err(Error::Hub(format!(
            "local path {} is a file but not a .zip; expected an AI Hub archive or a directory",
            path.display()
        )));
    }
    if !meta.is_dir() {
        return Err(Error::Hub(format!(
            "local path {} is neither a regular file nor a directory",
            path.display()
        )));
    }

    let mut has_bin = false;
    let mut has_metadata = false;
    let mut has_gguf = false;
    let mut has_safetensors = false;
    for entry in fs::read_dir(path)?.flatten() {
        let Ok(ft) = entry.file_type() else { continue };
        if !ft.is_file() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_ascii_lowercase) else {
            continue;
        };
        if name == AIHUB_METADATA_FILE {
            has_metadata = true;
        } else if name.ends_with(".bin") {
            has_bin = true;
        } else if name.ends_with(".gguf") {
            has_gguf = true;
        } else if name.ends_with(".safetensors") {
            has_safetensors = true;
        }
    }

    if has_metadata && has_bin {
        return Ok(LocalKind::AiHubExtracted);
    }
    if has_gguf {
        return Ok(LocalKind::HfGguf);
    }
    if has_safetensors {
        return Err(Error::Hub(format!(
            "local path {} looks like a HuggingFace safetensors snapshot, \
             which is not supported as a local pull source yet",
            path.display()
        )));
    }
    Err(Error::Hub(format!(
        "local path {} did not match any known layout: \
         expected a directory with *.gguf (HF GGUF), \
         a directory with metadata.json + *.bin (AI Hub extracted), \
         or a .zip file (AI Hub archive)",
        path.display()
    )))
}

fn has_extension(path: &Path, ext: &str) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|e| e.eq_ignore_ascii_case(ext))
        .unwrap_or(false)
}

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
        match detect_local_kind(&self.source_dir)? {
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
            for name in untagged_gguf_names(&file_names) {
                if let Some(q) = gguf::quant_from_file(&self.source_dir.join(name)) {
                    infer_hint.header_quants.insert(name.clone(), q);
                }
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
        // Lex-first `.bin` mirrors the remote AiHub puller.
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let display = self.source_dir.display().to_string();
        let (mut model_file, extra_files) = super::split_entrypoint_and_extras(
            &entries,
            || format!("AI Hub directory {display} has no .bin shard"),
            |s| *s as i64,
        )?;

        let metadata_bytes = std::fs::read(self.source_dir.join(AIHUB_METADATA_FILE)).ok();
        let model_type = metadata_bytes
            .as_deref()
            .and_then(classify_from_metadata_json)
            .or_else(|| self.hint.model_type.clone())
            .unwrap_or(ModelType::Llm);

        // Key by real precision, matching the remote AI Hub puller (#1242).
        if let Some(precision_label) = metadata_bytes
            .as_deref()
            .and_then(precision_from_metadata_json)
        {
            if let Some(entry) = model_file.remove("N/A") {
                model_file.insert(precision_label, entry);
            }
        }

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

        // Read metadata.json out of the zip once, for both modality and precision.
        let metadata_bytes = read_metadata_json_bytes(&flat, &transport, &dummy).await;
        let model_type = metadata_bytes
            .as_deref()
            .and_then(classify_from_metadata_json)
            .or_else(|| self.hint.model_type.clone())
            .unwrap_or(ModelType::Llm);

        let zip_display = zip_path.display().to_string();
        let (mut model_file, extra_files) = super::split_entrypoint_and_extras(
            &flat,
            || format!("AI Hub archive {zip_display} has no .bin shard"),
            |e| e.uncompressed_size as i64,
        )?;

        // Key by real precision, matching the remote AI Hub puller (#1242).
        if let Some(precision_label) = metadata_bytes
            .as_deref()
            .and_then(precision_from_metadata_json)
        {
            if let Some(entry) = model_file.remove("N/A") {
                model_file.insert(precision_label, entry);
            }
        }

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

/// Slurp `metadata.json` out of a flat zip listing; `None` on any failure
/// (missing entry, read/decode error) — callers derive modality and precision from the bytes.
async fn read_metadata_json_bytes(
    flat: &[(String, super::ai_hub::remote_zip::ZipEntry)],
    transport: &Arc<dyn HttpTransport>,
    url: &Url,
) -> Option<Vec<u8>> {
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
    Some(bytes)
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
    use crate::test_support::{build_zip, ZipMethod};
    use std::fs;

    const NO_VISION: &[u8] = br#"{"genie":{"supports_vision":false}}"#;

    fn write_all(dir: &Path, files: &[(&str, &[u8])]) {
        for (name, body) in files {
            fs::write(dir.join(name), body).unwrap();
        }
    }

    async fn plan_dir(files: &[(&str, &[u8])], name: &str) -> Result<Plan> {
        let tmp = tempfile::tempdir().unwrap();
        write_all(tmp.path(), files);
        LocalFsSource::new(
            tmp.path().to_path_buf(),
            name.to_string(),
            ManifestHint::default(),
        )
        .plan()
        .await
    }

    /// Plan a local zip; returns the zip path so callers can check sources.
    async fn plan_zip(
        entries: &[(&str, &[u8], ZipMethod)],
    ) -> (Result<Plan>, PathBuf, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let zip_path = tmp.path().join("model.zip");
        fs::write(&zip_path, build_zip(entries)).unwrap();
        let src = LocalFsSource::new(
            zip_path.clone(),
            "qualcomm/foo".to_string(),
            ManifestHint::default(),
        );
        (src.plan().await, zip_path, tmp)
    }

    fn extra_names(plan: &Plan) -> Vec<&str> {
        plan.manifest
            .extra_files
            .iter()
            .map(|f| f.name.as_str())
            .collect()
    }

    #[tokio::test]
    async fn gguf_dir_prefers_shipped_manifest_over_inference() {
        let shipped = br#"{
          "Name":"Org/Repo","ModelName":"tiny","ModelType":"llm","PluginId":"llama_cpp",
          "ModelFile":{"Q4_K_M":{"Name":"model-Q4_K_M.gguf","Downloaded":true,"Size":4}}
        }"#;
        let plan = plan_dir(
            &[("model-Q4_K_M.gguf", b"fake"), ("geniex.json", shipped)],
            "Org/Repo",
        )
        .await
        .unwrap();
        assert_eq!(plan.manifest.model_name, "tiny");
        assert_eq!(plan.files.len(), 1);
        assert!(matches!(
            &plan.files[0].bytes,
            BytesSource::Local { path } if path.ends_with("model-Q4_K_M.gguf")
        ));
    }

    #[tokio::test]
    async fn gguf_dir_infers_manifest_when_missing() {
        let plan = plan_dir(&[("model-Q4_K_M.gguf", b"x")], "Org/Repo-GGUF")
            .await
            .unwrap();
        assert_eq!(plan.manifest.name, "Org/Repo-GGUF");
        assert!(plan.manifest.model_file.contains_key("Q4_K_M"));
    }

    #[tokio::test]
    async fn gguf_dir_config_json_beats_stray_mmproj_file() {
        let files: &[(&str, &[u8])] = &[
            ("model-Q4_K_M.gguf", b"fake-weights"),
            ("mmproj-x.gguf", b"stray"),
            ("config.json", br#"{"architectures":["LlamaForCausalLM"]}"#),
        ];
        let plan = plan_dir(files, "Org/LLM").await.unwrap();
        assert_eq!(plan.manifest.model_type, ModelType::Llm);
    }

    #[tokio::test]
    async fn aihub_extracted_uses_qairt_and_lex_first_bin() {
        let files: &[(&str, &[u8])] = &[
            ("metadata.json", NO_VISION),
            ("weights_part_1.bin", b"abc"),
            ("weights_part_2.bin", b"defgh"),
            ("tokenizer.json", b"{}"),
        ];
        let plan = plan_dir(files, "qualcomm/llama").await.unwrap();
        assert_eq!(plan.manifest.plugin_id, QAIRT_PLUGIN_ID);
        assert_eq!(plan.manifest.model_file["N/A"].name, "weights_part_1.bin");
        for want in ["weights_part_2.bin", "tokenizer.json", "metadata.json"] {
            assert!(extra_names(&plan).contains(&want), "missing {want}");
        }
        assert!(plan
            .files
            .iter()
            .all(|f| matches!(f.bytes, BytesSource::Local { .. })));
    }

    #[tokio::test]
    async fn aihub_metadata_without_vision_flag_falls_back_to_llm() {
        for meta in [&br#"{"name":"x"}"#[..], b"not json"] {
            let plan = plan_dir(
                &[("metadata.json", meta), ("model.bin", b"x")],
                "qualcomm/foo",
            )
            .await
            .unwrap();
            assert_eq!(
                plan.manifest.model_type,
                ModelType::Llm,
                "{:?}",
                String::from_utf8_lossy(meta)
            );
        }
    }

    #[tokio::test]
    async fn aihub_keys_model_file_by_metadata_precision() {
        let meta: &[u8] = br#"{"precision":"w4a16","genie":{"supports_vision":false}}"#;
        let dir = plan_dir(
            &[("metadata.json", meta), ("weights_part_1.bin", b"abc")],
            "local/phi",
        )
        .await
        .unwrap();
        let (zip, _, _tmp) = plan_zip(&[
            ("metadata.json", meta, ZipMethod::Stored),
            ("weights_part_1.bin", b"abc", ZipMethod::Stored),
        ])
        .await;
        for (layout, plan) in [("dir", dir), ("zip", zip.unwrap())] {
            assert!(!plan.manifest.model_file.contains_key("N/A"), "{layout}");
            assert_eq!(
                plan.manifest.model_file["w4a16"].name, "weights_part_1.bin",
                "{layout}"
            );
        }
    }

    #[tokio::test]
    async fn aihub_zip_stored_emits_local_range() {
        let (plan, zip_path, _tmp) = plan_zip(&[
            ("metadata.json", NO_VISION, ZipMethod::Stored),
            ("shard_a.bin", b"hello-shard-a", ZipMethod::Stored),
            ("shard_b.bin", b"hello-shard-b", ZipMethod::Stored),
        ])
        .await;
        let plan = plan.unwrap();
        assert_eq!(plan.manifest.plugin_id, QAIRT_PLUGIN_ID);
        assert_eq!(plan.manifest.model_type, ModelType::Llm);
        assert_eq!(plan.manifest.model_file["N/A"].name, "shard_a.bin");
        for f in &plan.files {
            assert!(
                matches!(&f.bytes, BytesSource::LocalRange { path, .. } if *path == zip_path),
                "{:?}",
                f.bytes
            );
        }
    }

    #[tokio::test]
    async fn aihub_zip_deflate_emits_local_deflate_and_reads_metadata() {
        let big = vec![b'A'; 8192];
        let (plan, zip_path, _tmp) = plan_zip(&[
            (
                "metadata.json",
                br#"{"genie":{"supports_vision":true}}"#,
                ZipMethod::Deflate,
            ),
            ("weights.bin", &big, ZipMethod::Deflate),
        ])
        .await;
        let plan = plan.unwrap();
        assert_eq!(plan.manifest.model_type, ModelType::Vlm);
        let weights = plan.files.iter().find(|f| f.name == "weights.bin").unwrap();
        assert!(
            matches!(&weights.bytes, BytesSource::LocalDeflate { path, .. } if *path == zip_path),
            "{:?}",
            weights.bytes
        );
    }

    #[tokio::test]
    async fn aihub_zip_no_bin_returns_helpful_error() {
        let (plan, _, _tmp) = plan_zip(&[
            ("metadata.json", b"{}", ZipMethod::Stored),
            ("README.md", b"hi", ZipMethod::Stored),
        ])
        .await;
        let err = plan.unwrap_err();
        assert!(err.to_string().contains(".bin"), "{err}");
    }

    #[test]
    fn detect_local_kind_covers_layouts() {
        let tmp = tempfile::tempdir().unwrap();
        let mkdir = |name: &str, files: &[(&str, &[u8])]| {
            let dir = tmp.path().join(name);
            fs::create_dir_all(&dir).unwrap();
            write_all(&dir, files);
            dir
        };
        write_all(
            tmp.path(),
            &[("model.zip", b"PK\x03\x04"), ("Model.ZIP", b"x")],
        );
        let cases = [
            (tmp.path().join("model.zip"), LocalKind::AiHubZip),
            (tmp.path().join("Model.ZIP"), LocalKind::AiHubZip),
            (
                mkdir(
                    "aihub",
                    &[
                        ("metadata.json", b"{}"),
                        ("weights_part_1.bin", b"x"),
                        ("extra.gguf", b"y"),
                    ],
                ),
                LocalKind::AiHubExtracted,
            ),
            (
                mkdir("gguf", &[("model-Q4_K_M.gguf", b"x")]),
                LocalKind::HfGguf,
            ),
        ];
        for (path, want) in cases {
            assert_eq!(
                detect_local_kind(&path).unwrap(),
                want,
                "{}",
                path.display()
            );
        }
    }

    #[test]
    fn detect_local_kind_error_messages_are_actionable() {
        let tmp = tempfile::tempdir().unwrap();
        let mkdir = |name: &str, files: &[(&str, &[u8])]| {
            let dir = tmp.path().join(name);
            fs::create_dir_all(&dir).unwrap();
            write_all(&dir, files);
            dir
        };
        write_all(tmp.path(), &[("model.tar.gz", b"x")]);
        let cases: [(PathBuf, &[&str]); 4] = [
            (
                PathBuf::from("/nonexistent/path/12345xyz"),
                &["not accessible"],
            ),
            (tmp.path().join("model.tar.gz"), &["not a .zip"]),
            (
                mkdir("st", &[("config.json", b"{}"), ("model.safetensors", b"x")]),
                &["safetensors"],
            ),
            (
                mkdir("unk", &[("readme.txt", b"hi")]),
                &["AI Hub extracted", "HF GGUF", ".zip"],
            ),
        ];
        for (path, wants) in cases {
            let msg = detect_local_kind(&path).unwrap_err().to_string();
            for want in wants {
                assert!(
                    msg.contains(want),
                    "{}: want {want:?} in {msg}",
                    path.display()
                );
            }
        }
    }
}
