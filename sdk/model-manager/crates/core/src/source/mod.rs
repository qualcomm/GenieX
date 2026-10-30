// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Plan-then-fetch contract: a [`ModelSource`] resolves the manifest and
//! every byte recipe before any download; the [`crate::executor`] then only
//! moves bytes.

pub mod ai_hub;
pub mod dockerhub;
pub mod hf;
pub mod localfs;
pub mod modelscope;

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use url::Url;

use crate::error::{Error, Result};
use crate::manifest::{ModelFileInfo, ModelManifest, ModelType};
use crate::manifest_builder::{infer_manifest_from_names, untagged_gguf_names, ManifestHint};

pub(crate) const QAIRT_PLUGIN_ID: &str = "qairt";
pub(crate) const MANIFEST_FILE: &str = "geniex.json";
const CONFIG_FILE: &str = "config.json";

#[async_trait]
pub(crate) trait RepoFiles: Sync {
    async fn read(&self, name: &str) -> Result<Vec<u8>>;
    async fn header_quant(&self, name: &str, size: u64) -> Option<String>;
}

/// The repo's own `geniex.json` when it loads, else one inferred from `listing`.
pub(crate) async fn resolve_manifest(
    repo: &impl RepoFiles,
    name: &str,
    listing: &[(String, i64)],
    hint: &ManifestHint,
) -> Result<ModelManifest> {
    let has = |file: &str| listing.iter().any(|(n, _)| n == file);
    if has(MANIFEST_FILE) {
        let shipped = repo
            .read(MANIFEST_FILE)
            .await
            .and_then(|b| serde_json::from_slice::<ModelManifest>(&b).map_err(Error::Json));
        match shipped {
            Ok(mut m) => {
                m.name = name.to_string();
                return Ok(m);
            }
            Err(e) => crate::logging::warn(&format!(
                "{name}: shipped {MANIFEST_FILE} unusable ({e}); inferring"
            )),
        }
    }

    let files: Vec<&(String, i64)> = listing.iter().filter(|(n, _)| n != MANIFEST_FILE).collect();
    let names: Vec<String> = files.iter().map(|(n, _)| n.clone()).collect();
    let sizes: HashMap<String, i64> = files.into_iter().cloned().collect();
    let mut hint = hint.clone();
    if has(CONFIG_FILE) {
        hint.config_json_bytes = repo.read(CONFIG_FILE).await.ok();
    }
    for file in untagged_gguf_names(&names) {
        let size = sizes.get(file).copied().unwrap_or(0).max(0) as u64;
        if let Some(q) = repo.header_quant(file, size).await {
            hint.header_quants.insert(file.clone(), q);
        }
    }
    infer_manifest_from_names(name, &names, &sizes, hint)
}

pub(crate) fn basename(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or("")
}

pub(crate) fn files_for_manifest(
    manifest: &ModelManifest,
    mut spec: impl FnMut(&str) -> Result<(u64, BytesSource)>,
) -> Result<Vec<FileSpec>> {
    manifest
        .downloaded_files()
        .filter(|f| !f.name.is_empty())
        .map(|f| {
            let (size, bytes) = spec(&f.name)?;
            Ok(FileSpec {
                name: f.name.clone(),
                size,
                bytes,
            })
        })
        .collect()
}

/// The first `.bin` is the entrypoint, keyed by `precision` or `"N/A"`.
pub(crate) fn qairt_plan(
    name: &str,
    model_name: &str,
    model_type: ModelType,
    precision: &str,
    files: Vec<FileSpec>,
    no_bin: impl FnOnce() -> String,
) -> Result<Plan> {
    let entry = files
        .iter()
        .position(|f| f.name.to_ascii_lowercase().ends_with(".bin"))
        .ok_or_else(|| Error::Hub(no_bin()))?;
    let info = |f: &FileSpec| ModelFileInfo {
        name: f.name.clone(),
        downloaded: true,
        size: f.size as i64,
    };
    let key = if precision.is_empty() {
        "N/A"
    } else {
        precision
    };
    let manifest = ModelManifest {
        name: name.to_string(),
        model_name: model_name.to_string(),
        model_type,
        plugin_id: QAIRT_PLUGIN_ID.to_string(),
        precision: String::new(),
        model_file: HashMap::from([(key.to_string(), info(&files[entry]))]),
        mmproj_file: ModelFileInfo::default(),
        tokenizer_file: ModelFileInfo::default(),
        extra_files: files
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != entry)
            .map(|(_, f)| info(f))
            .collect(),
    };
    Ok(Plan { manifest, files })
}

#[async_trait]
pub trait ModelSource: Send + Sync {
    async fn plan(&self) -> Result<Plan>;
}

#[derive(Debug, Clone)]
pub struct Plan {
    /// Entry names must match the basenames in `files`.
    pub manifest: ModelManifest,
    pub files: Vec<FileSpec>,
}

#[derive(Debug, Clone)]
pub struct FileSpec {
    /// Basename only; every source lays files out flat.
    pub name: String,
    /// Size after decoding, not `compressed_len`.
    pub size: u64,
    pub bytes: BytesSource,
}

#[derive(Debug, Clone)]
pub enum BytesSource {
    Http {
        url: Url,
        auth: Option<String>,
    },
    /// STORED zip entry (method=0) inside an HTTP object.
    HttpRange {
        url: Url,
        auth: Option<String>,
        offset: u64,
        len: u64,
    },
    /// DEFLATE zip entry (method=8) inside an HTTP object.
    HttpDeflate {
        url: Url,
        auth: Option<String>,
        offset: u64,
        compressed_len: u64,
    },
    Local {
        path: PathBuf,
    },
    /// STORED zip entry inside a local file.
    LocalRange {
        path: PathBuf,
        offset: u64,
        len: u64,
    },
    /// DEFLATE zip entry inside a local file.
    LocalDeflate {
        path: PathBuf,
        offset: u64,
        compressed_len: u64,
    },
}
