// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Plan-then-fetch contract for a single model pull.
//!
//! A [`ModelSource`] fully resolves "what does this model consist of" —
//! the on-disk manifest and every byte-level recipe — *before* any
//! bulk download starts. Once [`ModelSource::plan`] returns, no further
//! "what's in the model" discovery is allowed; the [`crate::executor`]
//! just moves bytes from each [`BytesSource`] into `dest_dir/<name>`.
//!
//! Implementations live beside this file: [`hf`] (HuggingFace REST API
//! with siblings), [`localfs`] (on-disk directory walk), [`ai_hub`]
//! (Qualcomm AI Hub S3 protojson chain plus remote ZIP64 central-dir
//! parse), [`modelscope`] (ModelScope REST API), and [`dockerhub`]
//! (Docker Registry HTTP API V2, for models published under
//! `hub.docker.com/u/ai` and similar).

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

pub(crate) const QAIRT_PLUGIN_ID: &str = "qairt";

/// Last path component of `path`, treating both `/` and `\` as separators.
pub(crate) fn basename(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or("")
}

/// One [`FileSpec`] per file the manifest marks downloaded; `spec` maps a
/// file name to its size and byte source.
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

/// QAIRT plan over every file in `files`: the first `.bin` is the
/// entrypoint, keyed by `precision` (`"N/A"` when unknown); the rest are
/// extras.
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
    /// Resolve the full plan: final manifest + byte-level recipe for
    /// every file the caller will see on disk.
    ///
    /// All "pre-download discovery" happens here: HF
    /// `/api/models/{repo}` siblings, AI Hub manifest chain + remote
    /// zip central directory, LocalFS readdir, and any future hub's
    /// metadata APIs. After this returns, the [`crate::executor`] only
    /// does pure byte movement — no more HTTP "what files exist"
    /// lookups.
    async fn plan(&self) -> Result<Plan>;
}

/// Output of [`ModelSource::plan`]. The executor consumes `files`; the
/// caller (usually [`crate::pull::pull`]) publishes `manifest` after
/// every byte has landed.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Exactly what should land at `<dest_dir>/geniex.json` on success.
    /// Entry names inside the manifest are expected to match file
    /// basenames the executor produces.
    pub manifest: ModelManifest,
    /// Byte-level recipe per file. Order is meaningful only for
    /// progress display — the executor may download them in parallel.
    pub files: Vec<FileSpec>,
}

/// How a single file should be materialised on disk.
#[derive(Debug, Clone)]
pub struct FileSpec {
    /// Relative filename under the model's dest_dir. Basename only —
    /// the AI Hub path flat-extracts, HF already hands us flat names,
    /// and LocalFS is assumed to be flat at its source root.
    pub name: String,
    /// Final on-disk size after any decoding (so HttpDeflate carries
    /// the uncompressed size, not `compressed_len`).
    pub size: u64,
    pub bytes: BytesSource,
}

/// Byte source for a [`FileSpec`].
///
/// Variants cover HF, LocalFS, ModelScope, and AI Hub (remote and local
/// archives). A future Volces hub should be expressible with `Http` +
/// manifest-side overrides; if not, extend this enum.
#[derive(Debug, Clone)]
pub enum BytesSource {
    /// Full HTTP GET, size known (or discoverable via HEAD). Chunked
    /// parallel download + chunk-level resume via the `.progress`
    /// bitmap. HF files land here.
    Http { url: Url, auth: Option<String> },
    /// Byte range inside an HTTP object, no content decoding. STORED
    /// zip entries (method=0). Preserves chunk-level resume by adding
    /// `offset` to every range request.
    HttpRange {
        url: Url,
        auth: Option<String>,
        offset: u64,
        len: u64,
    },
    /// Byte range inside an HTTP object, DEFLATE-decoded inline.
    /// AI Hub `.bin` shards (method=8). Single-range fetch piped into
    /// a streaming `flate2::DeflateDecoder`. Resume is entry-granular
    /// (all or nothing) because DEFLATE isn't seekable — accepted
    /// tradeoff vs today's "download whole 4 GB zip" behaviour.
    HttpDeflate {
        url: Url,
        auth: Option<String>,
        offset: u64,
        compressed_len: u64,
    },
    /// Local file copy. `LocalFsSource` uses this when the source
    /// directory is already an unpacked tree; tests sometimes do too
    /// for offline fixtures.
    Local { path: PathBuf },
    /// Byte range inside a local file, no decoding. STORED zip entries
    /// inside an AI Hub archive that the user is pulling from disk.
    LocalRange {
        path: PathBuf,
        offset: u64,
        len: u64,
    },
    /// Byte range inside a local file, DEFLATE-decoded inline. DEFLATE
    /// zip entries inside an AI Hub archive on disk; counterpart to
    /// `HttpDeflate` for the local-zip path.
    LocalDeflate {
        path: PathBuf,
        offset: u64,
        compressed_len: u64,
    },
}
