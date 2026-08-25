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
//! parse), and [`dockerhub`] (Docker Registry HTTP API V2, for models
//! published under `hub.docker.com/u/ai` and similar).

pub mod ai_hub;
pub mod dockerhub;
pub mod hf;
pub mod localfs;

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use url::Url;

use crate::error::{Error, Result};
use crate::manifest::{ModelFileInfo, ModelManifest};

/// Last path component of `path`, treating both `/` and `\` as separators.
/// Empty path returns an empty string.
pub(crate) fn basename(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or("").to_string()
}

/// Split a flat list of `(name, T)` entries into a `model_file` map keyed
/// by `"N/A"` (AI Hub / QAIRT layout) and an `extra_files` vec. The
/// entrypoint is the lex-first entry whose name ends in `.bin`.
///
/// `size_of` extracts the on-disk size for each entry (kept generic so
/// remote `ZipEntry` and local `(name, u64)` tuples both work).
pub(crate) fn split_entrypoint_and_extras<T>(
    entries: &[(String, T)],
    missing_bin_err: impl FnOnce() -> String,
    size_of: impl Fn(&T) -> i64,
) -> Result<(HashMap<String, ModelFileInfo>, Vec<ModelFileInfo>)> {
    let entrypoint_idx = entries
        .iter()
        .position(|(name, _)| name.to_ascii_lowercase().ends_with(".bin"))
        .ok_or_else(|| Error::Hub(missing_bin_err()))?;
    let (entry_name, entry_val) = &entries[entrypoint_idx];
    let mut model_file = HashMap::new();
    model_file.insert(
        "N/A".to_string(),
        ModelFileInfo {
            name: entry_name.clone(),
            downloaded: true,
            size: size_of(entry_val),
        },
    );
    let extra_files: Vec<ModelFileInfo> = entries
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != entrypoint_idx)
        .map(|(_, (name, val))| ModelFileInfo {
            name: name.clone(),
            downloaded: true,
            size: size_of(val),
        })
        .collect();
    Ok((model_file, extra_files))
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
/// Variants cover HF, LocalFS, and AI Hub (remote and local archives).
/// A future ModelScope / Volces hub should be expressible with `Http` +
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
