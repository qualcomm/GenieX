//! Plan-then-fetch contract for a single model pull.
//!
//! A [`ModelSource`] fully resolves "what does this model consist of" —
//! the on-disk manifest and every byte-level recipe — *before* any
//! bulk download starts. Once [`ModelSource::plan`] returns, no further
//! "what's in the model" discovery is allowed; the [`crate::executor`]
//! just moves bytes from each [`BytesSource`] into `dest_dir/<name>`.
//!
//! Implementations live beside this file: [`hf`] (HuggingFace REST API
//! with siblings), [`localfs`] (on-disk directory walk), and
//! [`ai_hub`] (Qualcomm AI Hub S3 protojson chain plus remote ZIP64
//! central-dir parse).

pub mod ai_hub;
pub mod hf;
pub mod local_kind;
pub mod localfs;

use std::path::PathBuf;

use async_trait::async_trait;
use url::Url;

use crate::error::Result;
use crate::manifest::ModelManifest;

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
