// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// When adding a variant, also update the FFI code mapping in `crates/ffi/src/types.rs`.
#[derive(Debug, Error)]
pub enum Error {
    #[error("model '{0}' not found in local cache")]
    ModelNotFound(String),

    #[error("model {0} not found on hub")]
    HubModelNotFound(String),

    #[error("quantization '{0}' not found for model '{1}'")]
    QuantNotFound(String, String),

    #[error("quantization '{0}' exists but is not downloaded for model '{1}'")]
    QuantNotDownloaded(String, String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP {status} from {url}")]
    HttpStatus { url: String, status: u16 },

    /// Timeout / DNS / proxy / connection reset.
    #[error("network error: {0}")]
    HttpTimeout(String),

    #[error("failed to parse {what}: {source}")]
    ManifestParse {
        what: &'static str,
        #[source]
        source: serde_json::Error,
    },

    #[error(
        "chipset {requested:?} not available for this model; supported: {}",
        available.join(", ")
    )]
    ChipsetUnavailable {
        requested: String,
        available: Vec<String>,
    },

    #[error("hub error: {0}")]
    Hub(String),

    #[error("http error: {0}")]
    Http(String),

    #[error("invalid url ({context}): {source}")]
    InvalidUrl {
        context: String,
        #[source]
        source: url::ParseError,
    },

    #[error("download cancelled")]
    Cancelled,

    /// `required` sums full file sizes, overestimating for partially-resumed files.
    #[error(
        "not enough free disk space: this download needs {:.2} GiB but only {:.2} GiB is available",
        *required as f64 / (1024.0 * 1024.0 * 1024.0),
        *available as f64 / (1024.0 * 1024.0 * 1024.0)
    )]
    InsufficientDiskSpace { required: u64, available: u64 },

    #[error("invalid model name: '{0}' (must be 'org/repo' with no path traversal)")]
    InvalidModelName(String),

    #[error("could not infer a model manifest: {0}")]
    ManifestInferenceFailed(String),
}

impl Error {
    pub fn invalid_url(context: impl Into<String>, source: url::ParseError) -> Self {
        Error::InvalidUrl {
            context: context.into(),
            source,
        }
    }
}

pub fn parse_manifest<'a, T: serde::Deserialize<'a>>(
    what: &'static str,
    bytes: &'a [u8],
) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|source| Error::ManifestParse { what, source })
}
