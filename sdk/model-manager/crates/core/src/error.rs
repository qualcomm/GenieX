use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Library-internal error type.
///
/// Variants are the structured ones callers can match on for recovery;
/// [`Error::Hub`] and [`Error::Http`] remain as free-form fallbacks for
/// failures that don't fit any existing category. When adding a new
/// variant, also update the FFI code mapping in `crates/ffi/src/types.rs`.
#[derive(Debug, Error)]
pub enum Error {
    #[error("model '{0}' not found in local cache")]
    ModelNotFound(String),

    #[error("quantization '{0}' not found for model '{1}'")]
    QuantNotFound(String, String),

    #[error("quantization '{0}' exists but is not downloaded for model '{1}'")]
    QuantNotDownloaded(String, String),

    #[error("no downloaded quantization found for model '{0}'")]
    NoDownloadedQuant(String),

    #[error("model manager not initialized; call geniex_model_init() first")]
    NotInitialized,

    #[error("model manager already initialized; call geniex_model_deinit() first")]
    AlreadyInitialized,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// HTTP server returned a non-success status.
    #[error("HTTP {status} from {url}")]
    HttpStatus { url: String, status: u16 },

    /// Timeout / DNS / proxy / connection reset. Wraps the reqwest-level
    /// message so callers can log but still react uniformly on "network
    /// is down".
    #[error("network error: {0}")]
    HttpTimeout(String),

    /// JSON document we fetched didn't parse against the expected schema
    /// (AI Hub manifest, release-assets.json, platform.json, etc.). `what`
    /// is a short tag identifying which document for log grepping.
    #[error("failed to parse {what}: {source}")]
    ManifestParse {
        what: &'static str,
        #[source]
        source: serde_json::Error,
    },

    /// AI Hub has the model but no asset was published for the requested
    /// chipset. `available` lists the chipsets this model does ship for,
    /// so the caller can surface an actionable message.
    #[error(
        "chipset {requested:?} not available for this model; supported: {}",
        available.join(", ")
    )]
    ChipsetUnavailable {
        requested: String,
        available: Vec<String>,
    },

    /// Freeform hub / extraction / validation error that doesn't fit a
    /// structured variant. Prefer a dedicated variant when introducing
    /// a new failure that callers might want to recover from.
    #[error("hub error: {0}")]
    Hub(String),

    /// Freeform HTTP error that doesn't fit [`Error::HttpStatus`] /
    /// [`Error::HttpTimeout`].
    #[error("http error: {0}")]
    Http(String),

    #[error("download cancelled")]
    Cancelled,

    #[error("invalid model name: '{0}' (must be 'org/repo' with no path traversal)")]
    InvalidModelName(String),

    #[error("invalid file name: '{0}' (must be relative, no '..', no NUL)")]
    InvalidFileName(String),

    #[error("could not infer manifest from directory: {0}")]
    ManifestInferenceFailed(String),
}
