//! Concurrent + chunked download engine.
//!
//! Split out into:
//! - [`chunk`]: pure functions — chunk planner and `.progress` bitmap I/O.
//!   These carry no network or tokio dependency and are the foundation the
//!   rest of the engine builds on.
//! - (later commits) `engine` module: the async driver that composes a
//!   [`crate::transport::HttpTransport`] with a chunk plan.
//!
//! `.progress` format (byte-compatible with the Go CLI at
//! `cli/internal/model_hub/model_hub.go`): a file of length
//! `ceil(size / chunk_size)` where byte `i` is `0x01` iff chunk `i` is done.

pub mod chunk;
