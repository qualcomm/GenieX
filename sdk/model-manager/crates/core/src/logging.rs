//! Pluggable log sink for the core crate.
//!
//! Core has no logging facade and must not depend on the FFI crate —
//! that would invert the `ffi -> core` dependency. Instead it exposes a
//! global sink that the FFI layer installs at init time (see
//! `crates/ffi/src/logging.rs`), forwarding into geniex's C log callback.
//! Until a sink is installed — plain-Rust use, unit tests — messages
//! fall back to stderr so nothing is silently dropped.

use std::sync::OnceLock;

#[derive(Copy, Clone)]
pub enum Level {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

pub type LogFn = fn(Level, &str);

static SINK: OnceLock<LogFn> = OnceLock::new();

/// Install the process-wide log sink. Idempotent: the first install
/// wins, later calls are no-ops.
pub fn set_sink(f: LogFn) {
    let _ = SINK.set(f);
}

pub fn log(level: Level, msg: &str) {
    match SINK.get() {
        Some(f) => f(level, msg),
        None => eprintln!("[model-manager] {msg}"),
    }
}

#[inline]
#[allow(dead_code)]
pub fn debug(msg: &str) {
    log(Level::Debug, msg);
}

#[inline]
#[allow(dead_code)]
pub fn info(msg: &str) {
    log(Level::Info, msg);
}

#[inline]
pub fn warn(msg: &str) {
    log(Level::Warn, msg);
}

#[inline]
#[allow(dead_code)]
pub fn error(msg: &str) {
    log(Level::Error, msg);
}
