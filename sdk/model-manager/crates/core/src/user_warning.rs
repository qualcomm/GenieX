// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Pluggable sink for user-facing warnings.
//!
//! Distinct from [`crate::logging`]: `logging` is a firehose for the
//! `--log`-controlled developer stream (default off), whereas a
//! user_warning is a rare, actionable signal a caller wants to surface
//! even at the default log level — e.g. "AIHM release pointer
//! unreachable, using pinned fallback v0.60.0, new models may be
//! missing". The FFI layer bridges this into a thread-local slot that
//! [`geniex_model_last_warning_message`] hands out to bindings, so a
//! CLI / SDK caller can render it on stderr after a successful call.
//!
//! Core has no direct dependency on FFI (that would invert
//! `ffi -> core`); [`set_sink`] is installed once at init time.

use std::sync::OnceLock;

pub type WarningFn = fn(&str);

static SINK: OnceLock<WarningFn> = OnceLock::new();

/// Install the process-wide user-warning sink. Idempotent: the first
/// install wins, later calls are no-ops.
pub fn set_sink(f: WarningFn) {
    let _ = SINK.set(f);
}

/// Emit a user-facing warning. Silently dropped when no sink is
/// installed (e.g. plain-Rust use, unit tests) — the same message
/// should already be going through [`crate::logging::warn`] for the
/// developer stream.
pub fn emit(msg: &str) {
    if let Some(f) = SINK.get() {
        f(msg);
    }
}
