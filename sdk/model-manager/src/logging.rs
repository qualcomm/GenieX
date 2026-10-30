// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Global log sink: `geniex_model_init` routes it to the SDK's C logger;
//! until then (and in Rust tests) it falls back to stderr.

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

/// First install wins.
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
pub fn warn(msg: &str) {
    log(Level::Warn, msg);
}
