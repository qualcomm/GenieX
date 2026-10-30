// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Input validation — protects against path traversal and malformed names.

use crate::error::{Error, Result};

/// Validate a "org/repo" model name.
///
/// Rules:
/// - non-empty, <= 255 bytes total
/// - exactly one '/' separator
/// - each component is non-empty, not '.' or '..', and contains no
///   '\\', '\0', or literal slashes beyond the separator
pub fn validate_model_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > 255 {
        return Err(Error::InvalidModelName(name.to_string()));
    }
    if name.matches('/').count() != 1 {
        return Err(Error::InvalidModelName(name.to_string()));
    }
    for component in name.split('/') {
        if component.is_empty()
            || component == "."
            || component == ".."
            || component.contains('\\')
            || component.contains('\0')
        {
            return Err(Error::InvalidModelName(name.to_string()));
        }
    }
    Ok(())
}

/// Validate a file name relative to a model directory.
///
/// Rules:
/// - not absolute (no leading '/' or '\\')
/// - no component equal to '..'
/// - no NUL byte
pub fn validate_relative_file(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::InvalidFileName(name.to_string()));
    }
    if name.starts_with('/') || name.starts_with('\\') || name.contains('\0') {
        return Err(Error::InvalidFileName(name.to_string()));
    }
    for component in name.split(['/', '\\']) {
        if component == ".." {
            return Err(Error::InvalidFileName(name.to_string()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_name_rules() {
        for ok in ["Qwen/Qwen3-4B-Instruct", "a/b"] {
            assert!(validate_model_name(ok).is_ok(), "{ok:?}");
        }
        for bad in [
            "../etc", "a/..", "./b", "a\\b", "", "onlyone", "a/b/c", "a//b", "a/b\0",
        ] {
            assert!(validate_model_name(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn relative_file_rules() {
        for ok in ["model.gguf", "sub/model.gguf"] {
            assert!(validate_relative_file(ok).is_ok(), "{ok:?}");
        }
        for bad in ["/abs", "..", "a/../b", "foo\0.txt"] {
            assert!(validate_relative_file(bad).is_err(), "{bad:?}");
        }
    }
}
