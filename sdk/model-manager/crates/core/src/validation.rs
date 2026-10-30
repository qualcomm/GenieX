// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Input validation — protects against path traversal and malformed names.

use crate::error::{Error, Result};

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
}
