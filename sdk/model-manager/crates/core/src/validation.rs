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
    fn valid_names() {
        validate_model_name("Qwen/Qwen3-4B-Instruct").unwrap();
        validate_model_name("a/b").unwrap();
    }

    #[test]
    fn rejects_traversal() {
        assert!(validate_model_name("../etc").is_err());
        assert!(validate_model_name("a/..").is_err());
        assert!(validate_model_name("./b").is_err());
        assert!(validate_model_name("a\\b").is_err());
    }

    #[test]
    fn rejects_bad_shape() {
        assert!(validate_model_name("").is_err());
        assert!(validate_model_name("onlyone").is_err());
        assert!(validate_model_name("a/b/c").is_err());
        assert!(validate_model_name("a//b").is_err());
    }

    #[test]
    fn rejects_nul() {
        assert!(validate_model_name("a/b\0").is_err());
        assert!(validate_relative_file("foo\0.txt").is_err());
    }

    #[test]
    fn relative_file_rules() {
        validate_relative_file("model.gguf").unwrap();
        validate_relative_file("sub/model.gguf").unwrap();
        assert!(validate_relative_file("/abs").is_err());
        assert!(validate_relative_file("..").is_err());
        assert!(validate_relative_file("a/../b").is_err());
    }
}
