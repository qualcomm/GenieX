//! Auto-detect the layout of a `local_path` passed to [`LocalFsSource`].
//!
//! Three known shapes today:
//!
//! - `HfGguf` — a directory whose model weights are GGUF (the existing
//!   pre-AI-Hub path).
//! - `AiHubExtracted` — a directory matching what `aihub.ExtractFlat` /
//!   the user produces by unzipping an AI Hub asset: at least one `.bin`
//!   shard and a `metadata.json` at the root.
//! - `AiHubZip` — a `.zip` file straight off the AI Hub website.
//!
//! Order of detection matters: a `.zip` file is identified by its
//! extension before we look inside, and AI Hub directories take
//! precedence over GGUF because nothing prevents an AI Hub release from
//! eventually shipping a sibling GGUF file alongside `.bin` shards.

use std::ffi::OsStr;
use std::fs;
use std::path::Path;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalKind {
    HfGguf,
    AiHubExtracted,
    AiHubZip,
}

const AIHUB_METADATA_FILE: &str = "metadata.json";

/// Inspect `path` and decide which loader the [`LocalFsSource`] should
/// dispatch to.
///
/// Returns [`Error::Hub`] when `path` is neither a recognised directory
/// layout nor a `.zip` file; the message lists the three shapes the
/// loader knows how to handle so a user pointing at the wrong directory
/// gets actionable feedback.
pub fn detect(path: &Path) -> Result<LocalKind> {
    let meta = fs::metadata(path).map_err(|e| {
        Error::Hub(format!(
            "local path {} is not accessible: {e}",
            path.display()
        ))
    })?;

    if meta.is_file() {
        if has_extension(path, "zip") {
            return Ok(LocalKind::AiHubZip);
        }
        return Err(Error::Hub(format!(
            "local path {} is a file but not a .zip; expected an AI Hub archive or a directory",
            path.display()
        )));
    }

    if !meta.is_dir() {
        return Err(Error::Hub(format!(
            "local path {} is neither a regular file nor a directory",
            path.display()
        )));
    }

    let mut has_bin = false;
    let mut has_metadata = false;
    let mut has_gguf = false;
    let mut has_safetensors = false;
    for entry in fs::read_dir(path)?.flatten() {
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if !ft.is_file() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let lower = name.to_ascii_lowercase();
        if lower == AIHUB_METADATA_FILE {
            has_metadata = true;
        } else if lower.ends_with(".bin") {
            has_bin = true;
        } else if lower.ends_with(".gguf") {
            has_gguf = true;
        } else if lower.ends_with(".safetensors") {
            has_safetensors = true;
        }
    }

    if has_metadata && has_bin {
        return Ok(LocalKind::AiHubExtracted);
    }
    if has_gguf {
        return Ok(LocalKind::HfGguf);
    }
    if has_safetensors {
        return Err(Error::Hub(format!(
            "local path {} looks like a HuggingFace safetensors snapshot, \
             which is not supported as a local pull source yet",
            path.display()
        )));
    }
    Err(Error::Hub(format!(
        "local path {} did not match any known layout: \
         expected a directory with *.gguf (HF GGUF), \
         a directory with metadata.json + *.bin (AI Hub extracted), \
         or a .zip file (AI Hub archive)",
        path.display()
    )))
}

fn has_extension(path: &Path, ext: &str) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|e| e.eq_ignore_ascii_case(ext))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn detects_aihub_zip_by_extension() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("model.zip");
        fs::write(&p, b"PK\x03\x04not-a-real-zip").unwrap();
        assert_eq!(detect(&p).unwrap(), LocalKind::AiHubZip);
    }

    #[test]
    fn detects_aihub_zip_case_insensitive() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("Model.ZIP");
        fs::write(&p, b"x").unwrap();
        assert_eq!(detect(&p).unwrap(), LocalKind::AiHubZip);
    }

    #[test]
    fn detects_aihub_extracted_via_metadata_and_bin() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("metadata.json"), b"{}").unwrap();
        fs::write(tmp.path().join("weights_part_1.bin"), b"x").unwrap();
        fs::write(tmp.path().join("weights_part_2.bin"), b"y").unwrap();
        assert_eq!(detect(tmp.path()).unwrap(), LocalKind::AiHubExtracted);
    }

    #[test]
    fn detects_hf_gguf_dir() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("model-Q4_K_M.gguf"), b"x").unwrap();
        assert_eq!(detect(tmp.path()).unwrap(), LocalKind::HfGguf);
    }

    #[test]
    fn aihub_extracted_wins_over_gguf_sibling() {
        // Forward-compat: if AI Hub ever ships both, `metadata.json + .bin`
        // is the load-bearing signal — GGUF inference would get plugin_id
        // wrong.
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("metadata.json"), b"{}").unwrap();
        fs::write(tmp.path().join("weights.bin"), b"x").unwrap();
        fs::write(tmp.path().join("extra.gguf"), b"y").unwrap();
        assert_eq!(detect(tmp.path()).unwrap(), LocalKind::AiHubExtracted);
    }

    #[test]
    fn safetensors_only_dir_returns_unsupported_error() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), b"{}").unwrap();
        fs::write(tmp.path().join("model.safetensors"), b"x").unwrap();
        let err = detect(tmp.path()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("safetensors"), "msg: {msg}");
    }

    #[test]
    fn unknown_dir_lists_known_layouts() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("readme.txt"), b"hi").unwrap();
        let err = detect(tmp.path()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("AI Hub extracted"), "msg: {msg}");
        assert!(msg.contains("HF GGUF"), "msg: {msg}");
        assert!(msg.contains(".zip"), "msg: {msg}");
    }

    #[test]
    fn nonexistent_path_returns_helpful_error() {
        let err = detect(Path::new("/nonexistent/path/12345xyz")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not accessible"), "msg: {msg}");
    }

    #[test]
    fn non_zip_file_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("model.tar.gz");
        fs::write(&p, b"x").unwrap();
        let err = detect(&p).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not a .zip"), "msg: {msg}");
    }
}
