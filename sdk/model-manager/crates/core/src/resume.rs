//! Decide which files from a manifest still need fetching.
//!
//! The download engine does the final chunk-level resume decision from
//! the `.progress` bitmap, but a restarted pull still benefits from
//! short-circuiting files whose bitmap is already all 0x01 — otherwise
//! every resume round-trips one HEAD per file just to confirm size.
//!
//! Extracted from `pull::pull_locked` so the filter can be unit-tested
//! without mocking a hub, and so the orchestrator reads as an obvious
//! sequence (plan → fetch → publish) rather than an imperative lump.

use std::path::Path;

use crate::manifest::ModelManifest;

/// Suffix appended to a file name for its `.progress` marker. Kept in
/// lockstep with [`crate::pull::PROGRESS_SUFFIX`].
pub const PROGRESS_SUFFIX: &str = ".progress";

/// Build the list of file names from `manifest` that need to be sent to
/// the hub's `download()`. Files are included when any of the following
/// is true:
/// - The file is declared `downloaded` in the manifest (so the caller
///   actually wants it; non-downloaded entries in the manifest aren't
///   selected for this pull) AND
/// - There's no legacy "published but no marker" state (file present,
///   no `.progress` sibling — treat as already done) AND
/// - The `.progress` bitmap is either missing or not all-ones.
pub fn files_to_download(manifest: &ModelManifest, dest_dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();

    let mut push_if_needed = |name: &str| {
        if name.is_empty() {
            return;
        }
        let marker = dest_dir.join(format!("{name}{PROGRESS_SUFFIX}"));
        let output = dest_dir.join(name);
        // Legacy published state (file present, no marker): a prior
        // pull left this shape. Treat as done — the engine would
        // otherwise re-HEAD for no reason.
        if !marker.exists() && output.exists() {
            return;
        }
        if let Ok(data) = std::fs::read(&marker) {
            // Non-empty, all-0x01 bitmap means every chunk already
            // succeeded; skip. Partial bitmaps fall through so the
            // engine can decide what to refetch.
            if !data.is_empty() && data.iter().all(|b| *b == 0x01) {
                return;
            }
        }
        out.push(name.to_string());
    };

    for f in manifest.model_file.values() {
        if f.downloaded {
            push_if_needed(&f.name);
        }
    }
    if manifest.mmproj_file.downloaded {
        push_if_needed(&manifest.mmproj_file.name);
    }
    if manifest.tokenizer_file.downloaded {
        push_if_needed(&manifest.tokenizer_file.name);
    }
    for f in &manifest.extra_files {
        if f.downloaded {
            push_if_needed(&f.name);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelFileInfo, ModelType};
    use std::collections::HashMap;

    fn mf(name: &str, downloaded: bool) -> ModelFileInfo {
        ModelFileInfo {
            name: name.to_string(),
            downloaded,
            size: 10,
        }
    }

    fn base_manifest() -> ModelManifest {
        let mut model_file: HashMap<String, ModelFileInfo> = HashMap::new();
        model_file.insert("Q4_K_M".to_string(), mf("a.gguf", true));
        ModelManifest {
            name: "org/repo".to_string(),
            model_name: "org/repo".to_string(),
            model_type: ModelType::Llm,
            plugin_id: "llama_cpp".to_string(),
            device_id: String::new(),
            min_sdk_version: "0".to_string(),
            precision: String::new(),
            model_file,
            mmproj_file: ModelFileInfo::default(),
            tokenizer_file: ModelFileInfo::default(),
            extra_files: Vec::new(),
        }
    }

    #[test]
    fn includes_file_when_nothing_exists_yet() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = files_to_download(&base_manifest(), tmp.path());
        assert_eq!(plan, vec!["a.gguf".to_string()]);
    }

    #[test]
    fn skips_fully_complete_bitmap() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.gguf"), b"xxxxxxxxxx").unwrap();
        std::fs::write(tmp.path().join("a.gguf.progress"), [0x01, 0x01]).unwrap();
        let plan = files_to_download(&base_manifest(), tmp.path());
        assert!(plan.is_empty(), "fully done file must be skipped: {plan:?}");
    }

    #[test]
    fn includes_file_with_partial_bitmap() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.gguf"), b"xxxxxxxxxx").unwrap();
        std::fs::write(tmp.path().join("a.gguf.progress"), [0x01, 0x00, 0x01]).unwrap();
        let plan = files_to_download(&base_manifest(), tmp.path());
        assert_eq!(plan, vec!["a.gguf".to_string()]);
    }

    #[test]
    fn treats_published_without_marker_as_done() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.gguf"), b"xxxxxxxxxx").unwrap();
        // No `.progress` marker — legacy shape published by an older
        // Go CLI version. Don't refetch.
        let plan = files_to_download(&base_manifest(), tmp.path());
        assert!(plan.is_empty());
    }

    #[test]
    fn empty_file_names_are_silently_skipped() {
        // mmproj_file and tokenizer_file default to empty names when
        // the manifest doesn't carry them. We must never push "" into
        // the download list (would turn into dest_dir/"" and error in
        // the engine).
        let mut m = base_manifest();
        m.mmproj_file = ModelFileInfo {
            name: String::new(),
            downloaded: true,
            size: 0,
        };
        m.tokenizer_file = ModelFileInfo {
            name: String::new(),
            downloaded: true,
            size: 0,
        };
        let tmp = tempfile::tempdir().unwrap();
        let plan = files_to_download(&m, tmp.path());
        assert_eq!(plan, vec!["a.gguf".to_string()]);
    }

    #[test]
    fn manifest_entries_not_downloaded_are_ignored() {
        let mut m = base_manifest();
        m.model_file.clear();
        m.model_file.insert("Q4".to_string(), mf("notneeded.gguf", false));
        let tmp = tempfile::tempdir().unwrap();
        let plan = files_to_download(&m, tmp.path());
        assert!(plan.is_empty());
    }

    #[test]
    fn includes_extras_that_need_fetching() {
        let mut m = base_manifest();
        m.extra_files.push(mf("tokenizer.json", true));
        let tmp = tempfile::tempdir().unwrap();
        let plan = files_to_download(&m, tmp.path());
        assert!(plan.iter().any(|n| n == "a.gguf"));
        assert!(plan.iter().any(|n| n == "tokenizer.json"));
    }
}
