//! Filter `FileSpec`s that still need fetching on a resumed pull.
//!
//! The executor does the final chunk-level resume decision from the
//! `.progress` bitmap, but a restarted pull still benefits from
//! short-circuiting files whose bitmap is already all `0x01` — each
//! otherwise round-trips one HEAD per file just to confirm size.
//!
//! Also recognises a legacy "published but no marker" state: a prior
//! pull left the output file in place with no `.progress` sibling.
//! Treat that as done rather than re-HEADing.

use std::path::Path;

use crate::source::FileSpec;

/// Suffix appended to a file name for its `.progress` marker.
pub const PROGRESS_SUFFIX: &str = ".progress";

/// Return the subset of `files` that still need to be sent to the
/// executor. Files are included when the on-disk marker/payload state
/// does not already confirm them as done.
pub fn filter_pending<'a>(files: &'a [FileSpec], dest_dir: &Path) -> Vec<FileSpec> {
    files
        .iter()
        .filter(|f| needs_fetch(&f.name, dest_dir))
        .cloned()
        .collect()
}

fn needs_fetch(name: &str, dest_dir: &Path) -> bool {
    if name.is_empty() {
        return false;
    }
    let marker = dest_dir.join(format!("{name}{PROGRESS_SUFFIX}"));
    let output = dest_dir.join(name);
    // Legacy published state (file present, no marker): treat as done.
    if !marker.exists() && output.exists() {
        return false;
    }
    if let Ok(data) = std::fs::read(&marker) {
        // Non-empty, all-0x01 bitmap means every chunk already
        // succeeded; skip. Partial bitmaps fall through so the
        // executor can decide what to refetch.
        if !data.is_empty() && data.iter().all(|b| *b == 0x01) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{BytesSource, FileSpec};
    use std::path::PathBuf;

    fn local_spec(name: &str) -> FileSpec {
        FileSpec {
            name: name.to_string(),
            size: 10,
            bytes: BytesSource::Local {
                path: PathBuf::from("/nowhere"),
            },
        }
    }

    #[test]
    fn includes_file_when_nothing_exists_yet() {
        let tmp = tempfile::tempdir().unwrap();
        let pending = filter_pending(&[local_spec("a.gguf")], tmp.path());
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn skips_fully_complete_bitmap() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.gguf"), b"xxxxxxxxxx").unwrap();
        std::fs::write(tmp.path().join("a.gguf.progress"), [0x01, 0x01]).unwrap();
        let pending = filter_pending(&[local_spec("a.gguf")], tmp.path());
        assert!(pending.is_empty());
    }

    #[test]
    fn includes_file_with_partial_bitmap() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.gguf"), b"xxxxxxxxxx").unwrap();
        std::fs::write(tmp.path().join("a.gguf.progress"), [0x01, 0x00, 0x01]).unwrap();
        let pending = filter_pending(&[local_spec("a.gguf")], tmp.path());
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn treats_published_without_marker_as_done() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.gguf"), b"xxxxxxxxxx").unwrap();
        let pending = filter_pending(&[local_spec("a.gguf")], tmp.path());
        assert!(pending.is_empty());
    }
}
