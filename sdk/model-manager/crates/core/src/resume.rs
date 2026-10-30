// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

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

use std::path::{Path, PathBuf};

use crate::executor::chunk;
use crate::source::FileSpec;

/// Suffix appended to a file name for its `.progress` marker.
pub const PROGRESS_SUFFIX: &str = ".progress";

pub(crate) fn marker_path(dest_dir: &Path, name: &str) -> PathBuf {
    dest_dir.join(format!("{name}{PROGRESS_SUFFIX}"))
}

/// Return the subset of `files` that still need to be sent to the
/// executor. Files are included when the on-disk marker/payload state
/// does not already confirm them as done.
pub fn filter_pending(files: &[FileSpec], dest_dir: &Path) -> Vec<FileSpec> {
    files
        .iter()
        .filter(|f| needs_fetch(f, dest_dir))
        .cloned()
        .collect()
}

fn needs_fetch(spec: &FileSpec, dest_dir: &Path) -> bool {
    let name = &spec.name;
    if name.is_empty() {
        return false;
    }
    let marker = marker_path(dest_dir, name);
    let output = dest_dir.join(name);
    if !marker.exists() && output.exists() {
        let on_disk = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        return on_disk != spec.size;
    }
    !chunk::marker_complete(&marker, &output, spec.size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{BytesSource, FileSpec};
    use std::path::PathBuf;

    #[test]
    fn filter_pending_cases() {
        let spec = FileSpec {
            name: "a.gguf".to_string(),
            size: 10,
            bytes: BytesSource::Local {
                path: PathBuf::from("/nowhere"),
            },
        };
        let full: &[u8] = b"xxxxxxxxxx";
        type Bytes<'a> = Option<&'a [u8]>;
        let cases: [(&str, Bytes, Bytes, bool); 7] = [
            ("nothing on disk", None, None, true),
            ("complete bitmap", Some(full), Some(&[1, 1]), false),
            ("complete bitmap, output missing", None, Some(&[1]), true),
            (
                "complete bitmap, size mismatch",
                Some(b"xxxx"),
                Some(&[1]),
                true,
            ),
            ("partial bitmap", Some(full), Some(&[1, 0, 1]), true),
            ("published, no marker", Some(full), None, false),
            ("truncated, no marker", Some(b"xxxxx"), None, true),
        ];
        for (case, output, marker, want_pending) in cases {
            let tmp = tempfile::tempdir().unwrap();
            if let Some(b) = output {
                std::fs::write(tmp.path().join("a.gguf"), b).unwrap();
            }
            if let Some(m) = marker {
                std::fs::write(tmp.path().join("a.gguf.progress"), m).unwrap();
            }
            let pending = filter_pending(std::slice::from_ref(&spec), tmp.path());
            assert_eq!(!pending.is_empty(), want_pending, "{case}");
        }
    }
}
