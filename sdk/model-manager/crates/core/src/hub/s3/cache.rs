//! TTL-backed on-disk cache for AI Hub index JSONs.
//!
//! Matches `cli/internal/model_hub/aihub/aihub.go:fetchJSON`: a cache
//! entry is served when the file's mtime is younger than the TTL.
//! Write failures are logged to stderr and swallowed — the caller
//! always gets the freshly fetched bytes; the cache is best-effort.

use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

/// Default TTL for manifest / release-assets JSONs. Mirrors Go's
/// `DefaultCacheTTL = 24 * time.Hour`.
pub const DEFAULT_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Return cached bytes if the file exists and was modified more recently
/// than `ttl` ago. Any I/O error is treated as a cache miss.
pub fn read_if_fresh(path: &Path, ttl: Duration) -> Option<Vec<u8>> {
    let meta = fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?;
    let age = SystemTime::now().duration_since(mtime).ok()?;
    if age > ttl {
        return None;
    }
    fs::read(path).ok()
}

/// Best-effort write: creates parent dirs, logs and swallows errors.
/// Uses the `[model-manager]` prefix so downstream log scrapers can
/// pick it up, and to keep a single tag to replace when we migrate
/// to `tracing`.
pub fn write(path: &Path, data: &[u8]) {
    if let Some(parent) = path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            eprintln!(
                "[model-manager] aihub cache mkdir {}: {e}",
                parent.display()
            );
            return;
        }
    }
    if let Err(e) = fs::write(path, data) {
        eprintln!("[model-manager] aihub cache write {}: {e}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn miss_on_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("nope.json");
        assert!(read_if_fresh(&p, DEFAULT_TTL).is_none());
    }

    #[test]
    fn hit_on_fresh_file() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("x.json");
        write(&p, b"hello");
        let got = read_if_fresh(&p, DEFAULT_TTL).expect("hit");
        assert_eq!(got, b"hello");
    }

    #[test]
    fn miss_on_stale_file() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("x.json");
        write(&p, b"hello");
        assert!(read_if_fresh(&p, Duration::ZERO).is_none());
    }
}
