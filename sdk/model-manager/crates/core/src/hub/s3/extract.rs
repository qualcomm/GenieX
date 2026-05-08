//! Flat zip extraction for AI Hub qairt assets.
//!
//! Mirrors `cli/internal/model_hub/aihub/unzip.go:ExtractFlat`:
//! every entry lands at `dest_dir/<basename>`, directories and parent
//! paths are discarded, and basename collisions are rejected up front.
//! The lex-first `*.bin` becomes the model entrypoint.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

#[derive(Debug, Default)]
pub struct ExtractedFile {
    pub name: String,
    pub size: u64,
}

#[derive(Debug)]
pub struct ExtractResult {
    /// Lex-first `*.bin` shard; acts as the qairt model entrypoint.
    pub entrypoint_basename: String,
    pub files: Vec<ExtractedFile>,
    pub total_size: u64,
}

/// Unzip `zip_path` into `dest_dir`, flattening directories so every
/// file ends up at `dest_dir/<basename>`. Fails if two entries share a
/// basename, or if the archive contains no `*.bin` shard.
pub fn extract_flat(zip_path: &Path, dest_dir: &Path) -> Result<ExtractResult> {
    let file = fs::File::open(zip_path)
        .map_err(|e| Error::Hub(format!("open zip {}: {e}", zip_path.display())))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| Error::Hub(format!("read zip {}: {e}", zip_path.display())))?;

    // Pre-scan basenames so a collision aborts before writing anything.
    // Skip `__MACOSX/` branches and `._*` AppleDouble resource forks: real
    // AI Hub zips produced on macOS include those, and their basenames
    // (e.g. `._model.bin`) would otherwise collide with the real shards.
    let mut planned: Vec<(usize, String)> = Vec::new();
    let mut seen: HashMap<String, String> = HashMap::with_capacity(archive.len());
    for i in 0..archive.len() {
        let entry = archive
            .by_index(i)
            .map_err(|e| Error::Hub(format!("zip entry {i}: {e}")))?;
        if entry.is_dir() {
            continue;
        }
        let raw = entry.name().to_string();
        if is_macos_metadata(&raw) {
            continue;
        }
        let base = basename(&raw);
        if base.is_empty() || base == "." || base == ".." {
            continue;
        }
        if let Some(prev) = seen.get(&base) {
            return Err(Error::Hub(format!(
                "duplicate basename {base:?} in archive (from {prev:?} and {raw:?})"
            )));
        }
        seen.insert(base.clone(), raw);
        planned.push((i, base));
    }

    fs::create_dir_all(dest_dir)?;

    let mut files: Vec<ExtractedFile> = Vec::with_capacity(planned.len());
    let mut total_size: u64 = 0;
    for (idx, base) in planned {
        let mut entry = archive
            .by_index(idx)
            .map_err(|e| Error::Hub(format!("zip entry {idx}: {e}")))?;
        let out_path: PathBuf = dest_dir.join(&base);
        let mut out = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&out_path)
            .map_err(|e| Error::Hub(format!("create {}: {e}", out_path.display())))?;
        let n = io::copy(&mut entry, &mut out)
            .map_err(|e| Error::Hub(format!("extract {base}: {e}")))?;
        files.push(ExtractedFile {
            name: base,
            size: n,
        });
        total_size += n;
    }

    files.sort_by(|a, b| a.name.cmp(&b.name));
    let entrypoint_basename = files
        .iter()
        .find(|f| f.name.to_ascii_lowercase().ends_with(".bin"))
        .map(|f| f.name.clone())
        .ok_or_else(|| Error::Hub("no .bin shard in archive".to_string()))?;

    Ok(ExtractResult {
        entrypoint_basename,
        files,
        total_size,
    })
}

fn basename(path: &str) -> String {
    // Zip entry names are always '/'-separated per APPNOTE, but some
    // macOS / Windows zip tools emit '\'-tainted names — handle both.
    path.rsplit(['/', '\\'])
        .next()
        .unwrap_or("")
        .to_string()
}

/// AppleDouble metadata embedded by macOS Finder when zipping: `__MACOSX/`
/// siblings and `._*` resource-fork siblings. Never interesting to us,
/// and their basenames collide with real shards.
fn is_macos_metadata(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    if normalized.starts_with("__MACOSX/") || normalized.contains("/__MACOSX/") {
        return true;
    }
    basename(&normalized).starts_with("._")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_zip(path: &Path, entries: &[(&str, &[u8])]) {
        let file = fs::File::create(path).unwrap();
        let mut zw = zip::ZipWriter::new(file);
        let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, data) in entries {
            zw.start_file(*name, opts).unwrap();
            zw.write_all(data).unwrap();
        }
        zw.finish().unwrap();
    }

    #[test]
    fn flattens_and_picks_bin_entrypoint() {
        let tmp = tempfile::tempdir().unwrap();
        let zip_path = tmp.path().join("a.zip");
        write_zip(
            &zip_path,
            &[
                ("sub/model-01.bin", b"x"),
                ("sub/model-00.bin", b"yy"),
                ("tokenizer.json", b"{}"),
            ],
        );

        let dest = tmp.path().join("out");
        let res = extract_flat(&zip_path, &dest).unwrap();

        assert_eq!(res.entrypoint_basename, "model-00.bin");
        assert!(dest.join("model-00.bin").exists());
        assert!(dest.join("tokenizer.json").exists());
        assert_eq!(res.total_size, 1 + 2 + 2);
    }

    #[test]
    fn rejects_basename_collision() {
        let tmp = tempfile::tempdir().unwrap();
        let zip_path = tmp.path().join("a.zip");
        write_zip(&zip_path, &[("a/model.bin", b"xx"), ("b/model.bin", b"yy")]);

        let dest = tmp.path().join("out");
        assert!(extract_flat(&zip_path, &dest).is_err());
        // Ensure nothing was written.
        assert!(!dest.exists() || fs::read_dir(&dest).unwrap().next().is_none());
    }

    #[test]
    fn skips_macos_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let zip_path = tmp.path().join("a.zip");
        // Shape mirrors a real AI Hub qairt zip: the real shard plus a
        // `__MACOSX/.../._<shard>` AppleDouble sibling that would
        // otherwise collide on basename and win the lex-first entrypoint
        // pick.
        write_zip(
            &zip_path,
            &[
                ("dir/model.bin", b"XX"),
                ("__MACOSX/dir/._model.bin", b"junk"),
                ("dir/tokenizer.json", b"{}"),
                ("__MACOSX/dir/._tokenizer.json", b"junk"),
            ],
        );

        let dest = tmp.path().join("out");
        let res = extract_flat(&zip_path, &dest).unwrap();
        assert_eq!(res.entrypoint_basename, "model.bin");
        assert_eq!(res.files.len(), 2);
        assert!(!dest.join("._model.bin").exists());
    }

    #[test]
    fn rejects_archive_without_bin() {
        let tmp = tempfile::tempdir().unwrap();
        let zip_path = tmp.path().join("a.zip");
        write_zip(&zip_path, &[("readme.md", b"hello")]);

        let dest = tmp.path().join("out");
        assert!(extract_flat(&zip_path, &dest).is_err());
    }
}
