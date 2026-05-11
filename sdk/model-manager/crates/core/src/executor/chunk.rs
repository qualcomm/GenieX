//! Chunk planner + `.progress` bitmap.
//!
//! The chunking policy mirrors the Go CLI: `chunk_size =
//! max(MIN_CHUNK_SIZE, size / MAX_CHUNKS_PER_FILE)`, so small files are a
//! single chunk and very large files are capped at `MAX_CHUNKS_PER_FILE`
//! chunks. Users can override the floor via `GENIEX_DL_CHUNK_SIZE`.
//!
//! The bitmap format is `ceil(size / chunk_size)` bytes where byte `i` is
//! `0x01` iff chunk `i` has been fully written. This matches
//! `cli/internal/model_hub/model_hub.go` so Go and Rust agents can resume
//! each other's partial pulls.

use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::Result;

pub const MIN_CHUNK_SIZE: u64 = 16 * 1024 * 1024;
pub const MAX_CHUNKS_PER_FILE: u64 = 128;
pub const PROGRESS_DONE_BYTE: u8 = 0x01;

const ENV_CHUNK_SIZE: &str = "GENIEX_DL_CHUNK_SIZE";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkRange {
    pub index: usize,
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkPlan {
    pub file_size: u64,
    pub chunk_size: u64,
    pub chunks: Vec<ChunkRange>,
}

impl ChunkPlan {
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }
}

/// Build a chunk plan for `file_size`. A `file_size` of 0 yields a
/// zero-chunk plan; the caller is responsible for still creating an empty
/// output file.
pub fn plan_chunks(file_size: u64) -> ChunkPlan {
    plan_chunks_with_floor(file_size, effective_min_chunk_size())
}

/// Like [`plan_chunks`] but with an explicit floor. Exposed mainly for tests.
pub fn plan_chunks_with_floor(file_size: u64, min_chunk_size: u64) -> ChunkPlan {
    let min = min_chunk_size.max(1);
    let chunk_size = if file_size == 0 {
        min
    } else {
        core::cmp::max(min, file_size / MAX_CHUNKS_PER_FILE)
    };
    let mut chunks = Vec::new();
    if file_size > 0 {
        let n = (file_size + chunk_size - 1) / chunk_size;
        for i in 0..n {
            let offset = i * chunk_size;
            let len = core::cmp::min(chunk_size, file_size - offset);
            chunks.push(ChunkRange {
                index: i as usize,
                offset,
                len,
            });
        }
    }
    ChunkPlan {
        file_size,
        chunk_size,
        chunks,
    }
}

fn effective_min_chunk_size() -> u64 {
    std::env::var(ENV_CHUNK_SIZE)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(MIN_CHUNK_SIZE)
}

/// Load the `.progress` bitmap for a plan, creating a zero-initialised one
/// if the file is missing or length-mismatched (which signals either a
/// fresh pull or a chunk-size change since the last attempt).
pub fn load_or_init_bitmap(marker_path: &Path, plan: &ChunkPlan) -> Result<Vec<u8>> {
    let expected = plan.num_chunks();
    match fs::read(marker_path) {
        Ok(buf) if buf.len() == expected => Ok(buf),
        Ok(_) | Err(_) => {
            let buf = vec![0u8; expected];
            if let Some(parent) = marker_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(marker_path, &buf)?;
            Ok(buf)
        }
    }
}

/// Return the subset of chunks that still need to be fetched, given a
/// bitmap.
pub fn pending_chunks(plan: &ChunkPlan, bitmap: &[u8]) -> Vec<ChunkRange> {
    plan.chunks
        .iter()
        .filter(|c| bitmap.get(c.index).copied().unwrap_or(0) != PROGRESS_DONE_BYTE)
        .copied()
        .collect()
}

/// Count bytes already committed according to the bitmap.
pub fn bytes_already_done(plan: &ChunkPlan, bitmap: &[u8]) -> u64 {
    plan.chunks
        .iter()
        .filter(|c| bitmap.get(c.index).copied().unwrap_or(0) == PROGRESS_DONE_BYTE)
        .map(|c| c.len)
        .sum()
}

/// Flip the marker byte for `index` to `0x01` in the on-disk bitmap. Uses
/// positional write so concurrent workers do not step on each other.
pub fn mark_chunk_done(marker_path: &Path, index: usize) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .read(true)
        .open(marker_path)?;
    f.seek(SeekFrom::Start(index as u64))?;
    f.write_all(&[PROGRESS_DONE_BYTE])?;
    f.flush()?;
    Ok(())
}

/// True iff every chunk in the bitmap is marked done. Useful as the final
/// "is this file fully downloaded" check.
pub fn bitmap_complete(bitmap: &[u8]) -> bool {
    !bitmap.is_empty() && bitmap.iter().all(|b| *b == PROGRESS_DONE_BYTE)
}

/// Ensure the destination file exists and is at least `file_size` bytes,
/// so range workers can seek + write without races on file growth.
pub fn preallocate(output_path: &Path, file_size: u64) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let f = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .open(output_path)?;
    let cur = f.metadata()?.len();
    if cur < file_size {
        f.set_len(file_size)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn tiny_file_is_single_chunk() {
        let plan = plan_chunks_with_floor(1024, MIN_CHUNK_SIZE);
        assert_eq!(plan.num_chunks(), 1);
        assert_eq!(plan.chunks[0].offset, 0);
        assert_eq!(plan.chunks[0].len, 1024);
        assert_eq!(plan.chunk_size, MIN_CHUNK_SIZE);
    }

    #[test]
    fn exact_min_chunk_is_one_chunk() {
        let plan = plan_chunks_with_floor(MIN_CHUNK_SIZE, MIN_CHUNK_SIZE);
        assert_eq!(plan.num_chunks(), 1);
        assert_eq!(plan.chunks[0].len, MIN_CHUNK_SIZE);
    }

    #[test]
    fn thirty_two_mib_splits_into_two() {
        let plan = plan_chunks_with_floor(2 * MIN_CHUNK_SIZE, MIN_CHUNK_SIZE);
        assert_eq!(plan.num_chunks(), 2);
        assert_eq!(plan.chunks[0].len, MIN_CHUNK_SIZE);
        assert_eq!(plan.chunks[1].len, MIN_CHUNK_SIZE);
    }

    #[test]
    fn huge_file_caps_at_max_chunks() {
        // 4 GiB at the default floor → 128 chunks of 32 MiB each.
        let size = 4u64 * 1024 * 1024 * 1024;
        let plan = plan_chunks_with_floor(size, MIN_CHUNK_SIZE);
        assert_eq!(plan.num_chunks(), MAX_CHUNKS_PER_FILE as usize);
        assert_eq!(plan.chunk_size, size / MAX_CHUNKS_PER_FILE);
        assert_eq!(
            plan.chunks.iter().map(|c| c.len).sum::<u64>(),
            size,
            "chunks must tile the whole file"
        );
    }

    #[test]
    fn last_chunk_takes_the_remainder() {
        let size = MIN_CHUNK_SIZE + 123;
        let plan = plan_chunks_with_floor(size, MIN_CHUNK_SIZE);
        assert_eq!(plan.num_chunks(), 2);
        assert_eq!(plan.chunks[1].len, 123);
        assert_eq!(plan.chunks[1].offset, MIN_CHUNK_SIZE);
    }

    #[test]
    fn zero_size_yields_no_chunks() {
        let plan = plan_chunks_with_floor(0, MIN_CHUNK_SIZE);
        assert_eq!(plan.num_chunks(), 0);
    }

    #[test]
    fn bitmap_roundtrip_and_mark() {
        let dir = tempdir().unwrap();
        let marker = dir.path().join("m.progress");
        let plan = plan_chunks_with_floor(3 * MIN_CHUNK_SIZE, MIN_CHUNK_SIZE);
        let bitmap = load_or_init_bitmap(&marker, &plan).unwrap();
        assert_eq!(bitmap, vec![0, 0, 0]);
        assert_eq!(pending_chunks(&plan, &bitmap).len(), 3);
        assert_eq!(bytes_already_done(&plan, &bitmap), 0);

        mark_chunk_done(&marker, 1).unwrap();
        let bitmap = load_or_init_bitmap(&marker, &plan).unwrap();
        assert_eq!(bitmap, vec![0, PROGRESS_DONE_BYTE, 0]);
        assert_eq!(pending_chunks(&plan, &bitmap).len(), 2);
        assert_eq!(bytes_already_done(&plan, &bitmap), MIN_CHUNK_SIZE);
        assert!(!bitmap_complete(&bitmap));

        mark_chunk_done(&marker, 0).unwrap();
        mark_chunk_done(&marker, 2).unwrap();
        let bitmap = load_or_init_bitmap(&marker, &plan).unwrap();
        assert!(bitmap_complete(&bitmap));
        assert_eq!(pending_chunks(&plan, &bitmap).len(), 0);
    }

    #[test]
    fn length_mismatch_triggers_reinit() {
        let dir = tempdir().unwrap();
        let marker = dir.path().join("m.progress");
        // Stale bitmap from a previous pull with different chunk size.
        fs::write(&marker, vec![PROGRESS_DONE_BYTE; 7]).unwrap();
        let plan = plan_chunks_with_floor(3 * MIN_CHUNK_SIZE, MIN_CHUNK_SIZE);
        let bitmap = load_or_init_bitmap(&marker, &plan).unwrap();
        assert_eq!(bitmap, vec![0, 0, 0], "mismatched bitmap must reset");
    }

    #[test]
    fn preallocate_grows_but_never_truncates() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("f.bin");
        preallocate(&path, 1024).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), 1024);

        // Writing some data then preallocating to a smaller size is a no-op.
        fs::write(&path, vec![0xABu8; 2048]).unwrap();
        preallocate(&path, 512).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().len(),
            2048,
            "preallocate must not shrink",
        );
    }
}
