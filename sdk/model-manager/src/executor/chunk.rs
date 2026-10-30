// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Chunk planner + `.progress` bitmap: one byte per chunk, `0x01` = written.

use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::Result;

pub(crate) const MIN_CHUNK_SIZE: u64 = 16 * 1024 * 1024;
pub(crate) const MAX_CHUNKS_PER_FILE: u64 = 128;
pub(crate) const PROGRESS_DONE_BYTE: u8 = 0x01;

const ENV_CHUNK_SIZE: &str = "GENIEX_DL_CHUNK_SIZE";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkRange {
    pub index: usize,
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkPlan {
    pub chunk_size: u64,
    pub chunks: Vec<ChunkRange>,
}

/// A `file_size` of 0 yields a zero-chunk plan; the caller still creates the empty file.
pub fn plan_chunks(file_size: u64) -> ChunkPlan {
    plan_chunks_with_floor(file_size, effective_min_chunk_size())
}

/// An empty file still gets one zero-length chunk so its marker records completion.
pub(crate) fn single_chunk_plan(file_size: u64) -> ChunkPlan {
    ChunkPlan {
        chunk_size: file_size,
        chunks: vec![ChunkRange {
            index: 0,
            offset: 0,
            len: file_size,
        }],
    }
}

pub(crate) fn plan_chunks_with_floor(file_size: u64, min_chunk_size: u64) -> ChunkPlan {
    let min = min_chunk_size.max(1);
    let chunk_size = if file_size == 0 {
        min
    } else {
        core::cmp::max(min, file_size / MAX_CHUNKS_PER_FILE)
    };
    let mut chunks = Vec::new();
    if file_size > 0 {
        let n = file_size.div_ceil(chunk_size);
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
    ChunkPlan { chunk_size, chunks }
}

fn effective_min_chunk_size() -> u64 {
    crate::config::env_positive(ENV_CHUNK_SIZE).unwrap_or(MIN_CHUNK_SIZE)
}

/// A length mismatch means the chunk size changed since the last attempt; start over.
pub(crate) fn load_or_init_bitmap(marker_path: &Path, plan: &ChunkPlan) -> Result<Vec<u8>> {
    let expected = plan.chunks.len();
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

pub(crate) fn pending_chunks(plan: &ChunkPlan, bitmap: &[u8]) -> Vec<ChunkRange> {
    plan.chunks
        .iter()
        .filter(|c| bitmap.get(c.index).copied().unwrap_or(0) != PROGRESS_DONE_BYTE)
        .copied()
        .collect()
}

pub(crate) fn bytes_already_done(plan: &ChunkPlan, bitmap: &[u8]) -> u64 {
    plan.chunks
        .iter()
        .filter(|c| bitmap.get(c.index).copied().unwrap_or(0) == PROGRESS_DONE_BYTE)
        .map(|c| c.len)
        .sum()
}

/// Positional write so concurrent workers don't step on each other.
pub(crate) fn mark_chunk_done(marker_path: &Path, index: usize) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .read(true)
        .open(marker_path)?;
    f.seek(SeekFrom::Start(index as u64))?;
    f.write_all(&[PROGRESS_DONE_BYTE])?;
    f.flush()?;
    Ok(())
}

pub(crate) fn bitmap_complete(bitmap: &[u8]) -> bool {
    !bitmap.is_empty() && bitmap.iter().all(|b| *b == PROGRESS_DONE_BYTE)
}

pub(crate) fn marker_complete(marker: &Path, output: &Path, size: u64) -> bool {
    let bitmap = fs::read(marker).unwrap_or_default();
    bitmap_complete(&bitmap) && fs::metadata(output).map(|m| m.len()).ok() == Some(size)
}

/// Pre-sized so range workers can seek + write without racing on file growth.
pub(crate) fn preallocate(output_path: &Path, file_size: u64) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let f = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .open(output_path)?;
    let cur = f.metadata()?.len();
    if cur != file_size {
        f.set_len(file_size)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn plan_chunks_tiles_the_file() {
        const MIN: u64 = MIN_CHUNK_SIZE;
        const GIB4: u64 = 4 << 30;
        // (size, chunks, chunk_size, last chunk len)
        let cases = [
            (1024, 1, MIN, 1024),
            (MIN, 1, MIN, MIN),
            (2 * MIN, 2, MIN, MIN),
            (MIN + 123, 2, MIN, 123),
            (
                GIB4,
                MAX_CHUNKS_PER_FILE as usize,
                GIB4 / MAX_CHUNKS_PER_FILE,
                GIB4 / MAX_CHUNKS_PER_FILE,
            ),
        ];
        for (size, n, chunk_size, last_len) in cases {
            let plan = plan_chunks_with_floor(size, MIN);
            assert_eq!(plan.chunks.len(), n, "size {size}");
            assert_eq!(plan.chunk_size, chunk_size, "size {size}");
            assert_eq!(plan.chunks[0].offset, 0, "size {size}");
            let last = plan.chunks.last().unwrap();
            assert_eq!(
                (last.offset, last.len),
                ((n as u64 - 1) * chunk_size, last_len),
                "size {size}"
            );
            assert_eq!(
                plan.chunks.iter().map(|c| c.len).sum::<u64>(),
                size,
                "size {size}"
            );
        }
        assert!(plan_chunks_with_floor(0, MIN).chunks.is_empty());
    }

    #[test]
    fn bitmap_roundtrip_and_mark() {
        let dir = tempdir().unwrap();
        let marker = dir.path().join("m.progress");
        let plan = plan_chunks_with_floor(3 * MIN_CHUNK_SIZE, MIN_CHUNK_SIZE);
        let bitmap = load_or_init_bitmap(&marker, &plan).unwrap();
        assert_eq!(bitmap, [0, 0, 0]);
        assert_eq!(pending_chunks(&plan, &bitmap).len(), 3);
        assert_eq!(bytes_already_done(&plan, &bitmap), 0);

        mark_chunk_done(&marker, 1).unwrap();
        let bitmap = load_or_init_bitmap(&marker, &plan).unwrap();
        assert_eq!(bitmap, [0, PROGRESS_DONE_BYTE, 0]);
        assert_eq!(pending_chunks(&plan, &bitmap).len(), 2);
        assert_eq!(bytes_already_done(&plan, &bitmap), MIN_CHUNK_SIZE);
        assert!(!bitmap_complete(&bitmap));

        mark_chunk_done(&marker, 0).unwrap();
        mark_chunk_done(&marker, 2).unwrap();
        let bitmap = load_or_init_bitmap(&marker, &plan).unwrap();
        assert!(bitmap_complete(&bitmap));
        assert!(pending_chunks(&plan, &bitmap).is_empty());
    }

    #[test]
    fn length_mismatch_triggers_reinit() {
        let dir = tempdir().unwrap();
        let marker = dir.path().join("m.progress");
        fs::write(&marker, [PROGRESS_DONE_BYTE; 7]).unwrap();
        let plan = plan_chunks_with_floor(3 * MIN_CHUNK_SIZE, MIN_CHUNK_SIZE);
        assert_eq!(load_or_init_bitmap(&marker, &plan).unwrap(), [0, 0, 0]);
    }

    #[test]
    fn preallocate_sets_exact_size() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("f.bin");
        preallocate(&path, 1024).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), 1024);
        fs::write(&path, [0xABu8; 2048]).unwrap();
        preallocate(&path, 512).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), 512);
    }
}
