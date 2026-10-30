// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! ZIP64-aware central directory parser driven by HTTP Range reads.
//!
//! Given a `Url` to a remote zip, [`fetch_central_directory`] returns
//! the list of file entries with enough info to issue per-entry range
//! GETs. No byte payload is downloaded — only the EOCD footer, the
//! optional ZIP64 EOCD, the central directory itself (typically a few
//! KB even for 4 GB+ archives), and one 30-byte local file header per
//! entry.
//!
//! Scope is intentionally narrow:
//! - method 0 (STORED) and method 8 (DEFLATE) — every AI Hub qairt
//!   asset today is one or the other.
//! - ZIP64 extensions for entries ≥ 4 GB (Qwen2.5-VL-7B's zip, 4.6 GB
//!   compressed, uses them).
//! - Split archives, encrypted entries, or crazy extra-field layouts
//!   are rejected up front rather than guessed at.
//!
//! Wire-format references:
//! - APPNOTE.TXT 6.3.10
//! - End of central directory record (EOCD): signature 0x06054b50, fixed
//!   22 bytes + variable comment.
//! - ZIP64 EOCD locator: 0x07064b50, fixed 20 bytes, sits directly
//!   before the EOCD.
//! - ZIP64 EOCD: 0x06064b50, fixed 56 bytes + variable extensible data.
//! - Central directory header: 0x02014b50, fixed 46 bytes + name +
//!   extra + comment.
//! - Local file header: 0x04034b50, fixed 30 bytes + name + extra.

use std::sync::Arc;

use url::Url;

use crate::error::{Error, Result};
use crate::transport::HttpTransport;

const EOCD_SIG: u32 = 0x0605_4b50;
const ZIP64_EOCD_LOCATOR_SIG: u32 = 0x0706_4b50;
const ZIP64_EOCD_SIG: u32 = 0x0606_4b50;
const CD_HEADER_SIG: u32 = 0x0201_4b50;
const LOCAL_HEADER_SIG: u32 = 0x0403_4b50;

/// How many bytes from the end to scan for the EOCD signature. The EOCD
/// has a variable-length comment (up to 0xFFFF); practical archives
/// never include one, but we scan a buffer that comfortably covers both
/// it and the ZIP64 locator that sits in front.
const EOCD_SCAN_WINDOW: u64 = 64 * 1024 + 22;

/// Compression methods we understand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Stored,
    Deflate,
}

/// One entry inside a remote zip, reduced to what the executor needs
/// to issue a range GET for its payload bytes.
#[derive(Debug, Clone)]
pub struct ZipEntry {
    /// Raw path-like name from the central directory; the caller is
    /// expected to flatten to basename.
    pub name: String,
    pub method: Method,
    /// Byte offset of the compressed payload, already advanced past
    /// the local header's 30-byte prefix, name, and extra field.
    pub payload_offset: u64,
    pub compressed_size: u64,
    pub uncompressed_size: u64,
    pub is_dir: bool,
}

/// Fetch the central directory of the zip at `url` and return one
/// [`ZipEntry`] per file record.
pub async fn fetch_central_directory(
    transport: &Arc<dyn HttpTransport>,
    url: &Url,
) -> Result<Vec<ZipEntry>> {
    let head = transport.head(url, None).await?;
    let total = head.size;
    if total < 22 {
        return Err(Error::Hub(format!(
            "zip at {url} is {total} bytes; too small to be a valid archive"
        )));
    }

    // EOCD lives near the tail unless a trailing zip comment was used
    // (never in practice for AI Hub assets).
    let tail_len = total.min(EOCD_SCAN_WINDOW);
    let tail_start = total - tail_len;
    let mut tail = Vec::with_capacity(tail_len as usize);
    transport
        .get_range(url, None, tail_start, tail_len, &mut tail)
        .await?;

    let eocd_off_in_tail = find_eocd(&tail)
        .ok_or_else(|| Error::Hub(format!("zip at {url}: EOCD signature not found in tail")))?;
    let eocd = &tail[eocd_off_in_tail..];
    if eocd.len() < 22 {
        return Err(Error::Hub(format!(
            "zip at {url}: EOCD truncated ({} < 22 bytes)",
            eocd.len()
        )));
    }
    let disk_num = read_u16(eocd, 4);
    let disk_with_cd = read_u16(eocd, 6);
    if disk_num != 0 || disk_with_cd != 0 {
        return Err(Error::Hub(format!(
            "zip at {url}: split archives not supported (disk {disk_num}/{disk_with_cd})"
        )));
    }
    let mut cd_size = read_u32(eocd, 12) as u64;
    let mut cd_offset = read_u32(eocd, 16) as u64;
    let mut cd_entries = read_u16(eocd, 10) as u64;

    // Per APPNOTE, the ZIP64 locator sits immediately before the EOCD
    // and is present iff any of (cd_size, cd_offset, cd_entries) is
    // maxed out in the 32-bit EOCD slots.
    let needs_zip64 = cd_size == 0xFFFF_FFFF || cd_offset == 0xFFFF_FFFF || cd_entries == 0xFFFF;
    let eocd_abs_off = tail_start + eocd_off_in_tail as u64;
    if needs_zip64 {
        if eocd_abs_off < 20 {
            return Err(Error::Hub(format!(
                "zip at {url}: ZIP64 locator expected but no room before EOCD"
            )));
        }
        let locator_abs = eocd_abs_off - 20;
        let locator_rel = (locator_abs - tail_start) as usize;
        let locator = &tail[locator_rel..locator_rel + 20];
        if read_u32(locator, 0) != ZIP64_EOCD_LOCATOR_SIG {
            return Err(Error::Hub(format!(
                "zip at {url}: ZIP64 EOCD locator signature missing"
            )));
        }
        let z64_eocd_off = read_u64(locator, 8);

        // The ZIP64 EOCD has a 56-byte fixed prefix plus extensible
        // data we don't need — a 56-byte fetch is enough.
        let z64_end = z64_eocd_off.checked_add(56).ok_or_else(|| {
            Error::Hub(format!(
                "zip at {url}: ZIP64 EOCD offset {z64_eocd_off} overflows"
            ))
        })?;
        if z64_end > total {
            return Err(Error::Hub(format!(
                "zip at {url}: ZIP64 EOCD range ({z64_eocd_off}+56) exceeds archive size ({total})"
            )));
        }
        let mut z64 = Vec::with_capacity(56);
        transport
            .get_range(url, None, z64_eocd_off, 56, &mut z64)
            .await?;
        if z64.len() != 56 || read_u32(&z64, 0) != ZIP64_EOCD_SIG {
            return Err(Error::Hub(format!(
                "zip at {url}: ZIP64 EOCD signature missing"
            )));
        }
        cd_entries = read_u64(&z64, 32);
        cd_size = read_u64(&z64, 40);
        cd_offset = read_u64(&z64, 48);
    }

    if cd_size == 0 {
        return Ok(Vec::new());
    }
    let cd_end = cd_offset.checked_add(cd_size).ok_or_else(|| {
        Error::Hub(format!(
            "zip at {url}: central directory range ({cd_offset}+{cd_size}) overflows"
        ))
    })?;
    if cd_end > total {
        return Err(Error::Hub(format!(
            "zip at {url}: central directory range ({cd_offset}+{cd_size}) exceeds archive size ({total})"
        )));
    }

    // The CD itself is a few KB even for multi-GB archives (3.7 KB for
    // the 27-entry Qwen2.5-VL-7B asset), so one range fetch is fine.
    let mut cd = Vec::with_capacity(cd_size as usize);
    transport
        .get_range(url, None, cd_offset, cd_size, &mut cd)
        .await?;
    if cd.len() as u64 != cd_size {
        return Err(Error::Hub(format!(
            "zip at {url}: CD fetch short ({} < {cd_size})",
            cd.len()
        )));
    }

    // CD header layout: 46 fixed bytes + name_len + extra_len + comment_len.
    let mut entries: Vec<ZipEntry> = Vec::with_capacity(cd_entries.min(cd_size / 46) as usize);
    let mut cursor: usize = 0;
    while cursor + 46 <= cd.len() {
        if read_u32(&cd, cursor) != CD_HEADER_SIG {
            return Err(Error::Hub(format!(
                "zip at {url}: CD header signature missing at offset {cursor}"
            )));
        }
        let method_raw = read_u16(&cd, cursor + 10);
        let mut compressed_size = read_u32(&cd, cursor + 20) as u64;
        let mut uncompressed_size = read_u32(&cd, cursor + 24) as u64;
        let name_len = read_u16(&cd, cursor + 28) as usize;
        let extra_len = read_u16(&cd, cursor + 30) as usize;
        let comment_len = read_u16(&cd, cursor + 32) as usize;
        let mut local_header_offset = read_u32(&cd, cursor + 42) as u64;

        let total_len = 46 + name_len + extra_len + comment_len;
        if cursor + total_len > cd.len() {
            return Err(Error::Hub(format!(
                "zip at {url}: CD entry at offset {cursor} truncated"
            )));
        }

        let name_bytes = &cd[cursor + 46..cursor + 46 + name_len];
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| Error::Hub("zip entry name is not UTF-8".to_string()))?
            .to_string();

        // Per APPNOTE, a ZIP64 extra field (tag 0x0001) supplies 64-bit
        // values for any 32-bit CD slot that reads 0xFFFFFFFF — only
        // those overflowed slots are included, in this order:
        // uncompressed, compressed, local_header_offset, disk.
        let extra = &cd[cursor + 46 + name_len..cursor + 46 + name_len + extra_len];
        let mut ex_cursor = 0usize;
        while ex_cursor + 4 <= extra.len() {
            let tag = read_u16(extra, ex_cursor);
            let ex_size = read_u16(extra, ex_cursor + 2) as usize;
            if ex_cursor + 4 + ex_size > extra.len() {
                return Err(Error::Hub(format!(
                    "zip entry {name:?}: extra field at offset {ex_cursor} exceeds entry boundary"
                )));
            }
            if tag == 0x0001 {
                let read_zip64 = |offset| {
                    read_zip64_u64(extra, offset).ok_or_else(|| {
                        Error::Hub(format!("zip entry {name:?}: truncated ZIP64 extra field"))
                    })
                };
                let mut ez = ex_cursor + 4;
                if uncompressed_size == 0xFFFF_FFFF {
                    uncompressed_size = read_zip64(ez)?;
                    ez += 8;
                }
                if compressed_size == 0xFFFF_FFFF {
                    compressed_size = read_zip64(ez)?;
                    ez += 8;
                }
                if local_header_offset == 0xFFFF_FFFF {
                    local_header_offset = read_zip64(ez)?;
                }
                break;
            }
            ex_cursor += 4 + ex_size;
        }

        let method = match method_raw {
            0 => Method::Stored,
            8 => Method::Deflate,
            other => {
                return Err(Error::Hub(format!(
                    "zip entry {name:?}: unsupported compression method {other}"
                )))
            }
        };

        let is_dir = name.ends_with('/') && uncompressed_size == 0;

        entries.push(ZipEntry {
            name,
            method,
            // Local header offset for now; resolved to the payload
            // byte offset below after fetching each local header.
            payload_offset: local_header_offset,
            compressed_size,
            uncompressed_size,
            is_dir,
        });
        cursor += total_len;
    }

    // payload_offset = local_header_offset + 30 + name_len + extra_len;
    // those two length fields live in the local header, not the CD, so
    // one 30-byte GET per entry is unavoidable.
    let mut resolved: Vec<ZipEntry> = Vec::with_capacity(entries.len());
    for mut e in entries.drain(..) {
        if e.is_dir {
            resolved.push(e);
            continue;
        }
        let header_abs = e.payload_offset;
        let header_end = header_abs.checked_add(30).ok_or_else(|| {
            Error::Hub(format!(
                "zip entry {:?}: local header offset {header_abs} overflows",
                e.name
            ))
        })?;
        if header_end > total {
            return Err(Error::Hub(format!(
                "zip entry {:?}: local header range ({header_abs}+30) exceeds archive size ({total})",
                e.name
            )));
        }
        let mut hdr = Vec::with_capacity(30);
        transport
            .get_range(url, None, header_abs, 30, &mut hdr)
            .await?;
        if hdr.len() != 30 || read_u32(&hdr, 0) != LOCAL_HEADER_SIG {
            return Err(Error::Hub(format!(
                "zip entry {:?}: local header signature missing at offset {header_abs}",
                e.name
            )));
        }
        let lh_name_len = read_u16(&hdr, 26) as u64;
        let lh_extra_len = read_u16(&hdr, 28) as u64;
        e.payload_offset = header_abs
            .checked_add(30)
            .and_then(|offset| offset.checked_add(lh_name_len))
            .and_then(|offset| offset.checked_add(lh_extra_len))
            .ok_or_else(|| {
                Error::Hub(format!("zip entry {:?}: payload offset overflows", e.name))
            })?;
        let payload_end = e
            .payload_offset
            .checked_add(e.compressed_size)
            .ok_or_else(|| {
                Error::Hub(format!("zip entry {:?}: payload range overflows", e.name))
            })?;
        if payload_end > total {
            return Err(Error::Hub(format!(
                "zip entry {:?}: payload range ({}+{}) exceeds archive size ({total})",
                e.name, e.payload_offset, e.compressed_size
            )));
        }
        resolved.push(e);
    }

    Ok(resolved)
}

fn find_eocd(buf: &[u8]) -> Option<usize> {
    if buf.len() < 22 {
        return None;
    }
    let upper_start = buf.len() - 22;
    (0..=upper_start)
        .rev()
        .find(|&i| read_u32(buf, i) == EOCD_SIG)
}

fn read_u16(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn read_u64(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        buf[off],
        buf[off + 1],
        buf[off + 2],
        buf[off + 3],
        buf[off + 4],
        buf[off + 5],
        buf[off + 6],
        buf[off + 7],
    ])
}

fn read_zip64_u64(buf: &[u8], off: usize) -> Option<u64> {
    let bytes = buf.get(off..off.checked_add(8)?)?;
    Some(read_u64(bytes, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{build_zip, fast_transport, mount_ranged, url, ZipMethod};
    use crate::transport::LocalFileTransport;
    use wiremock::MockServer;

    fn stored_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let entries: Vec<_> = entries
            .iter()
            .map(|(n, d)| (*n, *d, ZipMethod::Stored))
            .collect();
        build_zip(&entries)
    }

    async fn parse_remote(body: Vec<u8>) -> Result<Vec<ZipEntry>> {
        let server = MockServer::start().await;
        mount_ranged(&server, "/a.zip", body).await;
        fetch_central_directory(&fast_transport(), &url(&server, "/a.zip")).await
    }

    async fn parse_local(body: Vec<u8>) -> Result<Vec<ZipEntry>> {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("malformed.zip");
        std::fs::write(&path, body).unwrap();
        let transport: Arc<dyn HttpTransport> = Arc::new(LocalFileTransport::open(&path).unwrap());
        let url = Url::parse("file:///malformed.zip").unwrap();
        fetch_central_directory(&transport, &url).await
    }

    fn find(body: &[u8], sig: &[u8; 4], last: bool) -> usize {
        let mut windows = body.windows(4);
        let hit = if last {
            windows.rposition(|w| w == sig)
        } else {
            windows.position(|w| w == sig)
        };
        hit.unwrap()
    }

    fn corrupt_zip64_extra(body: &mut Vec<u8>) {
        let central = find(body, b"PK\x01\x02", false);
        let name_len = u16::from_le_bytes([body[central + 28], body[central + 29]]) as usize;
        assert_eq!(
            u16::from_le_bytes([body[central + 30], body[central + 31]]),
            0
        );
        let extra_at = central + 46 + name_len;
        body.splice(extra_at..extra_at, [1, 0, 0, 0]);
        body[central + 20..central + 28].copy_from_slice(&[0xFF; 8]);
        body[central + 30..central + 32].copy_from_slice(&4u16.to_le_bytes());
        let eocd = find(body, b"PK\x05\x06", true);
        let cd_size = u32::from_le_bytes(body[eocd + 12..eocd + 16].try_into().unwrap());
        body[eocd + 12..eocd + 16].copy_from_slice(&(cd_size + 4).to_le_bytes());
    }

    fn add_zip64_eocd(body: &mut Vec<u8>, cd_entries: u64, locator_off: Option<u64>) {
        let eocd = find(body, b"PK\x05\x06", true);
        let cd_size = u32::from_le_bytes(body[eocd + 12..eocd + 16].try_into().unwrap());
        let cd_offset = u32::from_le_bytes(body[eocd + 16..eocd + 20].try_into().unwrap());
        let mut z64 = b"PK\x06\x06".to_vec();
        z64.extend_from_slice(&44u64.to_le_bytes());
        z64.extend_from_slice(&[45, 0, 45, 0]);
        z64.extend_from_slice(&[0; 8]);
        z64.extend_from_slice(&1u64.to_le_bytes());
        z64.extend_from_slice(&cd_entries.to_le_bytes());
        z64.extend_from_slice(&u64::from(cd_size).to_le_bytes());
        z64.extend_from_slice(&u64::from(cd_offset).to_le_bytes());
        z64.extend_from_slice(b"PK\x06\x07");
        z64.extend_from_slice(&0u32.to_le_bytes());
        z64.extend_from_slice(&locator_off.unwrap_or(eocd as u64).to_le_bytes());
        z64.extend_from_slice(&1u32.to_le_bytes());
        let new_eocd = eocd + z64.len();
        body.splice(eocd..eocd, z64);
        body[new_eocd + 12..new_eocd + 20].copy_from_slice(&[0xFF; 8]);
    }

    fn set_central_u32(body: &mut [u8], field: usize, v: u32) {
        let central = find(body, b"PK\x01\x02", false);
        body[central + field..central + field + 4].copy_from_slice(&v.to_le_bytes());
    }

    #[tokio::test]
    async fn parses_stored_and_deflate_zips() {
        let body = stored_zip(&[("model.bin", b"HELLO"), ("tokenizer.json", b"{}")]);
        let entries = parse_remote(body).await.unwrap();
        assert!(entries.iter().any(|e| e.name == "tokenizer.json"));
        let bin = entries.iter().find(|e| e.name == "model.bin").unwrap();
        assert_eq!(
            (bin.method, bin.uncompressed_size, bin.compressed_size),
            (Method::Stored, 5, 5)
        );

        let big = vec![b'A'; 8192];
        let body = build_zip(&[("shard.bin", &big, ZipMethod::Deflate)]);
        let entries = parse_remote(body).await.unwrap();
        let e = entries.iter().find(|e| e.name == "shard.bin").unwrap();
        assert_eq!((e.method, e.uncompressed_size), (Method::Deflate, 8192));
        assert!(e.compressed_size < e.uncompressed_size);
    }

    #[tokio::test]
    async fn rejects_malformed_tail() {
        let mut body = vec![0u8; 1024];
        body[..4].copy_from_slice(b"PK\x03\x04");
        let err = parse_remote(body).await.unwrap_err();
        assert!(err.to_string().contains("EOCD"), "{err}");
    }

    #[tokio::test]
    async fn rejects_malformed_metadata() {
        type Mutate = fn(&mut Vec<u8>);
        let cases: [(Mutate, &str); 5] = [
            (corrupt_zip64_extra, "ZIP64 extra"),
            (|b| add_zip64_eocd(b, 1, Some(u64::MAX - 10)), "overflows"),
            (
                |b| add_zip64_eocd(b, 1, Some(1 << 40)),
                "exceeds archive size",
            ),
            (
                |b| set_central_u32(b, 42, 0x7FFF_FFFF),
                "local header range",
            ),
            (|b| set_central_u32(b, 20, 0x7FFF_FFFF), "payload range"),
        ];
        for (mutate, want) in cases {
            let mut body = stored_zip(&[("model.bin", b"HELLO")]);
            mutate(&mut body);
            let err = parse_local(body).await.unwrap_err();
            assert!(err.to_string().contains(want), "want {want:?}: {err}");
        }
    }

    #[tokio::test]
    async fn tolerates_oversized_zip64_entry_count() {
        let mut body = stored_zip(&[("model.bin", b"HELLO")]);
        add_zip64_eocd(&mut body, u64::MAX, None);
        assert_eq!(parse_local(body).await.unwrap().len(), 1);
    }
}
