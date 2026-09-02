// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Read `general.file_type` out of a GGUF header when the filename carries no
//! quant tag. The key sits behind the tokenizer arrays — 2-7 MB into real
//! models — so the header is walked in appending ranged reads and abandoned at
//! [`MAX_PROBE_BYTES`].

use std::path::Path;

use url::Url;

use crate::transport::HttpTransport;

const CHUNK_BYTES: u64 = 2 * 1024 * 1024;
const MAX_PROBE_BYTES: u64 = 8 * 1024 * 1024;
const MAX_ARRAY_DEPTH: u32 = 4;

const MAGIC: &[u8] = b"GGUF";
const KEY_FILE_TYPE: &[u8] = b"general.file_type";

const TY_UINT8: u32 = 0;
const TY_INT8: u32 = 1;
const TY_UINT16: u32 = 2;
const TY_INT16: u32 = 3;
const TY_UINT32: u32 = 4;
const TY_INT32: u32 = 5;
const TY_FLOAT32: u32 = 6;
const TY_BOOL: u32 = 7;
const TY_STRING: u32 = 8;
const TY_ARRAY: u32 = 9;
const TY_UINT64: u32 = 10;
const TY_INT64: u32 = 11;
const TY_FLOAT64: u32 = 12;

pub(crate) async fn probe_quant(
    transport: &dyn HttpTransport,
    url: &Url,
    auth: Option<&str>,
    file_size: u64,
) -> Option<String> {
    let cap = MAX_PROBE_BYTES.min(file_size);
    let mut buf: Vec<u8> = Vec::new();
    while (buf.len() as u64) < cap {
        let have = buf.len() as u64;
        let want = cap.min(have + CHUNK_BYTES) - have;
        transport
            .get_range(url, auth, have, want, &mut buf)
            .await
            .ok()?;
        match scan(&buf) {
            Scan::Found(q) => return Some(q.to_string()),
            Scan::Absent => return None,
            Scan::NeedMore => continue,
        }
    }
    None
}

pub(crate) fn quant_from_file(path: &Path) -> Option<String> {
    use std::io::Read;
    let mut buf = Vec::new();
    std::fs::File::open(path)
        .ok()?
        .take(MAX_PROBE_BYTES)
        .read_to_end(&mut buf)
        .ok()?;
    match scan(&buf) {
        Scan::Found(q) => Some(q.to_string()),
        _ => None,
    }
}

enum Scan {
    Found(&'static str),
    Absent,
    NeedMore,
}

fn scan(buf: &[u8]) -> Scan {
    walk_kv(buf).unwrap_or(Scan::NeedMore)
}

fn walk_kv(buf: &[u8]) -> Option<Scan> {
    let mut r = Reader {
        buf,
        off: 0,
        wide: true,
    };
    if r.take(MAGIC.len())? != MAGIC {
        return Some(Scan::Absent);
    }
    let version = r.u32()?;
    if !(1..=3).contains(&version) {
        return Some(Scan::Absent);
    }
    r.wide = version >= 2;
    let _tensor_count = r.varlen()?;
    let kv_count = r.varlen()?;

    for _ in 0..kv_count {
        let key = r.str_bytes()?;
        let ty = r.u32()?;
        if key == KEY_FILE_TYPE && ty == TY_UINT32 {
            return Some(match ftype_name(r.u32()?) {
                Some(q) => Scan::Found(q),
                None => Scan::Absent,
            });
        }
        r.skip_value(ty, 0)?;
    }
    Some(Scan::Absent)
}

struct Reader<'a> {
    buf: &'a [u8],
    off: usize,
    wide: bool,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.off.checked_add(n)?;
        let s = self.buf.get(self.off..end)?;
        self.off = end;
        Some(s)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn varlen(&mut self) -> Option<u64> {
        if self.wide {
            Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
        } else {
            Some(self.u32()? as u64)
        }
    }

    fn str_bytes(&mut self) -> Option<&'a [u8]> {
        let n = usize::try_from(self.varlen()?).ok()?;
        self.take(n)
    }

    fn skip_value(&mut self, ty: u32, depth: u32) -> Option<()> {
        if let Some(w) = scalar_width(ty) {
            self.take(w)?;
            return Some(());
        }
        match ty {
            TY_STRING => {
                self.str_bytes()?;
            }
            TY_ARRAY => {
                if depth >= MAX_ARRAY_DEPTH {
                    return None;
                }
                let elem = self.u32()?;
                let count = self.varlen()?;
                if let Some(w) = scalar_width(elem) {
                    let total = count.checked_mul(w as u64)?;
                    self.take(usize::try_from(total).ok()?)?;
                } else {
                    for _ in 0..count {
                        self.skip_value(elem, depth + 1)?;
                    }
                }
            }
            _ => return None,
        }
        Some(())
    }
}

fn scalar_width(ty: u32) -> Option<usize> {
    Some(match ty {
        TY_UINT8 | TY_INT8 | TY_BOOL => 1,
        TY_UINT16 | TY_INT16 => 2,
        TY_UINT32 | TY_INT32 | TY_FLOAT32 => 4,
        TY_UINT64 | TY_INT64 | TY_FLOAT64 => 8,
        _ => return None,
    })
}

/// Mirrors `llama_ftype` in llama.cpp's `include/llama.h`.
fn ftype_name(v: u32) -> Option<&'static str> {
    Some(match v {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4",
        39 => "NVFP4",
        40 => "Q1_0",
        41 => "Q2_0",
        _ => return None,
    })
}

#[cfg(test)]
pub(crate) mod testdata {
    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    pub(crate) fn gguf_header_prefix(kv_count: u64) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&kv_count.to_le_bytes());
        out
    }

    pub(crate) fn gguf_head(ftype: u32, pad_bytes: usize) -> Vec<u8> {
        let mut out = gguf_header_prefix(2);
        put_str(&mut out, "general.name");
        out.extend_from_slice(&8u32.to_le_bytes());
        put_str(&mut out, &"x".repeat(pad_bytes));
        put_str(&mut out, "general.file_type");
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&ftype.to_le_bytes());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::testdata::{gguf_head, gguf_header_prefix};
    use super::*;

    fn found(buf: &[u8]) -> Option<&'static str> {
        match scan(buf) {
            Scan::Found(q) => Some(q),
            _ => None,
        }
    }

    #[test]
    fn reads_file_type_from_a_v3_head() {
        assert_eq!(found(&gguf_head(15, 0)), Some("Q4_K_M"));
        assert_eq!(found(&gguf_head(1, 0)), Some("F16"));
        assert_eq!(found(&gguf_head(32, 64)), Some("BF16"));
    }

    #[test]
    fn skips_over_a_preceding_token_array() {
        let mut out = gguf_header_prefix(2);
        let key = b"tokenizer.ggml.tokens";
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key);
        out.extend_from_slice(&TY_ARRAY.to_le_bytes());
        out.extend_from_slice(&TY_STRING.to_le_bytes());
        out.extend_from_slice(&512u64.to_le_bytes());
        for i in 0..512u64 {
            let tok = format!("tok{i}");
            out.extend_from_slice(&(tok.len() as u64).to_le_bytes());
            out.extend_from_slice(tok.as_bytes());
        }
        out.extend_from_slice(&(KEY_FILE_TYPE.len() as u64).to_le_bytes());
        out.extend_from_slice(KEY_FILE_TYPE);
        out.extend_from_slice(&TY_UINT32.to_le_bytes());
        out.extend_from_slice(&18u32.to_le_bytes());
        assert_eq!(found(&out), Some("Q6_K"));
    }

    #[test]
    fn truncated_head_asks_for_more() {
        let full = gguf_head(15, 0);
        assert!(matches!(scan(&full[..full.len() - 2]), Scan::NeedMore));
    }

    #[test]
    fn non_gguf_bytes_are_final() {
        assert!(matches!(scan(b"PK\x03\x04not a model"), Scan::Absent));
        let mut big_endian = b"GGUF".to_vec();
        big_endian.extend_from_slice(&3u32.to_be_bytes());
        assert!(matches!(scan(&big_endian), Scan::Absent));
    }

    #[test]
    fn unknown_or_absent_file_type_is_final() {
        assert!(matches!(scan(&gguf_head(9999, 0)), Scan::Absent));
        assert!(matches!(scan(&gguf_header_prefix(0)), Scan::Absent));
    }

    #[test]
    fn absurd_lengths_do_not_allocate() {
        let mut out = gguf_header_prefix(1);
        out.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(scan(&out), Scan::NeedMore));
    }

    #[tokio::test]
    #[ignore = "hits huggingface.co"]
    async fn live_probe_reads_real_headers() {
        const BASE: &str = "https://huggingface.co";
        let t = crate::transport::ReqwestTransport::new().unwrap();
        for (path, size, want) in [
            (
                "ggml-org/SmolVLM-Instruct-GGUF/resolve/main/SmolVLM-Instruct-Q4_K_M.gguf",
                1112242368u64,
                "Q4_K_M",
            ),
            (
                "ggml-org/SmolVLM-Instruct-GGUF/resolve/main/SmolVLM-Instruct-Q8_0.gguf",
                1927383680,
                "Q8_0",
            ),
            (
                "ggml-org/SmolVLM-Instruct-GGUF/resolve/main/SmolVLM-Instruct-f16.gguf",
                3626088320,
                "F16",
            ),
            (
                "unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
                5027783296,
                "Q4_K_M",
            ),
        ] {
            let url = Url::parse(&format!("{BASE}/{path}")).unwrap();
            assert_eq!(
                probe_quant(&t, &url, None, size).await.as_deref(),
                Some(want),
                "{path}"
            );
        }
    }
}
