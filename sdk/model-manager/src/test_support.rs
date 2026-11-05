// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Shared fixtures for unit and integration tests.

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex, MutexGuard, Once};
use std::time::Duration;

use url::Url;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use crate::config::StoreConfig;
use crate::manifest::{ModelFileInfo, ModelManifest, ModelType};
use crate::store::Store;
use crate::transport::{HttpTransport, ReqwestTransport, TransportConfig};

pub fn fast_transport_config() -> TransportConfig {
    TransportConfig {
        connect_timeout: Some(Duration::from_secs(2)),
        read_timeout: Some(Duration::from_secs(5)),
        retries: Some(0),
        retry_backoff: Some(Duration::from_millis(10)),
        proxy_override: None,
    }
}

pub fn fast_transport() -> Arc<dyn HttpTransport> {
    Arc::new(ReqwestTransport::with_config(fast_transport_config()).unwrap())
}

pub fn make_body(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (((i as u32).wrapping_mul(31).wrapping_add(seed as u32)) & 0xff) as u8)
        .collect()
}

pub fn deflate_raw(data: &[u8]) -> Vec<u8> {
    let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

pub fn parse_range(req: &Request) -> (usize, usize) {
    let hdr = req.headers.get("range").expect("Range header");
    let (s, e) = hdr
        .to_str()
        .unwrap()
        .strip_prefix("bytes=")
        .unwrap()
        .split_once('-')
        .unwrap();
    let (start, end): (usize, usize) = (s.parse().unwrap(), e.parse().unwrap());
    (start, end - start + 1)
}

/// 206 with the requested slice, or 200 with all of `body` when unranged.
pub fn range_slice(body: &[u8], req: &Request) -> ResponseTemplate {
    if !req.headers.contains_key("range") {
        return ResponseTemplate::new(200).set_body_bytes(body.to_vec());
    }
    let (start, len) = parse_range(req);
    ResponseTemplate::new(206).set_body_bytes(body[start..start + len].to_vec())
}

pub async fn mount_head(server: &MockServer, at: &str, len: usize) {
    Mock::given(method("HEAD"))
        .and(path(at))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("Content-Length", len.to_string())
                .append_header("Accept-Ranges", "bytes"),
        )
        .mount(server)
        .await;
}

/// Serve `body` at `at` for HEAD plus ranged GET.
pub async fn mount_ranged(server: &MockServer, at: &str, body: impl Into<Vec<u8>>) {
    let body: Arc<Vec<u8>> = Arc::new(body.into());
    mount_head(server, at, body.len()).await;
    Mock::given(method("GET"))
        .and(path(at))
        .respond_with(move |req: &Request| range_slice(&body, req))
        .mount(server)
        .await;
}

pub async fn mount_status(server: &MockServer, verb: &str, at: &str, status: u16) {
    Mock::given(method(verb))
        .and(path(at))
        .respond_with(ResponseTemplate::new(status))
        .mount(server)
        .await;
}

pub fn url(server: &MockServer, at: &str) -> Url {
    Url::parse(&format!("{}{at}", server.uri())).unwrap()
}

pub use crate::source::zip::Method as ZipMethod;

pub fn build_zip(entries: &[(&str, &[u8], ZipMethod)]) -> Vec<u8> {
    let mut zw = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, data, m) in entries {
        let method = match m {
            ZipMethod::Stored => zip::CompressionMethod::Stored,
            ZipMethod::Deflate => zip::CompressionMethod::Deflated,
        };
        let opts = zip::write::SimpleFileOptions::default().compression_method(method);
        zw.start_file(*name, opts).unwrap();
        zw.write_all(data).unwrap();
    }
    zw.finish().unwrap().into_inner()
}

pub mod gguf {
    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    pub fn header_prefix(kv_count: u64) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&kv_count.to_le_bytes());
        out
    }

    /// A v3 header whose `general.file_type` follows a `pad_bytes`-long name.
    pub fn head(ftype: u32, pad_bytes: usize) -> Vec<u8> {
        let mut out = header_prefix(2);
        put_str(&mut out, "general.name");
        out.extend_from_slice(&8u32.to_le_bytes());
        put_str(&mut out, &"x".repeat(pad_bytes));
        put_str(&mut out, "general.file_type");
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&ftype.to_le_bytes());
        out
    }
}

pub fn file_info(name: &str, size: i64) -> ModelFileInfo {
    ModelFileInfo {
        name: name.to_string(),
        downloaded: true,
        size,
    }
}

/// A llama_cpp manifest whose `model_file` holds `quants`.
pub fn manifest<'a>(
    name: &str,
    quants: impl IntoIterator<Item = (&'a str, ModelFileInfo)>,
) -> ModelManifest {
    ModelManifest {
        name: name.to_string(),
        model_name: name.rsplit('/').next().unwrap_or(name).to_string(),
        model_type: ModelType::Llm,
        plugin_id: "llama_cpp".to_string(),
        precision: String::new(),
        model_file: quants
            .into_iter()
            .map(|(q, f)| (q.to_string(), f))
            .collect::<HashMap<_, _>>(),
        mmproj_file: ModelFileInfo::default(),
        tokenizer_file: ModelFileInfo::default(),
        extra_files: Vec::new(),
    }
}

/// Keep the `TempDir` bound: dropping it deletes the store.
pub fn temp_store() -> (Store, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let store = Store::new(StoreConfig::new(tmp.path().to_path_buf())).unwrap();
    (store, tmp)
}

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Serialises env mutation within one test binary and restores every
/// touched variable on drop, including when the test panics.
pub struct EnvGuard {
    saved: Vec<(&'static str, Option<String>)>,
    _lock: MutexGuard<'static, ()>,
}

impl EnvGuard {
    pub fn lock() -> Self {
        Self {
            saved: Vec::new(),
            _lock: ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner()),
        }
    }

    fn save(&mut self, key: &'static str) {
        if !self.saved.iter().any(|(k, _)| *k == key) {
            self.saved.push((key, std::env::var(key).ok()));
        }
    }

    pub fn set(mut self, key: &'static str, value: &str) -> Self {
        self.save(key);
        std::env::set_var(key, value);
        self
    }

    pub fn unset(mut self, key: &'static str) -> Self {
        self.save(key);
        std::env::remove_var(key);
        self
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, prev) in self.saved.drain(..).rev() {
            match prev {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}

/// Pin 16 KiB chunks for the whole test binary. Never restored: executor
/// tests read it concurrently without holding [`EnvGuard`].
pub fn pin_small_chunks() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_var("GENIEX_DL_CHUNK_SIZE", "16384"));
}
