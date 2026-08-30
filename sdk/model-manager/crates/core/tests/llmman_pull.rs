// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! End-to-end pull through a mock `llmman serve` daemon.
//!
//! Exercises the full [`LlmmanSource`] contract against a wiremock stand-in
//! for the daemon plus a stub `llmman` executable:
//!
//! 1. `GET /api/version` liveness probe.
//! 2. `POST /api/pull` streaming NDJSON — manifest line, byte-count
//!    progress lines, terminal `{"status":"success"}`.
//! 3. `llmman resolve --no-pull <ref>` for the on-disk path.
//! 4. The executor adopting that path into GenieX's own store, and the
//!    published `geniex.json`.
//!
//! Also covers the two failure modes a user is most likely to hit: no
//! daemon listening, and an in-band error line mid-stream.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use model_manager_core::config::StoreConfig;
use model_manager_core::error::Error;
use model_manager_core::executor::{FileProgress, ProgressCallback};
use model_manager_core::manifest::{ModelManifest, ModelType};
use model_manager_core::manifest_builder::ManifestHint;
use model_manager_core::pull::pull_with_source;
use model_manager_core::source::llmman::LlmmanSource;
use model_manager_core::source::ModelSource;
use model_manager_core::store::Store;
use model_manager_core::transport::{HttpTransport, ReqwestTransport, TransportConfig};
use tempfile::{tempdir, TempDir};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const MODEL_BYTES: &[u8] = b"gguf-weights-payload-0123456789";
const MMPROJ_BYTES: &[u8] = b"mmproj-payload";

fn fast_transport() -> Arc<dyn HttpTransport> {
    Arc::new(
        ReqwestTransport::with_config(TransportConfig {
            connect_timeout: Some(Duration::from_secs(2)),
            read_timeout: Some(Duration::from_secs(5)),
            retries: Some(0),
            retry_backoff: Some(Duration::from_millis(10)),
            proxy_override: None,
        })
        .unwrap(),
    )
}

fn make_store(root: &std::path::Path) -> Store {
    Store::new(StoreConfig::new(root.to_path_buf())).unwrap()
}

/// A stub standing in for the real `llmman` CLI: prints the one JSON line
/// `llmman resolve` contracts to emit on stdout, plus a diagnostic on
/// stderr (which the source must ignore).
fn stub_llmman(dir: &TempDir, json: &str) -> String {
    #[cfg(unix)]
    {
        let script = dir.path().join("llmman");
        std::fs::write(
            &script,
            format!("#!/bin/sh\necho 'resolving...' >&2\necho '{json}'\n"),
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        script.display().to_string()
    }
    #[cfg(not(unix))]
    {
        let script = dir.path().join("llmman.bat");
        std::fs::write(&script, format!("@echo off\r\necho {json}\r\n")).unwrap();
        script.display().to_string()
    }
}

/// Mock daemon that answers the liveness probe and streams `body` from
/// `/api/pull`.
async fn mock_daemon(body: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/version"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(r#"{"version":"0.1.0","exe":"/usr/local/bin/llmman","pid":4242}"#),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/pull"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/x-ndjson")
                .set_body_string(body.to_string()),
        )
        .mount(&server)
        .await;
    server
}

#[tokio::test]
async fn pull_streams_progress_then_adopts_the_resolved_gguf() {
    // llmman's own wire format: aggregate byte counts, no per-layer digest.
    let stream = concat!(
        "{\"status\":\"pulling manifest\"}\n",
        "{\"status\":\"pulling\",\"total\":31,\"completed\":0}\n",
        "{\"status\":\"pulling\",\"total\":31,\"completed\":20}\n",
        "{\"status\":\"pulling\",\"total\":31,\"completed\":31}\n",
        "{\"status\":\"success\"}\n",
    );
    let server = mock_daemon(stream).await;

    // Stand in for llmman's own store: the bytes it "downloaded".
    let llmman_store = tempdir().unwrap();
    let model = llmman_store.path().join("gemma-3-4b-it-Q4_K_M.gguf");
    let mmproj = llmman_store.path().join("mmproj-F16.gguf");
    std::fs::write(&model, MODEL_BYTES).unwrap();
    std::fs::write(&mmproj, MMPROJ_BYTES).unwrap();

    let bin_dir = tempdir().unwrap();
    let binary = stub_llmman(
        &bin_dir,
        &format!(
            r#"{{"reference":"docker.io/ai/gemma3:latest","path":"{}","format":"gguf","mmproj":"{}"}}"#,
            model.display(),
            mmproj.display()
        ),
    );

    let src = LlmmanSource::new(
        "docker.io/ai/gemma3:latest".to_string(),
        "ai/gemma3".to_string(),
        ManifestHint::default(),
    )
    .with_endpoint_and_binary(server.uri(), binary);

    // Capture what the user's progress bar would have seen.
    let seen: Arc<std::sync::Mutex<Vec<(i64, i64)>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = seen.clone();
    let cb: ProgressCallback = Box::new(move |files: &[FileProgress]| {
        let mut g = sink.lock().unwrap();
        for f in files {
            g.push((f.downloaded_bytes, f.total_bytes));
        }
        true
    });

    let cache = tempdir().unwrap();
    let store = make_store(cache.path());
    pull_with_source(
        &store,
        "ai/gemma3",
        Box::new(src),
        fast_transport(),
        Some(&cb),
    )
    .await
    .expect("pull");

    // The daemon's aggregate progress reached the caller: -1 is the
    // "total unknown" sentinel emitted for the manifest-only line.
    let progress = seen.lock().unwrap().clone();
    assert!(
        progress.contains(&(0, -1)) && progress.contains(&(20, 31)) && progress.contains(&(31, 31)),
        "expected llmman's aggregate counts to be forwarded, got {progress:?}"
    );

    // Files landed in GenieX's store with their llmman-side names.
    let dir = cache.path().join("models").join("ai").join("gemma3");
    let dest_model = dir.join("gemma-3-4b-it-Q4_K_M.gguf");
    let dest_mmproj = dir.join("mmproj-F16.gguf");
    assert_eq!(std::fs::read(&dest_model).unwrap(), MODEL_BYTES);
    assert_eq!(std::fs::read(&dest_mmproj).unwrap(), MMPROJ_BYTES);

    // Adopted by hard link, not copied: one inode, two names, so a model
    // shared with llmman costs its bytes once.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let a = std::fs::metadata(&model).unwrap();
        let b = std::fs::metadata(&dest_model).unwrap();
        assert_eq!(
            (a.dev(), a.ino()),
            (b.dev(), b.ino()),
            "expected a hard link"
        );
        assert_eq!(b.nlink(), 2);
    }

    // Published manifest: quant read off the filename, VLM inferred from
    // the mmproj companion, GGUF served by llama_cpp.
    let raw = std::fs::read_to_string(dir.join("geniex.json")).unwrap();
    let manifest: ModelManifest = serde_json::from_str(&raw).unwrap();
    assert_eq!(manifest.name, "ai/gemma3");
    assert_eq!(manifest.model_name, "gemma3");
    assert_eq!(manifest.model_type, ModelType::Vlm);
    assert_eq!(manifest.plugin_id, "llama_cpp");
    assert_eq!(
        manifest.model_file["Q4_K_M"].name,
        "gemma-3-4b-it-Q4_K_M.gguf"
    );
    assert_eq!(manifest.model_file["Q4_K_M"].size, MODEL_BYTES.len() as i64);
    assert_eq!(manifest.mmproj_file.name, "mmproj-F16.gguf");

    // Resume markers cleaned up, no in-flight sentinel left behind.
    assert!(!dir.join(".inflight").exists());
    assert!(!dir.join("gemma-3-4b-it-Q4_K_M.gguf.progress").exists());
}

#[tokio::test]
async fn blob_without_an_extension_gets_a_usable_gguf_name() {
    // HuggingFace-sourced GGUFs are used in place as content-addressed
    // blobs, so llmman reports a path with no filename to speak of.
    let server = mock_daemon("{\"status\":\"success\"}\n").await;

    let llmman_store = tempdir().unwrap();
    let blob = llmman_store.path().join("9b7f3c1e2d4a5b6c");
    std::fs::write(&blob, MODEL_BYTES).unwrap();

    let bin_dir = tempdir().unwrap();
    let binary = stub_llmman(
        &bin_dir,
        &format!(
            r#"{{"reference":"hf.co/unsloth/Qwen3-GGUF:latest","path":"{}","format":"gguf"}}"#,
            blob.display()
        ),
    );

    let src = LlmmanSource::new(
        "hf.co/unsloth/Qwen3-GGUF:latest".to_string(),
        "unsloth/Qwen3-GGUF".to_string(),
        ManifestHint::default(),
    )
    .with_endpoint_and_binary(server.uri(), binary);

    let cache = tempdir().unwrap();
    let store = make_store(cache.path());
    pull_with_source(
        &store,
        "unsloth/Qwen3-GGUF",
        Box::new(src),
        fast_transport(),
        None,
    )
    .await
    .expect("pull");

    // A hex digest is not something llama.cpp will open, so the source
    // synthesizes a name; with no quant tag to read, the entry is N/A.
    let dir = cache
        .path()
        .join("models")
        .join("unsloth")
        .join("Qwen3-GGUF");
    assert_eq!(
        std::fs::read(dir.join("Qwen3-GGUF.gguf")).unwrap(),
        MODEL_BYTES
    );
    let manifest: ModelManifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("geniex.json")).unwrap()).unwrap();
    assert_eq!(manifest.model_type, ModelType::Llm);
    assert_eq!(manifest.model_file["N/A"].name, "Qwen3-GGUF.gguf");
}

#[tokio::test]
async fn no_daemon_listening_is_an_actionable_error() {
    // Claim an ephemeral port, note it, then release it — so the address
    // is well-formed but nothing is behind it.
    let uri = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", l.local_addr().unwrap())
    };

    let bin_dir = tempdir().unwrap();
    let binary = stub_llmman(&bin_dir, "{}");
    let src = LlmmanSource::new(
        "docker.io/ai/gemma3:latest".to_string(),
        "ai/gemma3".to_string(),
        ManifestHint::default(),
    )
    .with_endpoint_and_binary(uri, binary);

    let err = src.plan().await.unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("llmman serve"), "{msg}");
    assert!(msg.contains("LLMMAN_HOST"), "{msg}");
}

#[tokio::test]
async fn a_non_llmman_server_on_the_port_is_reported_the_same_way() {
    // Something holds LLMMAN_HOST but isn't llmman. The user fixes this
    // exactly as they'd fix "nothing is listening", so a bare "HTTP 404"
    // would just send them hunting for a missing model instead.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/version"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let bin_dir = tempdir().unwrap();
    let binary = stub_llmman(&bin_dir, "{}");
    let src = LlmmanSource::new(
        "docker.io/ai/gemma3:latest".to_string(),
        "ai/gemma3".to_string(),
        ManifestHint::default(),
    )
    .with_endpoint_and_binary(server.uri(), binary);

    let msg = format!("{}", src.plan().await.unwrap_err());
    assert!(msg.contains("llmman serve is not reachable"), "{msg}");
    assert!(msg.contains("HTTP 404"), "{msg}");
}

#[tokio::test]
async fn in_band_error_line_aborts_the_pull() {
    // llmman reports failures inside the stream at HTTP 200 — a client
    // that only checks the status code would treat this as success.
    let server = mock_daemon(concat!(
        "{\"status\":\"pulling manifest\"}\n",
        "{\"error\":\"pulling docker.io/ai/nope:latest: not found\"}\n",
    ))
    .await;

    let bin_dir = tempdir().unwrap();
    let binary = stub_llmman(&bin_dir, "{}");
    let src = LlmmanSource::new(
        "docker.io/ai/nope:latest".to_string(),
        "ai/nope".to_string(),
        ManifestHint::default(),
    )
    .with_endpoint_and_binary(server.uri(), binary);

    let err = src.plan().await.unwrap_err();
    assert!(
        matches!(err, Error::HubModelNotFound(ref r) if r == "docker.io/ai/nope:latest"),
        "{err:?}"
    );
}

#[tokio::test]
async fn stream_ending_without_success_is_a_failure() {
    // The daemon died mid-pull: no error line, no success line.
    let server = mock_daemon("{\"status\":\"pulling\",\"total\":31,\"completed\":4}\n").await;

    let bin_dir = tempdir().unwrap();
    let binary = stub_llmman(&bin_dir, "{}");
    let src = LlmmanSource::new(
        "docker.io/ai/gemma3:latest".to_string(),
        "ai/gemma3".to_string(),
        ManifestHint::default(),
    )
    .with_endpoint_and_binary(server.uri(), binary);

    let err = src.plan().await.unwrap_err();
    assert!(
        format!("{err}").contains("without reporting success"),
        "{err}"
    );
}

#[tokio::test]
async fn cancelling_from_the_progress_callback_stops_the_pull() {
    let server = mock_daemon(concat!(
        "{\"status\":\"pulling\",\"total\":31,\"completed\":4}\n",
        "{\"status\":\"success\"}\n",
    ))
    .await;

    let bin_dir = tempdir().unwrap();
    let binary = stub_llmman(&bin_dir, "{}");
    let src = LlmmanSource::new(
        "docker.io/ai/gemma3:latest".to_string(),
        "ai/gemma3".to_string(),
        ManifestHint::default(),
    )
    .with_endpoint_and_binary(server.uri(), binary);

    let called = Arc::new(AtomicBool::new(false));
    let flag = called.clone();
    let cb: ProgressCallback = Box::new(move |_: &[FileProgress]| {
        flag.store(true, Ordering::SeqCst);
        false // user hit Ctrl-C
    });

    let err = src.plan_with_progress(Some(&cb)).await.unwrap_err();
    assert!(called.load(Ordering::SeqCst), "callback was never invoked");
    assert!(matches!(err, Error::Cancelled), "{err:?}");
}
