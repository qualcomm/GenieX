// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! `resolve_ai_hub_version` against a wiremock'd `releases/latest.txt`.
//!
//! Covers the qcom-ai-hub/geniex#1434 contract: dynamic discovery via the
//! pointer AIHM's release pipeline writes
//! (qcom-ai-hub/ai-hub-models-internal#4284), a short-TTL disk cache so a
//! version bump propagates without re-hitting the network every call, an
//! explicit `GENIEX_AIHUBVERSION` override that skips the network
//! entirely, and a graceful fallback when the pointer is missing.

use model_manager_core::config::StoreConfig;
use model_manager_core::source::ai_hub::resolve_ai_hub_version;
use tempfile::tempdir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

/// Serializes access to `GENIEX_AIHUBVERSION` across this file's tests.
/// It's the only env var these tests touch, but `cargo test` runs test
/// functions on threads within one process, so two tests racing to
/// set/unset it could otherwise observe each other's value. `unwrap_or_else`
/// recovers from poisoning: a panic (i.e. assertion failure) in one test
/// while holding the guard must not cascade into unrelated failures on
/// every other test in this file via a "poisoned lock" panic of its own.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock_env() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn unset_override() -> std::sync::MutexGuard<'static, ()> {
    let guard = lock_env();
    std::env::remove_var("GENIEX_AIHUBVERSION");
    guard
}

/// `resolve_ai_hub_version` fetches via `fetch_direct`, which does a HEAD
/// (for size) before a ranged GET — mirrors `ai_hub_pull.rs`'s
/// `install_static` helper for the same reason.
fn parse_range(req: &Request) -> (u64, u64) {
    let hdr = req.headers.get("range").unwrap().to_str().unwrap();
    let rest = hdr.strip_prefix("bytes=").unwrap();
    let (s, e) = rest.split_once('-').unwrap();
    let start: u64 = s.parse().unwrap();
    let end: u64 = e.parse().unwrap();
    (start, end - start + 1)
}

/// `up_to_times`: caps how many times each of HEAD/GET may be served —
/// used by the cache-reuse test to prove a second `resolve_ai_hub_version`
/// call didn't hit the network at all.
async fn mount_latest_txt(server: &MockServer, body: Vec<u8>, up_to_times: Option<u64>) {
    let p = "/qai-hub-models/releases/latest.txt";
    let mut head_mock = Mock::given(method("HEAD")).and(path(p)).respond_with(
        ResponseTemplate::new(200)
            .append_header("Content-Length", body.len().to_string())
            .append_header("Accept-Ranges", "bytes"),
    );
    if let Some(n) = up_to_times {
        head_mock = head_mock.up_to_n_times(n);
    }
    head_mock.mount(server).await;

    let body = std::sync::Arc::new(body);
    let mut get_mock =
        Mock::given(method("GET"))
            .and(path(p))
            .respond_with(move |req: &Request| {
                let (start, len) = parse_range(req);
                let slice = body[start as usize..(start + len) as usize].to_vec();
                ResponseTemplate::new(206).set_body_bytes(slice)
            });
    if let Some(n) = up_to_times {
        get_mock = get_mock.up_to_n_times(n);
    }
    get_mock.mount(server).await;
}

#[tokio::test]
async fn resolves_latest_txt_with_v_prefix_normalized() {
    let _g = unset_override();
    let server = MockServer::start().await;
    mount_latest_txt(&server, b"0.61.0\n".to_vec(), None).await;

    let cache_dir = tempdir().unwrap();
    let endpoint = format!("{}/qai-hub-models", server.uri());
    let version = resolve_ai_hub_version(&endpoint, cache_dir.path()).await;

    assert_eq!(version, "v0.61.0");
}

#[tokio::test]
async fn falls_back_to_pinned_default_when_pointer_missing() {
    let _g = unset_override();
    // No route mounted: wiremock answers every request 404.
    let server = MockServer::start().await;
    let cache_dir = tempdir().unwrap();
    let endpoint = format!("{}/qai-hub-models", server.uri());

    let version = resolve_ai_hub_version(&endpoint, cache_dir.path()).await;

    assert_eq!(version, StoreConfig::ai_hub_version_fallback());
}

#[tokio::test]
async fn falls_back_when_pointer_body_is_empty() {
    let _g = unset_override();
    let server = MockServer::start().await;
    mount_latest_txt(&server, b"   \n".to_vec(), None).await;

    let cache_dir = tempdir().unwrap();
    let endpoint = format!("{}/qai-hub-models", server.uri());
    let version = resolve_ai_hub_version(&endpoint, cache_dir.path()).await;

    assert_eq!(version, StoreConfig::ai_hub_version_fallback());
}

#[tokio::test]
async fn env_override_wins_without_touching_the_network() {
    let guard = lock_env();
    std::env::set_var("GENIEX_AIHUBVERSION", "v0.42.0");

    // No server at all — if the override didn't short-circuit, this would
    // hang/fail against a connection nothing is listening on.
    let cache_dir = tempdir().unwrap();
    let version =
        resolve_ai_hub_version("http://127.0.0.1:0/qai-hub-models", cache_dir.path()).await;

    std::env::remove_var("GENIEX_AIHUBVERSION");
    drop(guard);

    assert_eq!(version, "v0.42.0");
}

#[tokio::test]
async fn second_call_reuses_disk_cache_instead_of_refetching() {
    let _g = unset_override();
    let server = MockServer::start().await;
    mount_latest_txt(&server, b"0.61.0\n".to_vec(), Some(1)).await;

    let cache_dir = tempdir().unwrap();
    let endpoint = format!("{}/qai-hub-models", server.uri());

    let first = resolve_ai_hub_version(&endpoint, cache_dir.path()).await;
    assert_eq!(first, "v0.61.0");

    // The mock above only answers once; a second network hit would 404
    // (unmatched request), so a cache hit is the only way this passes.
    let second = resolve_ai_hub_version(&endpoint, cache_dir.path()).await;
    assert_eq!(second, "v0.61.0");
}

// Reveal-gap probe: what happens when `latest.txt` is served but its body
// is not a version string at all — an HTML error page, a redirect body, an
// S3 XML error accidentally cached as 200, etc. Contract-wise this should
// fall back to the pinned default just like the empty/404 cases above.
#[tokio::test]
async fn falls_back_when_pointer_body_is_malformed() {
    let _g = unset_override();
    let server = MockServer::start().await;
    mount_latest_txt(
        &server,
        b"<!doctype html><html>404 not found</html>\n".to_vec(),
        None,
    )
    .await;

    let cache_dir = tempdir().unwrap();
    let endpoint = format!("{}/qai-hub-models", server.uri());
    let version = resolve_ai_hub_version(&endpoint, cache_dir.path()).await;

    assert_eq!(version, StoreConfig::ai_hub_version_fallback());
}
