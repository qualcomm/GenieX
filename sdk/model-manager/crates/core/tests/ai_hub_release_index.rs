// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! `releases/latest/` primary + `releases/<version>/` fallback, and
//! `GENIEX_AIHUBVERSION` skipping `latest/` (qcom-ai-hub/geniex#1434).

use std::sync::Arc;

use model_manager_core::source::ai_hub::{list_hub_models, AiHubConfig};
use tempfile::tempdir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

/// Serializes access to `GENIEX_AIHUBVERSION`. `unwrap_or_else` recovers
/// from poisoning so one panicking test can't cascade into the others.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock_env_unset() -> std::sync::MutexGuard<'static, ()> {
    let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("GENIEX_AIHUBVERSION");
    guard
}

fn parse_range(req: &Request) -> (u64, u64) {
    let hdr = req.headers.get("range").unwrap().to_str().unwrap();
    let rest = hdr.strip_prefix("bytes=").unwrap();
    let (s, e) = rest.split_once('-').unwrap();
    let start: u64 = s.parse().unwrap();
    let end: u64 = e.parse().unwrap();
    (start, end - start + 1)
}

/// Serve `body` at `p` for HEAD + ranged GET (matches `fetch_direct`).
async fn install_static(server: &MockServer, p: &str, body: Vec<u8>) {
    Mock::given(method("HEAD"))
        .and(path(p.to_string()))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("Content-Length", body.len().to_string())
                .append_header("Accept-Ranges", "bytes"),
        )
        .mount(server)
        .await;

    let body_arc = Arc::new(body);
    Mock::given(method("GET"))
        .and(path(p.to_string()))
        .respond_with(move |req: &Request| {
            let (start, len) = parse_range(req);
            let slice = body_arc[start as usize..(start + len) as usize].to_vec();
            ResponseTemplate::new(206).set_body_bytes(slice)
        })
        .mount(server)
        .await;
}

const LATEST_MF: &str = r#"{
  "version": "0.99.0",
  "models": [
    { "id": "latest_model", "display_name": "Latest",
      "domain": "MODEL_DOMAIN_GENERATIVE_AI",
      "supported_runtimes": ["RUNTIME_GENIEX_QAIRT"],
      "supported_chipsets": ["chipA"],
      "tags": [] }
  ]
}"#;

const PINNED_MF: &str = r#"{
  "version": "0.60.0",
  "models": [
    { "id": "pinned_model", "display_name": "Pinned",
      "domain": "MODEL_DOMAIN_GENERATIVE_AI",
      "supported_runtimes": ["RUNTIME_GENIEX_QAIRT"],
      "supported_chipsets": ["chipA"],
      "tags": [] }
  ]
}"#;

fn cfg_for(server: &MockServer, tmp: &tempfile::TempDir) -> AiHubConfig {
    AiHubConfig {
        endpoint: format!("{}/qai-hub-models", server.uri()),
        version: "v0.60.0".to_string(),
        chipset: String::new(),
        cache_dir: tmp.path().to_path_buf(),
        skip_cache: true,
    }
}

/// 200 on `latest/manifest.json` → use it, never touch the pinned dir.
#[tokio::test]
async fn latest_hit_short_circuits_pinned_fetch() {
    let _g = lock_env_unset();
    let server = MockServer::start().await;
    install_static(
        &server,
        "/qai-hub-models/releases/latest/manifest.json",
        LATEST_MF.as_bytes().to_vec(),
    )
    .await;

    let tmp = tempdir().unwrap();
    let cfg = cfg_for(&server, &tmp);

    let models = list_hub_models(&cfg, None).await.expect("list");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "latest_model");
}

/// 404 on `latest/` → transparent fallback to `releases/<version>/`.
#[tokio::test]
async fn latest_miss_falls_back_to_pinned_version() {
    let _g = lock_env_unset();
    let server = MockServer::start().await;
    // No route for `latest/manifest.json` → wiremock 404s automatically.
    install_static(
        &server,
        "/qai-hub-models/releases/v0.60.0/manifest.json",
        PINNED_MF.as_bytes().to_vec(),
    )
    .await;

    let tmp = tempdir().unwrap();
    let cfg = cfg_for(&server, &tmp);

    let models = list_hub_models(&cfg, None).await.expect("list");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "pinned_model");
}

/// `GENIEX_AIHUBVERSION` set → `latest/` is skipped entirely.
#[tokio::test]
async fn env_override_skips_latest_directory() {
    let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("GENIEX_AIHUBVERSION", "v0.60.0");

    let server = MockServer::start().await;
    // 500 on `latest/` so the test fails if the override doesn't skip it.
    Mock::given(method("HEAD"))
        .and(path("/qai-hub-models/releases/latest/manifest.json"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/qai-hub-models/releases/latest/manifest.json"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    install_static(
        &server,
        "/qai-hub-models/releases/v0.60.0/manifest.json",
        PINNED_MF.as_bytes().to_vec(),
    )
    .await;

    let tmp = tempdir().unwrap();
    let cfg = cfg_for(&server, &tmp);

    let models = list_hub_models(&cfg, None).await.expect("list");

    std::env::remove_var("GENIEX_AIHUBVERSION");
    drop(guard);

    assert_eq!(models[0].id, "pinned_model");
}
