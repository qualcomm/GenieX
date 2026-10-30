// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! `releases/latest/` primary + `releases/<version>/` fallback, and
//! `GENIEX_AIHUBVERSION` skipping `latest/`.

use geniex_model::source::ai_hub::{list_hub_models, AiHubConfig};
use geniex_model::test_support::{mount_ranged, mount_status, EnvGuard};
use wiremock::MockServer;

const LATEST: &str = "/qai-hub-models/releases/latest/manifest.json";
const PINNED: &str = "/qai-hub-models/releases/v0.60.0/manifest.json";

fn release_manifest(version: &str, id: &str) -> String {
    serde_json::json!({
        "version": version,
        "models": [{
            "id": id,
            "display_name": id,
            "domain": "MODEL_DOMAIN_GENERATIVE_AI",
            "supported_runtimes": ["RUNTIME_GENIEX_QAIRT"],
            "supported_chipsets": ["chipA"],
            "tags": [],
        }],
    })
    .to_string()
}

async fn listed_ids(server: &MockServer) -> Vec<String> {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = AiHubConfig {
        endpoint: format!("{}/qai-hub-models", server.uri()),
        version: "v0.60.0".to_string(),
        chipset: String::new(),
        cache_dir: tmp.path().to_path_buf(),
        skip_cache: true,
    };
    let models = list_hub_models(&cfg, None).await.expect("list");
    models.into_iter().map(|m| m.id).collect()
}

#[tokio::test]
async fn latest_hit_short_circuits_pinned_fetch() {
    let _env = EnvGuard::lock().unset("GENIEX_AIHUBVERSION");
    let server = MockServer::start().await;
    mount_ranged(&server, LATEST, release_manifest("0.99.0", "latest_model")).await;
    assert_eq!(listed_ids(&server).await, ["latest_model"]);
}

#[tokio::test]
async fn latest_miss_falls_back_to_pinned_version() {
    let _env = EnvGuard::lock().unset("GENIEX_AIHUBVERSION");
    let server = MockServer::start().await;
    mount_ranged(&server, PINNED, release_manifest("0.60.0", "pinned_model")).await;
    assert_eq!(listed_ids(&server).await, ["pinned_model"]);
}

#[tokio::test]
async fn env_override_skips_latest_directory() {
    let _env = EnvGuard::lock().set("GENIEX_AIHUBVERSION", "v0.60.0");
    let server = MockServer::start().await;
    mount_status(&server, "HEAD", LATEST, 500).await;
    mount_status(&server, "GET", LATEST, 500).await;
    mount_ranged(&server, PINNED, release_manifest("0.60.0", "pinned_model")).await;
    assert_eq!(listed_ids(&server).await, ["pinned_model"]);
}
