// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! End-to-end AI Hub pull against a wiremock'd release: the zip is
//! range-read straight into a flat model dir and never lands on disk.

use geniex_model::error::Result;
use geniex_model::pull::pull_with_source;
use geniex_model::source::ai_hub::{list_supported_chipsets, AiHubConfig, AiHubSource};
use geniex_model::store::Store;
use geniex_model::test_support::{build_zip, fast_transport, mount_ranged, temp_store, ZipMethod};
use serde_json::json;
use wiremock::MockServer;

const VERSION: &str = "v0.99.0";

fn release(server: &MockServer) -> String {
    format!("{}/qai-hub-models/releases/{VERSION}", server.uri())
}

fn rel_path(at: &str) -> String {
    format!("/qai-hub-models/releases/{VERSION}/{at}")
}

/// Serve a one-model `testnet` release with `chipsets` and `assets`.
async fn serve_release(
    server: &MockServer,
    chipsets: serde_json::Value,
    assets: serde_json::Value,
) {
    let base = release(server);
    let manifest = json!({
        "platform_url": format!("{base}/platform.json"),
        "models": [{
            "id": "testnet",
            "display_name": "TestNet",
            "domain": "MODEL_DOMAIN_GENERATIVE_AI",
            "manifest_urls": { "release_assets": format!("{base}/models/testnet/release-assets.json") },
        }],
    });
    let docs = [
        ("manifest.json", manifest),
        ("platform.json", json!({ "chipsets": chipsets })),
        (
            "models/testnet/release-assets.json",
            json!({ "model_id": "testnet", "assets": assets }),
        ),
    ];
    for (at, doc) in docs {
        mount_ranged(server, &rel_path(at), doc.to_string()).await;
    }
}

fn config(server: &MockServer, chipset: &str, cache: &std::path::Path) -> AiHubConfig {
    AiHubConfig {
        endpoint: format!("{}/qai-hub-models", server.uri()),
        version: VERSION.to_string(),
        chipset: chipset.to_string(),
        cache_dir: cache.join("aihub-cache"),
        skip_cache: true,
    }
}

async fn pull_testnet(
    server: &MockServer,
    chipset: &str,
) -> (Result<()>, Store, tempfile::TempDir) {
    let (store, tmp) = temp_store();
    let transport = fast_transport();
    let src = AiHubSource::with_transport(
        "TestNet".to_string(),
        "tests/TestNet".to_string(),
        config(server, chipset, tmp.path()),
        transport.clone(),
    );
    let res = pull_with_source(&store, "tests/TestNet", Box::new(src), transport, None).await;
    (res, store, tmp)
}

#[tokio::test]
async fn ai_hub_pull_writes_manifest_and_extracts_flat() {
    let server = MockServer::start().await;
    let zip = build_zip(&[
        (
            "weights/model-00.bin",
            b"BIN_SHARD_CONTENTS",
            ZipMethod::Stored,
        ),
        ("tokenizer.json", &[b'{'; 4096], ZipMethod::Deflate),
    ]);
    let asset_url = format!("{}/models/testnet/assets/SM8650.zip", release(&server));
    serve_release(
        &server,
        json!([{ "name": "SM8650", "aliases": ["Snapdragon 8 Gen 3", "sd8g3"] }]),
        json!([{
            "chipset": "SM8650",
            "runtime": "RUNTIME_GENIEX_QAIRT",
            "precision": "PRECISION_W4A16",
            "download_url": asset_url,
            "uncompressed_size": zip.len(),
        }]),
    )
    .await;
    mount_ranged(&server, &rel_path("models/testnet/assets/SM8650.zip"), zip).await;

    let (res, store, tmp) = pull_testnet(&server, "sd8g3").await;
    res.expect("ai hub pull");

    let model_dir = tmp.path().join("models/tests/TestNet");
    let mut on_disk: Vec<String> = std::fs::read_dir(&model_dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    on_disk.sort();
    assert!(on_disk.contains(&"model-00.bin".to_string()), "{on_disk:?}");
    assert!(
        on_disk.contains(&"tokenizer.json".to_string()),
        "{on_disk:?}"
    );
    assert!(!on_disk.iter().any(|n| n.ends_with(".zip")), "{on_disk:?}");

    let mf = store.get_manifest("tests/TestNet").unwrap();
    assert_eq!(
        (mf.plugin_id.as_str(), mf.precision.as_str()),
        ("qairt", "")
    );
    let entry = &mf.model_file["W4A16"];
    assert_eq!(entry.name, "model-00.bin");
    assert!(entry.downloaded);
    assert!(
        mf.extra_files.iter().any(|f| f.name == "tokenizer.json"),
        "{:?}",
        mf.extra_files
    );
}

#[tokio::test]
async fn ai_hub_pull_errors_when_chipset_unknown() {
    let server = MockServer::start().await;
    serve_release(
        &server,
        json!([{ "name": "SM8650", "aliases": [] }]),
        json!([{
            "chipset": "SM8650",
            "runtime": "RUNTIME_GENIE",
            "precision": "PRECISION_FP16",
            "download_url": format!("{}/unused.zip", server.uri()),
        }]),
    )
    .await;

    let (res, _store, _tmp) = pull_testnet(&server, "UnknownChip").await;
    let err = res.expect_err("expected chipset mismatch error");
    assert!(err.to_string().contains("not available"), "{err}");
}

#[tokio::test]
async fn list_supported_chipsets_returns_names_and_aliases() {
    let server = MockServer::start().await;
    let platform = json!({ "chipsets": [
        { "name": "SM8650", "aliases": ["Snapdragon 8 Gen 3", "sd8g3"] },
        { "name": "SM8750", "aliases": ["sd8elite"] },
    ]});
    mount_ranged(&server, &rel_path("platform.json"), platform.to_string()).await;

    let tmp = tempfile::tempdir().unwrap();
    let chipsets = list_supported_chipsets(&config(&server, "", tmp.path()))
        .await
        .expect("list chipsets");
    let got: Vec<(&str, Vec<&str>)> = chipsets
        .iter()
        .map(|c| {
            (
                c.name.as_str(),
                c.aliases.iter().map(String::as_str).collect(),
            )
        })
        .collect();
    assert_eq!(
        got,
        [
            ("SM8650", vec!["Snapdragon 8 Gen 3", "sd8g3"]),
            ("SM8750", vec!["sd8elite"])
        ]
    );
}
