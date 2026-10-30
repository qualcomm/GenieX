// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Live checks against the public Qualcomm AI Hub bucket (network
//! required; the end-to-end pull is ~2 GB):
//!
//!   cargo test --test ai_hub_live -- --ignored
//!
//! Phi-3.5-Mini-Instruct on snapdragon-x-elite is the smallest qairt asset.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use model_manager_core::executor::ProgressCallback;
use model_manager_core::pull::pull_with_source;
use model_manager_core::source::ai_hub::{AiHubConfig, AiHubSource};
use model_manager_core::source::ModelSource;
use model_manager_core::test_support::temp_store;
use model_manager_core::transport::{HttpTransport, ReqwestTransport};

const AI_HUB_BASE_URL: &str =
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models";
const PHI_STORED_NAME: &str = "qualcomm/Phi-3.5-Mini-Instruct";

fn phi_source(
    chipset: &str,
    cache: &std::path::Path,
    transport: Arc<dyn HttpTransport>,
) -> AiHubSource {
    let cfg = AiHubConfig::new(
        AI_HUB_BASE_URL.to_string(),
        "v0.60.0".to_string(),
        chipset.to_string(),
        cache.join("aihub"),
        true,
    );
    AiHubSource::with_transport(
        "Phi-3.5-Mini-Instruct".to_string(),
        PHI_STORED_NAME.to_string(),
        cfg,
        transport,
    )
}

/// manifest.json parses and still lists the model; no payload download.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn live_manifest_resolves_phi_model() {
    let tmp = tempfile::tempdir().unwrap();
    let transport = Arc::new(ReqwestTransport::new().unwrap());
    let err = phi_source("definitely-not-a-real-chipset", tmp.path(), transport)
        .plan()
        .await
        .expect_err("expected chipset mismatch error");
    let msg = err.to_string();
    assert!(
        msg.contains("not available") || msg.contains("not found in platform.json"),
        "{msg}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn live_phi_3_5_mini_e2e_pull() {
    let (store, tmp) = temp_store();
    let invoked = Arc::new(AtomicBool::new(false));
    let invoked_cl = invoked.clone();
    let cb: ProgressCallback = Box::new(move |_| {
        invoked_cl.store(true, Ordering::SeqCst);
        true
    });
    let transport: Arc<dyn HttpTransport> = Arc::new(ReqwestTransport::new().unwrap());
    let src = phi_source("qualcomm-snapdragon-x-elite", tmp.path(), transport.clone());
    pull_with_source(&store, PHI_STORED_NAME, Box::new(src), transport, Some(&cb))
        .await
        .expect("live AI Hub pull failed");
    assert!(
        invoked.load(Ordering::SeqCst),
        "progress callback never fired"
    );

    let mf = store.get_manifest(PHI_STORED_NAME).expect("manifest");
    assert_eq!(
        (mf.plugin_id.as_str(), mf.precision.as_str()),
        ("qairt", "")
    );
    let entry = mf.model_file.get("W4A16").expect("W4A16 precision entry");
    assert!(
        entry.name.ends_with(".bin") && entry.downloaded,
        "{entry:?}"
    );

    let model_dir = tmp.path().join("models").join(PHI_STORED_NAME);
    assert!(
        model_dir.join(&entry.name).exists(),
        "entrypoint missing on disk"
    );
    assert!(
        mf.extra_files.iter().any(|f| f.name == "tokenizer.json"),
        "{:?}",
        mf.extra_files
    );
    let on_disk: Vec<String> = std::fs::read_dir(&model_dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(!on_disk.iter().any(|n| n.ends_with(".zip")), "{on_disk:?}");
}
