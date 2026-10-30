// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Live pulls from the ModelScope Hub (network required):
//!
//!   cargo test --test modelscope_live -- --ignored

use model_manager_core::manifest::infer::ManifestHint;
use model_manager_core::pull::{pull, PullIntent, PullRequest};
use model_manager_core::store::Store;
use model_manager_core::test_support::temp_store;

const QWEN3_REPO: &str = "Qwen/Qwen3-0.6B-GGUF";

async fn pull_ms(quant: Option<&str>) -> (Store, tempfile::TempDir) {
    let (store, tmp) = temp_store();
    let req = PullRequest {
        model_name: QWEN3_REPO.to_string(),
        intent: PullIntent::ModelScope {
            repo: QWEN3_REPO.to_string(),
            token: None,
        },
        on_progress: None,
        hint: ManifestHint {
            quant: quant.map(str::to_string),
            ..Default::default()
        },
    };
    pull(&store, req).await.expect("pull failed");
    (store, tmp)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn end_to_end_pull_via_modelscope() {
    let (store, _tmp) = pull_ms(None).await;
    let list = store.list().expect("list");
    assert!(
        list.iter().any(|m| m.name == QWEN3_REPO),
        "not listed: {list:?}"
    );
    let (_, paths) = store.get_paths(QWEN3_REPO).expect("get_paths");
    assert!(paths.model_path.exists(), "model file missing: {paths:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn end_to_end_pull_qwen3_q8_0() {
    let (store, _tmp) = pull_ms(Some("Q8_0")).await;
    let (resolved, paths) = store
        .get_paths(&format!("{QWEN3_REPO}:Q8_0"))
        .expect("get_paths");
    assert_eq!(resolved, "Q8_0");
    assert!(paths.model_path.exists(), "model file missing: {paths:?}");
    let name = paths
        .model_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_lowercase();
    assert!(name.contains("q8_0"), "expected Q8_0 file, got {name}");
}
