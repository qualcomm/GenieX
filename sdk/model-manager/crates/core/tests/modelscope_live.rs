// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Live test against the ModelScope Hub. Requires network access, so it
//! is gated behind `--ignored` and only runs when explicitly requested:
//!
//!   cargo test --test modelscope_live -- --ignored
//!
//! Uses `Qwen/Qwen3-0.6B-GGUF` (the same repo the HuggingFace live test
//! exercises) so the file-list + resolve redirect path has something real
//! to chew on.

use model_manager_core::config::StoreConfig;
use model_manager_core::manifest_builder::ManifestHint;
use model_manager_core::pull::{pull, PullIntent, PullRequest};
use model_manager_core::store::Store;

const QWEN3_REPO: &str = "Qwen/Qwen3-0.6B-GGUF";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn end_to_end_pull_via_modelscope() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let cfg = StoreConfig::new(tmp.path().to_path_buf());
    let store = Store::new(cfg).expect("store init");

    let req = PullRequest {
        model_name: QWEN3_REPO.to_string(),
        intent: PullIntent::ModelScope {
            repo: QWEN3_REPO.to_string(),
            token: None,
        },
        on_progress: None,
        hint: ManifestHint::default(),
    };
    pull(&store, req).await.expect("pull failed");

    let list = store.list().expect("list failed");
    assert!(
        list.iter().any(|m| m.name == QWEN3_REPO),
        "pulled model not in list: {list:?}"
    );

    let (_quant, paths) = store
        .get_paths(QWEN3_REPO)
        .expect("get_paths after pull failed");
    assert!(paths.model_path.exists(), "model file missing: {paths:?}");
}

/// `:Q8_0` must pick the Q8_0 GGUF shard through manifest inference.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn end_to_end_pull_qwen3_q8_0() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let cfg = StoreConfig::new(tmp.path().to_path_buf());
    let store = Store::new(cfg).expect("store init");

    let req = PullRequest {
        model_name: QWEN3_REPO.to_string(),
        intent: PullIntent::ModelScope {
            repo: QWEN3_REPO.to_string(),
            token: None,
        },
        on_progress: None,
        hint: ManifestHint {
            quant: Some("Q8_0".to_string()),
            ..Default::default()
        },
    };
    pull(&store, req).await.expect("pull failed");

    let (resolved, paths) = store
        .get_paths(&format!("{QWEN3_REPO}:Q8_0"))
        .expect("get_paths after pull failed");
    assert_eq!(resolved, "Q8_0");
    assert!(paths.model_path.exists(), "model file missing: {paths:?}");
    let fname = paths.model_path.file_name().unwrap().to_string_lossy();
    assert!(
        fname.to_lowercase().contains("q8_0"),
        "expected Q8_0 file, got {fname}"
    );
}
