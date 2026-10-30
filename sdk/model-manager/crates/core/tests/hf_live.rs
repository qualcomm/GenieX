// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Live pulls from the HuggingFace Hub (network required):
//!
//!   cargo test --test hf_live -- --ignored

use model_manager_core::manifest::infer::ManifestHint;
use model_manager_core::pull::{pull, PullIntent, PullRequest};
use model_manager_core::store::Store;
use model_manager_core::test_support::temp_store;

async fn pull_hf(repo: &str, quant: Option<&str>) -> (Store, tempfile::TempDir) {
    let (store, tmp) = temp_store();
    let req = PullRequest {
        model_name: repo.to_string(),
        intent: PullIntent::HuggingFace {
            repo: repo.to_string(),
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

/// `get_paths(repo:quant)` resolves to `quant` and a file named like `file`.
fn assert_quant_file(store: &Store, repo: &str, quant: &str, file: &str) {
    let (resolved, paths) = store
        .get_paths(&format!("{repo}:{quant}"))
        .expect("get_paths");
    assert_eq!(resolved, quant);
    assert!(paths.model_path.exists(), "model file missing: {paths:?}");
    let name = paths
        .model_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_lowercase();
    assert!(name.contains(file), "expected {file} file, got {name}");
}

/// `ggml-org/tiny-llamas` (stories260K.gguf is ~1 MB).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn end_to_end_pull_via_hf() {
    let repo = "ggml-org/tiny-llamas";
    let (store, _tmp) = pull_hf(repo, None).await;
    let list = store.list().expect("list");
    assert!(list.iter().any(|m| m.name == repo), "not listed: {list:?}");
    let (_, paths) = store.get_paths(repo).expect("get_paths");
    assert!(paths.model_path.exists(), "model file missing: {paths:?}");
}

/// ~370 MB.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn end_to_end_pull_qwen3_q4_0() {
    let repo = "ggml-org/Qwen3-0.6B-GGUF";
    let (store, _tmp) = pull_hf(repo, Some("Q4_0")).await;
    assert_quant_file(&store, repo, "Q4_0", "q4_0");
}

/// Ships an `fp16` shard next to its Q* shards. ~3.1 GB.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn end_to_end_pull_qwen2_fp16() {
    let repo = "Qwen/Qwen2-1.5B-Instruct-GGUF";
    let (store, _tmp) = pull_hf(repo, Some("FP16")).await;
    assert_quant_file(&store, repo, "FP16", "fp16");
}

/// A single unsharded MXFP4 weight. ~12 GB.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn end_to_end_pull_gpt_oss_20b_mxfp4() {
    let repo = "ggml-org/gpt-oss-20b-GGUF";
    let (store, _tmp) = pull_hf(repo, Some("MXFP4")).await;
    assert_quant_file(&store, repo, "MXFP4", "gpt-oss-20b-mxfp4.gguf");
}
