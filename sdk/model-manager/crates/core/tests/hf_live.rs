//! Live test against the HuggingFace Hub. Requires network access, so
//! it is gated behind `--ignored` and only runs when explicitly
//! requested:
//!
//!   cargo test --test hf_live -- --ignored
//!
//! Uses `ggml-org/tiny-llamas` (stories260K.gguf is ~1 MB) so the
//! manifest inference path — which only recognises GGUF / ONNX /
//! tokenizer — has something real to chew on.

use model_manager_core::config::StoreConfig;
use model_manager_core::manifest_builder::ManifestHint;
use model_manager_core::pull::{pull, PullIntent, PullRequest};
use model_manager_core::store::Store;

const TINY_REPO: &str = "ggml-org/tiny-llamas";
const QWEN3_REPO: &str = "ggml-org/Qwen3-0.6B-GGUF";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn end_to_end_pull_via_hf() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let cfg = StoreConfig::new(tmp.path().to_path_buf());
    let store = Store::new(cfg).expect("store init");

    let req = PullRequest {
        model_name: TINY_REPO.to_string(),
        intent: PullIntent::HuggingFace {
            repo: TINY_REPO.to_string(),
            token: None,
        },
        on_progress: None,
        hint: ManifestHint::default(),
    };
    pull(&store, req).await.expect("pull failed");

    let list = store.list().expect("list failed");
    assert!(
        list.iter().any(|m| m.name == TINY_REPO),
        "pulled model not in list: {list:?}"
    );

    let (_quant, paths) = store
        .get_paths(TINY_REPO)
        .expect("get_paths after pull failed");
    assert!(paths.model_path.exists(), "model file missing: {paths:?}");
}

/// Resolving `Qwen3-0.6B-GGUF:Q4_0` against the real repo must pick the
/// Q4_0 GGUF — manifest inference reaches the upstream sibling list and
/// the upper-case hint hits the manifest_builder lookup path that this
/// branch is keeping case-sensitive at the core. Pulls ~370 MB.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn end_to_end_pull_qwen3_q4_0() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let cfg = StoreConfig::new(tmp.path().to_path_buf());
    let store = Store::new(cfg).expect("store init");

    let req = PullRequest {
        model_name: QWEN3_REPO.to_string(),
        intent: PullIntent::HuggingFace {
            repo: QWEN3_REPO.to_string(),
            token: None,
        },
        on_progress: None,
        hint: ManifestHint {
            quant: Some("Q4_0".to_string()),
            ..Default::default()
        },
    };
    pull(&store, req).await.expect("pull failed");

    let (resolved, paths) = store
        .get_paths(&format!("{QWEN3_REPO}:Q4_0"))
        .expect("get_paths after pull failed");
    assert_eq!(resolved, "Q4_0");
    assert!(paths.model_path.exists(), "model file missing: {paths:?}");
    let fname = paths.model_path.file_name().unwrap().to_string_lossy();
    assert!(
        fname.to_lowercase().contains("q4_0"),
        "expected Q4_0 file, got {fname}"
    );
}

/// `ggml-org/gpt-oss-20b-GGUF:mxfp4` ships a single ~12 GB MXFP4 weight —
/// exercises the new MXFP token recognition in `extract_quant`. Pulls
/// ~12 GB, so leave it off by default.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn end_to_end_pull_gpt_oss_20b_mxfp4() {
    let repo = "ggml-org/gpt-oss-20b-GGUF";
    let tmp = tempfile::tempdir().expect("tmpdir");
    let cfg = StoreConfig::new(tmp.path().to_path_buf());
    let store = Store::new(cfg).expect("store init");

    let req = PullRequest {
        model_name: repo.to_string(),
        intent: PullIntent::HuggingFace {
            repo: repo.to_string(),
            token: None,
        },
        on_progress: None,
        hint: ManifestHint {
            quant: Some("MXFP4".to_string()),
            ..Default::default()
        },
    };
    pull(&store, req).await.expect("pull failed");

    let (resolved, paths) = store
        .get_paths(&format!("{repo}:MXFP4"))
        .expect("get_paths after pull failed");
    assert_eq!(resolved, "MXFP4");
    assert!(paths.model_path.exists(), "model file missing: {paths:?}");
    let fname = paths.model_path.file_name().unwrap().to_string_lossy();
    assert_eq!(fname, "gpt-oss-20b-mxfp4.gguf");
}
