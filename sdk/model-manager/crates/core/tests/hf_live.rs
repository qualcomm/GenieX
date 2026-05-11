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
