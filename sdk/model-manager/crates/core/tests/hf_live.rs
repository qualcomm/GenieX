//! Live test against the HuggingFace Hub. Requires network access, so
//! it is gated behind `--ignored` and will only run when explicitly
//! requested:
//!
//!   cargo test --test hf_live -- --ignored
//!
//! The CI job does not pass `--ignored`; this test exists to give humans
//! a quick confidence check that the hf-hub crate + our pull pipeline
//! still work end-to-end against a real repo.
//!
//! Uses `ggml-org/tiny-llamas` (stories260K.gguf is ~1 MB) so the
//! manifest inference path — which only recognises GGUF / ONNX /
//! tokenizer — has something real to chew on.

use model_manager_core::config::StoreConfig;
use model_manager_core::hub::{hf::HfHub, HubSource, ModelHub};
use model_manager_core::manifest_builder::ManifestHint;
use model_manager_core::pull::{pull, PullRequest};
use model_manager_core::store::Store;

const TINY_REPO: &str = "ggml-org/tiny-llamas";

#[test]
#[ignore]
fn list_files_returns_something() {
    let hub = HfHub::new(None).unwrap();
    let (files, _manifest) = hub
        .list_files(TINY_REPO)
        .expect("hf list_files failed");
    assert!(
        !files.is_empty(),
        "expected at least one file in {TINY_REPO}"
    );
    assert!(
        files.iter().any(|f| f.name.ends_with(".gguf")),
        "no .gguf file in listing: {files:?}"
    );
}

#[test]
#[ignore]
fn end_to_end_pull_via_hf() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let cfg = StoreConfig::new(tmp.path().to_path_buf());
    let store = Store::new(cfg).expect("store init");

    let req = PullRequest {
        model_name: TINY_REPO.to_string(),
        hub: HubSource::HuggingFace,
        hf_token: None,
        on_progress: None,
        hint: ManifestHint::default(),
    };
    pull(&store, req).expect("pull failed");

    // After pull, the model should be listed and resolve to a real file.
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
