//! Live tests against the real Qualcomm AI Hub public bucket. Require
//! network access (and, for the end-to-end pull, ~2 GB of disk +
//! patience), so they are gated behind `--ignored` and only run when
//! explicitly requested:
//!
//!   cargo test --test ai_hub_live -- --ignored
//!
//! The CI job does not pass `--ignored`; these exist to give humans a
//! quick confidence check that our protojson parsing + selector +
//! remote-zip pipeline still work against production infrastructure.
//!
//! The chosen model is the smallest Genie (qairt) asset currently
//! published: Phi-3.5-Mini-Instruct on snapdragon-x-elite (~2 GB).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use model_manager_core::config::StoreConfig;
use model_manager_core::executor::{FileProgress, ProgressCallback};
use model_manager_core::pull::pull_with_source;
use model_manager_core::source::ai_hub::{AiHubConfig, AiHubSource};
use model_manager_core::source::ModelSource;
use model_manager_core::store::Store;
use model_manager_core::transport::ReqwestTransport;

const AI_HUB_BASE_URL: &str =
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models";
const AI_HUB_VERSION: &str = "v0.52.0";

const PHI_DISPLAY_NAME: &str = "Phi-3.5-Mini-Instruct";
const PHI_CHIPSET: &str = "qualcomm-snapdragon-x-elite";
const PHI_STORED_NAME: &str = "qualcomm/Phi-3.5-Mini-Instruct";

/// Cheap reachability + parsing check: fetch manifest.json, confirm
/// the model we rely on is still present. Runs in a couple hundred
/// milliseconds; no zip payload download.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn live_manifest_resolves_phi_model() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = AiHubConfig::new(
        AI_HUB_BASE_URL.to_string(),
        AI_HUB_VERSION.to_string(),
        "definitely-not-a-real-chipset".to_string(),
        tmp.path().join("aihub"),
        true,
    );
    let transport = Arc::new(ReqwestTransport::new().unwrap());
    let src = AiHubSource::with_transport(
        PHI_DISPLAY_NAME.to_string(),
        PHI_STORED_NAME.to_string(),
        cfg,
        transport,
    );
    let err = src
        .plan()
        .await
        .expect_err("expected chipset mismatch error");
    let msg = format!("{err}");
    assert!(
        msg.contains("not available") || msg.contains("not found in platform.json"),
        "expected chipset-mismatch error after successful manifest parse, got: {msg}"
    );
}

/// End-to-end: download the real Phi-3.5-Mini-Instruct qairt asset from
/// the public bucket via range reads, inflate each entry, write the
/// synthesised manifest, and verify the store reports the model.
/// ~2 GB payload; don't run in CI.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn live_phi_3_5_mini_e2e_pull() {
    let tmp = tempfile::tempdir().unwrap();
    let store = Store::new(StoreConfig::new(tmp.path().to_path_buf())).unwrap();

    let cfg = AiHubConfig::new(
        AI_HUB_BASE_URL.to_string(),
        AI_HUB_VERSION.to_string(),
        PHI_CHIPSET.to_string(),
        tmp.path().join("aihub"),
        true,
    );

    let invoked = Arc::new(AtomicBool::new(false));
    let invoked_cl = invoked.clone();
    let cb: ProgressCallback = Box::new(move |files: &[FileProgress]| -> bool {
        invoked_cl.store(true, Ordering::SeqCst);
        if let Some(f) = files.first() {
            if f.total_bytes > 0 {
                let pct = (f.downloaded_bytes as f64 / f.total_bytes as f64) * 100.0;
                eprintln!(
                    "[ai_hub_live] {}: {}/{} ({:.1}%)",
                    f.file_name, f.downloaded_bytes, f.total_bytes, pct
                );
            }
        }
        true
    });

    let transport = Arc::new(ReqwestTransport::new().unwrap());
    let src = AiHubSource::with_transport(
        PHI_DISPLAY_NAME.to_string(),
        PHI_STORED_NAME.to_string(),
        cfg,
        transport.clone(),
    );
    pull_with_source(&store, PHI_STORED_NAME, Box::new(src), transport, Some(&cb))
        .await
        .expect("live AI Hub pull failed");

    assert!(
        invoked.load(Ordering::SeqCst),
        "progress callback never fired"
    );

    let mf = store.get_manifest(PHI_STORED_NAME).expect("manifest");
    assert_eq!(mf.plugin_id, "qairt");
    assert_eq!(mf.precision, "W4A16");
    let entry = mf.model_file.get("N/A").expect("N/A quant");
    assert!(
        entry.name.ends_with(".bin"),
        "entrypoint not a .bin: {entry:?}"
    );
    assert!(entry.downloaded);

    let model_dir = tmp.path().join("models").join(PHI_STORED_NAME);
    assert!(
        model_dir.join(&entry.name).exists(),
        "entrypoint file missing on disk"
    );
    assert!(
        mf.extra_files.iter().any(|f| f.name == "tokenizer.json"),
        "tokenizer.json missing from extras: {:?}",
        mf.extra_files
    );
    // Regression guard for the whole point of this refactor: no zip
    // should ever land on disk with the range-read pipeline.
    let entries: Vec<_> = std::fs::read_dir(&model_dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        !entries.iter().any(|n| n.ends_with(".zip")),
        "unexpected .zip in model dir: {entries:?}"
    );
}
