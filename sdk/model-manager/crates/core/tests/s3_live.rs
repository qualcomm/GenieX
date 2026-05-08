//! Live tests against the real Qualcomm AI Hub public bucket. Require
//! network access (and, for the end-to-end pull, ~2 GB of disk + patience),
//! so they are gated behind `--ignored` and only run when explicitly
//! requested:
//!
//!   cargo test --test s3_live -- --ignored
//!
//! The CI job does not pass `--ignored`; these exist to give humans a
//! quick confidence check that our protojson parsing + selector + full
//! download pipeline still work against production infrastructure.
//!
//! The chosen model is the smallest Genie (qairt) asset currently
//! published: Phi-3.5-Mini-Instruct on snapdragon-x-elite (~2 GB).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use model_manager_core::config::StoreConfig;
use model_manager_core::hub::s3::{pull_ai_hub, S3Config};
use model_manager_core::hub::{FileProgress, ProgressCallback};
use model_manager_core::store::Store;

const AI_HUB_BASE_URL: &str =
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models";
const AI_HUB_VERSION: &str = "v0.52.0";

const PHI_DISPLAY_NAME: &str = "Phi-3.5-Mini-Instruct";
const PHI_CHIPSET: &str = "qualcomm-snapdragon-x-elite";
const PHI_STORED_NAME: &str = "qualcomm/Phi-3.5-Mini-Instruct";

/// Cheap reachability + parsing check: fetch manifest.json, confirm
/// the model we rely on is still present. Runs in a couple hundred
/// milliseconds; no zip download.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn live_manifest_resolves_phi_model() {
    use model_manager_core::hub::s3::pull_ai_hub_with_transport;
    use model_manager_core::transport::ReqwestTransport;

    // Reuse the pull entry point's HEAD-only code path indirectly by
    // pointing at a nonsense chipset and asserting we hit the "not
    // available" error after the manifest lookup succeeded. A failure
    // *before* that string means manifest parsing is broken.
    let tmp = tempfile::tempdir().unwrap();
    let store = Store::new(StoreConfig::new(tmp.path().to_path_buf())).unwrap();
    let cfg = S3Config::new(
        AI_HUB_BASE_URL,
        AI_HUB_VERSION,
        "definitely-not-a-real-chipset",
        tmp.path().join("aihub"),
        true,
    );

    let err = pull_ai_hub_with_transport(
        &store,
        PHI_STORED_NAME,
        PHI_DISPLAY_NAME,
        cfg,
        Arc::new(ReqwestTransport::new().unwrap()),
        None,
    )
    .await
    .expect_err("expected chipset mismatch error");
    let msg = format!("{err}");
    assert!(
        msg.contains("not available") || msg.contains("not found in platform.json"),
        "expected chipset-mismatch error after successful manifest parse, got: {msg}"
    );
}

/// End-to-end: download the real Phi-3.5-Mini-Instruct qairt zip from
/// the public bucket, extract, synthesise the manifest, and verify the
/// store reports the model. ~2 GB download; don't run in CI.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn live_phi_3_5_mini_e2e_pull() {
    let tmp = tempfile::tempdir().unwrap();
    let store = Store::new(StoreConfig::new(tmp.path().to_path_buf())).unwrap();

    let cfg = S3Config::new(
        AI_HUB_BASE_URL,
        AI_HUB_VERSION,
        PHI_CHIPSET,
        tmp.path().join("aihub"),
        true,
    );

    // Terse progress pinger so a slow run doesn't look hung when
    // invoked with `--nocapture`.
    let invoked = Arc::new(AtomicBool::new(false));
    let invoked_cl = invoked.clone();
    let cb: ProgressCallback = Box::new(move |files: &[FileProgress]| -> bool {
        invoked_cl.store(true, Ordering::SeqCst);
        if let Some(f) = files.first() {
            if f.total_bytes > 0 {
                let pct = (f.downloaded_bytes as f64 / f.total_bytes as f64) * 100.0;
                eprintln!(
                    "[s3_live] {}: {}/{} ({:.1}%)",
                    f.file_name, f.downloaded_bytes, f.total_bytes, pct
                );
            }
        }
        true
    });

    pull_ai_hub(&store, PHI_STORED_NAME, PHI_DISPLAY_NAME, cfg, Some(&cb))
        .await
        .expect("live AI Hub pull failed");

    assert!(
        invoked.load(Ordering::SeqCst),
        "progress callback never fired"
    );

    let mf = store.get_manifest(PHI_STORED_NAME).expect("manifest");
    assert_eq!(mf.plugin_id, "qairt");
    assert_eq!(mf.device_id, PHI_CHIPSET);
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
    // The real Phi qairt archive ships a tokenizer.json alongside the
    // weight shards.
    assert!(
        mf.extra_files.iter().any(|f| f.name == "tokenizer.json"),
        "tokenizer.json missing from extras: {:?}",
        mf.extra_files
    );
}

/// Same shape as `live_phi_3_5_mini_e2e_pull` but feeds an empty
/// chipset through `S3Config`, exercising the
/// `detect::detect_host_chipset` fallback inside `pull_ai_hub_inner`.
/// On non-Snapdragon-Windows hosts this fails with "host auto-detect
/// is not supported on this platform"; only real Snapdragon X Elite /
/// Plus / X2 Elite laptops can pass it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn live_e2e_with_auto_detect() {
    let tmp = tempfile::tempdir().unwrap();
    let store = Store::new(StoreConfig::new(tmp.path().to_path_buf())).unwrap();

    let cfg = S3Config::new(
        AI_HUB_BASE_URL,
        AI_HUB_VERSION,
        "", // <-- triggers host auto-detect
        tmp.path().join("aihub"),
        true,
    );

    pull_ai_hub(&store, PHI_STORED_NAME, PHI_DISPLAY_NAME, cfg, None)
        .await
        .expect("live AI Hub pull with auto-detect failed");

    let mf = store.get_manifest(PHI_STORED_NAME).expect("manifest");
    assert_eq!(
        mf.device_id, PHI_CHIPSET,
        "auto-detect resolved to unexpected chipset: {:?}",
        mf.device_id
    );
    assert_eq!(mf.plugin_id, "qairt");
}
