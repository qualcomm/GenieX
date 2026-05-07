//! End-to-end test for `pull()` resume at chunk granularity.
//!
//! Walks a real pull through the HfHub → engine → ReqwestTransport
//! stack against a wiremock upstream. The mock fails chunk #1 once on
//! the first GET; the first pull therefore errors out with a partial
//! `.progress` bitmap. A second pull must skip the completed chunks
//! and only re-fetch the one that failed.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use model_manager_core::config::StoreConfig;
use model_manager_core::hub::hf::HfHub;
use model_manager_core::hub::ModelHub;
use model_manager_core::manifest_builder::ManifestHint;
use model_manager_core::store::{Store, INFLIGHT_DIR, MANIFEST_FILE};
use model_manager_core::transport::{HttpTransport, ReqwestTransport, TransportConfig};
use tempfile::tempdir;
use tokio::runtime::{Builder, Runtime};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn fast_transport() -> Arc<dyn HttpTransport> {
    Arc::new(
        ReqwestTransport::with_config(TransportConfig {
            connect_timeout: Some(Duration::from_secs(2)),
            read_timeout: Some(Duration::from_secs(5)),
            retries: Some(0),
            retry_backoff: Some(Duration::from_millis(10)),
            proxy_override: None,
        })
        .unwrap(),
    )
}

fn make_body(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (((i as u32).wrapping_mul(31).wrapping_add(seed as u32)) & 0xff) as u8)
        .collect()
}

fn parse_range(req: &Request) -> (u64, u64) {
    let hdr = req.headers.get("range").unwrap().to_str().unwrap();
    let rest = hdr.strip_prefix("bytes=").unwrap();
    let (s, e) = rest.split_once('-').unwrap();
    let start: u64 = s.parse().unwrap();
    let end: u64 = e.parse().unwrap();
    (start, end - start + 1)
}

/// Stand up wiremock inside a leaked runtime, so the hyper server the
/// mock uses keeps running even while HfHub spins up its own short-lived
/// runtime below. Plain sync test body — no `#[tokio::test]` — so that
/// HfHub's `rt.block_on` inside `download()` doesn't nest runtimes.
fn setup_mock_server() -> (MockServer, &'static Runtime) {
    let rt: &'static Runtime = Box::leak(Box::new(
        Builder::new_multi_thread().enable_all().build().unwrap(),
    ));
    let server = rt.block_on(MockServer::start());
    (server, rt)
}

#[test]
fn pull_resumes_after_mid_download_failure() {
    std::env::set_var("GENIEX_DL_CHUNK_SIZE", "16384");

    let (server, rt) = setup_mock_server();
    let body = make_body(64 * 1024, 0xA5);
    let repo = "test/tiny";
    let file_name = "weights.gguf";

    // --- Mocks (installed on the wiremock runtime) -------------------
    let api_body = serde_json::json!({
        "siblings": [{ "rfilename": file_name }]
    })
    .to_string();

    let failed_once = Arc::new(AtomicUsize::new(0));
    let get_hits = Arc::new(AtomicUsize::new(0));

    rt.block_on(async {
        // API HEAD+GET.
        Mock::given(method("HEAD"))
            .and(path(format!("/api/models/{repo}")))
            .respond_with(
                ResponseTemplate::new(200)
                    .append_header("Content-Length", api_body.len().to_string())
                    .append_header("Accept-Ranges", "bytes"),
            )
            .mount(&server)
            .await;
        let api_body_cl = api_body.clone();
        Mock::given(method("GET"))
            .and(path(format!("/api/models/{repo}")))
            .respond_with(move |_req: &Request| {
                ResponseTemplate::new(206).set_body_string(api_body_cl.clone())
            })
            .mount(&server)
            .await;

        // File HEAD.
        Mock::given(method("HEAD"))
            .and(path(format!("/{repo}/resolve/main/{file_name}")))
            .respond_with(
                ResponseTemplate::new(200)
                    .append_header("Content-Length", body.len().to_string())
                    .append_header("Accept-Ranges", "bytes"),
            )
            .mount(&server)
            .await;

        // File GET with fail-once behavior on chunk at offset 16384.
        let failed_once_cl = failed_once.clone();
        let get_hits_cl = get_hits.clone();
        let body_arc = Arc::new(body.clone());
        let body_cl = body_arc.clone();
        Mock::given(method("GET"))
            .and(path(format!("/{repo}/resolve/main/{file_name}")))
            .respond_with(move |req: &Request| {
                get_hits_cl.fetch_add(1, Ordering::SeqCst);
                let (start, len) = parse_range(req);
                if start == 16384 && failed_once_cl.load(Ordering::SeqCst) == 0 {
                    failed_once_cl.store(1, Ordering::SeqCst);
                    return ResponseTemplate::new(500);
                }
                let slice = body_cl[start as usize..(start + len) as usize].to_vec();
                ResponseTemplate::new(206).set_body_bytes(slice)
            })
            .mount(&server)
            .await;
    });

    // --- Real store + hub -----------------------------------------------
    let tmp = tempdir().unwrap();
    let cfg = StoreConfig::new(tmp.path().to_path_buf());
    let store = Store::new(cfg).expect("store init");
    let hub = HfHub::with_endpoint(&server.uri(), None, fast_transport()).unwrap();

    // --- First pull: must fail due to 500 on one chunk -------------------
    let err = run_pull(&store, &hub, repo).expect_err("first pull must fail");
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("http") || msg.contains("500"),
        "expected an http/500 error, got: {msg}",
    );
    let after_first = get_hits.load(Ordering::SeqCst);

    let model_dir = tmp.path().join("models").join(repo);
    let marker_path = model_dir.join(format!("{file_name}.progress"));
    let bitmap = std::fs::read(&marker_path).expect(".progress must exist after partial");
    let done = bitmap.iter().filter(|b| **b == 0x01).count();
    assert_eq!(bitmap.len(), 4, "64 KiB / 16 KiB = 4 chunks");
    assert!(
        (1..bitmap.len()).contains(&done),
        "bitmap must be partial; got {done}/{} done",
        bitmap.len()
    );
    let remaining = bitmap.len() - done;

    // --- Second pull: should resume and only fetch the remaining chunks -
    run_pull(&store, &hub, repo).expect("second pull must succeed");
    let second_hits = get_hits.load(Ordering::SeqCst) - after_first;
    assert_eq!(
        second_hits, remaining,
        "expected {remaining} chunk GET(s) on resume, got {second_hits}",
    );

    // --- Final state: file correct, manifest published, marker gone ----
    let final_file = model_dir.join(file_name);
    assert_eq!(
        std::fs::read(&final_file).unwrap(),
        body,
        "bytes must match"
    );
    assert!(
        model_dir.join(MANIFEST_FILE).exists(),
        "geniex.json must be published"
    );
    assert!(
        !marker_path.exists(),
        ".progress should be cleaned up after success"
    );
    assert!(
        !model_dir.join(INFLIGHT_DIR).exists(),
        ".inflight sentinel should be cleared"
    );

    std::env::remove_var("GENIEX_DL_CHUNK_SIZE");
}

/// Thin reimplementation of the inner body of `pull::pull_locked`, so
/// the test can point at a custom HfHub endpoint while preserving the
/// lock + manifest-publish semantics the orchestrator guarantees.
fn run_pull(store: &Store, hub: &HfHub, repo: &str) -> model_manager_core::error::Result<()> {
    store.with_model_lock(repo, || {
        let dest_dir = store.model_file_path(repo, "")?;
        std::fs::create_dir_all(&dest_dir)?;
        let inflight = dest_dir.join(INFLIGHT_DIR);
        std::fs::create_dir_all(&inflight)?;

        let (remote_files, hub_manifest) = hub.list_files(repo)?;
        let mut manifest = match hub_manifest {
            Some(m) => m,
            None => {
                let names: Vec<String> = remote_files
                    .iter()
                    .filter(|f| f.name != "geniex.json")
                    .map(|f| f.name.clone())
                    .collect();
                let mut sizes = std::collections::HashMap::new();
                for f in &remote_files {
                    sizes.insert(f.name.clone(), f.size);
                }
                model_manager_core::manifest_builder::infer_manifest_from_names(
                    repo,
                    &names,
                    &sizes,
                    ManifestHint::default(),
                )?
            }
        };
        manifest.name = repo.to_string();

        let files: Vec<String> = manifest
            .model_file
            .values()
            .filter(|f| f.downloaded && !f.name.is_empty())
            .map(|f| f.name.clone())
            .collect();

        hub.download(repo, &files, &dest_dir, None)?;

        // Mirror pull_locked's publish step.
        let staged = inflight.join(MANIFEST_FILE);
        std::fs::write(&staged, serde_json::to_string(&manifest)?)?;
        let final_path = dest_dir.join(MANIFEST_FILE);
        std::fs::rename(&staged, &final_path)?;
        for f in manifest.model_file.values() {
            if !f.name.is_empty() {
                let marker = dest_dir.join(format!(
                    "{}{}",
                    f.name,
                    model_manager_core::pull::PROGRESS_SUFFIX
                ));
                let _ = std::fs::remove_file(marker);
            }
        }
        let _ = std::fs::remove_dir_all(&inflight);
        Ok(())
    })
}
