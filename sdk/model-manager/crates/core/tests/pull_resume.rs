//! End-to-end test for `pull()` resume at chunk granularity.
//!
//! Walks a real pull through the HfSource → Executor → ReqwestTransport
//! stack against a wiremock upstream. The mock fails chunk #1 once on
//! the first GET; the first pull therefore errors out with a partial
//! `.progress` bitmap. A second pull must skip the completed chunks
//! and only re-fetch the one that failed.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use model_manager_core::config::StoreConfig;
use model_manager_core::manifest_builder::ManifestHint;
use model_manager_core::pull::pull_with_source;
use model_manager_core::source::hf::HfSource;
use model_manager_core::store::{Store, INFLIGHT_DIR, MANIFEST_FILE};
use model_manager_core::transport::{HttpTransport, ReqwestTransport, TransportConfig};
use tempfile::tempdir;
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pull_resumes_after_mid_download_failure() {
    std::env::set_var("GENIEX_DL_CHUNK_SIZE", "16384");

    let server = MockServer::start().await;
    let body = make_body(64 * 1024, 0xA5);
    let repo = "test/tiny";
    let file_name = "weights.gguf";

    let api_body = serde_json::json!({
        "siblings": [{ "rfilename": file_name, "size": body.len() }]
    })
    .to_string();

    let failed_once = Arc::new(AtomicUsize::new(0));
    let get_hits = Arc::new(AtomicUsize::new(0));

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

    // HEAD for geniex.json probe → 404 (repo doesn't ship one).
    Mock::given(method("HEAD"))
        .and(path(format!("/{repo}/resolve/main/geniex.json")))
        .respond_with(ResponseTemplate::new(404))
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

    let tmp = tempdir().unwrap();
    let cfg = StoreConfig::new(tmp.path().to_path_buf());
    let store = Store::new(cfg).expect("store init");

    let endpoint = server.uri();
    let do_pull = || async {
        let transport = fast_transport();
        let src = HfSource::with_endpoint_and_transport(
            repo.to_string(),
            &endpoint,
            None,
            transport.clone(),
            ManifestHint::default(),
        )
        .unwrap();
        pull_with_source(&store, repo, Box::new(src), transport, None).await
    };

    // --- First pull: must fail due to 500 on one chunk -------------------
    let err = do_pull().await.expect_err("first pull must fail");
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
    do_pull().await.expect("second pull must succeed");
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
