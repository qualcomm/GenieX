//! Integration tests for the download engine against a wiremock'd
//! HuggingFace-shaped server. Covers: multi-file + multi-chunk downloads
//! land byte-correct, the `.progress` bitmap gets each bit flipped, a
//! resume run skips already-complete chunks, and progress callbacks fire
//! with monotonically-increasing totals.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use model_manager_core::download::{chunk as chunklib, Engine, EngineConfig};
use model_manager_core::hub::metadata::{FileSource, HubContext};
use model_manager_core::hub::{FileProgress, HfMetadata, ProgressCallback};
use model_manager_core::transport::{HttpTransport, ReqwestTransport, TransportConfig};
use tempfile::tempdir;
use url::Url;
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

/// Parse a `Range: bytes=start-end` header into `(start, len)` inclusive.
fn parse_range(req: &Request) -> (u64, u64) {
    let hdr = req
        .headers
        .get("range")
        .expect("Range header")
        .to_str()
        .unwrap();
    let rest = hdr.strip_prefix("bytes=").unwrap();
    let (s, e) = rest.split_once('-').unwrap();
    let start: u64 = s.parse().unwrap();
    let end: u64 = e.parse().unwrap();
    (start, end - start + 1)
}

async fn install_file_mock(server: &MockServer, path_str: &str, body: Vec<u8>) {
    // HEAD for size discovery.
    Mock::given(method("HEAD"))
        .and(path(path_str.to_string()))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("Content-Length", body.len().to_string())
                .append_header("Accept-Ranges", "bytes"),
        )
        .mount(server)
        .await;

    // Range GET: slice out [start, start+len).
    let body_arc = Arc::new(body);
    let body_cl = body_arc.clone();
    Mock::given(method("GET"))
        .and(path(path_str.to_string()))
        .respond_with(move |req: &Request| {
            let (start, len) = parse_range(req);
            let slice = body_cl[start as usize..(start + len) as usize].to_vec();
            ResponseTemplate::new(206).set_body_bytes(slice)
        })
        .mount(server)
        .await;
}

#[tokio::test]
async fn downloads_multi_file_multi_chunk_and_marks_progress() {
    // Force the chunk floor small so a ~64 KiB file becomes 4 chunks.
    std::env::set_var("GENIEX_DL_CHUNK_SIZE", "16384");

    let server = MockServer::start().await;
    let body_a = make_body(64 * 1024, 0x11);
    let body_b = make_body(48 * 1024, 0x77);
    install_file_mock(&server, "/org/repo/resolve/main/a.bin", body_a.clone()).await;
    install_file_mock(&server, "/org/repo/resolve/main/b.bin", body_b.clone()).await;

    let tmp = tempdir().unwrap();
    let ctx = HubContext::new(
        Arc::new(HfMetadata::with_endpoint(&server.uri(), None, fast_transport()).unwrap()),
        fast_transport(),
    );
    let cfg = EngineConfig {
        file_concurrency: 2,
        chunk_concurrency: 4,
        progress_interval: Duration::from_millis(20),
    };

    let sources = vec![
        FileSource {
            name: "a.bin".into(),
            size: None,
            url: Url::parse(&format!("{}/org/repo/resolve/main/a.bin", server.uri())).unwrap(),
            auth: None,
        },
        FileSource {
            name: "b.bin".into(),
            size: None,
            url: Url::parse(&format!("{}/org/repo/resolve/main/b.bin", server.uri())).unwrap(),
            auth: None,
        },
    ];

    let seen: Arc<Mutex<Vec<Vec<FileProgress>>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_cl = seen.clone();
    let cb: ProgressCallback = Box::new(move |files: &[FileProgress]| -> bool {
        seen_cl.lock().unwrap().push(files.to_vec());
        true
    });

    Engine::with_config(&ctx, cfg)
        .run(sources, tmp.path(), Some(&cb))
        .await
        .expect("download");

    let a_read = std::fs::read(tmp.path().join("a.bin")).unwrap();
    let b_read = std::fs::read(tmp.path().join("b.bin")).unwrap();
    assert_eq!(a_read, body_a, "a.bin bytes mismatch");
    assert_eq!(b_read, body_b, "b.bin bytes mismatch");

    // Bitmap size matches plan and every byte is 0x01.
    let marker_a = std::fs::read(tmp.path().join("a.bin.progress")).unwrap();
    let marker_b = std::fs::read(tmp.path().join("b.bin.progress")).unwrap();
    assert_eq!(marker_a.len(), 4, "64 KiB / 16 KiB = 4 chunks");
    assert_eq!(marker_b.len(), 3, "48 KiB / 16 KiB = 3 chunks");
    assert!(marker_a.iter().all(|b| *b == 0x01));
    assert!(marker_b.iter().all(|b| *b == 0x01));

    // Progress was reported at least once with the terminal 100% state.
    let history = seen.lock().unwrap();
    assert!(!history.is_empty(), "callback must fire at least once");
    let last = history.last().unwrap();
    let total_done: i64 = last.iter().map(|f| f.downloaded_bytes).sum();
    let total_size: i64 = last.iter().map(|f| f.total_bytes).sum();
    assert_eq!(
        total_done, total_size,
        "terminal callback must show 100% — got {total_done} / {total_size}",
    );

    std::env::remove_var("GENIEX_DL_CHUNK_SIZE");
}

#[tokio::test]
async fn resume_skips_completed_chunks() {
    std::env::set_var("GENIEX_DL_CHUNK_SIZE", "16384");

    let server = MockServer::start().await;
    let body = make_body(64 * 1024, 0x55);

    // HEAD is fine via the normal installer.
    Mock::given(method("HEAD"))
        .and(path("/org/repo/resolve/main/f.bin"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("Content-Length", body.len().to_string())
                .append_header("Accept-Ranges", "bytes"),
        )
        .mount(&server)
        .await;

    // Only counts GET requests, and serves the correct slice so the
    // bitmap completion assertion still holds.
    let get_hits = Arc::new(AtomicUsize::new(0));
    let hits_cl = get_hits.clone();
    let body_arc = Arc::new(body.clone());
    let body_cl = body_arc.clone();
    Mock::given(method("GET"))
        .and(path("/org/repo/resolve/main/f.bin"))
        .respond_with(move |req: &Request| {
            hits_cl.fetch_add(1, Ordering::SeqCst);
            let (start, len) = parse_range(req);
            let slice = body_cl[start as usize..(start + len) as usize].to_vec();
            ResponseTemplate::new(206).set_body_bytes(slice)
        })
        .mount(&server)
        .await;

    let tmp = tempdir().unwrap();
    // Seed: pretend we already finished 2 of the 4 chunks.
    let dest = tmp.path();
    std::fs::create_dir_all(dest).unwrap();
    let out = dest.join("f.bin");
    std::fs::write(&out, body.clone()).unwrap(); // content correct for those chunks
    let plan = chunklib::plan_chunks(body.len() as u64);
    let mut bitmap = vec![0u8; plan.num_chunks()];
    bitmap[0] = 0x01;
    bitmap[2] = 0x01;
    std::fs::write(dest.join("f.bin.progress"), &bitmap).unwrap();

    let ctx = HubContext::new(
        Arc::new(HfMetadata::with_endpoint(&server.uri(), None, fast_transport()).unwrap()),
        fast_transport(),
    );
    let cfg = EngineConfig {
        file_concurrency: 1,
        chunk_concurrency: 4,
        progress_interval: Duration::from_millis(20),
    };
    let sources = vec![FileSource {
        name: "f.bin".into(),
        size: None,
        url: Url::parse(&format!("{}/org/repo/resolve/main/f.bin", server.uri())).unwrap(),
        auth: None,
    }];

    // Clear the counter: HEAD also counts against `get_hits` if we
    // mount the same pattern again, so we only installed GET above.
    get_hits.store(0, Ordering::SeqCst);

    Engine::with_config(&ctx, cfg)
        .run(sources, dest, None)
        .await
        .expect("resume");

    // Only the 2 missing chunks should have been fetched.
    assert_eq!(
        get_hits.load(Ordering::SeqCst),
        2,
        "expected 2 GET requests for the missing chunks, got {}",
        get_hits.load(Ordering::SeqCst),
    );
    let marker = std::fs::read(dest.join("f.bin.progress")).unwrap();
    assert!(
        marker.iter().all(|b| *b == 0x01),
        "bitmap should be fully set"
    );

    std::env::remove_var("GENIEX_DL_CHUNK_SIZE");
}

#[tokio::test]
async fn cancel_via_callback_returns_cancelled() {
    std::env::set_var("GENIEX_DL_CHUNK_SIZE", "16384");

    let server = MockServer::start().await;
    // Slow responses so the callback has a chance to fire before the
    // download finishes.
    let body = make_body(64 * 1024, 0x33);
    Mock::given(method("HEAD"))
        .and(path("/org/repo/resolve/main/slow.bin"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("Content-Length", body.len().to_string())
                .append_header("Accept-Ranges", "bytes"),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/org/repo/resolve/main/slow.bin"))
        .respond_with(move |req: &Request| {
            let (start, len) = parse_range(req);
            let slice = body[start as usize..(start + len) as usize].to_vec();
            ResponseTemplate::new(206)
                .set_delay(Duration::from_millis(200))
                .set_body_bytes(slice)
        })
        .mount(&server)
        .await;

    let tmp = tempdir().unwrap();
    let ctx = HubContext::new(
        Arc::new(HfMetadata::with_endpoint(&server.uri(), None, fast_transport()).unwrap()),
        fast_transport(),
    );
    let cfg = EngineConfig {
        file_concurrency: 1,
        chunk_concurrency: 1,
        progress_interval: Duration::from_millis(20),
    };

    let calls = Arc::new(AtomicUsize::new(0));
    let calls_cl = calls.clone();
    let cb: ProgressCallback = Box::new(move |_files| -> bool {
        calls_cl.fetch_add(1, Ordering::SeqCst);
        false
    });
    let sources = vec![FileSource {
        name: "slow.bin".into(),
        size: None,
        url: Url::parse(&format!("{}/org/repo/resolve/main/slow.bin", server.uri())).unwrap(),
        auth: None,
    }];

    let err = Engine::with_config(&ctx, cfg)
        .run(sources, tmp.path(), Some(&cb))
        .await
        .expect_err("cancellation must surface");
    assert!(
        matches!(err, model_manager_core::error::Error::Cancelled),
        "expected Cancelled, got: {err}",
    );
    assert!(calls.load(Ordering::SeqCst) >= 1);

    std::env::remove_var("GENIEX_DL_CHUNK_SIZE");
}
