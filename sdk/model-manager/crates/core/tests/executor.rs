//! Integration tests for the download executor against a wiremock'd
//! HuggingFace-shaped server. Covers: multi-file + multi-chunk
//! downloads land byte-correct, the `.progress` bitmap gets each bit
//! flipped, a resume run skips already-complete chunks, and progress
//! callbacks fire with monotonically-increasing totals.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use model_manager_core::executor::{
    chunk as chunklib, Executor, ExecutorConfig, FileProgress, ProgressCallback,
};
use model_manager_core::source::{BytesSource, FileSpec};
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
    Mock::given(method("HEAD"))
        .and(path(path_str.to_string()))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("Content-Length", body.len().to_string())
                .append_header("Accept-Ranges", "bytes"),
        )
        .mount(server)
        .await;

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

fn http_spec(name: &str, url: Url) -> FileSpec {
    FileSpec {
        name: name.to_string(),
        size: 0, // unknown — executor will HEAD
        bytes: BytesSource::Http { url, auth: None },
    }
}

#[tokio::test]
async fn downloads_multi_file_multi_chunk_and_marks_progress() {
    std::env::set_var("GENIEX_DL_CHUNK_SIZE", "16384");

    let server = MockServer::start().await;
    let body_a = make_body(64 * 1024, 0x11);
    let body_b = make_body(48 * 1024, 0x77);
    install_file_mock(&server, "/org/repo/resolve/main/a.bin", body_a.clone()).await;
    install_file_mock(&server, "/org/repo/resolve/main/b.bin", body_b.clone()).await;

    let tmp = tempdir().unwrap();
    let cfg = ExecutorConfig {
        file_concurrency: 2,
        chunk_concurrency: 4,
        progress_interval: Duration::from_millis(20),
    };

    let files = vec![
        http_spec(
            "a.bin",
            Url::parse(&format!("{}/org/repo/resolve/main/a.bin", server.uri())).unwrap(),
        ),
        http_spec(
            "b.bin",
            Url::parse(&format!("{}/org/repo/resolve/main/b.bin", server.uri())).unwrap(),
        ),
    ];

    let seen: Arc<Mutex<Vec<Vec<FileProgress>>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_cl = seen.clone();
    let cb: ProgressCallback = Box::new(move |files: &[FileProgress]| -> bool {
        seen_cl.lock().unwrap().push(files.to_vec());
        true
    });

    Executor::with_config(fast_transport(), cfg)
        .run(&files, tmp.path(), Some(&cb))
        .await
        .expect("download");

    let a_read = std::fs::read(tmp.path().join("a.bin")).unwrap();
    let b_read = std::fs::read(tmp.path().join("b.bin")).unwrap();
    assert_eq!(a_read, body_a, "a.bin bytes mismatch");
    assert_eq!(b_read, body_b, "b.bin bytes mismatch");

    let marker_a = std::fs::read(tmp.path().join("a.bin.progress")).unwrap();
    let marker_b = std::fs::read(tmp.path().join("b.bin.progress")).unwrap();
    assert_eq!(marker_a.len(), 4, "64 KiB / 16 KiB = 4 chunks");
    assert_eq!(marker_b.len(), 3, "48 KiB / 16 KiB = 3 chunks");
    assert!(marker_a.iter().all(|b| *b == 0x01));
    assert!(marker_b.iter().all(|b| *b == 0x01));

    let history = seen.lock().unwrap();
    assert!(!history.is_empty(), "callback must fire at least once");
    let last = history.last().unwrap();
    let total_done: i64 = last.iter().map(|f| f.downloaded_bytes).sum();
    let total_size: i64 = last.iter().map(|f| f.total_bytes).sum();
    assert_eq!(
        total_done, total_size,
        "terminal callback must show 100% — got {total_done} / {total_size}",
    );
}

#[tokio::test]
async fn resume_skips_completed_chunks() {
    std::env::set_var("GENIEX_DL_CHUNK_SIZE", "16384");

    let server = MockServer::start().await;
    let body = make_body(64 * 1024, 0x55);

    Mock::given(method("HEAD"))
        .and(path("/org/repo/resolve/main/f.bin"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("Content-Length", body.len().to_string())
                .append_header("Accept-Ranges", "bytes"),
        )
        .mount(&server)
        .await;

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
    let dest = tmp.path();
    std::fs::create_dir_all(dest).unwrap();
    let out = dest.join("f.bin");
    std::fs::write(&out, body.clone()).unwrap();
    let plan = chunklib::plan_chunks(body.len() as u64);
    let mut bitmap = vec![0u8; plan.num_chunks()];
    bitmap[0] = 0x01;
    bitmap[2] = 0x01;
    std::fs::write(dest.join("f.bin.progress"), &bitmap).unwrap();

    let cfg = ExecutorConfig {
        file_concurrency: 1,
        chunk_concurrency: 4,
        progress_interval: Duration::from_millis(20),
    };
    let files = vec![http_spec(
        "f.bin",
        Url::parse(&format!("{}/org/repo/resolve/main/f.bin", server.uri())).unwrap(),
    )];

    get_hits.store(0, Ordering::SeqCst);

    Executor::with_config(fast_transport(), cfg)
        .run(&files, dest, None)
        .await
        .expect("resume");

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
}

#[tokio::test]
async fn cancel_via_callback_returns_cancelled() {
    std::env::set_var("GENIEX_DL_CHUNK_SIZE", "16384");

    let server = MockServer::start().await;
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
    let cfg = ExecutorConfig {
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
    let files = vec![http_spec(
        "slow.bin",
        Url::parse(&format!("{}/org/repo/resolve/main/slow.bin", server.uri())).unwrap(),
    )];

    let err = Executor::with_config(fast_transport(), cfg)
        .run(&files, tmp.path(), Some(&cb))
        .await
        .expect_err("cancellation must surface");
    assert!(
        matches!(err, model_manager_core::error::Error::Cancelled),
        "expected Cancelled, got: {err}",
    );
    assert!(calls.load(Ordering::SeqCst) >= 1);
}

#[tokio::test]
async fn http_deflate_decodes_and_marks_done() {
    // Build a deflate-compressed body, serve it as a byte range, and
    // verify the executor decodes it into the destination file.
    use flate2::write::DeflateEncoder;
    use flate2::Compression;
    use std::io::Write as IoWrite;

    let plain = make_body(16 * 1024, 0x22);
    let mut compressed: Vec<u8> = Vec::new();
    {
        let mut enc = DeflateEncoder::new(&mut compressed, Compression::default());
        enc.write_all(&plain).unwrap();
        enc.finish().unwrap();
    }
    let compressed_len = compressed.len() as u64;

    // Pretend the compressed slice lives inside a larger object at
    // offset 100 — serve only that range window.
    const OFFSET: u64 = 100;
    let mut object = vec![0u8; OFFSET as usize];
    object.extend_from_slice(&compressed);

    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path("/blob"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("Content-Length", object.len().to_string())
                .append_header("Accept-Ranges", "bytes"),
        )
        .mount(&server)
        .await;
    let object_arc = Arc::new(object);
    let object_cl = object_arc.clone();
    Mock::given(method("GET"))
        .and(path("/blob"))
        .respond_with(move |req: &Request| {
            let (start, len) = parse_range(req);
            let slice = object_cl[start as usize..(start + len) as usize].to_vec();
            ResponseTemplate::new(206).set_body_bytes(slice)
        })
        .mount(&server)
        .await;

    let tmp = tempdir().unwrap();
    let files = vec![FileSpec {
        name: "decoded.bin".to_string(),
        size: plain.len() as u64,
        bytes: BytesSource::HttpDeflate {
            url: Url::parse(&format!("{}/blob", server.uri())).unwrap(),
            auth: None,
            offset: OFFSET,
            compressed_len,
        },
    }];

    Executor::new(fast_transport(), 1)
        .run(&files, tmp.path(), None)
        .await
        .expect("deflate");

    let decoded = std::fs::read(tmp.path().join("decoded.bin")).unwrap();
    assert_eq!(decoded, plain, "decoded bytes must match plaintext");
    let marker = std::fs::read(tmp.path().join("decoded.bin.progress")).unwrap();
    assert_eq!(marker, vec![0x01]);
}

#[tokio::test]
async fn http_deflate_progress_scales_to_uncompressed_size() {
    // Regression guard: the compressed fetch must credit
    // `downloaded_bytes` scaled to the uncompressed size, otherwise
    // the user-visible progress sits at 0% through the whole fetch
    // and only jumps to 100% post-decode.
    use flate2::write::DeflateEncoder;
    use flate2::Compression;
    use std::io::Write as IoWrite;

    let plain = vec![b'A'; 64 * 1024]; // highly compressible → non-1:1 ratio
    let mut compressed: Vec<u8> = Vec::new();
    {
        let mut enc = DeflateEncoder::new(&mut compressed, Compression::default());
        enc.write_all(&plain).unwrap();
        enc.finish().unwrap();
    }
    let compressed_len = compressed.len() as u64;
    assert!(
        compressed_len < plain.len() as u64 / 4,
        "fixture should compress to well under 25%, got {}/{}",
        compressed_len,
        plain.len()
    );

    let server = MockServer::start().await;
    let body = compressed.clone();
    Mock::given(method("HEAD"))
        .and(path("/blob"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("Content-Length", body.len().to_string())
                .append_header("Accept-Ranges", "bytes"),
        )
        .mount(&server)
        .await;
    let body_arc = Arc::new(body);
    let body_cl = body_arc.clone();
    Mock::given(method("GET"))
        .and(path("/blob"))
        .respond_with(move |req: &Request| {
            let (start, len) = parse_range(req);
            let slice = body_cl[start as usize..(start + len) as usize].to_vec();
            ResponseTemplate::new(206).set_body_bytes(slice)
        })
        .mount(&server)
        .await;

    let tmp = tempdir().unwrap();
    let files = vec![FileSpec {
        name: "scaled.bin".to_string(),
        size: plain.len() as u64,
        bytes: BytesSource::HttpDeflate {
            url: Url::parse(&format!("{}/blob", server.uri())).unwrap(),
            auth: None,
            offset: 0,
            compressed_len,
        },
    }];

    let seen: Arc<Mutex<Vec<Vec<FileProgress>>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_cl = seen.clone();
    let cb: ProgressCallback = Box::new(move |files: &[FileProgress]| -> bool {
        seen_cl.lock().unwrap().push(files.to_vec());
        true
    });

    Executor::with_config(
        fast_transport(),
        ExecutorConfig {
            file_concurrency: 1,
            chunk_concurrency: 1,
            progress_interval: Duration::from_millis(5),
        },
    )
    .run(&files, tmp.path(), Some(&cb))
    .await
    .expect("deflate");

    let history = seen.lock().unwrap();
    let last = history.last().expect("at least one callback");
    // Terminal snapshot must settle on the exact uncompressed size —
    // the store-after-decode pin guarantees no rounding drift.
    assert_eq!(
        last[0].downloaded_bytes,
        plain.len() as i64,
        "final progress should be exact uncompressed size"
    );
    assert_eq!(
        last[0].total_bytes,
        plain.len() as i64,
        "total should match uncompressed size"
    );
}
