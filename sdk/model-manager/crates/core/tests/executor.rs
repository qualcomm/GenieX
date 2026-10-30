// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Executor against a wiremock'd upstream: byte-correct multi-chunk
//! downloads, `.progress` bitmaps, resume, cancellation and DEFLATE.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use model_manager_core::error::Error;
use model_manager_core::executor::{
    chunk as chunklib, Executor, ExecutorConfig, FileProgress, ProgressCallback,
};
use model_manager_core::source::{BytesSource, FileSpec};
use model_manager_core::test_support::{
    deflate_raw, fast_transport, make_body, mount_head, mount_ranged, pin_small_chunks,
    range_slice, url,
};
use tempfile::tempdir;
use url::Url;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn http_spec(name: &str, url: Url) -> FileSpec {
    FileSpec {
        name: name.to_string(),
        size: 0,
        bytes: BytesSource::Http { url, auth: None },
    }
}

fn deflate_spec(name: &str, url: Url, size: usize, offset: u64, compressed_len: usize) -> FileSpec {
    FileSpec {
        name: name.to_string(),
        size: size as u64,
        bytes: BytesSource::HttpDeflate {
            url,
            auth: None,
            offset,
            compressed_len: compressed_len as u64,
        },
    }
}

fn config(files: usize, chunks: usize, interval_ms: u64) -> ExecutorConfig {
    ExecutorConfig {
        file_concurrency: files,
        chunk_concurrency: chunks,
        progress_interval: Duration::from_millis(interval_ms),
    }
}

fn recording_callback() -> (ProgressCallback, Arc<Mutex<Vec<Vec<FileProgress>>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_cl = seen.clone();
    let cb: ProgressCallback = Box::new(move |files: &[FileProgress]| {
        seen_cl.lock().unwrap().push(files.to_vec());
        true
    });
    (cb, seen)
}

/// Ranged GET that counts hits and lets `respond` override a slice.
async fn mount_counted(
    server: &MockServer,
    at: &str,
    body: Vec<u8>,
    respond: impl Fn(&Request, &[u8]) -> ResponseTemplate + Send + Sync + 'static,
) -> Arc<AtomicUsize> {
    mount_head(server, at, body.len()).await;
    let hits = Arc::new(AtomicUsize::new(0));
    let hits_cl = hits.clone();
    Mock::given(method("GET"))
        .and(path(at))
        .respond_with(move |req: &Request| {
            hits_cl.fetch_add(1, Ordering::SeqCst);
            respond(req, &body)
        })
        .mount(server)
        .await;
    hits
}

#[tokio::test]
async fn downloads_multi_file_multi_chunk_and_marks_progress() {
    pin_small_chunks();
    let server = MockServer::start().await;
    let body_a = make_body(64 * 1024, 0x11);
    let body_b = make_body(48 * 1024, 0x77);
    mount_ranged(&server, "/a.bin", body_a.clone()).await;
    mount_ranged(&server, "/b.bin", body_b.clone()).await;

    let tmp = tempdir().unwrap();
    let files = [
        http_spec("a.bin", url(&server, "/a.bin")),
        http_spec("b.bin", url(&server, "/b.bin")),
    ];
    let (cb, seen) = recording_callback();
    Executor::with_config(fast_transport(), config(2, 4, 20))
        .run(&files, tmp.path(), Some(&cb))
        .await
        .expect("download");

    for (name, body, chunks) in [("a.bin", &body_a, 4), ("b.bin", &body_b, 3)] {
        assert_eq!(
            &std::fs::read(tmp.path().join(name)).unwrap(),
            body,
            "{name}"
        );
        let marker = std::fs::read(tmp.path().join(format!("{name}.progress"))).unwrap();
        assert_eq!(marker, vec![0x01; chunks], "{name} marker");
    }

    let history = seen.lock().unwrap();
    let last = history.last().expect("callback must fire");
    let done: i64 = last.iter().map(|f| f.downloaded_bytes).sum();
    let total: i64 = last.iter().map(|f| f.total_bytes).sum();
    assert_eq!(done, total, "terminal callback must show 100%");
}

#[tokio::test]
async fn resume_skips_completed_chunks() {
    pin_small_chunks();
    let server = MockServer::start().await;
    let body = make_body(64 * 1024, 0x55);
    let hits = mount_counted(&server, "/f.bin", body.clone(), |req, b| {
        range_slice(b, req)
    })
    .await;

    let tmp = tempdir().unwrap();
    std::fs::write(tmp.path().join("f.bin"), &body).unwrap();
    let mut bitmap = vec![0u8; chunklib::plan_chunks(body.len() as u64).chunks.len()];
    bitmap[0] = 0x01;
    bitmap[2] = 0x01;
    std::fs::write(tmp.path().join("f.bin.progress"), &bitmap).unwrap();

    Executor::with_config(fast_transport(), config(1, 4, 20))
        .run(
            &[http_spec("f.bin", url(&server, "/f.bin"))],
            tmp.path(),
            None,
        )
        .await
        .expect("resume");

    assert_eq!(hits.load(Ordering::SeqCst), 2, "only the missing chunks");
    let marker = std::fs::read(tmp.path().join("f.bin.progress")).unwrap();
    assert!(marker.iter().all(|b| *b == 0x01));
}

#[tokio::test]
async fn complete_bitmap_without_payload_is_refetched() {
    pin_small_chunks();
    let body = make_body(64 * 1024, 0xA6);
    let server = MockServer::start().await;
    mount_ranged(&server, "/missing.bin", body.clone()).await;

    let tmp = tempdir().unwrap();
    let chunks = chunklib::plan_chunks(body.len() as u64).chunks.len();
    std::fs::write(tmp.path().join("missing.bin.progress"), vec![0x01; chunks]).unwrap();

    Executor::new(fast_transport(), 1)
        .run(
            &[http_spec("missing.bin", url(&server, "/missing.bin"))],
            tmp.path(),
            None,
        )
        .await
        .expect("refetch stale bitmap");
    assert!(std::fs::read(tmp.path().join("missing.bin")).unwrap() == body);
}

#[tokio::test]
async fn cancel_via_callback_returns_cancelled() {
    pin_small_chunks();
    let server = MockServer::start().await;
    mount_counted(
        &server,
        "/slow.bin",
        make_body(64 * 1024, 0x33),
        |req, b| range_slice(b, req).set_delay(Duration::from_millis(200)),
    )
    .await;

    let calls = Arc::new(AtomicUsize::new(0));
    let calls_cl = calls.clone();
    let cb: ProgressCallback = Box::new(move |_| {
        calls_cl.fetch_add(1, Ordering::SeqCst);
        false
    });
    let tmp = tempdir().unwrap();
    let err = Executor::with_config(fast_transport(), config(1, 1, 20))
        .run(
            &[http_spec("slow.bin", url(&server, "/slow.bin"))],
            tmp.path(),
            Some(&cb),
        )
        .await
        .expect_err("cancellation must surface");
    assert!(matches!(err, Error::Cancelled), "got: {err}");
    assert!(calls.load(Ordering::SeqCst) >= 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chunk_failure_stops_queued_chunks_from_starting() {
    pin_small_chunks();
    let total_chunks = 32;
    let server = MockServer::start().await;
    let hits = mount_counted(
        &server,
        "/flaky.bin",
        make_body(total_chunks * 16 * 1024, 0x5C),
        |_, _| ResponseTemplate::new(500),
    )
    .await;

    let tmp = tempdir().unwrap();
    let err = Executor::with_config(fast_transport(), config(1, 2, 100))
        .run(
            &[http_spec("flaky.bin", url(&server, "/flaky.bin"))],
            tmp.path(),
            None,
        )
        .await
        .expect_err("the chunk failure must surface");
    assert!(!matches!(err, Error::Cancelled), "got {err}");
    let hits = hits.load(Ordering::SeqCst);
    assert!(hits <= 2, "queued chunks must bail ({hits} GETs)");
}

#[tokio::test]
async fn http_deflate_decodes_and_marks_done() {
    let plain = make_body(16 * 1024, 0x22);
    let compressed = deflate_raw(&plain);
    const OFFSET: usize = 100;
    let mut object = vec![0u8; OFFSET];
    object.extend_from_slice(&compressed);
    let server = MockServer::start().await;
    mount_ranged(&server, "/blob", object).await;

    let tmp = tempdir().unwrap();
    let spec = deflate_spec(
        "decoded.bin",
        url(&server, "/blob"),
        plain.len(),
        OFFSET as u64,
        compressed.len(),
    );
    Executor::new(fast_transport(), 1)
        .run(&[spec], tmp.path(), None)
        .await
        .expect("deflate");

    assert_eq!(
        std::fs::read(tmp.path().join("decoded.bin")).unwrap(),
        plain
    );
    assert_eq!(
        std::fs::read(tmp.path().join("decoded.bin.progress")).unwrap(),
        vec![0x01]
    );
}

#[tokio::test]
async fn empty_stored_zip_entry_is_written_without_fetching() {
    let server = MockServer::start().await;
    mount_ranged(&server, "/archive.zip", make_body(4096, 0x10)).await;
    let spec = FileSpec {
        name: "empty.txt".to_string(),
        size: 0,
        bytes: BytesSource::HttpRange {
            url: url(&server, "/archive.zip"),
            auth: None,
            offset: 100,
            len: 0,
        },
    };
    let tmp = tempdir().unwrap();
    Executor::new(fast_transport(), 1)
        .run(&[spec], tmp.path(), None)
        .await
        .expect("empty entry");
    assert_eq!(std::fs::read(tmp.path().join("empty.txt")).unwrap(), b"");
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn deflate_entry_with_stale_chunk_marker_is_refetched() {
    let plain = make_body(16 * 1024, 0x2B);
    let compressed = deflate_raw(&plain);
    let server = MockServer::start().await;
    mount_ranged(&server, "/blob", compressed.clone()).await;

    // A partial chunk bitmap left by a STORED download of the same name,
    // next to an output that happens to be the right size.
    let tmp = tempdir().unwrap();
    std::fs::write(tmp.path().join("entry.bin"), vec![0u8; plain.len()]).unwrap();
    std::fs::write(tmp.path().join("entry.bin.progress"), [0x01, 0x00]).unwrap();

    let spec = deflate_spec(
        "entry.bin",
        url(&server, "/blob"),
        plain.len(),
        0,
        compressed.len(),
    );
    Executor::new(fast_transport(), 1)
        .run(&[spec], tmp.path(), None)
        .await
        .expect("deflate");
    assert!(
        std::fs::read(tmp.path().join("entry.bin")).unwrap() == plain,
        "stale entry was kept"
    );
}

#[tokio::test]
async fn local_sources_copy_bytes_and_mark_done() {
    let src = tempdir().unwrap();
    let plain = make_body(40 * 1024, 0x42);
    let compressed = deflate_raw(&plain);
    let mut archive = b"prefix-".to_vec();
    let range_at = archive.len();
    archive.extend_from_slice(&plain[..1000]);
    let deflate_at = archive.len();
    archive.extend_from_slice(&compressed);
    let archive_path = src.path().join("archive.zip");
    std::fs::write(&archive_path, &archive).unwrap();
    let whole_path = src.path().join("whole.bin");
    std::fs::write(&whole_path, &plain).unwrap();

    let spec = |name: &str, size: usize, bytes| FileSpec {
        name: name.to_string(),
        size: size as u64,
        bytes,
    };
    let files = [
        spec(
            "whole.bin",
            plain.len(),
            BytesSource::Local { path: whole_path },
        ),
        spec(
            "range.bin",
            1000,
            BytesSource::LocalRange {
                path: archive_path.clone(),
                offset: range_at as u64,
                len: 1000,
            },
        ),
        spec(
            "empty.bin",
            0,
            BytesSource::LocalRange {
                path: archive_path.clone(),
                offset: range_at as u64,
                len: 0,
            },
        ),
        spec(
            "inflated.bin",
            plain.len(),
            BytesSource::LocalDeflate {
                path: archive_path,
                offset: deflate_at as u64,
                compressed_len: compressed.len() as u64,
            },
        ),
    ];
    let dest = tempdir().unwrap();
    let (cb, seen) = recording_callback();
    Executor::with_config(fast_transport(), config(4, 4, 5))
        .run(&files, dest.path(), Some(&cb))
        .await
        .expect("local copy");

    let want: [(&str, &[u8]); 4] = [
        ("whole.bin", &plain),
        ("range.bin", &plain[..1000]),
        ("empty.bin", b""),
        ("inflated.bin", &plain),
    ];
    for (name, bytes) in want {
        assert_eq!(
            std::fs::read(dest.path().join(name)).unwrap(),
            bytes,
            "{name}"
        );
        let marker = std::fs::read(dest.path().join(format!("{name}.progress"))).unwrap();
        assert_eq!(marker, [0x01], "{name} marker");
    }
    let history = seen.lock().unwrap();
    for f in history.last().expect("callback must fire") {
        assert_eq!(f.downloaded_bytes, f.total_bytes, "{}", f.file_name);
    }
}

#[tokio::test]
async fn http_deflate_progress_scales_to_uncompressed_size() {
    let plain = vec![b'A'; 64 * 1024];
    let compressed = deflate_raw(&plain);
    assert!(compressed.len() < plain.len() / 4, "fixture must compress");
    let server = MockServer::start().await;
    mount_ranged(&server, "/blob", compressed.clone()).await;

    let tmp = tempdir().unwrap();
    let spec = deflate_spec(
        "scaled.bin",
        url(&server, "/blob"),
        plain.len(),
        0,
        compressed.len(),
    );
    let (cb, seen) = recording_callback();
    Executor::with_config(fast_transport(), config(1, 1, 5))
        .run(&[spec], tmp.path(), Some(&cb))
        .await
        .expect("deflate");

    let history = seen.lock().unwrap();
    let last = &history.last().expect("callback must fire")[0];
    let want = plain.len() as i64;
    assert_eq!((last.downloaded_bytes, last.total_bytes), (want, want));
}
