// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! `pull()` resume at chunk granularity through HfSource → Executor →
//! ReqwestTransport: one chunk fails once, and the retry pull must
//! fetch only the chunks the first attempt left undone.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use model_manager_core::manifest_builder::ManifestHint;
use model_manager_core::pull::pull_with_source;
use model_manager_core::source::hf::HfSource;
use model_manager_core::store::{INFLIGHT_DIR, MANIFEST_FILE};
use model_manager_core::test_support::{
    fast_transport, make_body, mount_head, mount_ranged, mount_status, parse_range,
    pin_small_chunks, range_slice, temp_store,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pull_resumes_after_mid_download_failure() {
    pin_small_chunks();
    let server = MockServer::start().await;
    let body = make_body(64 * 1024, 0xA5);
    let repo = "test/tiny";
    let file_name = "weights-Q4_0.gguf";
    let file_path = format!("/{repo}/resolve/main/{file_name}");

    let api = serde_json::json!({ "siblings": [{ "rfilename": file_name, "size": body.len() }] });
    mount_ranged(&server, &format!("/api/models/{repo}"), api.to_string()).await;
    mount_status(
        &server,
        "HEAD",
        &format!("/{repo}/resolve/main/geniex.json"),
        404,
    )
    .await;
    mount_head(&server, &file_path, body.len()).await;

    let failed_once = Arc::new(AtomicBool::new(false));
    let get_hits = Arc::new(AtomicUsize::new(0));
    let (failed_cl, hits_cl, body_cl) = (failed_once.clone(), get_hits.clone(), body.clone());
    Mock::given(method("GET"))
        .and(path(file_path.as_str()))
        .respond_with(move |req: &Request| {
            hits_cl.fetch_add(1, Ordering::SeqCst);
            if parse_range(req).0 == 16384 && !failed_cl.swap(true, Ordering::SeqCst) {
                return ResponseTemplate::new(500);
            }
            range_slice(&body_cl, req)
        })
        .mount(&server)
        .await;

    let (store, tmp) = temp_store();
    let do_pull = || async {
        let transport = fast_transport();
        let src = HfSource::with_endpoint_and_transport(
            repo.to_string(),
            &server.uri(),
            None,
            transport.clone(),
            ManifestHint::default(),
        )
        .unwrap();
        pull_with_source(&store, repo, Box::new(src), transport, None).await
    };

    let err = do_pull().await.expect_err("first pull must fail");
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("http") || msg.contains("500"),
        "{msg}"
    );
    let after_first = get_hits.load(Ordering::SeqCst);

    let model_dir = tmp.path().join("models").join(repo);
    let marker_path = model_dir.join(format!("{file_name}.progress"));
    let bitmap = std::fs::read(&marker_path).expect(".progress must exist after partial");
    assert_eq!(bitmap.len(), 4, "64 KiB / 16 KiB = 4 chunks");
    let done = bitmap.iter().filter(|b| **b == 0x01).count();
    assert!(
        (1..bitmap.len()).contains(&done),
        "bitmap must be partial: {done}/4"
    );

    do_pull().await.expect("second pull must succeed");
    assert_eq!(
        get_hits.load(Ordering::SeqCst) - after_first,
        bitmap.len() - done
    );

    assert_eq!(std::fs::read(model_dir.join(file_name)).unwrap(), body);
    assert!(model_dir.join(MANIFEST_FILE).exists());
    assert!(!marker_path.exists(), ".progress must be cleaned up");
    assert!(
        !model_dir.join(INFLIGHT_DIR).exists(),
        ".inflight must be cleared"
    );
}
