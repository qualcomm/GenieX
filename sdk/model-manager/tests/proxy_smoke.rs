// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! [`ReqwestTransport`] against a fake upstream: HEAD parsing, range GET
//! semantics and retries, and `proxy_override` routing (the same plumbing
//! reqwest uses for `HTTPS_PROXY`).

use std::time::Duration;

use geniex_model::test_support::fast_transport_config as fast_cfg;
use geniex_model::transport::{HttpTransport, ReqwestTransport};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use url::Url;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn head_returns_size() {
    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path("/file.bin"))
        .respond_with(ResponseTemplate::new(200).append_header("Content-Length", "1024"))
        .mount(&server)
        .await;

    let t = ReqwestTransport::with_config(fast_cfg()).unwrap();
    let url = Url::parse(&format!("{}/file.bin", server.uri())).unwrap();
    let info = t.head(&url, None).await.unwrap();
    assert_eq!(info.size, 1024);
}

#[tokio::test]
async fn get_range_writes_exact_bytes() {
    let server = MockServer::start().await;
    let body = (0u8..=255).cycle().take(1024).collect::<Vec<u8>>();
    let slice = body[100..200].to_vec();

    Mock::given(method("GET"))
        .and(path("/f"))
        .respond_with(ResponseTemplate::new(206).set_body_bytes(slice.clone()))
        .mount(&server)
        .await;

    let t = ReqwestTransport::with_config(fast_cfg()).unwrap();
    let url = Url::parse(&format!("{}/f", server.uri())).unwrap();

    let mut sink: Vec<u8> = Vec::new();
    t.get_range(&url, None, 100, 100, &mut sink).await.unwrap();
    sink.flush().await.unwrap();

    assert_eq!(sink, slice);
}

#[tokio::test]
async fn get_range_continues_short_read_with_suffix_range() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/continuation"))
        .and(header("range", "bytes=0-7"))
        .respond_with(ResponseTemplate::new(206).set_body_bytes(b"abc".to_vec()))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/continuation"))
        .and(header("range", "bytes=3-7"))
        .respond_with(ResponseTemplate::new(206).set_body_bytes(b"defgh".to_vec()))
        .expect(1)
        .mount(&server)
        .await;

    let mut cfg = fast_cfg();
    cfg.retries = Some(1);
    let t = ReqwestTransport::with_config(cfg).unwrap();
    let url = Url::parse(&format!("{}/continuation", server.uri())).unwrap();
    let mut sink: Vec<u8> = Vec::new();
    t.get_range(&url, None, 0, 8, &mut sink).await.unwrap();
    assert_eq!(sink, b"abcdefgh");
}

async fn accept_range_request(listener: &TcpListener, expected_range: &str) -> TcpStream {
    let (mut stream, _) = tokio::time::timeout(Duration::from_secs(2), listener.accept())
        .await
        .expect("timed out waiting for ranged GET")
        .unwrap();
    let mut request = Vec::new();
    let mut buffer = [0u8; 1024];
    while !request.windows(4).any(|window| window == b"\r\n\r\n") {
        let read = stream.read(&mut buffer).await.unwrap();
        assert_ne!(read, 0, "client closed before sending request headers");
        request.extend_from_slice(&buffer[..read]);
    }
    let request = String::from_utf8_lossy(&request).to_ascii_lowercase();
    assert!(
        request
            .lines()
            .any(|line| line == format!("range: {expected_range}")),
        "expected Range {expected_range}, got request:\n{request}"
    );
    stream
}

#[tokio::test]
async fn get_range_retries_mid_stream_error_from_written_offset() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let mut first = accept_range_request(&listener, "bytes=100-107").await;
        first
            .write_all(b"HTTP/1.1 206 Partial Content\r\nContent-Length: 8\r\nConnection: close\r\n\r\nabc")
            .await
            .unwrap();
        first.flush().await.unwrap();
        // Let reqwest receive and expose the first bytes before the server
        // closes without satisfying Content-Length.
        tokio::time::sleep(Duration::from_millis(50)).await;
        drop(first);

        let mut continuation = accept_range_request(&listener, "bytes=103-107").await;
        continuation
            .write_all(b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nConnection: close\r\n\r\ndefgh")
            .await
            .unwrap();
    });

    let mut cfg = fast_cfg();
    cfg.retries = Some(1);
    let transport = ReqwestTransport::with_config(cfg).unwrap();
    let url = Url::parse(&format!("http://{address}/interrupted")).unwrap();
    let mut sink = Vec::new();
    transport
        .get_range(&url, None, 100, 8, &mut sink)
        .await
        .unwrap();
    server.await.unwrap();
    assert_eq!(sink, b"abcdefgh");
}

#[tokio::test]
async fn get_range_stream_error_exhausts_retries() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        for (range, content_length, body) in [
            ("bytes=0-7", 8, b"abc".as_slice()),
            ("bytes=3-7", 5, b"de".as_slice()),
        ] {
            let mut stream = accept_range_request(&listener, range).await;
            let headers = format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
            );
            stream.write_all(headers.as_bytes()).await.unwrap();
            stream.write_all(body).await.unwrap();
            stream.flush().await.unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    });

    let mut cfg = fast_cfg();
    cfg.retries = Some(1);
    let transport = ReqwestTransport::with_config(cfg).unwrap();
    let url = Url::parse(&format!("http://{address}/interrupted-twice")).unwrap();
    let mut sink = Vec::new();
    let err = transport
        .get_range(&url, None, 0, 8, &mut sink)
        .await
        .unwrap_err();
    server.await.unwrap();
    assert!(
        err.to_string().contains("stream"),
        "expected stream error, got: {err}"
    );
    assert_eq!(sink, b"abcde");
}

#[tokio::test]
async fn get_range_short_read_errors() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/short"))
        .respond_with(ResponseTemplate::new(206).set_body_bytes(vec![0u8; 50]))
        .mount(&server)
        .await;

    let t = ReqwestTransport::with_config(fast_cfg()).unwrap();
    let url = Url::parse(&format!("{}/short", server.uri())).unwrap();
    let mut sink: Vec<u8> = Vec::new();
    let err = t
        .get_range(&url, None, 0, 100, &mut sink)
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("short read"),
        "expected short read, got: {msg}"
    );
}

#[tokio::test]
async fn proxy_override_routes_through_configured_host() {
    let proxy = MockServer::start().await;
    let upstream = MockServer::start().await;

    Mock::given(method("HEAD"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("Content-Length", "42")
                .append_header("Accept-Ranges", "bytes"),
        )
        .expect(1)
        .mount(&proxy)
        .await;

    Mock::given(method("HEAD"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&upstream)
        .await;

    let mut cfg = fast_cfg();
    cfg.proxy_override = Some(proxy.uri());
    let t = ReqwestTransport::with_config(cfg).unwrap();

    let url = Url::parse(&format!("{}/somewhere", upstream.uri())).unwrap();
    let info = t.head(&url, None).await.unwrap();
    assert_eq!(info.size, 42);
}

#[tokio::test]
async fn bearer_token_is_forwarded() {
    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path("/auth"))
        .and(header("authorization", "Bearer hf_xxx"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("Content-Length", "7")
                .append_header("Accept-Ranges", "bytes"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let t = ReqwestTransport::with_config(fast_cfg()).unwrap();
    let url = Url::parse(&format!("{}/auth", server.uri())).unwrap();
    let info = t.head(&url, Some("hf_xxx")).await.unwrap();
    assert_eq!(info.size, 7);
}
