//! Integration tests for [`ReqwestTransport`]: proxy routing, HEAD parsing,
//! and range GET semantics. Uses `wiremock` to stand up a fake upstream.
//!
//! The "proxy" test cheats a little — a real forward HTTP proxy is heavier
//! than we need here. Instead we verify the lower-level claim: setting
//! `TransportConfig::proxy_override` makes reqwest route the request to
//! that endpoint instead of the URL's host. That's the same plumbing
//! reqwest uses for `HTTPS_PROXY`, so if this works, env-based proxy works.

use std::time::Duration;

use model_manager_core::transport::{
    HttpTransport, ReqwestTransport, TransportConfig,
};
use tokio::io::AsyncWriteExt;
use url::Url;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn fast_cfg() -> TransportConfig {
    TransportConfig {
        connect_timeout: Some(Duration::from_secs(2)),
        read_timeout: Some(Duration::from_secs(5)),
        retries: Some(0),
        retry_backoff: Some(Duration::from_millis(10)),
        proxy_override: None,
    }
}

#[tokio::test]
async fn head_returns_size_and_accepts_ranges() {
    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path("/file.bin"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("Content-Length", "1024")
                .append_header("Accept-Ranges", "bytes")
                .append_header("ETag", "\"abc\""),
        )
        .mount(&server)
        .await;

    let t = ReqwestTransport::with_config(fast_cfg()).unwrap();
    let url = Url::parse(&format!("{}/file.bin", server.uri())).unwrap();
    let info = t.head(&url, None).await.unwrap();
    assert_eq!(info.size, 1024);
    assert!(info.accepts_ranges);
    assert_eq!(info.etag.as_deref(), Some("\"abc\""));
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
async fn get_range_short_read_errors() {
    let server = MockServer::start().await;
    // Claim 100 bytes, send only 50 → engine must surface an error so the
    // outer loop can retry or fail loudly. (We set retries=0 above.)
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
    assert!(msg.contains("short read"), "expected short read, got: {msg}");
}

#[tokio::test]
async fn proxy_override_routes_through_configured_host() {
    // The "proxy server" is just a wiremock that answers HEADs. When
    // reqwest is configured with proxy=all, every request goes there —
    // URL host is irrelevant. We assert the proxy received the hit by
    // counting matched requests on the proxy mock, and verify that a
    // stub upstream on a *different* mock server was NEVER hit.
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

    // Upstream would answer 500 if it were hit — but the proxy should
    // intercept first, so it should never be reached.
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
    // Drops of proxy + upstream run the `.expect(n)` assertions in their
    // Drop impls.
}

#[tokio::test]
async fn bearer_token_is_forwarded() {
    use wiremock::matchers::header;

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
