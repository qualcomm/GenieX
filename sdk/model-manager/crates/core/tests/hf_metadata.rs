//! Offline tests for `HfMetadata`: the JSON listing path and the
//! `resolve/main/` URL shape. We stub the HF API with wiremock so these
//! run without network access.

use std::sync::Arc;

use model_manager_core::hub::metadata::HubMetadata;
use model_manager_core::hub::HfMetadata;
use model_manager_core::transport::{HttpTransport, ReqwestTransport, TransportConfig};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn fast_transport() -> Arc<dyn HttpTransport> {
    Arc::new(
        ReqwestTransport::with_config(TransportConfig {
            connect_timeout: Some(std::time::Duration::from_secs(2)),
            read_timeout: Some(std::time::Duration::from_secs(5)),
            retries: Some(0),
            retry_backoff: Some(std::time::Duration::from_millis(10)),
            proxy_override: None,
        })
        .unwrap(),
    )
}

#[tokio::test]
async fn list_files_parses_siblings_and_skips_missing_manifest() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "siblings": [
            { "rfilename": "model.gguf", "size": 123456 },
            { "rfilename": "tokenizer.model" }
        ]
    });
    Mock::given(method("HEAD"))
        .and(path("/api/models/org/repo"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("Content-Length", body.to_string().len().to_string())
                .append_header("Accept-Ranges", "bytes"),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models/org/repo"))
        .respond_with(ResponseTemplate::new(206).set_body_string(body.to_string()))
        .mount(&server)
        .await;

    let md = HfMetadata::with_endpoint(&server.uri(), None, fast_transport()).unwrap();
    let (files, manifest) = md.list_files("org/repo").await.unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].name, "model.gguf");
    assert_eq!(files[0].size, 123456);
    assert_eq!(files[1].name, "tokenizer.model");
    assert_eq!(files[1].size, -1, "missing size encodes as -1");
    assert!(manifest.is_none(), "no geniex.json in listing");
}

#[tokio::test]
async fn resolve_builds_expected_url_shape() {
    let server = MockServer::start().await;
    let md =
        HfMetadata::with_endpoint(&server.uri(), Some("tok".into()), fast_transport()).unwrap();
    let got = md
        .resolve("meta-llama/Llama-3", &["weights.safetensors".to_string()])
        .await
        .unwrap();
    assert_eq!(got.len(), 1);
    let expected = format!(
        "{}/meta-llama/Llama-3/resolve/main/weights.safetensors",
        server.uri()
    );
    assert_eq!(got[0].url.as_str(), expected);
    assert_eq!(got[0].auth.as_deref(), Some("tok"));
}

#[tokio::test]
async fn default_file_concurrency_matches_token_presence() {
    let server = MockServer::start().await;
    let anon = HfMetadata::with_endpoint(&server.uri(), None, fast_transport()).unwrap();
    assert_eq!(anon.default_file_concurrency(), 1);
    let authed =
        HfMetadata::with_endpoint(&server.uri(), Some("t".into()), fast_transport()).unwrap();
    assert_eq!(authed.default_file_concurrency(), 8);
}
