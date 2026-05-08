//! End-to-end S3 (AI Hub) pull against a wiremock'd server.
//!
//! Serves three protojson documents (manifest.json, release-assets.json,
//! platform.json) plus a small zip containing a `.bin` shard and a
//! tokenizer. Confirms the resulting model directory matches what the
//! Go CLI's `Store.PullZipAsset` would produce: entrypoint basename in
//! `ModelFile["N/A"]`, extras populated, plugin_id = `"qairt"`.

use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use model_manager_core::config::StoreConfig;
use model_manager_core::hub::s3::{pull_ai_hub_with_transport, S3Config};
use model_manager_core::store::Store;
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

async fn install_static(server: &MockServer, p: &str, body: Vec<u8>) {
    Mock::given(method("HEAD"))
        .and(path(p.to_string()))
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
        .and(path(p.to_string()))
        .respond_with(move |req: &Request| {
            let (start, len) = parse_range(req);
            let slice = body_cl[start as usize..(start + len) as usize].to_vec();
            ResponseTemplate::new(206).set_body_bytes(slice)
        })
        .mount(server)
        .await;
}

fn build_zip() -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut buf);
        let mut zw = zip::ZipWriter::new(cursor);
        let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        zw.start_file("weights/model-00.bin", opts).unwrap();
        zw.write_all(b"BIN_SHARD_CONTENTS").unwrap();
        zw.start_file("tokenizer.json", opts).unwrap();
        zw.write_all(b"{\"vocab\":[]}").unwrap();
        zw.finish().unwrap();
    }
    buf
}

#[tokio::test]
async fn s3_pull_writes_manifest_and_extracts_flat() {
    let server = MockServer::start().await;
    let base = server.uri();
    let version = "v0.99.0";

    let manifest_json = format!(
        r#"{{
          "platform_url": "{base}/qai-hub-models/releases/{version}/platform.json",
          "models": [
            {{
              "id": "testnet",
              "display_name": "TestNet",
              "domain": "MODEL_DOMAIN_GENERATIVE_AI",
              "manifest_urls": {{
                "release_assets": "{base}/qai-hub-models/releases/{version}/models/testnet/release-assets.json"
              }}
            }}
          ]
        }}"#
    );
    let platform_json = r#"{
      "chipsets": [
        { "name": "SM8650", "aliases": ["Snapdragon 8 Gen 3", "sd8g3"] }
      ]
    }"#;
    let zip_bytes = build_zip();
    let asset_url =
        format!("{base}/qai-hub-models/releases/{version}/models/testnet/assets/SM8650.zip");
    let release_assets_json = format!(
        r#"{{
          "model_id": "testnet",
          "assets": [
            {{
              "chipset": "SM8650",
              "runtime": "RUNTIME_GENIE",
              "precision": "PRECISION_W4A16",
              "download_url": "{asset_url}",
              "uncompressed_size": {}
            }}
          ]
        }}"#,
        zip_bytes.len()
    );

    install_static(
        &server,
        &format!("/qai-hub-models/releases/{version}/manifest.json"),
        manifest_json.into_bytes(),
    )
    .await;
    install_static(
        &server,
        &format!("/qai-hub-models/releases/{version}/platform.json"),
        platform_json.as_bytes().to_vec(),
    )
    .await;
    install_static(
        &server,
        &format!("/qai-hub-models/releases/{version}/models/testnet/release-assets.json"),
        release_assets_json.into_bytes(),
    )
    .await;
    install_static(
        &server,
        &format!("/qai-hub-models/releases/{version}/models/testnet/assets/SM8650.zip"),
        zip_bytes,
    )
    .await;

    let tmp = tempdir().unwrap();
    let store_cfg = StoreConfig::new(tmp.path().to_path_buf());
    let store = Store::new(store_cfg).unwrap();

    let s3_cfg = S3Config {
        endpoint: format!("{base}/qai-hub-models"),
        version: version.to_string(),
        chipset: "sd8g3".to_string(),
        cache_dir: tmp.path().join("aihub-cache"),
        skip_cache: true,
    };

    pull_ai_hub_with_transport(
        &store,
        "NexaAI/TestNet",
        "TestNet",
        s3_cfg,
        fast_transport(),
        None,
    )
    .await
    .expect("s3 pull");

    // Flat-extracted payload + synthesised manifest.
    let model_dir = tmp.path().join("models/NexaAI/TestNet");
    assert!(
        model_dir.join("model-00.bin").exists(),
        "entrypoint .bin missing"
    );
    assert!(
        model_dir.join("tokenizer.json").exists(),
        "extra tokenizer missing"
    );
    assert!(
        !model_dir.join("NexaAI-TestNet.zip").exists() && !model_dir.join("TestNet.zip").exists(),
        "zip should have been removed after extract"
    );

    let mf = store.get_manifest("NexaAI/TestNet").unwrap();
    assert_eq!(mf.plugin_id, "qairt");
    assert_eq!(mf.device_id, "SM8650");
    assert_eq!(mf.precision, "W4A16");
    let entry = mf.model_file.get("N/A").expect("N/A quant entry");
    assert_eq!(entry.name, "model-00.bin");
    assert!(entry.downloaded);
    // The extras slot should carry the tokenizer (non-entrypoint files).
    assert!(
        mf.extra_files.iter().any(|f| f.name == "tokenizer.json"),
        "tokenizer.json missing from extras: {:?}",
        mf.extra_files
    );
}

#[tokio::test]
async fn s3_pull_errors_when_chipset_unknown() {
    let server = MockServer::start().await;
    let base = server.uri();
    let version = "v0.99.0";

    let manifest_json = format!(
        r#"{{
          "platform_url": "{base}/qai-hub-models/releases/{version}/platform.json",
          "models": [
            {{
              "id": "testnet",
              "display_name": "TestNet",
              "domain": "MODEL_DOMAIN_GENERATIVE_AI",
              "manifest_urls": {{
                "release_assets": "{base}/qai-hub-models/releases/{version}/models/testnet/release-assets.json"
              }}
            }}
          ]
        }}"#
    );
    let platform_json = r#"{ "chipsets": [ { "name": "SM8650", "aliases": [] } ] }"#;
    let release_assets_json = format!(
        r#"{{
          "model_id": "testnet",
          "assets": [
            {{
              "chipset": "SM8650",
              "runtime": "RUNTIME_GENIE",
              "precision": "PRECISION_FP16",
              "download_url": "{base}/unused.zip"
            }}
          ]
        }}"#
    );

    install_static(
        &server,
        &format!("/qai-hub-models/releases/{version}/manifest.json"),
        manifest_json.into_bytes(),
    )
    .await;
    install_static(
        &server,
        &format!("/qai-hub-models/releases/{version}/platform.json"),
        platform_json.as_bytes().to_vec(),
    )
    .await;
    install_static(
        &server,
        &format!("/qai-hub-models/releases/{version}/models/testnet/release-assets.json"),
        release_assets_json.into_bytes(),
    )
    .await;

    let tmp = tempdir().unwrap();
    let store_cfg = StoreConfig::new(tmp.path().to_path_buf());
    let store = Store::new(store_cfg).unwrap();

    let s3_cfg = S3Config {
        endpoint: format!("{base}/qai-hub-models"),
        version: version.to_string(),
        chipset: "UnknownChip".to_string(),
        cache_dir: tmp.path().join("aihub-cache"),
        skip_cache: true,
    };

    let err = pull_ai_hub_with_transport(
        &store,
        "NexaAI/TestNet",
        "TestNet",
        s3_cfg,
        fast_transport(),
        None,
    )
    .await
    .expect_err("expected chipset mismatch error");
    assert!(
        format!("{err}").contains("not available"),
        "unexpected error: {err}"
    );
}
