//! End-to-end pull from a local AI Hub source.
//!
//! Two flows are exercised: pulling from an already-extracted directory
//! (lex-first `.bin` + metadata.json + extras), and pulling from a `.zip`
//! containing a STORED `.bin` shard, a STORED `metadata.json`, and a
//! DEFLATE tokenizer. The on-disk layout produced by either path must
//! match what the AI Hub remote source produces: `model_file["N/A"]`
//! holds the entrypoint basename, the rest go into `extra_files`, and
//! `plugin_id == "qairt"`.

use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use model_manager_core::config::StoreConfig;
use model_manager_core::manifest::ModelType;
use model_manager_core::manifest_builder::ManifestHint;
use model_manager_core::pull::pull_with_source;
use model_manager_core::source::localfs::LocalFsSource;
use model_manager_core::store::Store;
use model_manager_core::transport::{HttpTransport, ReqwestTransport, TransportConfig};
use tempfile::tempdir;

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

fn write_extracted(dir: &std::path::Path, supports_vision: bool) -> (Vec<u8>, Vec<u8>) {
    let bin_a = b"shard-a-payload-AAAAAAAA".to_vec();
    let bin_b = b"shard-b-payload-BBBB".to_vec();
    let meta = format!(r#"{{"genie":{{"supports_vision":{supports_vision}}}}}"#);
    std::fs::write(dir.join("metadata.json"), meta.as_bytes()).unwrap();
    std::fs::write(dir.join("shard_a.bin"), &bin_a).unwrap();
    std::fs::write(dir.join("shard_b.bin"), &bin_b).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    (bin_a, bin_b)
}

fn make_store(root: &std::path::Path) -> Store {
    Store::new(StoreConfig::new(root.to_path_buf())).unwrap()
}

#[tokio::test]
async fn pull_from_extracted_dir_produces_qairt_manifest() {
    let staging = tempdir().unwrap();
    let (bin_a, bin_b) = write_extracted(staging.path(), false);

    let cache = tempdir().unwrap();
    let store = make_store(cache.path());
    let model_name = "qualcomm/foo";
    let src = LocalFsSource::new(
        staging.path().to_path_buf(),
        model_name.to_string(),
        ManifestHint::default(),
    );
    let transport = fast_transport();
    pull_with_source(&store, model_name, Box::new(src), transport, None)
        .await
        .unwrap();

    let dest = cache.path().join("models").join("qualcomm").join("foo");
    let mf = store.get_manifest(model_name).unwrap();
    assert_eq!(mf.plugin_id, "qairt");
    assert_eq!(mf.model_type, ModelType::Llm);
    let entry = mf.model_file.get("N/A").expect("entrypoint");
    assert_eq!(entry.name, "shard_a.bin");
    assert!(entry.downloaded);
    let extras: Vec<&str> = mf.extra_files.iter().map(|f| f.name.as_str()).collect();
    assert!(extras.contains(&"shard_b.bin"));
    assert!(extras.contains(&"tokenizer.json"));
    assert!(extras.contains(&"metadata.json"));

    assert_eq!(std::fs::read(dest.join("shard_a.bin")).unwrap(), bin_a);
    assert_eq!(std::fs::read(dest.join("shard_b.bin")).unwrap(), bin_b);
    assert!(dest.join("geniex.json").exists());
}

#[tokio::test]
async fn pull_from_extracted_dir_supports_vision_yields_vlm() {
    let staging = tempdir().unwrap();
    write_extracted(staging.path(), true);

    let cache = tempdir().unwrap();
    let store = make_store(cache.path());
    let model_name = "qualcomm/qwen-vl";
    let src = LocalFsSource::new(
        staging.path().to_path_buf(),
        model_name.to_string(),
        ManifestHint::default(),
    );
    let transport = fast_transport();
    pull_with_source(&store, model_name, Box::new(src), transport, None)
        .await
        .unwrap();
    let mf = store.get_manifest(model_name).unwrap();
    assert_eq!(mf.model_type, ModelType::Vlm);
}

fn build_zip(entries: &[(&str, &[u8], bool)]) -> Vec<u8> {
    // Each tuple: (name, data, deflated). We build per-entry options so
    // we can mix compression methods in one archive — the case the
    // executor really needs to handle (LocalRange + LocalDeflate in the
    // same Plan).
    let mut buf: Vec<u8> = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut buf);
        let mut zw = zip::ZipWriter::new(cursor);
        for (name, data, deflated) in entries {
            let method = if *deflated {
                zip::CompressionMethod::Deflated
            } else {
                zip::CompressionMethod::Stored
            };
            let opts: zip::write::SimpleFileOptions =
                zip::write::SimpleFileOptions::default().compression_method(method);
            zw.start_file(*name, opts).unwrap();
            zw.write_all(data).unwrap();
        }
        zw.finish().unwrap();
    }
    buf
}

#[tokio::test]
async fn pull_from_local_zip_extracts_stored_and_deflate_entries() {
    let big_payload: Vec<u8> = (0..16_384u32).map(|i| (i & 0xff) as u8).collect();
    let body = build_zip(&[
        // STORED — small files round-trip identically.
        (
            "metadata.json",
            br#"{"genie":{"supports_vision":true}}"#,
            false,
        ),
        ("shard_a.bin", b"hello-shard", false),
        // DEFLATE — a larger payload so the zip writer actually
        // compresses (very small inputs may end up STORED).
        ("weights.bin", &big_payload, true),
    ]);

    let staging = tempdir().unwrap();
    let zip_path = staging.path().join("model.zip");
    std::fs::write(&zip_path, &body).unwrap();

    let cache = tempdir().unwrap();
    let store = make_store(cache.path());
    let model_name = "qualcomm/zip-vlm";
    let src = LocalFsSource::new(zip_path, model_name.to_string(), ManifestHint::default());
    let transport = fast_transport();
    pull_with_source(&store, model_name, Box::new(src), transport, None)
        .await
        .unwrap();

    let mf = store.get_manifest(model_name).unwrap();
    assert_eq!(mf.plugin_id, "qairt");
    assert_eq!(mf.model_type, ModelType::Vlm);
    let entry = mf.model_file.get("N/A").expect("entrypoint");
    // Lex-first .bin among {shard_a.bin, weights.bin} is "shard_a.bin".
    assert_eq!(entry.name, "shard_a.bin");
    let extras: Vec<&str> = mf.extra_files.iter().map(|f| f.name.as_str()).collect();
    assert!(extras.contains(&"weights.bin"));
    assert!(extras.contains(&"metadata.json"));

    let dest = cache.path().join("models").join("qualcomm").join("zip-vlm");
    assert_eq!(
        std::fs::read(dest.join("shard_a.bin")).unwrap(),
        b"hello-shard"
    );
    assert_eq!(
        std::fs::read(dest.join("weights.bin")).unwrap(),
        big_payload
    );
}
