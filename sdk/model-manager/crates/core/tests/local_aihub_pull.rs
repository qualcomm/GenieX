// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! End-to-end pull from a local AI Hub source, either an extracted
//! directory or a `.zip` mixing STORED and DEFLATE entries. Local pulls
//! carry no precision label, so the entrypoint sits under `"N/A"`.

use std::path::Path;

use model_manager_core::manifest::{ModelManifest, ModelType};
use model_manager_core::manifest_builder::ManifestHint;
use model_manager_core::pull::pull_with_source;
use model_manager_core::source::localfs::LocalFsSource;
use model_manager_core::test_support::{build_zip, fast_transport, temp_store, ZipMethod};
use tempfile::tempdir;

const BIN_A: &[u8] = b"shard-a-payload-AAAAAAAA";
const BIN_B: &[u8] = b"shard-b-payload-BBBB";

fn write_extracted(dir: &Path, supports_vision: bool) {
    let meta = format!(r#"{{"genie":{{"supports_vision":{supports_vision}}}}}"#);
    std::fs::write(dir.join("metadata.json"), meta).unwrap();
    std::fs::write(dir.join("shard_a.bin"), BIN_A).unwrap();
    std::fs::write(dir.join("shard_b.bin"), BIN_B).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
}

/// Pull `source` into a fresh store; returns its manifest and model dir.
async fn pull_local(
    source: &Path,
    model_name: &str,
) -> (ModelManifest, std::path::PathBuf, tempfile::TempDir) {
    let (store, cache) = temp_store();
    let src = LocalFsSource::new(
        source.to_path_buf(),
        model_name.to_string(),
        ManifestHint::default(),
    );
    pull_with_source(&store, model_name, Box::new(src), fast_transport(), None)
        .await
        .unwrap();
    let dest = cache.path().join("models").join(model_name);
    (store.get_manifest(model_name).unwrap(), dest, cache)
}

fn extra_names(mf: &ModelManifest) -> Vec<&str> {
    mf.extra_files.iter().map(|f| f.name.as_str()).collect()
}

#[tokio::test]
async fn pull_from_extracted_dir_produces_qairt_manifest() {
    let staging = tempdir().unwrap();
    write_extracted(staging.path(), false);
    let (mf, dest, _cache) = pull_local(staging.path(), "qualcomm/foo").await;

    assert_eq!(
        (mf.plugin_id.as_str(), &mf.model_type),
        ("qairt", &ModelType::Llm)
    );
    let entry = &mf.model_file["N/A"];
    assert_eq!(entry.name, "shard_a.bin");
    assert!(entry.downloaded);
    // Own size only; siblings carry theirs in extra_files (#952).
    assert_eq!(entry.size, BIN_A.len() as i64);
    for want in ["shard_b.bin", "tokenizer.json", "metadata.json"] {
        assert!(extra_names(&mf).contains(&want), "missing {want}");
    }

    let on_disk: u64 = std::fs::read_dir(&dest)
        .unwrap()
        .flatten()
        .filter(|e| e.file_type().unwrap().is_file())
        .filter(|e| !matches!(e.file_name().to_str(), Some("geniex.json" | ".lock")))
        .map(|e| e.metadata().unwrap().len())
        .sum();
    assert_eq!(
        mf.total_size() as u64,
        on_disk,
        "manifest total vs on-disk bytes"
    );
    assert_eq!(std::fs::read(dest.join("shard_a.bin")).unwrap(), BIN_A);
    assert_eq!(std::fs::read(dest.join("shard_b.bin")).unwrap(), BIN_B);
    assert!(dest.join("geniex.json").exists());
}

#[tokio::test]
async fn pull_from_extracted_dir_supports_vision_yields_vlm() {
    let staging = tempdir().unwrap();
    write_extracted(staging.path(), true);
    let (mf, _dest, _cache) = pull_local(staging.path(), "qualcomm/qwen-vl").await;
    assert_eq!(mf.model_type, ModelType::Vlm);
}

#[tokio::test]
async fn pull_from_local_zip_extracts_stored_and_deflate_entries() {
    let big: Vec<u8> = (0..16_384u32).map(|i| i as u8).collect();
    let body = build_zip(&[
        (
            "metadata.json",
            br#"{"genie":{"supports_vision":true}}"#,
            ZipMethod::Stored,
        ),
        ("shard_a.bin", b"hello-shard", ZipMethod::Stored),
        ("weights.bin", &big, ZipMethod::Deflate),
    ]);
    let staging = tempdir().unwrap();
    let zip_path = staging.path().join("model.zip");
    std::fs::write(&zip_path, body).unwrap();
    let (mf, dest, _cache) = pull_local(&zip_path, "qualcomm/zip-vlm").await;

    assert_eq!(
        (mf.plugin_id.as_str(), &mf.model_type),
        ("qairt", &ModelType::Vlm)
    );
    assert_eq!(mf.model_file["N/A"].name, "shard_a.bin");
    for want in ["weights.bin", "metadata.json"] {
        assert!(extra_names(&mf).contains(&want), "missing {want}");
    }
    assert_eq!(
        std::fs::read(dest.join("shard_a.bin")).unwrap(),
        b"hello-shard"
    );
    assert_eq!(std::fs::read(dest.join("weights.bin")).unwrap(), big);
}
