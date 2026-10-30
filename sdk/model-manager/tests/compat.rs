// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! A Go-CLI-written `geniex.json` (fixture from a real `geniex pull qwen3`)
//! round-trips through the Rust manifest without loss.

use geniex_model::manifest::{ModelManifest, ModelType};

const GO_MANIFEST: &str = include_str!("fixtures/go_manifest.json");

#[test]
fn parses_go_generated_manifest() {
    let m: ModelManifest =
        serde_json::from_str(GO_MANIFEST).expect("Go-produced manifest must parse");

    assert_eq!(m.name, "Qwen/Qwen3-4B-GGUF");
    assert_eq!(m.model_name, "qwen3-4b");
    assert_eq!(m.model_type, ModelType::Llm);
    assert_eq!(m.plugin_id, "llama_cpp");

    assert_eq!(m.model_file.len(), 2);
    let q4 = m.model_file.get("Q4_K_M").expect("Q4_K_M must survive");
    assert!(q4.downloaded);
    assert_eq!(q4.size, 2_593_504_256);

    let q8 = m.model_file.get("Q8_0").expect("Q8_0 must survive");
    assert!(!q8.downloaded);

    assert_eq!(m.mmproj_file.name, "");
    assert_eq!(m.tokenizer_file.name, "");
    assert!(m.extra_files.is_empty());
}

#[test]
fn tolerates_null_optional_fields() {
    let json = r#"{
      "Name":"Org/Repo",
      "ModelName":"tiny",
      "ModelType":"llm",
      "PluginId":"llama_cpp",
      "ModelFile":{"Q4_K_M":{"Name":"m.gguf","Downloaded":true,"Size":1}},
      "MMProjFile":null,
      "TokenizerFile":null,
      "ExtraFiles":null
    }"#;
    let m: ModelManifest = serde_json::from_str(json).expect("null optionals must parse");
    assert_eq!(m.name, "Org/Repo");
    assert_eq!(m.mmproj_file.name, "");
    assert_eq!(m.tokenizer_file.name, "");
    assert!(m.extra_files.is_empty());
}

#[test]
fn round_trip_back_to_go_shape() {
    let m: ModelManifest = serde_json::from_str(GO_MANIFEST).unwrap();
    let rustified = serde_json::to_string(&m).unwrap();

    for key in [
        "\"Name\"",
        "\"ModelName\"",
        "\"ModelType\"",
        "\"PluginId\"",
        "\"ModelFile\"",
        "\"MMProjFile\"",
        "\"TokenizerFile\"",
        "\"ExtraFiles\"",
        "\"Downloaded\"",
        "\"Size\"",
    ] {
        assert!(
            rustified.contains(key),
            "Rust output missing expected Go-style key {key}: {rustified}"
        );
    }

    let _: ModelManifest = serde_json::from_str(&rustified).unwrap();
}
