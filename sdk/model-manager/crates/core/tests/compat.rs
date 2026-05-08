//! Cross-language manifest compatibility: verify that a `geniex.json`
//! produced by the Go CLI (which uses `sonic.Marshal` with PascalCase
//! struct field names) round-trips through the Rust model manager
//! without data loss or parse failures.
//!
//! The fixture under `tests/fixtures/go_manifest.json` was captured
//! from a real `geniex pull qwen3` run. If the Go side ever changes the
//! manifest schema this test should fail first, making the breakage
//! visible before it lands in production.

use model_manager_core::manifest::{ModelManifest, ModelType};

const GO_MANIFEST: &str = include_str!("fixtures/go_manifest.json");

#[test]
fn parses_go_generated_manifest() {
    let m: ModelManifest =
        serde_json::from_str(GO_MANIFEST).expect("Go-produced manifest must parse");

    assert_eq!(m.name, "Qwen/Qwen3-4B-GGUF");
    assert_eq!(m.model_name, "qwen3-4b");
    assert_eq!(m.model_type, ModelType::Llm);
    assert_eq!(m.plugin_id, "llama_cpp");

    // Quantization table preserved.
    assert_eq!(m.model_file.len(), 2);
    let q4 = m.model_file.get("Q4_K_M").expect("Q4_K_M must survive");
    assert!(q4.downloaded);
    assert_eq!(q4.size, 2_593_504_256);

    let q8 = m.model_file.get("Q8_0").expect("Q8_0 must survive");
    assert!(!q8.downloaded);

    // Empty-by-default fields come through with their zero values.
    assert_eq!(m.mmproj_file.name, "");
    assert_eq!(m.tokenizer_file.name, "");
    assert!(m.extra_files.is_empty());
}

#[test]
fn round_trip_back_to_go_shape() {
    // Parse, serialize, reparse — ensuring we emit the same PascalCase
    // shape Go expects.
    let m: ModelManifest = serde_json::from_str(GO_MANIFEST).unwrap();
    let rustified = serde_json::to_string(&m).unwrap();

    // Check the exact Go-compatible field names are emitted.
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

    // And Rust can reparse its own output.
    let _: ModelManifest = serde_json::from_str(&rustified).unwrap();
}
