// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Infer a `ModelManifest` for sources that ship no `geniex.json`.

use std::collections::HashMap;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::manifest::{ModelFileInfo, ModelManifest, ModelType};

#[derive(Debug, Default, Clone)]
pub struct ManifestHint {
    pub model_type: Option<ModelType>,
    /// An unrecognised quant is an error, not a silent no-op.
    pub quant: Option<String>,
    /// Transformers-style `config.json`, used to tell LLM from VLM.
    pub config_json_bytes: Option<Vec<u8>>,
    /// Quant from a GGUF's `general.file_type`, keyed by filename.
    pub header_quants: HashMap<String, String>,
}

/// Preferred quantizations, best first.
pub const QUANT_PRIORITY: &[&str] = &["Q4_0", "Q4_K_M", "Q8_0"];

/// A bare `org/repo` loads the head of this order.
pub fn quant_sort_key(quant: &str) -> (usize, &str) {
    let rank = QUANT_PRIORITY
        .iter()
        .position(|p| *p == quant)
        .unwrap_or(QUANT_PRIORITY.len());
    (rank, quant)
}

pub fn infer_manifest_from_names(
    name: &str,
    file_names: &[String],
    sizes: &HashMap<String, i64>,
    hint: ManifestHint,
) -> Result<ModelManifest> {
    let mut ggufs: HashMap<String, Vec<&String>> = HashMap::new();
    let mut mmprojs: Vec<&String> = Vec::new();
    let mut tokenizers: Vec<&String> = Vec::new();
    let mut onnx_files: Vec<&String> = Vec::new();
    let mut geniex_files: Vec<&String> = Vec::new();
    let mut npy_files: Vec<&String> = Vec::new();

    for n in file_names {
        let lname = n.to_lowercase();
        if is_weight_gguf(&lname) {
            let quant = extract_quant(n)
                .or_else(|| hint.header_quants.get(n).cloned())
                .unwrap_or_else(|| "DEFAULT".to_string());
            ggufs.entry(quant).or_default().push(n);
        } else if lname.ends_with(".gguf") {
            if is_mmproj_filename(&lname) {
                mmprojs.push(n);
            }
        } else if lname.ends_with("tokenizer.json") {
            tokenizers.push(n);
        } else if lname.ends_with(".onnx") {
            onnx_files.push(n);
        } else if lname.ends_with(".geniex") {
            geniex_files.push(n);
        } else if lname.ends_with(".npy") {
            npy_files.push(n);
        }
    }

    if ggufs.is_empty() && onnx_files.is_empty() && geniex_files.is_empty() {
        return Err(Error::ManifestInferenceFailed(format!(
            "'{name}' has no model files GenieX can load; supported formats \
             are GGUF (llama.cpp) and QAIRT (Qualcomm AI Hub)"
        )));
    }

    if let Some(ref wanted) = hint.quant {
        if !ggufs.is_empty() && !ggufs.contains_key(wanted) {
            let mut available: Vec<&String> = ggufs.keys().collect();
            available.sort();
            return Err(Error::ManifestInferenceFailed(format!(
                "requested quant {:?} not found; available: {:?}",
                wanted, available
            )));
        }
        ggufs.retain(|k, _| k == wanted);
    }

    // The first shard is the entrypoint and keeps only its own size, or total_size() double-counts.
    let mut model_file: HashMap<String, ModelFileInfo> = HashMap::new();
    let mut shard_extras: Vec<&String> = Vec::new();
    for (quant, mut files) in ggufs {
        files.sort();
        model_file.insert(quant, file_info(files[0], sizes));
        shard_extras.extend_from_slice(&files[1..]);
    }

    let mmproj_file = match mmprojs.len() {
        0 => {
            if onnx_files.len() == 1 {
                file_info(onnx_files[0], sizes)
            } else if geniex_files.len() == 1 {
                file_info(geniex_files[0], sizes)
            } else {
                ModelFileInfo::default()
            }
        }
        1 => file_info(mmprojs[0], sizes),
        _ => {
            let chosen = mmprojs
                .iter()
                .max_by_key(|n| {
                    (
                        is_preferred_mmproj_precision(n),
                        sizes.get(n.as_str()).copied().unwrap_or(0),
                    )
                })
                .unwrap();
            file_info(chosen, sizes)
        }
    };

    let tokenizer_file = match tokenizers.len() {
        0 => ModelFileInfo::default(),
        1 => file_info(tokenizers[0], sizes),
        _ => {
            return Err(Error::ManifestInferenceFailed(format!(
                "multiple tokenizer files found: {:?}",
                tokenizers
            )));
        }
    };

    let mut extra_files: Vec<ModelFileInfo> = Vec::new();
    extra_files.extend(shard_extras.iter().map(|n| file_info(n, sizes)));
    extra_files.extend(npy_files.iter().map(|n| file_info(n, sizes)));
    extra_files.extend(
        geniex_files
            .iter()
            .filter(|n| mmproj_file.name != ***n)
            .map(|n| file_info(n, sizes)),
    );

    let model_type = infer_model_type(&hint, file_names);

    let model_name = {
        let repo = name.rsplit('/').next().unwrap_or(name);
        repo.trim_end_matches("-GGUF")
            .trim_end_matches("-gguf")
            .to_string()
    };

    let plugin_id = "llama_cpp".to_string();

    Ok(ModelManifest {
        name: name.to_string(),
        model_name,
        model_type,
        plugin_id,
        precision: String::new(),
        model_file,
        mmproj_file,
        tokenizer_file,
        extra_files,
    })
}

/// Defaults to LLM: a wrong VLM label fails loudly, a wrong LLM label silently drops images.
pub(crate) fn infer_model_type(hint: &ManifestHint, file_names: &[String]) -> ModelType {
    if let Some(t) = hint.model_type.clone() {
        return t;
    }
    if let Some(bytes) = hint.config_json_bytes.as_deref() {
        if let Some(t) = classify_from_config(bytes) {
            return t;
        }
    }
    if classify_from_filenames(file_names) == Some(ModelType::Vlm) {
        return ModelType::Vlm;
    }
    ModelType::Llm
}

#[derive(Debug, Default, Deserialize)]
struct ConfigJson {
    #[serde(default)]
    architectures: Option<Vec<String>>,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    vision_config: Option<serde_json::Value>,
    #[serde(default)]
    mm_vision_tower: Option<serde_json::Value>,
}

const VLM_ARCH_TOKENS: &[&str] = &[
    "VL",
    "Vision",
    "Llava",
    "MLlama",
    "PaliGemma",
    "MiniCPM-V",
    "MiniCPMV",
    "Idefics",
    "Aria",
    "Fuyu",
    "InternVL",
    "Llama4",
];

/// Checked before the VLM substring scan so text-only Gemma3 isn't promoted to VLM.
const LLM_ARCH_EXPLICIT: &[&str] = &["Gemma3ForCausalLM"];

fn classify_from_config(bytes: &[u8]) -> Option<ModelType> {
    let cfg: ConfigJson = serde_json::from_slice(bytes).ok()?;

    if let Some(mt) = cfg.model_type.as_deref() {
        if mt.eq_ignore_ascii_case("gemma3_text") {
            return Some(ModelType::Llm);
        }
    }

    let archs = cfg.architectures.as_deref().unwrap_or(&[]);
    for arch in archs {
        if LLM_ARCH_EXPLICIT.iter().any(|e| arch == *e) {
            return Some(ModelType::Llm);
        }
    }

    if matches!(&cfg.vision_config, Some(v) if v.is_object()) {
        return Some(ModelType::Vlm);
    }
    if matches!(&cfg.mm_vision_tower, Some(v) if !v.is_null()) {
        return Some(ModelType::Vlm);
    }

    for arch in archs {
        if arch.ends_with("ForConditionalGeneration") {
            let lower = arch.to_lowercase();
            if VLM_ARCH_TOKENS
                .iter()
                .any(|tok| lower.contains(&tok.to_lowercase()))
            {
                return Some(ModelType::Vlm);
            }
        }
    }

    for arch in archs {
        if arch.ends_with("ForCausalLM") {
            return Some(ModelType::Llm);
        }
    }

    None
}

/// Fallback for llama.cpp GGUF repos that ship no `config.json`.
fn classify_from_filenames(file_names: &[String]) -> Option<ModelType> {
    if file_names
        .iter()
        .any(|n| is_mmproj_filename(&n.to_lowercase()))
    {
        Some(ModelType::Vlm)
    } else {
        None
    }
}

fn is_mmproj_filename(lname: &str) -> bool {
    debug_assert_eq!(
        lname,
        lname.to_ascii_lowercase(),
        "is_mmproj_filename expects lowercased input"
    );
    let stem = lname.strip_suffix(".gguf").unwrap_or(lname);
    stem == "mmproj"
        || stem.starts_with("mmproj-")
        || stem.starts_with("mmproj.")
        || stem.starts_with("mmproj_")
        || stem.ends_with("-mmproj")
        || stem.ends_with("_mmproj")
        || stem.contains("-mmproj-")
        || stem.contains("_mmproj_")
}

fn is_preferred_mmproj_precision(name: &str) -> bool {
    matches!(extract_quant(name).as_deref(), Some("F16") | Some("FP16"))
}

/// Not a vision projector or an MTP draft head. Expects a lowercased name.
fn is_weight_gguf(lname: &str) -> bool {
    lname.ends_with(".gguf") && !is_mmproj_filename(lname) && !lname.contains("mtp")
}

pub(crate) fn untagged_gguf_names(file_names: &[String]) -> Vec<&String> {
    file_names
        .iter()
        .filter(|n| is_weight_gguf(&n.to_lowercase()) && extract_quant(n).is_none())
        .collect()
}

/// Upper-cased quant tag; [`QUANT_PRIORITY`] wins when several match.
pub(crate) fn extract_quant(name: &str) -> Option<String> {
    // Anchor at token boundaries so `IQ4_XS` isn't scanned as `Q4_XS`.
    let upper = name.to_ascii_uppercase();
    let bytes = upper.as_bytes();
    let mut composite: Vec<String> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !is_token_start(bytes, i) {
            i += 1;
            continue;
        }
        // Longer prefixes first so BF16/FP16 don't collapse to F16.
        let mut matched = false;
        for prefix in [b"MXFP" as &[u8], b"BF", b"FP", b"F", b"I"] {
            if let Some(end) = scan_token(bytes, i, prefix) {
                if let Ok(s) = std::str::from_utf8(&bytes[i..end]) {
                    composite.push(s.to_string());
                }
                i = end;
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }
        // `IQ*` i-quants and `TQ*` ternary quants.
        let mut j = i;
        if (bytes[j] == b'I' || bytes[j] == b'T') && j + 1 < bytes.len() && bytes[j + 1] == b'Q' {
            j += 1;
        }
        if !(j < bytes.len()
            && bytes[j] == b'Q'
            && j + 1 < bytes.len()
            && bytes[j + 1].is_ascii_digit())
        {
            i += 1;
            continue;
        }
        let start = i;
        i = j + 2;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        i = consume_short_suffix_segments(bytes, i);
        if let Ok(s) = std::str::from_utf8(&bytes[start..i]) {
            composite.push(s.to_string());
        }
    }
    if composite.is_empty() {
        return None;
    }
    for pref in QUANT_PRIORITY {
        if composite.iter().any(|c| c == *pref) {
            return Some((*pref).to_string());
        }
    }
    Some(composite.remove(0))
}

fn scan_token(bytes: &[u8], start: usize, prefix: &[u8]) -> Option<usize> {
    if !bytes[start..].starts_with(prefix) {
        return None;
    }
    let mut i = start + prefix.len();
    if !(i < bytes.len() && bytes[i].is_ascii_digit()) {
        return None;
    }
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    Some(consume_short_suffix_segments(bytes, i))
}

/// Segments cap at 3 chars so words like `_FINETUNE` aren't part of the tag.
fn consume_short_suffix_segments(bytes: &[u8], mut i: usize) -> usize {
    while i + 1 < bytes.len() && bytes[i] == b'_' {
        let seg_start = i + 1;
        let mut seg_end = seg_start;
        while seg_end < bytes.len()
            && (bytes[seg_end].is_ascii_uppercase() || bytes[seg_end].is_ascii_digit())
        {
            seg_end += 1;
        }
        let seg_len = seg_end - seg_start;
        if seg_len == 0 || seg_len > 3 {
            break;
        }
        i = seg_end;
    }
    i
}

fn is_token_start(bytes: &[u8], i: usize) -> bool {
    if i == 0 {
        return true;
    }
    matches!(bytes[i - 1], b'-' | b'_' | b'.' | b'/' | b'\\')
}

fn file_info(name: &str, sizes: &HashMap<String, i64>) -> ModelFileInfo {
    ModelFileInfo {
        name: name.to_string(),
        downloaded: true,
        size: sizes.get(name).copied().unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn infer_with(name: &str, files: &[(&str, i64)], hint: ManifestHint) -> Result<ModelManifest> {
        let names: Vec<String> = files.iter().map(|(n, _)| n.to_string()).collect();
        let sizes = files.iter().map(|(n, s)| (n.to_string(), *s)).collect();
        infer_manifest_from_names(name, &names, &sizes, hint)
    }

    fn infer(files: &[(&str, i64)]) -> ModelManifest {
        infer_with("Org/Repo-GGUF", files, ManifestHint::default()).unwrap()
    }

    fn with_quant(quant: &str) -> ManifestHint {
        ManifestHint {
            quant: Some(quant.to_string()),
            ..Default::default()
        }
    }

    fn with_config(bytes: &str) -> ManifestHint {
        ManifestHint {
            config_json_bytes: Some(bytes.as_bytes().to_vec()),
            ..Default::default()
        }
    }

    fn quants(m: &ModelManifest) -> Vec<&str> {
        let mut q: Vec<&str> = m.model_file.keys().map(String::as_str).collect();
        q.sort();
        q
    }

    #[test]
    fn quant_sort_key_ranks_priority_first_then_alphabetical() {
        let mut tags = vec!["Q8_0", "IQ4_XS", "Q4_K_M", "BF16", "Q4_0"];
        tags.sort_by_key(|t| quant_sort_key(t));
        assert_eq!(tags, ["Q4_0", "Q4_K_M", "Q8_0", "BF16", "IQ4_XS"]);
    }

    #[test]
    fn extract_quant_cases() {
        let cases = [
            ("model-Q4_K_M.gguf", Some("Q4_K_M")),
            ("model-Q8_0.gguf", Some("Q8_0")),
            ("model.gguf", None),
            ("gemma-4-E2B-q4_0-override.gguf", Some("Q4_0")),
            ("model-q4_k_m.gguf", Some("Q4_K_M")),
            ("model-IQ4_XS.gguf", Some("IQ4_XS")),
            ("model-iq1_s.gguf", Some("IQ1_S")),
            ("model-TQ1_0.gguf", Some("TQ1_0")),
            ("aQ4_0.gguf", None),
            ("noQuant.gguf", None),
            ("dir/Q4_0.gguf", Some("Q4_0")),
            ("dir\\Q4_0.gguf", Some("Q4_0")),
            ("model-Q4_K_M_finetune.gguf", Some("Q4_K_M")),
            ("model-MXFP4.gguf", Some("MXFP4")),
            ("model-mxfp4.gguf", Some("MXFP4")),
            ("gpt-oss-20b-mxfp4-00001-of-00002.gguf", Some("MXFP4")),
            ("model-MXFP4_MOE.gguf", Some("MXFP4_MOE")),
            ("aMXFP4.gguf", None),
            ("model-F16.gguf", Some("F16")),
            ("model-F32.gguf", Some("F32")),
            ("model-F64.gguf", Some("F64")),
            ("model-f16.gguf", Some("F16")),
            ("model-BF16.gguf", Some("BF16")),
            ("model-bf16.gguf", Some("BF16")),
            ("qwen2-1_5b-instruct-fp16.gguf", Some("FP16")),
            ("model-FP32.gguf", Some("FP32")),
            ("model-F32-00001-of-00002.gguf", Some("F32")),
            ("model-I8.gguf", Some("I8")),
            ("model-I16.gguf", Some("I16")),
            ("model-I32.gguf", Some("I32")),
            ("model-i64.gguf", Some("I64")),
            ("aF16.gguf", None),
            ("aBF16.gguf", None),
            ("aI8.gguf", None),
            ("modelI8.gguf", None),
        ];
        for (name, want) in cases {
            assert_eq!(extract_quant(name).as_deref(), want, "{name}");
        }
    }

    #[test]
    fn is_mmproj_filename_cases() {
        for name in [
            "mmproj-f16.gguf",
            "mmproj.gguf",
            "mmproj_f16.gguf",
            "model_mmproj.gguf",
            "gemma-4-e2b-it-mmproj.gguf",
            "model-mmproj-f16.gguf",
        ] {
            assert!(is_mmproj_filename(name), "{name}");
        }
        for name in ["model-q4_0-override.gguf", "projector-only-no-mm.gguf"] {
            assert!(!is_mmproj_filename(name), "{name}");
        }
    }

    #[test]
    fn untagged_gguf_names_skips_tagged_projectors_and_drafts() {
        let names: Vec<String> = [
            "model.gguf",
            "model-Q4_K_M.gguf",
            "mmproj.gguf",
            "model-mtp.gguf",
            "README.md",
        ]
        .map(String::from)
        .into();
        assert_eq!(untagged_gguf_names(&names), ["model.gguf"]);
    }

    #[test]
    fn gemma4_qat_layout_picks_correct_entrypoint() {
        let m = infer_with(
            "qualcomm-ai-hub-community/gemma-4-E2B-it-qat-GGUF",
            &[
                ("gemma-4-E2B-it-mmproj.gguf", 500_000),
                ("gemma-4-E2B-q4_0-override.gguf", 2_000_000),
            ],
            ManifestHint::default(),
        )
        .unwrap();
        assert_eq!(m.model_type, ModelType::Vlm);
        assert_eq!(m.model_file["Q4_0"].name, "gemma-4-E2B-q4_0-override.gguf");
        assert_eq!(m.mmproj_file.name, "gemma-4-E2B-it-mmproj.gguf");
    }

    #[test]
    fn quant_buckets() {
        type Files<'a> = &'a [(&'a str, i64)];
        let cases: [(Files, &[&str]); 3] = [
            (
                &[
                    ("qwen2-1_5b-instruct-q4_0.gguf", 1),
                    ("qwen2-1_5b-instruct-fp16.gguf", 3),
                ],
                &["FP16", "Q4_0"],
            ),
            (&[("Qwen3-35B-A3B-REAP-48-v2.gguf", 8)], &["DEFAULT"]),
            // The MTP draft's Q4_0 would outrank Q4_K_XL but can't load standalone.
            (
                &[
                    ("gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf", 2),
                    ("MTP/mtp-gemma-4-E2B-it-Q4_0.gguf", 1),
                    ("mmproj-F32.gguf", 1),
                ],
                &["Q4_K_XL"],
            ),
        ];
        for (files, want) in cases {
            assert_eq!(quants(&infer(files)), want, "{files:?}");
        }
    }

    #[test]
    fn quant_hint_filters_or_rejects() {
        let files = [
            ("model-Q4_0.gguf", 9),
            ("model-Q4_K_M.gguf", 10),
            ("model-Q8_0.gguf", 18),
        ];
        let m = infer_with("Org/Repo-GGUF", &files, with_quant("Q4_0")).unwrap();
        assert_eq!(quants(&m), ["Q4_0"]);
        assert_eq!(m.model_file["Q4_0"].name, "model-Q4_0.gguf");
        assert!(infer_with("Org/Repo-GGUF", &files, with_quant("Q2_K")).is_err());

        let m = infer_with(
            "Org/R",
            &[("gpt-oss-20b-mxfp4.gguf", 12_109_566_560)],
            with_quant("MXFP4"),
        )
        .unwrap();
        assert_eq!(m.model_file["MXFP4"].name, "gpt-oss-20b-mxfp4.gguf");
        assert!(m.extra_files.is_empty(), "{:?}", m.extra_files);
    }

    #[test]
    fn sharded_gguf_keeps_every_shard() {
        let m = infer(&[
            ("model-Q4_0-00001-of-00003.gguf", 100),
            ("model-Q4_0-00002-of-00003.gguf", 200),
            ("model-Q4_0-00003-of-00003.gguf", 300),
        ]);
        let entry = &m.model_file["Q4_0"];
        assert_eq!(entry.name, "model-Q4_0-00001-of-00003.gguf");
        assert_eq!(entry.size, 100, "entrypoint stores its own size only");
        let mut extra: Vec<&str> = m.extra_files.iter().map(|f| f.name.as_str()).collect();
        extra.sort();
        assert_eq!(
            extra,
            [
                "model-Q4_0-00002-of-00003.gguf",
                "model-Q4_0-00003-of-00003.gguf"
            ]
        );
        assert_eq!(m.total_size(), 600, "every shard counted exactly once");
    }

    #[test]
    fn header_quant_names_untagged_files_only() {
        let header = |file: &str, q: &str| ManifestHint {
            header_quants: HashMap::from([(file.to_string(), q.to_string())]),
            ..Default::default()
        };
        let untagged = "Qwen3-35B-A3B-REAP-48-v2.gguf";
        let m = infer_with(
            "peonist/REAP-48",
            &[(untagged, 8)],
            header(untagged, "Q4_K_M"),
        )
        .unwrap();
        assert_eq!(m.model_file["Q4_K_M"].name, untagged);
        let m = infer_with(
            "Org/Repo",
            &[("model-Q4_0.gguf", 1)],
            header("model-Q4_0.gguf", "Q8_0"),
        )
        .unwrap();
        assert_eq!(quants(&m), ["Q4_0"]);
    }

    #[test]
    fn mmproj_and_tokenizer_selection() {
        let m = infer(&[
            ("model-Q4_K_M.gguf", 1_000_000),
            ("mmproj-F16.gguf", 200_000),
        ]);
        assert_eq!(
            (&m.model_type, m.mmproj_file.name.as_str()),
            (&ModelType::Vlm, "mmproj-F16.gguf")
        );
        assert_eq!(m.model_name, "Repo");

        let m = infer(&[("model-Q4_K_M.gguf", 1_000_000), ("tokenizer.json", 2_000)]);
        assert_eq!(
            (&m.model_type, m.tokenizer_file.name.as_str()),
            (&ModelType::Llm, "tokenizer.json")
        );

        // F16 beats larger BF16/F32 (no native BF16 HTP kernels); else largest wins.
        let pick = |projectors: &[(&'static str, i64)]| {
            let mut files = vec![("model-Q4_K_M.gguf", 1_000_000)];
            files.extend_from_slice(projectors);
            infer(&files).mmproj_file.name
        };
        let bf16_f32 = [
            ("mmproj-BF16.gguf", 992_000_000),
            ("mmproj-F32.gguf", 1_910_000_000),
        ];
        assert_eq!(
            pick(&[("mmproj-F16.gguf", 990_000_000), bf16_f32[0], bf16_f32[1]]),
            "mmproj-F16.gguf"
        );
        assert_eq!(pick(&bf16_f32), "mmproj-F32.gguf");
    }

    #[test]
    fn rejects_ambiguous_or_unloadable_listings() {
        let multi_tokenizer = [
            ("model-Q4_0.gguf", 1),
            ("a/tokenizer.json", 1),
            ("b/tokenizer.json", 2),
        ];
        assert!(infer_with("Org/X", &multi_tokenizer, ManifestHint::default()).is_err());
        assert!(infer_with("Org/X", &[], ManifestHint::default()).is_err());

        let repo = "openai/gpt-oss-safeguard-20b";
        let safetensors = [
            ("config.json", 1000),
            ("model-00001-of-00002.safetensors", 8_000_000),
            ("model-00002-of-00002.safetensors", 8_000_000),
            ("tokenizer.json", 2000),
        ];
        let err = infer_with(repo, &safetensors, ManifestHint::default())
            .unwrap_err()
            .to_string();
        for want in [repo, "GGUF", "QAIRT"] {
            assert!(err.contains(want), "missing {want:?}: {err}");
        }
    }

    const QWEN25_VL: &str = r#"{"architectures":["Qwen2_5_VLForConditionalGeneration"],"model_type":"qwen2_5_vl","vision_config":{"depth":32}}"#;
    const LLAMA31: &str = r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#;

    #[test]
    fn infer_model_type_cases() {
        use ModelType::{Llm, Vlm};
        let gguf = "model-Q4_K_M.gguf";
        let cases: [(&str, Option<&str>, &[&str], ModelType); 10] = [
            (
                "vision_config",
                Some(QWEN25_VL),
                &["model.safetensors"],
                Vlm,
            ),
            ("llama", Some(LLAMA31), &[gguf], Llm),
            (
                "qwen3",
                Some(r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}"#),
                &[],
                Llm,
            ),
            (
                "gpt-oss",
                Some(r#"{"architectures":["GptOssForCausalLM"],"model_type":"gpt_oss"}"#),
                &[],
                Llm,
            ),
            // gemma3_text must short-circuit before the `gemma3` substring.
            (
                "gemma3 1b",
                Some(r#"{"architectures":["Gemma3ForCausalLM"],"model_type":"gemma3_text"}"#),
                &[],
                Llm,
            ),
            (
                "gemma3 4b",
                Some(
                    r#"{"architectures":["Gemma3ForConditionalGeneration"],"model_type":"gemma3","vision_config":{}}"#,
                ),
                &[],
                Vlm,
            ),
            ("mmproj fallback", None, &[gguf, "mmproj-f16.gguf"], Vlm),
            ("pure gguf", None, &[gguf], Llm),
            (
                "config beats stray mmproj",
                Some(LLAMA31),
                &[gguf, "mmproj-x.gguf"],
                Llm,
            ),
            ("corrupt config", Some("{not json"), &[], Llm),
        ];
        for (case, config, names, want) in cases {
            let hint = config.map(with_config).unwrap_or_default();
            let names: Vec<String> = names.iter().map(|n| n.to_string()).collect();
            assert_eq!(infer_model_type(&hint, &names), want, "{case}");
        }

        let mut hint = with_config(LLAMA31);
        hint.model_type = Some(Vlm);
        assert_eq!(
            infer_model_type(&hint, &[]),
            Vlm,
            "user override beats config"
        );
    }

    #[test]
    fn config_vlm_reaches_the_manifest_without_mmproj() {
        let files = [("model-Q4_K_M.gguf", 1_000_000), ("tokenizer.json", 2_000)];
        let m = infer_with("Org/Repo", &files, with_config(QWEN25_VL)).unwrap();
        assert_eq!(m.model_type, ModelType::Vlm);
    }
}
