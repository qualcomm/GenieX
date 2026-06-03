//! Infer a `ModelManifest` from a directory or a list of files.
//!
//! Used when the source (LocalFS or a HuggingFace repo) does not ship a
//! `geniex.json` — we synthesize a minimal manifest so the rest of the
//! pipeline can operate uniformly.
//!
//! Rules mirror the Go CLI's `cli/cmd/geniex/model.go:chooseFiles` so that
//! a model pulled either way is interchangeable.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::manifest::{ModelFileInfo, ModelManifest, ModelType};

/// Optional caller-supplied overrides used when inferring a manifest.
#[derive(Debug, Default, Clone)]
pub struct ManifestHint {
    /// Force this `ModelType`; otherwise inferred via [`infer_model_type`].
    pub model_type: Option<ModelType>,
    /// Restrict the inferred manifest to a single quantization. When set,
    /// GGUFs whose extracted quant tag doesn't match are excluded from
    /// `model_file`, so `pull` only fetches the requested quant. An
    /// unrecognised value is an error rather than a silent no-op.
    pub quant: Option<String>,
    /// Raw bytes of a transformers-style `config.json` sibling, when the
    /// source has one. Parsed by [`classify_from_config`] to decide LLM
    /// vs VLM more reliably than the mmproj filename heuristic.
    pub config_json_bytes: Option<Vec<u8>>,
}

/// Quantization priority order (earlier = preferred). Prefers the smaller,
/// faster `Q4_0` first — the historical Go CLI default that all bindings now
/// share.
pub(crate) const QUANT_PRIORITY: &[&str] = &["Q4_0", "Q4_K_M", "Q8_0"];

/// Infer a manifest by scanning `src_dir` for model files.
pub fn infer_manifest_from_dir(
    name: &str,
    src_dir: &Path,
    hint: ManifestHint,
) -> Result<ModelManifest> {
    let mut file_names: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(src_dir)?.flatten() {
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if !ft.is_file() {
            continue;
        }
        if let Some(n) = entry.file_name().to_str().map(str::to_string) {
            file_names.push(n);
        }
    }

    let mut sizes: HashMap<String, i64> = HashMap::new();
    for n in &file_names {
        let size = std::fs::metadata(src_dir.join(n))
            .map(|m| m.len() as i64)
            .unwrap_or(0);
        sizes.insert(n.clone(), size);
    }

    infer_manifest_from_names(name, &file_names, &sizes, hint)
}

/// Same logic as [`infer_manifest_from_dir`] but driven by an explicit
/// file list and size map — useful when the caller already has the remote
/// listing in hand (e.g. `HfHub::model_info`).
pub fn infer_manifest_from_names(
    name: &str,
    file_names: &[String],
    sizes: &HashMap<String, i64>,
    hint: ManifestHint,
) -> Result<ModelManifest> {
    let mut ggufs: HashMap<String, Vec<&String>> = HashMap::new(); // quant -> files
    let mut mmprojs: Vec<&String> = Vec::new();
    let mut tokenizers: Vec<&String> = Vec::new();
    let mut onnx_files: Vec<&String> = Vec::new();
    let mut geniex_files: Vec<&String> = Vec::new();
    let mut npy_files: Vec<&String> = Vec::new();

    for n in file_names {
        let lname = n.to_lowercase();
        if lname.ends_with(".gguf") {
            if lname.starts_with("mmproj") {
                mmprojs.push(n);
            } else {
                let quant = extract_quant(n).unwrap_or_else(|| "default".to_string());
                ggufs.entry(quant).or_default().push(n);
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
            "no recognizable model files found for '{}'",
            name
        )));
    }

    // If the caller asked for a specific quant, drop everything else.
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

    // Pick one entrypoint file per quant. Sharded GGUFs (`*-NNNNN-of-MMMMM.gguf`)
    // are split across several files that share one quant; the first shard is
    // the manifest entrypoint, its recorded size is the sum of every shard, and
    // the remaining shards go to `extra_files` so the executor fetches them all.
    let mut model_file: HashMap<String, ModelFileInfo> = HashMap::new();
    let mut shard_extras: Vec<&String> = Vec::new();
    for (quant, mut files) in ggufs {
        files.sort();
        let total: i64 = files
            .iter()
            .map(|n| sizes.get(n.as_str()).copied().unwrap_or(0))
            .sum();
        let entry = files[0];
        model_file.insert(
            quant,
            ModelFileInfo {
                name: entry.clone(),
                downloaded: true,
                size: total,
            },
        );
        shard_extras.extend_from_slice(&files[1..]);
    }

    // MMProj: 0 -> try single onnx/geniex; 1 -> use; >1 -> largest.
    let mmproj_file = match mmprojs.len() {
        0 => {
            if onnx_files.len() == 1 {
                let n = onnx_files[0];
                ModelFileInfo {
                    name: n.clone(),
                    downloaded: true,
                    size: sizes.get(n.as_str()).copied().unwrap_or(0),
                }
            } else if geniex_files.len() == 1 {
                let n = geniex_files[0];
                ModelFileInfo {
                    name: n.clone(),
                    downloaded: true,
                    size: sizes.get(n.as_str()).copied().unwrap_or(0),
                }
            } else {
                ModelFileInfo::default()
            }
        }
        1 => {
            let n = mmprojs[0];
            ModelFileInfo {
                name: n.clone(),
                downloaded: true,
                size: sizes.get(n.as_str()).copied().unwrap_or(0),
            }
        }
        _ => {
            let chosen = mmprojs
                .iter()
                .max_by_key(|n| sizes.get(n.as_str()).copied().unwrap_or(0))
                .unwrap();
            ModelFileInfo {
                name: (*chosen).clone(),
                downloaded: true,
                size: sizes.get(chosen.as_str()).copied().unwrap_or(0),
            }
        }
    };

    // Tokenizer: 0 -> none; 1 -> use; >1 -> error (ambiguous).
    let tokenizer_file = match tokenizers.len() {
        0 => ModelFileInfo::default(),
        1 => {
            let n = tokenizers[0];
            ModelFileInfo {
                name: n.clone(),
                downloaded: true,
                size: sizes.get(n.as_str()).copied().unwrap_or(0),
            }
        }
        _ => {
            return Err(Error::ManifestInferenceFailed(format!(
                "multiple tokenizer files found: {:?}",
                tokenizers
            )));
        }
    };

    // ExtraFiles: trailing GGUF shards (the entrypoint shard lives in
    // model_file) + all .npy + .geniex not used as mmproj.
    let mut extra_files: Vec<ModelFileInfo> = Vec::new();
    for n in &shard_extras {
        extra_files.push(ModelFileInfo {
            name: (*n).clone(),
            downloaded: true,
            size: sizes.get(n.as_str()).copied().unwrap_or(0),
        });
    }
    for n in &npy_files {
        extra_files.push(ModelFileInfo {
            name: (*n).clone(),
            downloaded: true,
            size: sizes.get(n.as_str()).copied().unwrap_or(0),
        });
    }
    for n in &geniex_files {
        if mmproj_file.name != **n {
            extra_files.push(ModelFileInfo {
                name: (*n).clone(),
                downloaded: true,
                size: sizes.get(n.as_str()).copied().unwrap_or(0),
            });
        }
    }

    let model_type = infer_model_type(&hint, file_names);

    // Derive model_name: last path component of `name`, with -GGUF suffix
    // stripped. e.g. "Qwen/Qwen3-4B-GGUF" -> "Qwen3-4B".
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

/// Tiered modality classifier. Order is deliberate: explicit user
/// override wins, then positive signals from `config.json`, then the
/// legacy `mmproj` filename rule (kept so pure-GGUF repos still work),
/// then a LLM-biased default.
///
/// Why LLM-biased: mis-labeling an LLM as VLM breaks inference loudly
/// (mmproj missing), while mis-labeling a VLM as LLM silently drops
/// image input. Users can always override via `--model-type`, so we'd
/// rather fail loud than carry image tensors nowhere.
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

/// Subset of HuggingFace `config.json` we inspect. Extra fields are
/// ignored so we don't break on schema drift.
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

/// Architecture substrings that mark a `*ForConditionalGeneration`
/// class as vision-capable. Case-insensitive match.
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

/// Explicit LLM architectures — checked before the VLM substring scan
/// so names like "Gemma3ForCausalLM" don't match the generic "Gemma3"
/// substring and get promoted to VLM.
const LLM_ARCH_EXPLICIT: &[&str] = &["Gemma3ForCausalLM"];

/// Positive-signal classifier over a `config.json` byte blob. Returns
/// `None` when the file parses but carries no modality-determining
/// fields, so the caller can fall through to other tiers.
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

/// Legacy mmproj-filename heuristic, retained as Tier 2 fallback for
/// llama.cpp-converted GGUF repos that ship no `config.json`.
fn classify_from_filenames(file_names: &[String]) -> Option<ModelType> {
    if file_names
        .iter()
        .any(|n| n.to_lowercase().starts_with("mmproj"))
    {
        Some(ModelType::Vlm)
    } else {
        None
    }
}

/// Extract a quant tag like `Q4_K_M` or `Q8_0` from a filename.
/// Returns the highest-priority match if multiple are present, else the
/// first match, else None.
fn extract_quant(name: &str) -> Option<String> {
    // Scan for tokens matching `Q<digit>[_A-Z0-9]*`.
    let mut matches: Vec<String> = Vec::new();
    for part in name.split(|c: char| c == '-' || c == '.' || c == '_') {
        if part.starts_with('Q')
            && part.len() >= 2
            && part[1..2]
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
        {
            matches.push(part.to_string());
        }
    }
    // Rejoin composite quants like "Q4_K_M" which get split by '_'.
    // Simpler: regex-lite — walk the name, match ASCII pattern.
    // We'll use a lightweight hand-rolled scan for `Q[0-9]+(_[A-Z0-9]+)*`.
    let mut composite: Vec<String> = Vec::new();
    let bytes = name.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'Q' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
            let start = i;
            i += 2;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            // optional `_XXX` segments (letters and/or digits), e.g. "Q8_0", "Q4_K_M"
            while i + 1 < bytes.len()
                && bytes[i] == b'_'
                && (bytes[i + 1].is_ascii_uppercase() || bytes[i + 1].is_ascii_digit())
            {
                i += 1;
                while i < bytes.len()
                    && (bytes[i].is_ascii_uppercase() || bytes[i].is_ascii_digit())
                {
                    i += 1;
                }
            }
            if let Ok(s) = std::str::from_utf8(&bytes[start..i]) {
                composite.push(s.to_string());
            }
        } else {
            i += 1;
        }
    }
    let _ = matches; // above naive split is unused in favor of composite scanner
    if composite.is_empty() {
        return None;
    }
    // Prefer highest-priority quant among the matches.
    for pref in QUANT_PRIORITY {
        if composite.iter().any(|c| c == *pref) {
            return Some((*pref).to_string());
        }
    }
    Some(composite.remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sizes_of(names: &[(&str, i64)]) -> (Vec<String>, HashMap<String, i64>) {
        let mut m = HashMap::new();
        let mut v = Vec::new();
        for (n, s) in names {
            m.insert(n.to_string(), *s);
            v.push(n.to_string());
        }
        (v, m)
    }

    #[test]
    fn extract_quant_finds_common_formats() {
        assert_eq!(
            extract_quant("model-Q4_K_M.gguf"),
            Some("Q4_K_M".to_string())
        );
        assert_eq!(extract_quant("model-Q8_0.gguf"), Some("Q8_0".to_string()));
        assert_eq!(extract_quant("model.gguf"), None);
    }

    #[test]
    fn quant_hint_filters_ggufs() {
        let (names, sizes) = sizes_of(&[
            ("model-Q4_0.gguf", 900_000),
            ("model-Q4_K_M.gguf", 1_000_000),
            ("model-Q8_0.gguf", 1_800_000),
        ]);
        let hint = ManifestHint {
            quant: Some("Q4_0".to_string()),
            ..Default::default()
        };
        let m = infer_manifest_from_names("Org/Repo-GGUF", &names, &sizes, hint).unwrap();
        assert_eq!(m.model_file.len(), 1);
        assert_eq!(m.model_file.get("Q4_0").unwrap().name, "model-Q4_0.gguf");
    }

    #[test]
    fn sharded_gguf_keeps_every_shard() {
        // A 3-shard Q4_0 model: entrypoint goes to model_file with the summed
        // size; the other two shards must land in extra_files so they download.
        let (names, sizes) = sizes_of(&[
            ("model-Q4_0-00001-of-00003.gguf", 100),
            ("model-Q4_0-00002-of-00003.gguf", 200),
            ("model-Q4_0-00003-of-00003.gguf", 300),
        ]);
        let m =
            infer_manifest_from_names("Org/Repo-GGUF", &names, &sizes, Default::default()).unwrap();
        let entry = m.model_file.get("Q4_0").unwrap();
        assert_eq!(entry.name, "model-Q4_0-00001-of-00003.gguf");
        assert_eq!(entry.size, 600, "model_file size must sum all shards");
        let extra: Vec<&str> = m.extra_files.iter().map(|f| f.name.as_str()).collect();
        assert!(extra.contains(&"model-Q4_0-00002-of-00003.gguf"));
        assert!(extra.contains(&"model-Q4_0-00003-of-00003.gguf"));
    }

    #[test]
    fn quant_hint_rejects_unknown_quant() {
        let (names, sizes) = sizes_of(&[("model-Q4_K_M.gguf", 1_000_000)]);
        let hint = ManifestHint {
            quant: Some("Q2_K".to_string()),
            ..Default::default()
        };
        assert!(infer_manifest_from_names("Org/Repo-GGUF", &names, &sizes, hint).is_err());
    }

    #[test]
    fn infers_vlm_when_mmproj_present() {
        let (names, sizes) = sizes_of(&[
            ("model-Q4_K_M.gguf", 1_000_000),
            ("mmproj-F16.gguf", 200_000),
        ]);
        let m =
            infer_manifest_from_names("Org/Repo-GGUF", &names, &sizes, Default::default()).unwrap();
        assert_eq!(m.model_type, ModelType::Vlm);
        assert!(m.model_file.contains_key("Q4_K_M"));
        assert_eq!(m.mmproj_file.name, "mmproj-F16.gguf");
        assert_eq!(m.model_name, "Repo");
    }

    #[test]
    fn infers_llm_without_mmproj() {
        let (names, sizes) =
            sizes_of(&[("model-Q4_K_M.gguf", 1_000_000), ("tokenizer.json", 2_000)]);
        let m =
            infer_manifest_from_names("Org/Repo-GGUF", &names, &sizes, Default::default()).unwrap();
        assert_eq!(m.model_type, ModelType::Llm);
        assert_eq!(m.tokenizer_file.name, "tokenizer.json");
    }

    #[test]
    fn rejects_multiple_tokenizers() {
        let (names, sizes) = sizes_of(&[
            ("model-Q4_0.gguf", 1_000_000),
            ("a/tokenizer.json", 1000),
            ("b/tokenizer.json", 2000),
        ]);
        assert!(infer_manifest_from_names("Org/X", &names, &sizes, Default::default()).is_err());
    }

    #[test]
    fn rejects_empty_dir() {
        let (names, sizes) = sizes_of(&[]);
        assert!(infer_manifest_from_names("Org/X", &names, &sizes, Default::default()).is_err());
    }

    // -- modality classifier ------------------------------------------

    /// Trimmed Qwen2.5-VL config — only the fields the classifier reads.
    const QWEN25_VL_CONFIG: &str = r#"{
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "model_type": "qwen2_5_vl",
        "vision_config": {"depth": 32, "hidden_size": 1280}
    }"#;

    const LLAMA31_CONFIG: &str = r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama"
    }"#;

    const QWEN3_CONFIG: &str = r#"{
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3"
    }"#;

    const GPT_OSS_CONFIG: &str = r#"{
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss"
    }"#;

    const GEMMA3_1B_CONFIG: &str = r#"{
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3_text"
    }"#;

    const GEMMA3_4B_CONFIG: &str = r#"{
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "vision_config": {"hidden_size": 1152}
    }"#;

    fn hint_with_config(bytes: &[u8]) -> ManifestHint {
        ManifestHint {
            config_json_bytes: Some(bytes.to_vec()),
            ..Default::default()
        }
    }

    #[test]
    fn config_vlm_with_vision_config() {
        // #1 — Qwen2.5-VL safetensors, no mmproj anywhere.
        let names = vec!["model.safetensors".to_string()];
        let t = infer_model_type(&hint_with_config(QWEN25_VL_CONFIG.as_bytes()), &names);
        assert_eq!(t, ModelType::Vlm);
    }

    #[test]
    fn config_llm_llama31() {
        // #2 — LLaMA-3.1 GGUF.
        let names = vec!["model-Q4_K_M.gguf".to_string()];
        let t = infer_model_type(&hint_with_config(LLAMA31_CONFIG.as_bytes()), &names);
        assert_eq!(t, ModelType::Llm);
    }

    #[test]
    fn config_llm_qwen3() {
        // #3 — Qwen3 text-only.
        let t = infer_model_type(&hint_with_config(QWEN3_CONFIG.as_bytes()), &[]);
        assert_eq!(t, ModelType::Llm);
    }

    #[test]
    fn config_llm_gpt_oss() {
        // #4 — gpt-oss.
        let t = infer_model_type(&hint_with_config(GPT_OSS_CONFIG.as_bytes()), &[]);
        assert_eq!(t, ModelType::Llm);
    }

    #[test]
    fn config_llm_gemma3_1b_text() {
        // #5 — Gemma-3-1B text-only: model_type="gemma3_text" must short-
        // circuit before the generic `gemma3` substring promotes VLM.
        let t = infer_model_type(&hint_with_config(GEMMA3_1B_CONFIG.as_bytes()), &[]);
        assert_eq!(t, ModelType::Llm);
    }

    #[test]
    fn config_vlm_gemma3_4b() {
        // #6 — Gemma-3-4B-IT multimodal.
        let t = infer_model_type(&hint_with_config(GEMMA3_4B_CONFIG.as_bytes()), &[]);
        assert_eq!(t, ModelType::Vlm);
    }

    #[test]
    fn mmproj_fallback_preserved_without_config() {
        // #7 — pure GGUF VLM repo, no config.json.
        let names = vec!["model-Q4_K_M.gguf".into(), "mmproj-f16.gguf".into()];
        let t = infer_model_type(&ManifestHint::default(), &names);
        assert_eq!(t, ModelType::Vlm);
    }

    #[test]
    fn pure_gguf_llm_without_config() {
        // #8 — pure GGUF LLM: no config.json, no mmproj → LLM.
        let names = vec!["model-Q4_K_M.gguf".into()];
        let t = infer_model_type(&ManifestHint::default(), &names);
        assert_eq!(t, ModelType::Llm);
    }

    #[test]
    fn config_llm_wins_over_stray_mmproj_file() {
        // #9 — conflict case: config says LLM, directory has a stray
        // mmproj file. Config wins (Tier 1 beats Tier 2).
        let names = vec!["model-Q4_K_M.gguf".into(), "mmproj-x.gguf".into()];
        let t = infer_model_type(&hint_with_config(LLAMA31_CONFIG.as_bytes()), &names);
        assert_eq!(t, ModelType::Llm);
    }

    #[test]
    fn corrupt_config_defaults_to_llm() {
        // #10 — unparseable config degrades to Tier 2/3.
        let t = infer_model_type(&hint_with_config(b"{not json"), &[]);
        assert_eq!(t, ModelType::Llm);
    }

    #[test]
    fn user_override_beats_config() {
        // #11 — --model-type flag always wins.
        let mut hint = hint_with_config(LLAMA31_CONFIG.as_bytes());
        hint.model_type = Some(ModelType::Vlm);
        let t = infer_model_type(&hint, &[]);
        assert_eq!(t, ModelType::Vlm);
    }

    #[test]
    fn full_pipeline_vlm_config_with_gguf_files() {
        // End-to-end check that `infer_manifest_from_names` plumbs the
        // config-derived VLM through to ModelManifest.model_type even
        // when the directory has no mmproj file.
        let (names, sizes) =
            sizes_of(&[("model-Q4_K_M.gguf", 1_000_000), ("tokenizer.json", 2_000)]);
        let m = infer_manifest_from_names(
            "Org/Repo",
            &names,
            &sizes,
            hint_with_config(QWEN25_VL_CONFIG.as_bytes()),
        )
        .unwrap();
        assert_eq!(m.model_type, ModelType::Vlm);
    }
}
