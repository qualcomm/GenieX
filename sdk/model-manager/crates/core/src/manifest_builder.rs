//! Infer a `ModelManifest` from a directory or a list of files.
//!
//! Used when the source (LocalFS or a HuggingFace repo) does not ship a
//! `geniex.json` — we synthesize a minimal manifest so the rest of the
//! pipeline can operate uniformly.

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
            if is_mmproj_filename(&lname) {
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
    // the manifest entrypoint, the rest go to `extra_files` so the executor
    // fetches them. Each bucket records its own single-file size; total_size()
    // sums every bucket and would double-count if the entrypoint aggregated.
    let mut model_file: HashMap<String, ModelFileInfo> = HashMap::new();
    let mut shard_extras: Vec<&String> = Vec::new();
    for (quant, mut files) in ggufs {
        files.sort();
        let entry = files[0];
        let entry_size = sizes.get(entry.as_str()).copied().unwrap_or(0);
        model_file.insert(
            quant,
            ModelFileInfo {
                name: entry.clone(),
                downloaded: true,
                size: entry_size,
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
        .any(|n| is_mmproj_filename(&n.to_lowercase()))
    {
        Some(ModelType::Vlm)
    } else {
        None
    }
}

/// Returns true when a `.gguf` filename refers to a multi-modal projector.
/// Matches `mmproj`, `mmproj-*`, `mmproj.*`, `mmproj_*` (the llama.cpp
/// default and underscore variants community repos sometimes use),
/// `*-mmproj` / `*_mmproj` (qualcomm-ai-hub-community-style suffix), and
/// `*-mmproj-*` / `*_mmproj_*` (infix). The check is case-insensitive —
/// callers must pass a lowercased name.
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

/// Extract a quant tag like `Q4_K_M`, `Q8_0`, `IQ4_XS`, `TQ1_0`, or `MXFP4`
/// from a filename. The match is case-insensitive (HF repos publish both
/// `q4_0` and `Q4_0`); the returned tag is upper-cased so it lines up with
/// [`QUANT_PRIORITY`]. Returns the highest-priority match if multiple are
/// present, else the first match, else None.
fn extract_quant(name: &str) -> Option<String> {
    // Walk the (uppercased, ASCII-only) name token by token. A quant tag
    // must sit on a token boundary so that `IQ4_XS` doesn't get scanned as
    // `Q4_XS` from index 1 — we anchor each candidate at a separator.
    let upper = name.to_ascii_uppercase();
    let bytes = upper.as_bytes();
    let mut composite: Vec<String> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !is_token_start(bytes, i) {
            i += 1;
            continue;
        }
        // Try MXFP-anchored tags first (`MXFP4`, `MXFP4_MOE`) — ggml-org's
        // gpt-oss GGUFs use them and they don't start with Q.
        if let Some(end) = scan_token(bytes, i, b"MXFP") {
            if let Ok(s) = std::str::from_utf8(&bytes[i..end]) {
                composite.push(s.to_string());
            }
            i = end;
            continue;
        }
        // Optional 1-byte prefix used by i-quants (`IQ*`) and ternary
        // quants (`TQ*`). Anything else means this token isn't a quant.
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

/// If the slice at `start` begins with `prefix` followed by ≥1 digit, return
/// the end index of the full tag (including any trailing short `_XXX`
/// segments); otherwise None.
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

/// Trailing `_XXX` segments. Each segment must be 1..=3 ASCII upper/digit
/// chars — long alphabetic words like `_FINETUNE` are NOT part of the tag.
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

/// True when index `i` of `bytes` sits at the start of a filename token —
/// either the beginning of the string, or right after one of the common
/// separators (`-`, `_`, `.`, path separators).
fn is_token_start(bytes: &[u8], i: usize) -> bool {
    if i == 0 {
        return true;
    }
    matches!(bytes[i - 1], b'-' | b'_' | b'.' | b'/' | b'\\')
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
    fn extract_quant_is_case_insensitive() {
        // qualcomm-ai-hub-community/gemma-4-E2B-it-qat-GGUF ships a lowercase
        // `q4_0` filename; without folding it landed in the "default" bucket
        // alongside the projector and got swapped at sort time.
        assert_eq!(
            extract_quant("gemma-4-E2B-q4_0-override.gguf"),
            Some("Q4_0".to_string())
        );
        assert_eq!(
            extract_quant("model-q4_k_m.gguf"),
            Some("Q4_K_M".to_string())
        );
    }

    #[test]
    fn extract_quant_recognises_i_and_ternary_prefixes() {
        // i-quants and ternary quants are real upstream tags — they must
        // be returned as-is, not silently truncated to a Q* tag.
        assert_eq!(
            extract_quant("model-IQ4_XS.gguf"),
            Some("IQ4_XS".to_string())
        );
        assert_eq!(extract_quant("model-iq1_s.gguf"), Some("IQ1_S".to_string()));
        assert_eq!(extract_quant("model-TQ1_0.gguf"), Some("TQ1_0".to_string()));
    }

    #[test]
    fn extract_quant_anchors_on_token_boundary() {
        // Mid-token Q (no separator before it) is not a quant.
        assert_eq!(extract_quant("aQ4_0.gguf"), None);
        assert_eq!(extract_quant("noQuant.gguf"), None);
        // Path separators count as token boundaries.
        assert_eq!(extract_quant("dir/Q4_0.gguf"), Some("Q4_0".to_string()));
        assert_eq!(extract_quant("dir\\Q4_0.gguf"), Some("Q4_0".to_string()));
    }

    #[test]
    fn extract_quant_does_not_glue_long_alphabetic_segments() {
        // A trailing `_FINETUNE` is not part of the quant tag — the inner
        // segment loop must cap segment length so the tag stays Q4_K_M.
        assert_eq!(
            extract_quant("model-Q4_K_M_finetune.gguf"),
            Some("Q4_K_M".to_string())
        );
    }

    #[test]
    fn extract_quant_recognises_mxfp() {
        // ggml-org/gpt-oss-*-GGUF publishes MXFP4 shards; the tag must be
        // recognised regardless of case and survive trailing shard suffixes.
        assert_eq!(extract_quant("model-MXFP4.gguf"), Some("MXFP4".to_string()));
        assert_eq!(extract_quant("model-mxfp4.gguf"), Some("MXFP4".to_string()));
        assert_eq!(
            extract_quant("gpt-oss-20b-mxfp4-00001-of-00002.gguf"),
            Some("MXFP4".to_string())
        );
        assert_eq!(
            extract_quant("model-MXFP4_MOE.gguf"),
            Some("MXFP4_MOE".to_string())
        );
        // Mid-token MXFP (no separator before it) is not a quant.
        assert_eq!(extract_quant("aMXFP4.gguf"), None);
    }

    #[test]
    fn is_mmproj_filename_matches_known_layouts() {
        // Standard llama.cpp prefix layout.
        assert!(is_mmproj_filename("mmproj-f16.gguf"));
        // Bare projector with no variant tag.
        assert!(is_mmproj_filename("mmproj.gguf"));
        // Underscore-separated variants seen in some community uploads.
        assert!(is_mmproj_filename("mmproj_f16.gguf"));
        assert!(is_mmproj_filename("model_mmproj.gguf"));
        // Suffix layout shipped by qualcomm-ai-hub-community/gemma-4-*-qat-GGUF.
        assert!(is_mmproj_filename("gemma-4-e2b-it-mmproj.gguf"));
        // Infix layout.
        assert!(is_mmproj_filename("model-mmproj-f16.gguf"));
        // Negative: a regular weight whose name happens to contain "proj".
        assert!(!is_mmproj_filename("model-q4_0-override.gguf"));
        assert!(!is_mmproj_filename("projector-only-no-mm.gguf"));
    }

    #[test]
    fn gemma4_qat_layout_picks_correct_entrypoint() {
        // End-to-end regression for #1013: the projector uses an `-mmproj`
        // suffix and the weight uses a lowercase `q4_0` tag. The real weight
        // must become the Q4_0 entrypoint and the projector must populate
        // mmproj_file.
        let (names, sizes) = sizes_of(&[
            ("gemma-4-E2B-it-mmproj.gguf", 500_000),
            ("gemma-4-E2B-q4_0-override.gguf", 2_000_000),
        ]);
        let m = infer_manifest_from_names(
            "qualcomm-ai-hub-community/gemma-4-E2B-it-qat-GGUF",
            &names,
            &sizes,
            Default::default(),
        )
        .unwrap();
        assert_eq!(m.model_type, ModelType::Vlm);
        assert_eq!(
            m.model_file.get("Q4_0").map(|f| f.name.as_str()),
            Some("gemma-4-E2B-q4_0-override.gguf")
        );
        assert_eq!(m.mmproj_file.name, "gemma-4-E2B-it-mmproj.gguf");
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
        // A 3-shard Q4_0 model: entrypoint records its own single-shard size,
        // the trailing shards land in extra_files so the executor still fetches
        // them. total_size() must count each shard exactly once.
        let (names, sizes) = sizes_of(&[
            ("model-Q4_0-00001-of-00003.gguf", 100),
            ("model-Q4_0-00002-of-00003.gguf", 200),
            ("model-Q4_0-00003-of-00003.gguf", 300),
        ]);
        let m =
            infer_manifest_from_names("Org/Repo-GGUF", &names, &sizes, Default::default()).unwrap();
        let entry = m.model_file.get("Q4_0").unwrap();
        assert_eq!(entry.name, "model-Q4_0-00001-of-00003.gguf");
        assert_eq!(entry.size, 100, "entrypoint stores its own size only");
        let extra: Vec<&str> = m.extra_files.iter().map(|f| f.name.as_str()).collect();
        assert!(extra.contains(&"model-Q4_0-00002-of-00003.gguf"));
        assert!(extra.contains(&"model-Q4_0-00003-of-00003.gguf"));
        assert_eq!(m.total_size(), 600, "every shard counted exactly once");
    }

    #[test]
    fn gpt_oss_20b_mxfp4_layout() {
        // ggml-org/gpt-oss-20b-GGUF ships a single 12 GB MXFP4 weight (no
        // shards). Before MXFP was recognised the file landed in the
        // `"default"` bucket and `:mxfp4` failed with QuantNotFound.
        let (names, sizes) = sizes_of(&[("gpt-oss-20b-mxfp4.gguf", 12_109_566_560)]);
        let hint = ManifestHint {
            quant: Some("MXFP4".to_string()),
            ..Default::default()
        };
        let m =
            infer_manifest_from_names("ggml-org/gpt-oss-20b-GGUF", &names, &sizes, hint).unwrap();
        let entry = m.model_file.get("MXFP4").expect("MXFP4 key present");
        assert_eq!(entry.name, "gpt-oss-20b-mxfp4.gguf");
        assert!(
            m.extra_files.is_empty(),
            "single-file repo has no extras: {:?}",
            m.extra_files
        );
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
