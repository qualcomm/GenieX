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

use crate::error::{Error, Result};
use crate::manifest::{ModelFileInfo, ModelManifest, ModelType};

/// Optional caller-supplied overrides used when inferring a manifest.
#[derive(Debug, Default, Clone)]
pub struct ManifestHint {
    /// Force this `ModelType`; otherwise inferred from presence of mmproj.
    pub model_type: Option<ModelType>,
    /// Force this plugin id; otherwise defaults to "llama_cpp".
    pub plugin_id: Option<String>,
    /// Force this model_name; otherwise derived from `name`.
    pub model_name: Option<String>,
    /// Restrict the inferred manifest to a single quantization. When set,
    /// GGUFs whose extracted quant tag doesn't match are excluded from
    /// `model_file`, so `pull` only fetches the requested quant. An
    /// unrecognised value is an error rather than a silent no-op.
    pub quant: Option<String>,
}

/// Quantization priority order (earlier = preferred). Mirrors the Go CLI.
const QUANT_PRIORITY: &[&str] = &["Q8_0", "Q4_K_M", "Q4_0"];

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

    // Pick one file per quant (prefer the largest — the actual weights).
    let mut model_file: HashMap<String, ModelFileInfo> = HashMap::new();
    for (quant, files) in ggufs {
        let chosen = files
            .iter()
            .max_by_key(|n| sizes.get(n.as_str()).copied().unwrap_or(0))
            .unwrap();
        model_file.insert(
            quant,
            ModelFileInfo {
                name: (*chosen).clone(),
                downloaded: true,
                size: sizes.get(chosen.as_str()).copied().unwrap_or(0),
            },
        );
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

    // ExtraFiles: all .npy + .geniex not used as mmproj.
    let mut extra_files: Vec<ModelFileInfo> = Vec::new();
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

    // Derive model_type: mmproj present => VLM, else LLM.
    let model_type = hint.model_type.unwrap_or_else(|| {
        if !mmproj_file.name.is_empty() {
            ModelType::Vlm
        } else {
            ModelType::Llm
        }
    });

    // Derive model_name: last path component of `name`, with -GGUF suffix
    // stripped. e.g. "NexaAI/Qwen3-4B-GGUF" -> "Qwen3-4B".
    let model_name = hint.model_name.unwrap_or_else(|| {
        let repo = name.rsplit('/').next().unwrap_or(name);
        repo.trim_end_matches("-GGUF")
            .trim_end_matches("-gguf")
            .to_string()
    });

    let plugin_id = hint.plugin_id.unwrap_or_else(|| "llama_cpp".to_string());

    Ok(ModelManifest {
        name: name.to_string(),
        model_name,
        model_type,
        plugin_id,
        device_id: String::new(),
        min_sdk_version: String::new(),
        model_file,
        mmproj_file,
        tokenizer_file,
        extra_files,
    })
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
}
