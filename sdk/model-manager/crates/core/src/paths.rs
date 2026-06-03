use crate::error::{Error, Result};
use crate::manifest::{ModelManifest, ModelType};
use crate::manifest_builder::QUANT_PRIORITY;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// Absolute path to the main model file.
    pub model_path: PathBuf,
    pub mmproj_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub model_dir: PathBuf,
    pub model_name: String,
    pub plugin_id: String,
    pub model_type: ModelType,
}

/// Resolve file paths from a manifest + local base directory + optional quant hint.
///
/// Replicates the logic in cli/server/service/keepalive.go:121-141:
/// - If `quant` is Some, look up that exact key; error if not downloaded.
/// - If `quant` is None, prefer the highest-ranked entry in
///   [`QUANT_PRIORITY`]; fall back to lexicographic min when none of the
///   downloaded quants appear in the priority list.
pub fn resolve_model_paths(
    manifest: &ModelManifest,
    base_dir: &Path,
    quant: Option<&str>,
) -> Result<(String, ModelPaths)> {
    let model_dir = base_dir.to_path_buf();

    let (resolved_quant, model_path) = {
        let (q, file_info) = if let Some(q) = quant {
            let fi = manifest
                .model_file
                .get(q)
                .ok_or_else(|| Error::QuantNotFound(q.to_string(), manifest.name.clone()))?;
            if !fi.downloaded {
                return Err(Error::QuantNotDownloaded(
                    q.to_string(),
                    manifest.name.clone(),
                ));
            }
            (q.to_string(), fi)
        } else {
            let downloaded: Vec<&str> = manifest
                .model_file
                .iter()
                .filter(|(_, v)| v.downloaded)
                .map(|(k, _)| k.as_str())
                .collect();
            if downloaded.is_empty() {
                return Err(Error::NoDownloadedQuant(manifest.name.clone()));
            }
            let q = pick_default_quant(&downloaded).to_string();
            let fi = &manifest.model_file[&q];
            (q, fi)
        };
        (q, model_dir.join(&file_info.name))
    };

    let mmproj_path = if !manifest.mmproj_file.name.is_empty() {
        Some(model_dir.join(&manifest.mmproj_file.name))
    } else {
        None
    };

    let tokenizer_path = if !manifest.tokenizer_file.name.is_empty() {
        Some(model_dir.join(&manifest.tokenizer_file.name))
    } else {
        None
    };

    Ok((
        resolved_quant,
        ModelPaths {
            model_path,
            mmproj_path,
            tokenizer_path,
            model_dir,
            model_name: manifest.model_name.clone(),
            plugin_id: manifest.plugin_id.clone(),
            model_type: manifest.model_type.clone(),
        },
    ))
}

/// Pick a default quant from a non-empty slice of available quants.
/// `QUANT_PRIORITY` wins; otherwise lexicographic min keeps the legacy
/// `slices.Min` behavior for unrecognised quants.
pub(crate) fn pick_default_quant<'a>(available: &'a [&'a str]) -> &'a str {
    for pref in QUANT_PRIORITY {
        if let Some(hit) = available.iter().find(|q| **q == *pref) {
            return hit;
        }
    }
    available.iter().min().copied().expect("non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelFileInfo, ModelManifest, ModelType};
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[test]
    fn pick_default_quant_uses_priority() {
        assert_eq!(pick_default_quant(&["Q4_0", "Q4_K_M", "Q8_0"]), "Q4_0");
        assert_eq!(pick_default_quant(&["Q4_K_M", "Q8_0"]), "Q4_K_M");
    }

    #[test]
    fn pick_default_quant_falls_back_to_lex_min() {
        assert_eq!(pick_default_quant(&["Q6_K", "Q5_K_M"]), "Q5_K_M");
        assert_eq!(pick_default_quant(&["IQ4_XS"]), "IQ4_XS");
    }

    #[test]
    fn pick_default_quant_partial_priority() {
        // Q4_0 is in priority, Q5_K_M is not — priority wins regardless of
        // lex order (Q4_0 < Q5_K_M lexicographically anyway, but that is
        // incidental).
        assert_eq!(pick_default_quant(&["Q4_0", "Q5_K_M"]), "Q4_0");
    }

    fn manifest_with(quants: &[(&str, bool)]) -> ModelManifest {
        let mut model_file: HashMap<String, ModelFileInfo> = HashMap::new();
        for (q, downloaded) in quants {
            model_file.insert(
                (*q).to_string(),
                ModelFileInfo {
                    name: format!("model-{q}.gguf"),
                    downloaded: *downloaded,
                    size: 0,
                },
            );
        }
        ModelManifest {
            name: "owner/repo".to_string(),
            model_name: "repo".to_string(),
            model_type: ModelType::Llm,
            plugin_id: "llama_cpp".to_string(),
            precision: String::new(),
            model_file,
            mmproj_file: ModelFileInfo::default(),
            tokenizer_file: ModelFileInfo::default(),
            extra_files: Vec::new(),
        }
    }

    #[test]
    fn resolve_paths_no_quant_picks_priority() {
        let m = manifest_with(&[("Q4_0", true), ("Q4_K_M", true), ("Q8_0", true)]);
        let (q, paths) = resolve_model_paths(&m, &PathBuf::from("/cache"), None).unwrap();
        assert_eq!(q, "Q4_0");
        assert_eq!(
            paths.model_path,
            PathBuf::from("/cache").join("model-Q4_0.gguf")
        );
    }

    #[test]
    fn resolve_paths_no_quant_falls_back_to_lex_min() {
        let m = manifest_with(&[("Q6_K", true), ("Q5_K_M", true)]);
        let (q, _) = resolve_model_paths(&m, &PathBuf::from("/cache"), None).unwrap();
        assert_eq!(q, "Q5_K_M");
    }

    #[test]
    fn resolve_paths_no_quant_skips_undownloaded_priority_member() {
        let m = manifest_with(&[("Q4_0", false), ("Q4_K_M", true), ("Q8_0", true)]);
        let (q, _) = resolve_model_paths(&m, &PathBuf::from("/cache"), None).unwrap();
        assert_eq!(q, "Q4_K_M");
    }
}
