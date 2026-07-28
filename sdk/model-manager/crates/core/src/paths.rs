// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

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

/// Placeholder key QAIRT (AI Hub / localfs) manifests use for their single
/// `model_file` entry; the human-meaningful precision (e.g. "W4A16") lives on
/// the manifest's top-level `Precision` field instead.
pub(crate) const PRECISION_NA: &str = "N/A";

/// Resolve file paths from a manifest + local base directory + optional quant hint.
///
/// Replicates the logic in cli/server/service/keepalive.go:121-141:
/// - If `quant` is Some, look up that exact key; error if not downloaded.
///   The manifest's top-level precision is accepted as an alias for the
///   [`PRECISION_NA`] entry: listings (`geniex list`, serve `/v1/models`)
///   surface that precision as the id suffix, so the advertised
///   `name:precision` must resolve back here (#1242).
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
            let (q, fi) = match manifest.model_file.get(q) {
                Some(fi) => (q.to_string(), fi),
                None if !manifest.precision.is_empty()
                    && q.eq_ignore_ascii_case(&manifest.precision) =>
                {
                    let fi = manifest.model_file.get(PRECISION_NA).ok_or_else(|| {
                        Error::QuantNotFound(q.to_string(), manifest.name.clone())
                    })?;
                    (PRECISION_NA.to_string(), fi)
                }
                None => {
                    return Err(Error::QuantNotFound(q.to_string(), manifest.name.clone()));
                }
            };
            if !fi.downloaded {
                return Err(Error::QuantNotDownloaded(q, manifest.name.clone()));
            }
            (q, fi)
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

    /// QAIRT-shaped manifest: single "N/A" model_file entry, real precision on
    /// the top-level field (what AI Hub / localfs pulls write).
    fn qairt_manifest(precision: &str, downloaded: bool) -> ModelManifest {
        let mut m = manifest_with(&[(PRECISION_NA, downloaded)]);
        m.plugin_id = "qairt".to_string();
        m.precision = precision.to_string();
        m
    }

    // #1242: /v1/models advertises "name:W4A16" (the top-level precision), so
    // that exact id must resolve.
    #[test]
    fn resolve_paths_accepts_top_level_precision_alias() {
        let m = qairt_manifest("W4A16", true);
        let (q, paths) = resolve_model_paths(&m, &PathBuf::from("/cache"), Some("W4A16")).unwrap();
        assert_eq!(q, PRECISION_NA);
        assert_eq!(
            paths.model_path,
            PathBuf::from("/cache").join("model-N/A.gguf")
        );
    }

    #[test]
    fn resolve_paths_precision_alias_is_case_insensitive() {
        let m = qairt_manifest("W4A16", true);
        let (q, _) = resolve_model_paths(&m, &PathBuf::from("/cache"), Some("w4a16")).unwrap();
        assert_eq!(q, PRECISION_NA);
    }

    #[test]
    fn resolve_paths_precision_alias_requires_download() {
        let m = qairt_manifest("W4A16", false);
        let err = resolve_model_paths(&m, &PathBuf::from("/cache"), Some("W4A16")).unwrap_err();
        assert!(matches!(err, Error::QuantNotDownloaded(..)), "got {err:?}");
    }

    #[test]
    fn resolve_paths_unknown_quant_still_errors() {
        let m = qairt_manifest("W4A16", true);
        let err = resolve_model_paths(&m, &PathBuf::from("/cache"), Some("Q4_0")).unwrap_err();
        assert!(matches!(err, Error::QuantNotFound(..)), "got {err:?}");
    }

    #[test]
    fn resolve_paths_exact_key_wins_over_precision_alias() {
        // A real model_file key equal to the request must resolve directly,
        // even when the manifest also carries a top-level precision.
        let mut m = manifest_with(&[("Q4_0", true), (PRECISION_NA, true)]);
        m.precision = "Q4_0".to_string();
        let (q, _) = resolve_model_paths(&m, &PathBuf::from("/cache"), Some("Q4_0")).unwrap();
        assert_eq!(q, "Q4_0");
    }
}
