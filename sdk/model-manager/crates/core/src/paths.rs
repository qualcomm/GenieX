use crate::error::{Error, Result};
use crate::manifest::ModelManifest;
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
    pub device_id: Option<String>,
}

/// Resolve file paths from a manifest + local base directory + optional quant hint.
///
/// Replicates the logic in cli/server/service/keepalive.go:121-141:
/// - If `quant` is Some, look up that exact key; error if not downloaded.
/// - If `quant` is None, pick the lexicographically smallest downloaded quant
///   (matches Go's `slices.Min`).
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
            // Pick the lexicographically smallest downloaded quant.
            let mut quants: Vec<&str> = manifest
                .model_file
                .iter()
                .filter(|(_, v)| v.downloaded)
                .map(|(k, _)| k.as_str())
                .collect();
            if quants.is_empty() {
                return Err(Error::NoDownloadedQuant(manifest.name.clone()));
            }
            quants.sort();
            let q = quants[0].to_string();
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

    let device_id = if manifest.device_id.is_empty() {
        None
    } else {
        Some(manifest.device_id.clone())
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
            device_id,
        },
    ))
}
