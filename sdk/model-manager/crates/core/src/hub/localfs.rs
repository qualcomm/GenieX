use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::manifest::ModelManifest;
use crate::validation::validate_relative_file;

use super::{FileProgress, ModelHub, ProgressCallback, RemoteFile};

pub struct LocalFsHub {
    source_dir: PathBuf,
}

impl LocalFsHub {
    pub fn new(source_dir: PathBuf) -> Self {
        Self { source_dir }
    }

    pub fn source_dir(&self) -> &Path {
        &self.source_dir
    }
}

impl ModelHub for LocalFsHub {
    fn list_files(&self, _repo_id: &str) -> Result<(Vec<RemoteFile>, Option<ModelManifest>)> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&self.source_dir)?.flatten() {
            let ft = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if !ft.is_file() {
                continue;
            }
            if let Some(name) = entry.file_name().to_str().map(str::to_string) {
                let size = std::fs::metadata(entry.path())
                    .map(|m| m.len() as i64)
                    .unwrap_or(-1);
                files.push(RemoteFile { name, size });
            }
        }

        // If the source directory has a geniex.json, use it as the canonical manifest.
        let manifest_path = self.source_dir.join("geniex.json");
        let manifest = if manifest_path.exists() {
            std::fs::read_to_string(&manifest_path)
                .ok()
                .and_then(|data| serde_json::from_str(&data).ok())
        } else {
            None
        };

        Ok((files, manifest))
    }

    fn download(
        &self,
        _repo_id: &str,
        files: &[String],
        dest_dir: &Path,
        on_progress: Option<&ProgressCallback>,
    ) -> Result<()> {
        std::fs::create_dir_all(dest_dir)?;

        let mut tracked: Vec<FileProgress> = files
            .iter()
            .map(|n| FileProgress {
                file_name: n.clone(),
                downloaded_bytes: 0,
                total_bytes: -1,
            })
            .collect();

        for (idx, file_name) in files.iter().enumerate() {
            validate_relative_file(file_name)?;
            let src = self.source_dir.join(file_name);
            if !src.exists() {
                return Err(Error::Hub(format!(
                    "local file not found: {}",
                    src.display()
                )));
            }
            let dest = dest_dir.join(file_name);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let size = std::fs::metadata(&src)
                .map(|m| m.len() as i64)
                .unwrap_or(-1);
            std::fs::copy(&src, &dest)?;

            tracked[idx].downloaded_bytes = size.max(0);
            tracked[idx].total_bytes = size;

            if let Some(cb) = on_progress {
                if !cb(&tracked) {
                    return Err(Error::Hub("download cancelled by caller".to_string()));
                }
            }
        }
        Ok(())
    }
}
