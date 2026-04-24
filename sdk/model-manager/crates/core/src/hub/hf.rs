use std::path::Path;

use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

use crate::error::{Error, Result};
use crate::manifest::ModelManifest;
use crate::validation::validate_relative_file;

use super::{FileProgress, ModelHub, ProgressCallback, RemoteFile};

pub struct HfHub {
    api: hf_hub::api::sync::Api,
}

impl HfHub {
    pub fn new(token: Option<String>) -> Result<Self> {
        let mut builder = ApiBuilder::new();
        if let Some(t) = token {
            builder = builder.with_token(Some(t));
        }
        let api = builder.build().map_err(|e| Error::Hub(e.to_string()))?;
        Ok(Self { api })
    }
}

impl ModelHub for HfHub {
    fn list_files(
        &self,
        repo_id: &str,
    ) -> Result<(Vec<RemoteFile>, Option<ModelManifest>)> {
        let repo = self.api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
        let info = repo.info().map_err(|e| Error::Hub(e.to_string()))?;

        let files: Vec<RemoteFile> = info
            .siblings
            .iter()
            .map(|s| RemoteFile {
                name: s.rfilename.clone(),
                // hf-hub Siblings doesn't expose size in v0.4; use -1 (unknown).
                size: -1,
            })
            .collect();

        // If the repo ships a geniex.json, fetch and parse it.
        let manifest = if files.iter().any(|f| f.name == "geniex.json") {
            match repo.get("geniex.json") {
                Ok(cached) => std::fs::read_to_string(&cached)
                    .ok()
                    .and_then(|data| serde_json::from_str(&data).ok()),
                Err(_) => None,
            }
        } else {
            None
        };

        Ok((files, manifest))
    }

    fn download(
        &self,
        repo_id: &str,
        files: &[String],
        dest_dir: &Path,
        on_progress: Option<&ProgressCallback>,
    ) -> Result<()> {
        std::fs::create_dir_all(dest_dir)?;
        let repo = self.api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

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

            let cached = repo.get(file_name).map_err(|e| Error::Hub(e.to_string()))?;
            let dest = dest_dir.join(file_name);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let size = std::fs::metadata(&cached).map(|m| m.len() as i64).unwrap_or(-1);
            std::fs::copy(&cached, &dest)?;

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
