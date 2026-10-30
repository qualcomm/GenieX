// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Plan, fetch, then publish the manifest; `.inflight/` marks a pull in
//! progress and `geniex.json` appears only once every file is complete.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::config::StoreConfig;
use crate::error::{Error, Result};
use crate::executor::resume;
use crate::executor::{Executor, ProgressCallback};
use crate::manifest::infer::ManifestHint;
use crate::model_ref::canonicalize_model_name;
use crate::source::ai_hub::{AiHubConfig, AiHubSource};
use crate::source::dockerhub::DockerHubSource;
use crate::source::hf::HfSource;
use crate::source::localfs::LocalFsSource;
use crate::source::modelscope::ModelScopeSource;
use crate::source::ModelSource;
use crate::store::{Store, INFLIGHT_DIR, MANIFEST_FILE};
use crate::transport::{HttpTransport, ReqwestTransport};
use crate::validation::validate_model_name;

pub struct PullRequest {
    pub model_name: String,
    pub intent: PullIntent,
    pub on_progress: Option<ProgressCallback>,
    pub hint: ManifestHint,
}

pub enum PullIntent {
    HuggingFace {
        repo: String,
        token: Option<String>,
    },
    LocalFs {
        source_dir: PathBuf,
    },
    ModelScope {
        repo: String,
        token: Option<String>,
    },
    AiHub {
        display_name: String,
        chipset: String,
    },
    DockerHub {
        repo: String,
        /// Tag or `sha256:<hex>` digest; empty means `"latest"`.
        reference: String,
    },
}

pub async fn pull(store: &Store, mut req: PullRequest) -> Result<()> {
    let (source, transport) = prepare(store, &mut req)?;
    pull_with_source(
        store,
        &req.model_name,
        source,
        transport,
        req.on_progress.as_ref(),
    )
    .await
}

pub async fn pull_with_source(
    store: &Store,
    model_name: &str,
    source: Box<dyn ModelSource>,
    transport: Arc<dyn HttpTransport>,
    on_progress: Option<&ProgressCallback>,
) -> Result<()> {
    validate_model_name(model_name)?;
    let model_name_owned = model_name.to_string();
    store
        .with_model_lock_async(model_name, || async move {
            let dest_dir = store.model_dir(&model_name_owned)?;
            let inflight_dir = dest_dir.join(INFLIGHT_DIR);
            fs::create_dir_all(&inflight_dir)?;

            let mut plan = source.plan().await?;
            plan.manifest.name = model_name_owned.clone();

            // A quant-specific plan carries only that quant; keep the ones on disk.
            let final_path = dest_dir.join(MANIFEST_FILE);
            if let Ok(existing) = store.get_manifest(&model_name_owned) {
                plan.manifest.merge_existing(existing);
            }

            let pending = resume::filter_pending(&plan.files, &dest_dir);
            let required: u64 = pending.iter().map(|f| f.size).sum();
            let available = fs2::available_space(&dest_dir)?;
            check_disk_space(required, available)?;
            if !pending.is_empty() {
                let executor = Executor::new(transport.clone(), 4);
                executor.run(&pending, &dest_dir, on_progress).await?;
            }

            let staged = inflight_dir.join(MANIFEST_FILE);
            fs::write(&staged, serde_json::to_string(&plan.manifest)?)?;
            fs::rename(&staged, &final_path)?;

            // The Go CLI drops these markers too; keeps cache layouts interchangeable.
            for f in &plan.files {
                drop_marker(&dest_dir, &f.name);
            }

            remove_inflight_dir(&inflight_dir);

            Ok(())
        })
        .await
}

pub(crate) fn prepare(
    store: &Store,
    req: &mut PullRequest,
) -> Result<(Box<dyn ModelSource>, Arc<dyn HttpTransport>)> {
    req.model_name = canonicalize_model_name(&req.model_name);
    validate_model_name(&req.model_name)?;
    let transport: Arc<dyn HttpTransport> = Arc::new(ReqwestTransport::new()?);
    Ok((build_source(req, store, transport.clone())?, transport))
}

fn build_source(
    req: &PullRequest,
    store: &Store,
    transport: Arc<dyn HttpTransport>,
) -> Result<Box<dyn ModelSource>> {
    match &req.intent {
        PullIntent::HuggingFace { repo, token } => {
            let endpoint = StoreConfig::hf_endpoint();
            let src = HfSource::with_endpoint_and_transport(
                repo.clone(),
                &endpoint,
                token.clone(),
                transport,
                req.hint.clone(),
            )?;
            Ok(Box::new(src))
        }
        PullIntent::LocalFs { source_dir } => Ok(Box::new(LocalFsSource::new(
            source_dir.clone(),
            req.model_name.clone(),
            req.hint.clone(),
        ))),
        PullIntent::ModelScope { repo, token } => {
            let src =
                ModelScopeSource::new(repo.clone(), token.clone(), transport, req.hint.clone())?;
            Ok(Box::new(src))
        }
        PullIntent::AiHub {
            display_name,
            chipset,
        } => Ok(Box::new(AiHubSource::with_transport(
            display_name.clone(),
            req.model_name.clone(),
            AiHubConfig::from_store(store, chipset.clone()),
            transport,
        ))),
        PullIntent::DockerHub { repo, reference } => {
            let src = DockerHubSource::new(repo.clone(), reference.clone(), transport)?;
            Ok(Box::new(src))
        }
    }
}

// AV scans / the indexer can briefly hold handles on Windows; a leftover
// `.inflight/` would hide the model from `Store::list`, so retry.
fn remove_inflight_dir(inflight_dir: &Path) {
    const DELAYS_MS: [u64; 3] = [50, 100, 200];
    for delay in DELAYS_MS {
        match fs::remove_dir_all(inflight_dir) {
            Ok(()) => return,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(delay)),
        }
    }
    if let Err(e) = fs::remove_dir_all(inflight_dir) {
        if e.kind() != std::io::ErrorKind::NotFound {
            crate::logging::warn(&format!(
                "failed to clean up {} after successful pull: {e}",
                inflight_dir.display()
            ));
        }
    }
}

fn check_disk_space(required: u64, available: u64) -> Result<()> {
    if required > available {
        return Err(Error::InsufficientDiskSpace {
            required,
            available,
        });
    }
    Ok(())
}

fn drop_marker(dest_dir: &Path, file_name: &str) {
    if file_name.is_empty() {
        return;
    }
    let _ = fs::remove_file(resume::marker_path(dest_dir, file_name));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remove_inflight_dir_handles_empty_nonempty_and_missing() {
        for (create, leftover) in [(true, false), (true, true), (false, false)] {
            let tmp = tempfile::tempdir().unwrap();
            let inflight = tmp.path().join(INFLIGHT_DIR);
            if create {
                fs::create_dir_all(&inflight).unwrap();
            }
            if leftover {
                fs::write(inflight.join("leftover.bin"), b"stale").unwrap();
            }
            remove_inflight_dir(&inflight);
            assert!(!inflight.exists(), "create={create} leftover={leftover}");
        }
    }

    #[test]
    fn check_disk_space_allows_up_to_the_boundary() {
        assert!(check_disk_space(100, 200).is_ok());
        assert!(check_disk_space(100, 100).is_ok());
        assert!(matches!(
            check_disk_space(200, 100),
            Err(Error::InsufficientDiskSpace {
                required: 200,
                available: 100
            })
        ));
    }
}
