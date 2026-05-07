//! HuggingFace [`ModelHub`] — thin sync adapter that composes
//! [`HfMetadata`] + [`ReqwestTransport`] and drives them through the
//! [`Engine`]. The `ModelHub` trait is sync for backward compatibility
//! with [`crate::pull::pull_locked`] and the FFI layer; we run the async
//! engine on a scoped multi-thread tokio runtime that is built and
//! dropped inside `download()` so no global reactor leaks out.

use std::path::Path;
use std::sync::Arc;

use tokio::runtime::Builder;

use crate::download::{Engine, EngineConfig};
use crate::error::{Error, Result};
use crate::hub::hf_metadata::HfMetadata;
use crate::hub::metadata::HubContext;
use crate::hub::{ModelHub, ProgressCallback, RemoteFile};
use crate::manifest::ModelManifest;
use crate::transport::{HttpTransport, ReqwestTransport};
use crate::validation::validate_relative_file;

pub struct HfHub {
    ctx: HubContext,
}

impl HfHub {
    pub fn new(token: Option<String>) -> Result<Self> {
        let transport: Arc<dyn HttpTransport> = Arc::new(ReqwestTransport::new()?);
        let metadata = Arc::new(HfMetadata::new(token, transport.clone())?);
        Ok(Self {
            ctx: HubContext::new(metadata, transport),
        })
    }

    /// Escape hatch for tests / alternate endpoints. The metadata layer
    /// points at `endpoint`, and both layers share the same transport
    /// (so proxy settings apply uniformly).
    pub fn with_endpoint(
        endpoint: &str,
        token: Option<String>,
        transport: Arc<dyn HttpTransport>,
    ) -> Result<Self> {
        let metadata = Arc::new(HfMetadata::with_endpoint(
            endpoint,
            token,
            transport.clone(),
        )?);
        Ok(Self {
            ctx: HubContext::new(metadata, transport),
        })
    }

    fn worker_threads() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(8)
    }
}

impl ModelHub for HfHub {
    fn list_files(&self, repo_id: &str) -> Result<(Vec<RemoteFile>, Option<ModelManifest>)> {
        // Single short-lived current-thread runtime is enough for a
        // one-shot metadata call — no need to spin up the worker pool.
        let rt = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::Http(format!("build current-thread runtime: {e}")))?;
        rt.block_on(self.ctx.metadata.list_files(repo_id))
    }

    fn download(
        &self,
        repo_id: &str,
        files: &[String],
        dest_dir: &Path,
        on_progress: Option<&ProgressCallback>,
    ) -> Result<()> {
        for f in files {
            validate_relative_file(f)?;
        }
        let names: Vec<String> = files.to_vec();

        let rt = Builder::new_multi_thread()
            .worker_threads(Self::worker_threads())
            .enable_all()
            .build()
            .map_err(|e| Error::Http(format!("build multi-thread runtime: {e}")))?;

        rt.block_on(async {
            let sources = self.ctx.metadata.resolve(repo_id, &names).await?;
            let engine = Engine::with_config(&self.ctx, EngineConfig::resolve(&self.ctx));
            engine.run(sources, dest_dir, on_progress).await
        })
    }
}
