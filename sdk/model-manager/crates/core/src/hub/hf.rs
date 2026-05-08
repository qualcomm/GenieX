//! HuggingFace [`ModelHub`] — composes [`HfMetadata`] +
//! [`ReqwestTransport`] and drives them through the [`Engine`].
//!
//! This module is pure async. Sync callers should go through the FFI
//! crate (`crates/ffi`), which owns a single process-global runtime
//! rather than spinning one up per pull.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

use crate::download::{Engine, EngineConfig};
use crate::error::Result;
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
}

#[async_trait]
impl ModelHub for HfHub {
    async fn list_files(
        &self,
        repo_id: &str,
    ) -> Result<(Vec<RemoteFile>, Option<ModelManifest>)> {
        self.ctx.metadata.list_files(repo_id).await
    }

    async fn download(
        &self,
        repo_id: &str,
        files: &[String],
        dest_dir: &Path,
        on_progress: Option<&ProgressCallback>,
    ) -> Result<()> {
        for f in files {
            validate_relative_file(f)?;
        }
        let sources = self.ctx.metadata.resolve(repo_id, files).await?;
        let engine = Engine::with_config(&self.ctx, EngineConfig::resolve(&self.ctx));
        engine.run(sources, dest_dir, on_progress).await
    }
}
