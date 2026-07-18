// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Docker Hub [`ModelSource`] — pulls GGUF-packaged model artifacts
//! published under <https://hub.docker.com/u/ai> (and any other
//! Docker-format OCI registry) using the plain Docker Registry HTTP
//! API V2, the same protocol `docker model pull ai/<repo>` uses under
//! the hood (see `docker/model-runner`'s `pkg/distribution` package).
//!
//! Three network calls resolve a full [`Plan`] without ever streaming
//! a weight blob:
//!  1. Anonymous bearer token exchange (`GET {auth}?service=...&scope=...`).
//!  2. Content-negotiated manifest GET (`GET /v2/{repo}/manifests/{ref}`).
//!  3. A small config blob GET (`GET /v2/{repo}/blobs/{digest}`), which
//!     carries the GGUF quantization tag and tells us whether the
//!     model is a VLM (an `mmproj` layer is present).
//!
//! Only the Docker proprietary format (`application/vnd.docker.ai.*`
//! media types, "v0.1" single-layer-per-format and "v0.2"
//! layer-per-file with `org.cncf.model.filepath` annotations) is
//! supported — this is what `docker model package` / the official
//! `ai/*` Docker Hub images publish. CNCF ModelPack artifacts
//! (`application/vnd.cncf.model.*`) are not yet handled.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client;
use serde::Deserialize;
use url::Url;

use crate::error::{Error, Result};
use crate::manifest::{ModelFileInfo, ModelManifest, ModelType};
use crate::manifest_builder::extract_quant;
use crate::transport::{build_tls_config, HttpTransport, USER_AGENT};

use super::{BytesSource, FileSpec, ModelSource, Plan};

pub const DEFAULT_REGISTRY_ENDPOINT: &str = "https://registry-1.docker.io";
pub const DEFAULT_AUTH_ENDPOINT: &str = "https://auth.docker.io/token";
pub const DEFAULT_AUTH_SERVICE: &str = "registry.docker.io";
const DEFAULT_REFERENCE: &str = "latest";
const MAX_MANIFEST_BYTES: usize = 4 * 1024 * 1024;
const MAX_CONFIG_BYTES: u64 = 4 * 1024 * 1024;
/// An index is only ever chased one level deep — a Docker-format model
/// image points straight at its manifest, never a nested index of
/// indices.
const MAX_INDEX_REDIRECTS: u32 = 1;

mod media {
    // Docker proprietary model media types (application/vnd.docker.ai.*).
    // CONFIG_V01 (legacy, single-layer-per-format) is only "not
    // CONFIG_V02" in production code, but the constant is kept for
    // clarity in tests that build a v0.1-shaped manifest fixture.
    #[cfg(test)]
    pub const CONFIG_V01: &str = "application/vnd.docker.ai.model.config.v0.1+json";
    pub const CONFIG_V02: &str = "application/vnd.docker.ai.model.config.v0.2+json";
    pub const GGUF: &str = "application/vnd.docker.ai.gguf.v3";
    pub const SAFETENSORS: &str = "application/vnd.docker.ai.safetensors";
    pub const DDUF: &str = "application/vnd.docker.ai.dduf";
    pub const MMPROJ: &str = "application/vnd.docker.ai.mmproj";
    pub const CHAT_TEMPLATE: &str = "application/vnd.docker.ai.chat.template.jinja";
    pub const LICENSE: &str = "application/vnd.docker.ai.license";
    pub const MODEL_FILE: &str = "application/vnd.docker.ai.model.file";

    // Manifest / index media types accepted from the registry.
    pub const OCI_MANIFEST: &str = "application/vnd.oci.image.manifest.v1+json";
    pub const OCI_INDEX: &str = "application/vnd.oci.image.index.v1+json";
    pub const DOCKER_MANIFEST_V2: &str = "application/vnd.docker.distribution.manifest.v2+json";
    pub const DOCKER_MANIFEST_LIST: &str =
        "application/vnd.docker.distribution.manifest.list.v2+json";
}

/// The `org.cncf.model.filepath` annotation the "v0.2" layer-per-file
/// packaging format stamps on each layer descriptor.
const ANNOTATION_FILEPATH: &str = "org.cncf.model.filepath";

#[derive(Debug, Clone)]
pub struct DockerHubConfig {
    pub registry_endpoint: String,
    pub auth_endpoint: String,
    pub service: String,
}

impl Default for DockerHubConfig {
    fn default() -> Self {
        Self {
            registry_endpoint: DEFAULT_REGISTRY_ENDPOINT.to_string(),
            auth_endpoint: DEFAULT_AUTH_ENDPOINT.to_string(),
            service: DEFAULT_AUTH_SERVICE.to_string(),
        }
    }
}

pub struct DockerHubSource {
    /// "ai/gemma3" — the Docker Hub repository, no registry host prefix.
    repo: String,
    /// Tag or `sha256:<hex>` digest. Empty is treated as `"latest"`.
    reference: String,
    cfg: DockerHubConfig,
    /// Reused for blob GETs so chunked download / retry behaviour stays
    /// identical to every other source.
    transport: Arc<dyn HttpTransport>,
    /// Manifest + token exchange need per-request `Accept` headers the
    /// [`HttpTransport`] trait doesn't expose, so this source keeps a
    /// small dedicated client for just those two calls.
    http: Client,
}

impl DockerHubSource {
    pub fn new(repo: String, reference: String, transport: Arc<dyn HttpTransport>) -> Result<Self> {
        Self::with_config(repo, reference, DockerHubConfig::default(), transport)
    }

    pub fn with_config(
        repo: String,
        reference: String,
        cfg: DockerHubConfig,
        transport: Arc<dyn HttpTransport>,
    ) -> Result<Self> {
        let http = Client::builder()
            .user_agent(USER_AGENT)
            .use_preconfigured_tls(build_tls_config()?)
            .build()
            .map_err(|e| Error::Http(format!("build docker registry client: {e}")))?;
        let reference = if reference.trim().is_empty() {
            DEFAULT_REFERENCE.to_string()
        } else {
            reference
        };
        Ok(Self {
            repo,
            reference,
            cfg,
            transport,
            http,
        })
    }

    /// Exchange an anonymous pull scope for a short-lived bearer token.
    /// Public Docker Hub repositories (including everything under
    /// `hub.docker.com/u/ai`) don't require credentials for this —  a
    /// registry that doesn't require auth at all is tolerated by
    /// treating a failed exchange as "no token" and letting the
    /// manifest/blob request surface the real error if one turns out
    /// to be required.
    async fn fetch_token(&self) -> Option<String> {
        let scope = format!("repository:{}:pull", self.repo);
        let url = Url::parse_with_params(
            &self.cfg.auth_endpoint,
            &[
                ("service", self.cfg.service.as_str()),
                ("scope", scope.as_str()),
            ],
        )
        .ok()?;
        let resp = self.http.get(url).send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let bytes = resp.bytes().await.ok()?;
        let body: TokenResponse = serde_json::from_slice(&bytes).ok()?;
        body.token.or(body.access_token)
    }

    async fn fetch_manifest(
        &self,
        token: Option<&str>,
        reference: &str,
    ) -> Result<RegistryManifest> {
        let url = Url::parse(&format!(
            "{}/v2/{}/manifests/{reference}",
            self.cfg.registry_endpoint, self.repo
        ))
        .map_err(|e| Error::Hub(format!("invalid docker manifest url: {e}")))?;

        let accept = [
            media::OCI_MANIFEST,
            media::DOCKER_MANIFEST_V2,
            media::OCI_INDEX,
            media::DOCKER_MANIFEST_LIST,
        ]
        .join(", ");
        let mut req = self.http.get(url.clone()).header(ACCEPT, accept);
        if let Some(tok) = token {
            req = req.header(AUTHORIZATION, format!("Bearer {tok}"));
        }
        let resp = req
            .send()
            .await
            .map_err(|e| Error::HttpTimeout(format!("GET {url}: {e}")))?;

        let status = resp.status();
        if status.as_u16() == 404 {
            return Err(Error::HubModelNotFound(format!(
                "{}:{reference}",
                self.repo
            )));
        }
        if !status.is_success() {
            return Err(Error::HttpStatus {
                url: url.to_string(),
                status: status.as_u16(),
            });
        }

        let content_type = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| Error::Http(format!("read manifest {url}: {e}")))?;
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(Error::Hub(format!(
                "manifest at {url} is {} bytes, exceeds {MAX_MANIFEST_BYTES} byte cap",
                bytes.len()
            )));
        }
        let mut parsed: RegistryManifest =
            serde_json::from_slice(&bytes).map_err(|source| Error::ManifestParse {
                what: "docker registry manifest",
                source,
            })?;
        if parsed.media_type.is_empty() {
            parsed.media_type = content_type;
        }
        Ok(parsed)
    }

    fn blob_url(&self, digest: &str) -> Result<Url> {
        Url::parse(&format!(
            "{}/v2/{}/blobs/{digest}",
            self.cfg.registry_endpoint, self.repo
        ))
        .map_err(|e| Error::Hub(format!("invalid docker blob url: {e}")))
    }

    async fn fetch_blob(&self, digest: &str, token: Option<&str>, cap: u64) -> Result<Vec<u8>> {
        let url = self.blob_url(digest)?;
        let head = self.transport.head(&url, token).await?;
        if head.size > cap {
            return Err(Error::Hub(format!(
                "blob {digest} is {} bytes, exceeds {cap} byte cap",
                head.size
            )));
        }
        let mut buf: Vec<u8> = Vec::with_capacity(head.size as usize);
        self.transport
            .get_range(&url, token, 0, head.size, &mut buf)
            .await?;
        Ok(buf)
    }
}

#[async_trait]
impl ModelSource for DockerHubSource {
    async fn plan(&self) -> Result<Plan> {
        let token = self.fetch_token().await;

        let mut manifest = self
            .fetch_manifest(token.as_deref(), &self.reference)
            .await?;
        let mut redirects = 0;
        while manifest.is_index() {
            if redirects >= MAX_INDEX_REDIRECTS {
                return Err(Error::Hub(format!(
                    "docker manifest index for {} nested too deep",
                    self.repo
                )));
            }
            let first = manifest.manifests.first().ok_or_else(|| {
                Error::Hub(format!(
                    "docker manifest index for {} lists no manifests",
                    self.repo
                ))
            })?;
            manifest = self.fetch_manifest(token.as_deref(), &first.digest).await?;
            redirects += 1;
        }

        let config_desc = manifest.config.clone().ok_or_else(|| {
            Error::Hub(format!(
                "docker manifest for {} has no config descriptor",
                self.repo
            ))
        })?;
        let config_bytes = self
            .fetch_blob(&config_desc.digest, token.as_deref(), MAX_CONFIG_BYTES)
            .await?;
        let config_file: DockerConfigFile =
            serde_json::from_slice(&config_bytes).map_err(|source| Error::ManifestParse {
                what: "docker model config",
                source,
            })?;

        let is_layer_per_file = config_desc.media_type == media::CONFIG_V02;
        build_plan(
            &self.repo,
            &self.reference,
            &config_file.config,
            &manifest.layers,
            is_layer_per_file,
            token,
            |digest| self.blob_url(digest),
        )
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Descriptor {
    #[serde(rename = "mediaType", default)]
    media_type: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    digest: String,
    #[serde(default)]
    annotations: HashMap<String, String>,
}

#[derive(Debug, Default, Deserialize)]
struct RegistryManifest {
    #[serde(rename = "mediaType", default)]
    media_type: String,
    #[serde(default)]
    config: Option<Descriptor>,
    #[serde(default)]
    layers: Vec<Descriptor>,
    /// Only present on an index/manifest-list response.
    #[serde(default)]
    manifests: Vec<Descriptor>,
}

impl RegistryManifest {
    fn is_index(&self) -> bool {
        self.media_type == media::OCI_INDEX
            || self.media_type == media::DOCKER_MANIFEST_LIST
            || (self.config.is_none() && !self.manifests.is_empty())
    }
}

#[derive(Debug, Default, Deserialize)]
struct DockerConfig {
    #[serde(default)]
    quantization: String,
}

#[derive(Debug, Default, Deserialize)]
struct DockerConfigFile {
    #[serde(default)]
    config: DockerConfig,
}

#[derive(Debug, Default, Deserialize)]
struct TokenResponse {
    #[serde(default)]
    token: Option<String>,
    #[serde(default)]
    access_token: Option<String>,
}

/// Classify `layers` into a [`ModelManifest`] + [`FileSpec`] list.
///
/// `is_layer_per_file` selects the naming strategy: "v0.2" packaging
/// names every file via its `org.cncf.model.filepath` annotation; the
/// legacy "v0.1" packaging has exactly one layer per format and names
/// them positionally (`model.gguf`, or `model-%05d-of-%05d.gguf` for
/// shards), matching `docker/model-runner`'s `unpackLegacy`.
fn build_plan(
    repo: &str,
    reference: &str,
    config: &DockerConfig,
    layers: &[Descriptor],
    is_layer_per_file: bool,
    token: Option<String>,
    blob_url: impl Fn(&str) -> Result<Url>,
) -> Result<Plan> {
    let mut gguf_layers: Vec<&Descriptor> = Vec::new();
    let mut mmproj_layers: Vec<&Descriptor> = Vec::new();
    let mut chat_template_layers: Vec<&Descriptor> = Vec::new();
    let mut license_layers: Vec<&Descriptor> = Vec::new();
    let mut model_file_layers: Vec<&Descriptor> = Vec::new();
    let mut has_unsupported_weights = false;

    for l in layers {
        match l.media_type.as_str() {
            media::GGUF => gguf_layers.push(l),
            media::MMPROJ => mmproj_layers.push(l),
            media::CHAT_TEMPLATE => chat_template_layers.push(l),
            media::LICENSE => license_layers.push(l),
            media::MODEL_FILE => model_file_layers.push(l),
            media::SAFETENSORS | media::DDUF => has_unsupported_weights = true,
            _ => {
                crate::logging::warn(&format!(
                    "docker hub: skipping layer with unhandled media type {:?} for {repo}",
                    l.media_type
                ));
            }
        }
    }

    if gguf_layers.is_empty() {
        return Err(Error::Hub(format!(
            "docker hub model {repo}:{reference} has no GGUF layer{}; only GGUF-format models are supported",
            if has_unsupported_weights {
                " (it publishes safetensors/dduf weights instead)"
            } else {
                ""
            }
        )));
    }

    let mut files: Vec<FileSpec> = Vec::new();
    let mut push = |name: String, desc: &Descriptor| -> Result<()> {
        files.push(FileSpec {
            name,
            size: desc.size,
            bytes: BytesSource::Http {
                url: blob_url(&desc.digest)?,
                auth: token.clone(),
            },
        });
        Ok(())
    };

    let layer_name = |desc: &Descriptor, fallback: String| -> String {
        if is_layer_per_file {
            if let Some(fp) = desc.annotations.get(ANNOTATION_FILEPATH) {
                if !fp.is_empty() {
                    return basename(fp);
                }
            }
        }
        fallback
    };

    // GGUF weight(s): single file -> "model.gguf"; multiple -> numbered
    // shards, matching the Go CLI's `unpackGGUFs` naming exactly so a
    // cache populated by either agent is interchangeable.
    let mut model_file: HashMap<String, ModelFileInfo> = HashMap::new();
    let mut extra_files: Vec<ModelFileInfo> = Vec::new();
    let n = gguf_layers.len();
    let mut entrypoint: Option<ModelFileInfo> = None;
    for (i, desc) in gguf_layers.iter().enumerate() {
        let fallback = if n == 1 {
            "model.gguf".to_string()
        } else {
            format!("model-{:05}-of-{:05}.gguf", i + 1, n)
        };
        let name = layer_name(desc, fallback);
        push(name.clone(), desc)?;
        let info = ModelFileInfo {
            name,
            downloaded: true,
            size: desc.size as i64,
        };
        if i == 0 {
            entrypoint = Some(info);
        } else {
            extra_files.push(info);
        }
    }
    let quant = if !config.quantization.is_empty() {
        config.quantization.to_ascii_uppercase()
    } else if let Some(entry) = &entrypoint {
        extract_quant(&entry.name).unwrap_or_else(|| "DEFAULT".to_string())
    } else {
        "DEFAULT".to_string()
    };
    model_file.insert(quant, entrypoint.expect("gguf_layers is non-empty"));

    let mmproj_file = if let Some(desc) = mmproj_layers.first() {
        let name = layer_name(desc, "model.mmproj".to_string());
        push(name.clone(), desc)?;
        ModelFileInfo {
            name,
            downloaded: true,
            size: desc.size as i64,
        }
    } else {
        ModelFileInfo::default()
    };

    for desc in &chat_template_layers[..chat_template_layers.len().min(1)] {
        let name = layer_name(desc, "template.jinja".to_string());
        push(name.clone(), desc)?;
        extra_files.push(ModelFileInfo {
            name,
            downloaded: true,
            size: desc.size as i64,
        });
    }
    for desc in &license_layers[..license_layers.len().min(1)] {
        let name = layer_name(desc, "LICENSE".to_string());
        push(name.clone(), desc)?;
        extra_files.push(ModelFileInfo {
            name,
            downloaded: true,
            size: desc.size as i64,
        });
    }
    for desc in &model_file_layers {
        let short = desc.digest.rsplit(':').next().unwrap_or(&desc.digest);
        let fallback = format!("file-{}", &short[..short.len().min(12)]);
        let name = layer_name(desc, fallback);
        push(name.clone(), desc)?;
        extra_files.push(ModelFileInfo {
            name,
            downloaded: true,
            size: desc.size as i64,
        });
    }

    let model_type = if mmproj_file.downloaded {
        ModelType::Vlm
    } else {
        ModelType::Llm
    };
    let model_name = repo.rsplit('/').next().unwrap_or(repo).to_string();

    let manifest = ModelManifest {
        name: repo.to_string(),
        model_name,
        model_type,
        plugin_id: "llama_cpp".to_string(),
        precision: config.quantization.clone(),
        qairt_version: String::new(),
        model_file,
        mmproj_file,
        tokenizer_file: ModelFileInfo::default(),
        extra_files,
    };

    Ok(Plan { manifest, files })
}

fn basename(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::{HttpTransport, ReqwestTransport, TransportConfig};
    use std::time::Duration;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn fast_transport() -> Arc<dyn HttpTransport> {
        Arc::new(
            ReqwestTransport::with_config(TransportConfig {
                connect_timeout: Some(Duration::from_secs(2)),
                read_timeout: Some(Duration::from_secs(5)),
                retries: Some(0),
                retry_backoff: Some(Duration::from_millis(10)),
                proxy_override: None,
            })
            .unwrap(),
        )
    }

    fn descriptor(media_type: &str, digest: &str, size: u64) -> serde_json::Value {
        serde_json::json!({
            "mediaType": media_type,
            "digest": digest,
            "size": size,
        })
    }

    async fn mount_auth(server: &MockServer) {
        Mock::given(method("GET"))
            .and(path("/token"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"token": "tok"})),
            )
            .mount(server)
            .await;
    }

    async fn mount_blob(server: &MockServer, repo: &str, digest: &str, body: &str) {
        Mock::given(method("HEAD"))
            .and(path(format!("/v2/{repo}/blobs/{digest}")))
            .respond_with(
                ResponseTemplate::new(200)
                    .append_header("Content-Length", body.len().to_string())
                    .append_header("Accept-Ranges", "bytes"),
            )
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path(format!("/v2/{repo}/blobs/{digest}")))
            .respond_with(ResponseTemplate::new(206).set_body_bytes(body.as_bytes()))
            .mount(server)
            .await;
    }

    fn cfg_for(server: &MockServer) -> DockerHubConfig {
        DockerHubConfig {
            registry_endpoint: server.uri(),
            auth_endpoint: format!("{}/token", server.uri()),
            service: "registry.docker.io".to_string(),
        }
    }

    #[tokio::test]
    async fn plan_single_gguf_v01_layout() {
        let server = MockServer::start().await;
        mount_auth(&server).await;

        let config_body = serde_json::json!({
            "config": {"format": "gguf", "quantization": "Q4_K_M", "architecture": "gemma3"}
        })
        .to_string();
        mount_blob(&server, "ai/gemma3", "sha256:cfg", &config_body).await;

        let weight_body = "gguf-bytes";
        mount_blob(&server, "ai/gemma3", "sha256:weight", weight_body).await;

        let manifest = serde_json::json!({
            "schemaVersion": 2,
            "mediaType": media::OCI_MANIFEST,
            "config": descriptor(media::CONFIG_V01, "sha256:cfg", config_body.len() as u64),
            "layers": [descriptor(media::GGUF, "sha256:weight", weight_body.len() as u64)],
        });
        Mock::given(method("GET"))
            .and(path("/v2/ai/gemma3/manifests/latest"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&manifest))
            .mount(&server)
            .await;

        let src = DockerHubSource::with_config(
            "ai/gemma3".to_string(),
            "latest".to_string(),
            cfg_for(&server),
            fast_transport(),
        )
        .unwrap();
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.model_type, ModelType::Llm);
        assert_eq!(plan.manifest.model_name, "gemma3");
        assert_eq!(
            plan.manifest
                .model_file
                .get("Q4_K_M")
                .map(|f| f.name.as_str()),
            Some("model.gguf")
        );
        assert_eq!(plan.files.len(), 1);
        assert_eq!(plan.files[0].name, "model.gguf");
    }

    #[tokio::test]
    async fn plan_detects_vlm_via_mmproj_layer() {
        let server = MockServer::start().await;
        mount_auth(&server).await;

        let config_body = serde_json::json!({"config": {"quantization": "F16"}}).to_string();
        mount_blob(&server, "ai/gemma3-vl", "sha256:cfg", &config_body).await;
        mount_blob(&server, "ai/gemma3-vl", "sha256:weight", "gguf-bytes").await;
        mount_blob(&server, "ai/gemma3-vl", "sha256:mmproj", "mmproj-bytes").await;

        let manifest = serde_json::json!({
            "schemaVersion": 2,
            "mediaType": media::OCI_MANIFEST,
            "config": descriptor(media::CONFIG_V01, "sha256:cfg", config_body.len() as u64),
            "layers": [
                descriptor(media::GGUF, "sha256:weight", 10),
                descriptor(media::MMPROJ, "sha256:mmproj", 5),
            ],
        });
        Mock::given(method("GET"))
            .and(path("/v2/ai/gemma3-vl/manifests/latest"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&manifest))
            .mount(&server)
            .await;

        let src = DockerHubSource::with_config(
            "ai/gemma3-vl".to_string(),
            String::new(),
            cfg_for(&server),
            fast_transport(),
        )
        .unwrap();
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.model_type, ModelType::Vlm);
        assert_eq!(plan.manifest.mmproj_file.name, "model.mmproj");
        assert_eq!(plan.files.len(), 2);
    }

    #[tokio::test]
    async fn plan_shards_multiple_gguf_layers() {
        let server = MockServer::start().await;
        mount_auth(&server).await;

        let config_body = serde_json::json!({"config": {"quantization": "Q8_0"}}).to_string();
        mount_blob(&server, "ai/big", "sha256:cfg", &config_body).await;
        mount_blob(&server, "ai/big", "sha256:w1", "aaa").await;
        mount_blob(&server, "ai/big", "sha256:w2", "bbb").await;

        let manifest = serde_json::json!({
            "schemaVersion": 2,
            "mediaType": media::OCI_MANIFEST,
            "config": descriptor(media::CONFIG_V01, "sha256:cfg", config_body.len() as u64),
            "layers": [
                descriptor(media::GGUF, "sha256:w1", 3),
                descriptor(media::GGUF, "sha256:w2", 3),
            ],
        });
        Mock::given(method("GET"))
            .and(path("/v2/ai/big/manifests/latest"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&manifest))
            .mount(&server)
            .await;

        let src = DockerHubSource::with_config(
            "ai/big".to_string(),
            "latest".to_string(),
            cfg_for(&server),
            fast_transport(),
        )
        .unwrap();
        let plan = src.plan().await.unwrap();
        let entry = plan.manifest.model_file.get("Q8_0").unwrap();
        assert_eq!(entry.name, "model-00001-of-00002.gguf");
        assert_eq!(plan.manifest.extra_files.len(), 1);
        assert_eq!(
            plan.manifest.extra_files[0].name,
            "model-00002-of-00002.gguf"
        );
    }

    #[tokio::test]
    async fn plan_v02_uses_filepath_annotation() {
        let server = MockServer::start().await;
        mount_auth(&server).await;

        let config_body = serde_json::json!({"config": {"quantization": "Q4_0"}}).to_string();
        mount_blob(&server, "ai/nested", "sha256:cfg", &config_body).await;
        mount_blob(&server, "ai/nested", "sha256:weight", "gguf-bytes").await;

        let mut weight_desc = descriptor(media::GGUF, "sha256:weight", 10);
        weight_desc["annotations"] =
            serde_json::json!({ANNOTATION_FILEPATH: "weights/model-Q4_0.gguf"});

        let manifest = serde_json::json!({
            "schemaVersion": 2,
            "mediaType": media::OCI_MANIFEST,
            "config": descriptor(media::CONFIG_V02, "sha256:cfg", config_body.len() as u64),
            "layers": [weight_desc],
        });
        Mock::given(method("GET"))
            .and(path("/v2/ai/nested/manifests/latest"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&manifest))
            .mount(&server)
            .await;

        let src = DockerHubSource::with_config(
            "ai/nested".to_string(),
            "latest".to_string(),
            cfg_for(&server),
            fast_transport(),
        )
        .unwrap();
        let plan = src.plan().await.unwrap();
        // Annotation path is flattened to its basename — our on-disk layout
        // is flat, same as the AI Hub and HF sources.
        assert_eq!(
            plan.manifest.model_file.get("Q4_0").unwrap().name,
            "model-Q4_0.gguf"
        );
    }

    #[tokio::test]
    async fn plan_rejects_safetensors_only_model() {
        let server = MockServer::start().await;
        mount_auth(&server).await;

        let config_body = serde_json::json!({"config": {}}).to_string();
        mount_blob(&server, "ai/st-only", "sha256:cfg", &config_body).await;

        let manifest = serde_json::json!({
            "schemaVersion": 2,
            "mediaType": media::OCI_MANIFEST,
            "config": descriptor(media::CONFIG_V01, "sha256:cfg", config_body.len() as u64),
            "layers": [descriptor(media::SAFETENSORS, "sha256:weight", 10)],
        });
        Mock::given(method("GET"))
            .and(path("/v2/ai/st-only/manifests/latest"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&manifest))
            .mount(&server)
            .await;

        let src = DockerHubSource::with_config(
            "ai/st-only".to_string(),
            "latest".to_string(),
            cfg_for(&server),
            fast_transport(),
        )
        .unwrap();
        assert!(src.plan().await.is_err());
    }

    #[tokio::test]
    async fn plan_returns_hub_model_not_found_on_404() {
        let server = MockServer::start().await;
        mount_auth(&server).await;
        Mock::given(method("GET"))
            .and(path("/v2/ai/missing/manifests/latest"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let src = DockerHubSource::with_config(
            "ai/missing".to_string(),
            "latest".to_string(),
            cfg_for(&server),
            fast_transport(),
        )
        .unwrap();
        let err = src.plan().await.unwrap_err();
        assert!(matches!(err, Error::HubModelNotFound(_)));
    }

    #[test]
    fn empty_reference_defaults_to_latest() {
        let src =
            DockerHubSource::new("ai/gemma3".to_string(), String::new(), fast_transport()).unwrap();
        assert_eq!(src.reference, "latest");
    }
}
