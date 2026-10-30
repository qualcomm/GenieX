// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Docker Hub [`ModelSource`] over the Docker Registry HTTP API V2, as used
//! by `docker model pull`. Only `application/vnd.docker.ai.*` artifacts
//! (v0.1 and v0.2 packaging) are supported, not CNCF ModelPack.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client;
use serde::Deserialize;
use url::Url;

use crate::error::{parse_manifest, Error, Result};
use crate::manifest::{ModelFileInfo, ModelManifest, ModelType};
use crate::manifest_builder::extract_quant;
use crate::transport::{default_client, fetch_capped, HttpTransport};

use super::{basename, BytesSource, FileSpec, ModelSource, Plan};

pub const DEFAULT_REGISTRY_ENDPOINT: &str = "https://registry-1.docker.io";
pub const DEFAULT_AUTH_ENDPOINT: &str = "https://auth.docker.io/token";
pub const DEFAULT_AUTH_SERVICE: &str = "registry.docker.io";
const DEFAULT_REFERENCE: &str = "latest";
const MAX_MANIFEST_BYTES: usize = 4 * 1024 * 1024;
const MAX_CONFIG_BYTES: u64 = 4 * 1024 * 1024;
/// A Docker-format model index points straight at its manifest, never at another index.
const MAX_INDEX_REDIRECTS: u32 = 1;

mod media {
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

    pub const OCI_MANIFEST: &str = "application/vnd.oci.image.manifest.v1+json";
    pub const OCI_INDEX: &str = "application/vnd.oci.image.index.v1+json";
    pub const DOCKER_MANIFEST_V2: &str = "application/vnd.docker.distribution.manifest.v2+json";
    pub const DOCKER_MANIFEST_LIST: &str =
        "application/vnd.docker.distribution.manifest.list.v2+json";
}

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
    repo: String,
    reference: String,
    cfg: DockerHubConfig,
    transport: Arc<dyn HttpTransport>,
    /// Manifest and token calls need `Accept` headers [`HttpTransport`] lacks.
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
        let http = default_client()?;
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

    /// A failed exchange means "no token"; the manifest GET surfaces any real auth error.
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
        .map_err(|e| Error::invalid_url("docker manifest url", e))?;

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
        let mut parsed: RegistryManifest = parse_manifest("docker registry manifest", &bytes)?;
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
        .map_err(|e| Error::invalid_url("docker blob url", e))
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
        let config_bytes = fetch_capped(
            self.transport.as_ref(),
            &self.blob_url(&config_desc.digest)?,
            token.as_deref(),
            MAX_CONFIG_BYTES,
        )
        .await?;
        let config_file: DockerConfigFile = parse_manifest("docker model config", &config_bytes)?;

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

/// v0.2 names files by their filepath annotation; v0.1 names them
/// positionally, matching `docker/model-runner`'s `unpackLegacy`.
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
    let mut add_layer = |desc: &Descriptor, fallback: String| -> Result<ModelFileInfo> {
        let name = if is_layer_per_file {
            desc.annotations
                .get(ANNOTATION_FILEPATH)
                .filter(|fp| !fp.is_empty())
                .map(|fp| basename(fp).to_string())
                .unwrap_or(fallback)
        } else {
            fallback
        };
        files.push(FileSpec {
            name: name.clone(),
            size: desc.size,
            bytes: BytesSource::Http {
                url: blob_url(&desc.digest)?,
                auth: token.clone(),
            },
        });
        Ok(ModelFileInfo {
            name,
            downloaded: true,
            size: desc.size as i64,
        })
    };

    // Shard names must match the Go CLI's unpackGGUFs.
    let mut extra_files: Vec<ModelFileInfo> = Vec::new();
    let n = gguf_layers.len();
    let mut gguf_entries = gguf_layers.iter().enumerate().map(|(i, desc)| {
        let fallback = if n == 1 {
            "model.gguf".to_string()
        } else {
            format!("model-{:05}-of-{:05}.gguf", i + 1, n)
        };
        add_layer(desc, fallback)
    });
    let entrypoint = gguf_entries
        .next()
        .expect("gguf_layers is non-empty (checked above)")?;
    for rest in gguf_entries {
        extra_files.push(rest?);
    }

    let mmproj_file = if let Some(desc) = mmproj_layers.first() {
        add_layer(desc, "model.mmproj".to_string())?
    } else {
        ModelFileInfo::default()
    };

    if let Some(desc) = chat_template_layers.first() {
        extra_files.push(add_layer(desc, "template.jinja".to_string())?);
    }
    if let Some(desc) = license_layers.first() {
        extra_files.push(add_layer(desc, "LICENSE".to_string())?);
    }
    for desc in &model_file_layers {
        let short = desc.digest.rsplit(':').next().unwrap_or(&desc.digest);
        let fallback = format!("file-{}", &short[..short.len().min(12)]);
        extra_files.push(add_layer(desc, fallback)?);
    }

    let quant = if !config.quantization.is_empty() {
        config.quantization.to_ascii_uppercase()
    } else {
        extract_quant(&entrypoint.name).unwrap_or_else(|| "DEFAULT".to_string())
    };
    let mut model_file: HashMap<String, ModelFileInfo> = HashMap::new();
    model_file.insert(quant, entrypoint);

    let model_type = if mmproj_file.downloaded {
        ModelType::Vlm
    } else {
        ModelType::Llm
    };
    let manifest = ModelManifest {
        name: repo.to_string(),
        model_name: basename(repo).to_string(),
        model_type,
        plugin_id: "llama_cpp".to_string(),
        precision: config.quantization.clone(),
        model_file,
        mmproj_file,
        tokenizer_file: ModelFileInfo::default(),
        extra_files,
    };

    Ok(Plan { manifest, files })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{fast_transport, mount_ranged};
    use serde_json::{json, Value};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn layer(media_type: &str, digest: &str) -> Value {
        json!({"mediaType": media_type, "digest": digest, "size": 10})
    }

    async fn source_for(server: &MockServer, repo: &str) -> DockerHubSource {
        Mock::given(method("GET"))
            .and(path("/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"token": "tok"})))
            .mount(server)
            .await;
        let cfg = DockerHubConfig {
            registry_endpoint: server.uri(),
            auth_endpoint: format!("{}/token", server.uri()),
            service: "registry.docker.io".to_string(),
        };
        DockerHubSource::with_config(repo.to_string(), String::new(), cfg, fast_transport())
            .unwrap()
    }

    async fn plan_for(
        name: &str,
        config_media: &str,
        config: Value,
        layers: Vec<Value>,
    ) -> Result<Plan> {
        let server = MockServer::start().await;
        let repo = format!("ai/{name}");
        let config = json!({ "config": config }).to_string();
        mount_ranged(
            &server,
            &format!("/v2/{repo}/blobs/sha256:cfg"),
            config.clone(),
        )
        .await;
        let manifest = json!({
            "schemaVersion": 2,
            "mediaType": media::OCI_MANIFEST,
            "config": {"mediaType": config_media, "digest": "sha256:cfg", "size": config.len()},
            "layers": layers,
        });
        Mock::given(method("GET"))
            .and(path(format!("/v2/{repo}/manifests/latest")))
            .respond_with(ResponseTemplate::new(200).set_body_json(&manifest))
            .mount(&server)
            .await;
        source_for(&server, &repo).await.plan().await
    }

    fn entry(plan: &Plan, quant: &str) -> String {
        plan.manifest.model_file[quant].name.clone()
    }

    #[tokio::test]
    async fn plan_single_gguf_v01_layout() {
        let config = json!({"format": "gguf", "quantization": "Q4_K_M", "architecture": "gemma3"});
        let plan = plan_for(
            "gemma3",
            media::CONFIG_V01,
            config,
            vec![layer(media::GGUF, "sha256:w")],
        )
        .await
        .unwrap();
        assert_eq!(plan.manifest.model_type, ModelType::Llm);
        assert_eq!(plan.manifest.model_name, "gemma3");
        assert_eq!(entry(&plan, "Q4_K_M"), "model.gguf");
        let names: Vec<&str> = plan.files.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["model.gguf"]);
    }

    #[tokio::test]
    async fn plan_detects_vlm_via_mmproj_layer() {
        let layers = vec![
            layer(media::GGUF, "sha256:w"),
            layer(media::MMPROJ, "sha256:m"),
        ];
        let plan = plan_for(
            "gemma3-vl",
            media::CONFIG_V01,
            json!({"quantization": "F16"}),
            layers,
        )
        .await
        .unwrap();
        assert_eq!(plan.manifest.model_type, ModelType::Vlm);
        assert_eq!(plan.manifest.mmproj_file.name, "model.mmproj");
        assert_eq!(plan.files.len(), 2);
    }

    #[tokio::test]
    async fn plan_shards_multiple_gguf_layers() {
        let layers = vec![
            layer(media::GGUF, "sha256:w1"),
            layer(media::GGUF, "sha256:w2"),
        ];
        let plan = plan_for(
            "big",
            media::CONFIG_V01,
            json!({"quantization": "Q8_0"}),
            layers,
        )
        .await
        .unwrap();
        assert_eq!(entry(&plan, "Q8_0"), "model-00001-of-00002.gguf");
        let extras: Vec<&str> = plan
            .manifest
            .extra_files
            .iter()
            .map(|f| f.name.as_str())
            .collect();
        assert_eq!(extras, ["model-00002-of-00002.gguf"]);
    }

    #[tokio::test]
    async fn plan_v02_flattens_filepath_annotation() {
        let mut weight = layer(media::GGUF, "sha256:w");
        weight["annotations"] = json!({ANNOTATION_FILEPATH: "weights/model-Q4_0.gguf"});
        let plan = plan_for(
            "nested",
            media::CONFIG_V02,
            json!({"quantization": "Q4_0"}),
            vec![weight],
        )
        .await
        .unwrap();
        assert_eq!(entry(&plan, "Q4_0"), "model-Q4_0.gguf");
    }

    #[tokio::test]
    async fn plan_rejects_safetensors_only_model() {
        let layers = vec![layer(media::SAFETENSORS, "sha256:w")];
        let err = plan_for("st-only", media::CONFIG_V01, json!({}), layers)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("safetensors"), "{err}");
    }

    #[tokio::test]
    async fn plan_returns_hub_model_not_found_on_404() {
        let server = MockServer::start().await;
        let err = source_for(&server, "ai/missing")
            .await
            .plan()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::HubModelNotFound(_)), "{err:?}");
    }

    #[test]
    fn empty_reference_defaults_to_latest() {
        let src =
            DockerHubSource::new("ai/gemma3".to_string(), String::new(), fast_transport()).unwrap();
        assert_eq!(src.reference, "latest");
    }
}
