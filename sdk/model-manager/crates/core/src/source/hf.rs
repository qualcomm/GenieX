//! HuggingFace [`ModelSource`].
//!
//! One call to `/api/models/{repo}` gives siblings + sizes. If the
//! repo ships a `geniex.json` we use it verbatim; otherwise we
//! synthesise via [`crate::manifest_builder::infer_manifest_from_names`].

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use url::Url;

use crate::error::{Error, Result};
use crate::manifest::ModelManifest;
use crate::manifest_builder::{infer_manifest_from_names, ManifestHint};
use crate::transport::{HttpTransport, ReqwestTransport};

use super::{BytesSource, FileSpec, ModelSource, Plan};

pub const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
const MANIFEST_FILE: &str = "geniex.json";
const CONFIG_FILE: &str = "config.json";
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

pub struct HfSource {
    repo: String,
    endpoint: Url,
    token: Option<String>,
    transport: Arc<dyn HttpTransport>,
    hint: ManifestHint,
}

impl HfSource {
    pub fn new(repo: String, token: Option<String>, hint: ManifestHint) -> Result<Self> {
        let transport: Arc<dyn HttpTransport> = Arc::new(ReqwestTransport::new()?);
        Self::with_endpoint_and_transport(repo, DEFAULT_HF_ENDPOINT, token, transport, hint)
    }

    /// Escape hatch for tests / alternate endpoints / shared transports.
    pub fn with_endpoint_and_transport(
        repo: String,
        endpoint: &str,
        token: Option<String>,
        transport: Arc<dyn HttpTransport>,
        hint: ManifestHint,
    ) -> Result<Self> {
        let endpoint = Url::parse(endpoint)
            .map_err(|e| Error::Hub(format!("invalid HF endpoint {endpoint}: {e}")))?;
        Ok(Self {
            repo,
            endpoint,
            token,
            transport,
            hint,
        })
    }

    fn api_url(&self) -> Result<Url> {
        self.endpoint
            .join(&format!("api/models/{}?blobs=true", self.repo))
            .map_err(|e| Error::Hub(format!("join api url for {}: {e}", self.repo)))
    }

    fn file_url(&self, name: &str) -> Result<Url> {
        self.endpoint
            .join(&format!("{}/resolve/main/{name}", self.repo))
            .map_err(|e| Error::Hub(format!("join resolve url for {}/{name}: {e}", self.repo)))
    }

    async fn fetch_small(&self, url: &Url, limit: u64) -> Result<Vec<u8>> {
        let head = self.transport.head(url, self.token.as_deref()).await?;
        if head.size > limit {
            return Err(Error::Hub(format!(
                "file at {url} is {} bytes, exceeds {limit} byte cap",
                head.size
            )));
        }
        let mut buf: Vec<u8> = Vec::with_capacity(head.size as usize);
        self.transport
            .get_range(url, self.token.as_deref(), 0, head.size, &mut buf)
            .await?;
        Ok(buf)
    }
}

#[derive(Debug, Deserialize)]
struct ApiModelResponse {
    #[serde(default)]
    siblings: Vec<ApiSibling>,
}

#[derive(Debug, Deserialize)]
struct ApiSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

#[async_trait]
impl ModelSource for HfSource {
    async fn plan(&self) -> Result<Plan> {
        let api_url = self.api_url()?;
        let body = self.fetch_small(&api_url, MAX_MANIFEST_BYTES).await?;
        let parsed: ApiModelResponse = serde_json::from_slice(&body)?;

        let mut file_sizes: HashMap<String, i64> = HashMap::new();
        let mut file_names: Vec<String> = Vec::new();
        for s in parsed.siblings {
            if s.rfilename == MANIFEST_FILE {
                continue;
            }
            file_sizes.insert(s.rfilename.clone(), s.size.map(|n| n as i64).unwrap_or(0));
            file_names.push(s.rfilename);
        }

        // HEAD rather than GET so we don't pull a nonexistent manifest's
        // HTML 404 page into memory before failing JSON parse.
        let ships_manifest = {
            let url = self.file_url(MANIFEST_FILE)?;
            self.transport
                .head(&url, self.token.as_deref())
                .await
                .is_ok()
        };
        // Fetch `config.json` when siblings advertise one. Used by the
        // modality classifier; fetch failure is non-fatal (we degrade to
        // the mmproj filename heuristic).
        let config_bytes: Option<Vec<u8>> =
            if !ships_manifest && file_names.iter().any(|n| n == CONFIG_FILE) {
                match self.file_url(CONFIG_FILE) {
                    Ok(url) => self.fetch_small(&url, MAX_MANIFEST_BYTES).await.ok(),
                    Err(_) => None,
                }
            } else {
                None
            };
        let mut infer_hint = self.hint.clone();
        infer_hint.config_json_bytes = config_bytes;

        let mut manifest: ModelManifest = if ships_manifest {
            let url = self.file_url(MANIFEST_FILE)?;
            match self.fetch_small(&url, MAX_MANIFEST_BYTES).await {
                Ok(bytes) => serde_json::from_slice(&bytes).map_err(Error::Json)?,
                Err(_) => {
                    infer_manifest_from_names(&self.repo, &file_names, &file_sizes, infer_hint)?
                }
            }
        } else {
            infer_manifest_from_names(&self.repo, &file_names, &file_sizes, infer_hint)?
        };
        manifest.name = self.repo.clone();

        // Only materialise files the manifest actually uses; readmes and
        // unused quantization shards in `siblings` are left behind.
        let mut files: Vec<FileSpec> = Vec::new();
        let mut push = |name: &str, size: Option<u64>| -> Result<()> {
            if name.is_empty() {
                return Ok(());
            }
            let url = self.file_url(name)?;
            files.push(FileSpec {
                name: name.to_string(),
                size: size.unwrap_or(0),
                bytes: BytesSource::Http {
                    url,
                    auth: self.token.clone(),
                },
            });
            Ok(())
        };

        for f in manifest.model_file.values() {
            if f.downloaded {
                push(&f.name, size_for(&file_sizes, &f.name))?;
            }
        }
        if manifest.mmproj_file.downloaded {
            push(
                &manifest.mmproj_file.name,
                size_for(&file_sizes, &manifest.mmproj_file.name),
            )?;
        }
        if manifest.tokenizer_file.downloaded {
            push(
                &manifest.tokenizer_file.name,
                size_for(&file_sizes, &manifest.tokenizer_file.name),
            )?;
        }
        for f in &manifest.extra_files {
            if f.downloaded {
                push(&f.name, size_for(&file_sizes, &f.name))?;
            }
        }

        Ok(Plan { manifest, files })
    }
}

fn size_for(map: &HashMap<String, i64>, name: &str) -> Option<u64> {
    let v = map.get(name).copied().unwrap_or(-1);
    if v < 0 {
        None
    } else {
        Some(v as u64)
    }
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

    /// Register `HEAD` + `GET` handlers for a sibling URL so
    /// `fetch_small` can pull its bytes.
    async fn mount_file(server: &MockServer, rel_path: &str, body: &str) {
        Mock::given(method("HEAD"))
            .and(path(rel_path))
            .respond_with(
                ResponseTemplate::new(200)
                    .append_header("Content-Length", body.len().to_string())
                    .append_header("Accept-Ranges", "bytes"),
            )
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path(rel_path))
            .respond_with(ResponseTemplate::new(206).set_body_bytes(body.as_bytes()))
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn plan_classifies_vlm_via_config_json() {
        // #17 — HF repo with safetensors + config.json carrying
        // `vision_config` is classified as VLM without any mmproj file.
        let server = MockServer::start().await;
        // infer_manifest_from_names requires at least one recognised
        // weight file; include a GGUF alongside safetensors so inference
        // succeeds and we can observe the modality verdict.
        let api_body = r#"{
          "siblings": [
            {"rfilename": "config.json", "size": 128},
            {"rfilename": "model-Q4_K_M.gguf", "size": 1024}
          ]
        }"#;
        let config_body =
            r#"{"architectures":["Qwen2_5_VLForConditionalGeneration"],"vision_config":{}}"#;
        mount_file(&server, "/api/models/tests/VLM", api_body).await;
        mount_file(&server, "/tests/VLM/resolve/main/config.json", config_body).await;
        Mock::given(method("HEAD"))
            .and(path("/tests/VLM/resolve/main/geniex.json"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let src = HfSource::with_endpoint_and_transport(
            "tests/VLM".to_string(),
            &server.uri(),
            None,
            fast_transport(),
            ManifestHint::default(),
        )
        .unwrap();
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.model_type, crate::manifest::ModelType::Vlm);
    }

    #[tokio::test]
    async fn plan_classifies_llm_when_config_overrides_stray_mmproj() {
        // #18 — config says LlamaForCausalLM; a stray `mmproj-x.gguf`
        // sibling must NOT promote this to VLM.
        let server = MockServer::start().await;
        let api_body = r#"{
          "siblings": [
            {"rfilename": "config.json", "size": 64},
            {"rfilename": "model-Q4_K_M.gguf", "size": 1024},
            {"rfilename": "model-Q8_0.gguf", "size": 2048},
            {"rfilename": "mmproj-x.gguf", "size": 128}
          ]
        }"#;
        let config_body = r#"{"architectures":["LlamaForCausalLM"]}"#;
        mount_file(&server, "/api/models/tests/LLM", api_body).await;
        mount_file(&server, "/tests/LLM/resolve/main/config.json", config_body).await;
        Mock::given(method("HEAD"))
            .and(path("/tests/LLM/resolve/main/geniex.json"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let src = HfSource::with_endpoint_and_transport(
            "tests/LLM".to_string(),
            &server.uri(),
            None,
            fast_transport(),
            ManifestHint::default(),
        )
        .unwrap();
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.model_type, crate::manifest::ModelType::Llm);
    }

    #[tokio::test]
    async fn plan_synthesises_manifest_when_repo_lacks_one() {
        let server = MockServer::start().await;
        let body = r#"{
          "siblings": [
            {"rfilename": "model-Q4_K_M.gguf", "size": 1024}
          ]
        }"#;
        Mock::given(method("HEAD"))
            .and(path("/api/models/tests/Tiny-GGUF"))
            .respond_with(
                ResponseTemplate::new(200)
                    .append_header("Content-Length", body.len().to_string())
                    .append_header("Accept-Ranges", "bytes"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/models/tests/Tiny-GGUF"))
            .respond_with(ResponseTemplate::new(206).set_body_bytes(body.as_bytes()))
            .mount(&server)
            .await;
        // HEAD for geniex.json → 404 means "repo does not ship one".
        Mock::given(method("HEAD"))
            .and(path("/tests/Tiny-GGUF/resolve/main/geniex.json"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let src = HfSource::with_endpoint_and_transport(
            "tests/Tiny-GGUF".to_string(),
            &server.uri(),
            None,
            fast_transport(),
            ManifestHint::default(),
        )
        .unwrap();
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.name, "tests/Tiny-GGUF");
        assert!(plan.manifest.model_file.contains_key("Q4_K_M"));
        assert_eq!(plan.files.len(), 1);
        assert_eq!(plan.files[0].name, "model-Q4_K_M.gguf");
        match &plan.files[0].bytes {
            BytesSource::Http { url, .. } => {
                assert!(url.path().ends_with("model-Q4_K_M.gguf"));
            }
            _ => panic!("HF file should be BytesSource::Http"),
        }
    }
}
