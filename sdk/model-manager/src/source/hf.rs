// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! HuggingFace [`ModelSource`].

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use url::Url;

use crate::error::{Error, Result};
use crate::manifest::gguf;
use crate::manifest::infer::ManifestHint;
use crate::transport::{fetch_capped, HttpTransport};

use super::{files_for_manifest, resolve_manifest, BytesSource, ModelSource, Plan, RepoFiles};

pub const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

pub struct HfSource {
    repo: String,
    endpoint: Url,
    token: Option<String>,
    transport: Arc<dyn HttpTransport>,
    hint: ManifestHint,
}

impl HfSource {
    pub fn with_endpoint_and_transport(
        repo: String,
        endpoint: &str,
        token: Option<String>,
        transport: Arc<dyn HttpTransport>,
        hint: ManifestHint,
    ) -> Result<Self> {
        let endpoint = Url::parse(endpoint)
            .map_err(|e| Error::invalid_url(format!("HF endpoint {endpoint}"), e))?;
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
            .map_err(|e| Error::invalid_url(format!("HF api for {}", self.repo), e))
    }

    fn file_url(&self, name: &str) -> Result<Url> {
        self.endpoint
            .join(&format!("{}/resolve/main/{name}", self.repo))
            .map_err(|e| Error::invalid_url(format!("HF file {}/{name}", self.repo), e))
    }

    async fn fetch_small(&self, url: &Url) -> Result<Vec<u8>> {
        fetch_capped(
            self.transport.as_ref(),
            url,
            self.token.as_deref(),
            MAX_MANIFEST_BYTES,
        )
        .await
    }
}

#[async_trait]
impl RepoFiles for HfSource {
    async fn read(&self, name: &str) -> Result<Vec<u8>> {
        self.fetch_small(&self.file_url(name)?).await
    }

    async fn header_quant(&self, name: &str, size: u64) -> Option<String> {
        let url = self.file_url(name).ok()?;
        gguf::probe_quant(self.transport.as_ref(), &url, self.token.as_deref(), size).await
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
        let body = self.fetch_small(&self.api_url()?).await?;
        let parsed: ApiModelResponse = serde_json::from_slice(&body)?;
        let listing: Vec<(String, i64)> = parsed
            .siblings
            .into_iter()
            .map(|s| (s.rfilename, s.size.unwrap_or(0) as i64))
            .collect();

        let manifest = resolve_manifest(self, &self.repo, &listing, &self.hint).await?;
        let files = files_for_manifest(&manifest, |name| {
            let size = listing
                .iter()
                .find(|(n, _)| n == name)
                .map_or(0, |(_, s)| *s as u64);
            let url = self.file_url(name)?;
            let auth = self.token.clone();
            Ok((size, BytesSource::Http { url, auth }))
        })?;
        Ok(Plan { manifest, files })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::ModelType;
    use crate::test_support::{fast_transport, gguf, mount_ranged};
    use wiremock::MockServer;

    async fn plan_for(repo: &str, siblings: &[(&str, &[u8])]) -> Result<Plan> {
        let server = MockServer::start().await;
        let listing: Vec<_> = siblings
            .iter()
            .map(|(n, b)| serde_json::json!({"rfilename": n, "size": b.len()}))
            .collect();
        let api = serde_json::json!({ "siblings": listing }).to_string();
        mount_ranged(&server, &format!("/api/models/{repo}"), api).await;
        for (name, body) in siblings {
            mount_ranged(
                &server,
                &format!("/{repo}/resolve/main/{name}"),
                body.to_vec(),
            )
            .await;
        }
        HfSource::with_endpoint_and_transport(
            repo.to_string(),
            &server.uri(),
            None,
            fast_transport(),
            ManifestHint::default(),
        )?
        .plan()
        .await
    }

    #[tokio::test]
    async fn plan_synthesises_manifest_when_repo_lacks_one() {
        let plan = plan_for("tests/Tiny-GGUF", &[("model-Q4_K_M.gguf", &[0; 1024])])
            .await
            .unwrap();
        assert_eq!(plan.manifest.name, "tests/Tiny-GGUF");
        assert!(plan.manifest.model_file.contains_key("Q4_K_M"));
        assert_eq!(plan.files.len(), 1);
        assert_eq!(plan.files[0].name, "model-Q4_K_M.gguf");
        assert!(matches!(
            &plan.files[0].bytes,
            BytesSource::Http { url, .. } if url.path().ends_with("model-Q4_K_M.gguf")
        ));
    }

    #[tokio::test]
    async fn plan_rejects_safetensors_only_repo_with_a_format_hint() {
        let repo = "openai/gpt-oss-safeguard-20b";
        let err = plan_for(
            repo,
            &[
                ("config.json", br#"{"architectures":["GptOssForCausalLM"]}"#),
                ("model-00001-of-00002.safetensors", &[0; 1024]),
                ("tokenizer.json", b"{}"),
            ],
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::ManifestInferenceFailed(_)), "{err:?}");
        let msg = err.to_string();
        for want in [repo, "GGUF", "QAIRT"] {
            assert!(msg.contains(want), "missing {want:?}: {msg}");
        }
    }

    #[tokio::test]
    async fn plan_classifies_modality_from_config_json() {
        let vlm_config =
            br#"{"architectures":["Qwen2_5_VLForConditionalGeneration"],"vision_config":{}}"#;
        let plan = plan_for(
            "tests/VLM",
            &[
                ("config.json", vlm_config),
                ("model-Q4_K_M.gguf", &[0; 1024]),
            ],
        )
        .await
        .unwrap();
        assert_eq!(plan.manifest.model_type, ModelType::Vlm);
    }

    #[tokio::test]
    async fn plan_uses_shipped_manifest_and_infers_past_a_broken_one() {
        let shipped: &[u8] = br#"{"Name":"x","ModelName":"Shipped","ModelType":"llm","PluginId":"llama_cpp","ModelFile":{"Q8_0":{"Name":"model-Q8_0.gguf","Downloaded":true,"Size":1}}}"#;
        for (manifest, want) in [(shipped, "Shipped"), (b"not json", "Repo")] {
            let plan = plan_for(
                "org/Repo",
                &[("geniex.json", manifest), ("model-Q8_0.gguf", &[0; 16])],
            )
            .await
            .unwrap();
            assert_eq!(plan.manifest.model_name, want);
            assert_eq!(plan.manifest.name, "org/Repo");
            assert!(plan.manifest.model_file.contains_key("Q8_0"));
        }
    }

    #[tokio::test]
    async fn plan_reads_quant_from_the_header_when_the_name_lacks_one() {
        let head = gguf::head(15, 0);
        let plan = plan_for("tests/Untagged", &[("REAP-48-v2.gguf", &head)])
            .await
            .unwrap();
        assert_eq!(
            plan.manifest
                .model_file
                .get("Q4_K_M")
                .map(|f| f.name.as_str()),
            Some("REAP-48-v2.gguf")
        );
    }
}
