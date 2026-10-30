// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

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
use crate::gguf;
use crate::manifest::ModelManifest;
use crate::manifest_builder::{infer_manifest_from_names, untagged_gguf_names, ManifestHint};
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
        let endpoint = crate::config::StoreConfig::hf_endpoint();
        Self::with_endpoint_and_transport(repo, &endpoint, token, transport, hint)
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

    /// Recover the quant of GGUFs whose filename doesn't carry one. Usually
    /// zero extra requests; a failure leaves the file to the `DEFAULT` bucket.
    async fn probe_header_quants(
        &self,
        file_names: &[String],
        sizes: &HashMap<String, i64>,
    ) -> HashMap<String, String> {
        let mut out = HashMap::new();
        for name in untagged_gguf_names(file_names) {
            let Ok(url) = self.file_url(name) else {
                continue;
            };
            let size = sizes.get(name).copied().unwrap_or(0).max(0) as u64;
            let quant =
                gguf::probe_quant(self.transport.as_ref(), &url, self.token.as_deref(), size).await;
            if let Some(q) = quant {
                out.insert(name.clone(), q);
            }
        }
        out
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
        if !ships_manifest {
            infer_hint.header_quants = self.probe_header_quants(&file_names, &file_sizes).await;
        }

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
    use crate::manifest::ModelType;
    use crate::test_support::{fast_transport, gguf, mount_ranged, mount_status};
    use wiremock::MockServer;

    /// Serve an HF repo listing `siblings` (name, bytes) with no `geniex.json`.
    async fn plan_for(repo: &str, siblings: &[(&str, &[u8])]) -> Result<Plan> {
        let server = MockServer::start().await;
        let listing: Vec<_> = siblings
            .iter()
            .map(|(n, b)| serde_json::json!({"rfilename": n, "size": b.len()}))
            .collect();
        let api = serde_json::json!({ "siblings": listing }).to_string();
        mount_ranged(&server, &format!("/api/models/{repo}"), api).await;
        mount_status(
            &server,
            "HEAD",
            &format!("/{repo}/resolve/main/geniex.json"),
            404,
        )
        .await;
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
