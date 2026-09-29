// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! ModelScope [`ModelSource`].
//!
//! One call to `/api/v1/models/{repo}/repo/files` (with `Recursive=true`)
//! lists every file + size. If the repo ships a `geniex.json` we use it
//! verbatim; otherwise we synthesise via
//! [`crate::manifest_builder::infer_manifest_from_names`]. Each file is
//! fetched from `/api/v1/models/{repo}/resolve/{revision}/{path}`, which
//! redirects to the underlying Alibaba OSS object.
//!
//! Unlike HuggingFace, ModelScope does **not** answer `HEAD` on its API
//! or resolve endpoints (the former 404s, the latter drops
//! `Content-Length`), so the small JSON documents are fetched with plain
//! `GET` through a dedicated client; the weight bytes flow through the
//! executor's shared transport range machinery with sizes taken from the
//! listing, so no `HEAD` is ever needed there.

use std::collections::HashMap;

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use url::Url;

use crate::error::{Error, Result};
use crate::manifest::ModelManifest;
use crate::manifest_builder::{infer_manifest_from_names, ManifestHint};
use crate::transport::{build_tls_config, USER_AGENT};

use super::{BytesSource, FileSpec, ModelSource, Plan};

pub const DEFAULT_MODELSCOPE_ENDPOINT: &str = "https://modelscope.cn";
const DEFAULT_REVISION: &str = "master";
const MANIFEST_FILE: &str = "geniex.json";
const CONFIG_FILE: &str = "config.json";
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

pub struct ModelScopeSource {
    repo: String,
    endpoint: Url,
    revision: String,
    token: Option<String>,
    /// Plain-GET client for the small JSON documents (listing,
    /// `geniex.json`, `config.json`) — ModelScope doesn't support HEAD on
    /// these endpoints. Weight bytes flow through the executor's shared
    /// transport instead.
    http: Client,
    hint: ManifestHint,
}

impl ModelScopeSource {
    pub fn new(repo: String, token: Option<String>, hint: ManifestHint) -> Result<Self> {
        let endpoint = crate::config::StoreConfig::modelscope_endpoint();
        Self::with_endpoint(repo, &endpoint, token, hint)
    }

    /// Escape hatch for tests / alternate endpoints.
    pub fn with_endpoint(
        repo: String,
        endpoint: &str,
        token: Option<String>,
        hint: ManifestHint,
    ) -> Result<Self> {
        let endpoint = Url::parse(endpoint)
            .map_err(|e| Error::invalid_url(format!("ModelScope endpoint {endpoint}"), e))?;
        let http = Client::builder()
            .user_agent(USER_AGENT)
            .use_preconfigured_tls(build_tls_config()?)
            .build()
            .map_err(|e| Error::Http(format!("build modelscope client: {e}")))?;
        Ok(Self {
            repo,
            endpoint,
            revision: DEFAULT_REVISION.to_string(),
            token,
            http,
            hint,
        })
    }

    fn api_url(&self) -> Result<Url> {
        self.endpoint
            .join(&format!(
                "api/v1/models/{}/repo/files?Recursive=true&Revision={}",
                self.repo, self.revision
            ))
            .map_err(|e| Error::invalid_url(format!("ModelScope api for {}", self.repo), e))
    }

    fn file_url(&self, path: &str) -> Result<Url> {
        self.endpoint
            .join(&format!(
                "api/v1/models/{}/resolve/{}/{}",
                self.repo, self.revision, path
            ))
            .map_err(|e| Error::invalid_url(format!("ModelScope file {}/{path}", self.repo), e))
    }

    /// Fetch a small JSON document with a plain GET, capped at `limit`
    /// bytes. ModelScope rejects HEAD on its API endpoints, so the shared
    /// transport's HEAD-then-range flow can't be used here.
    async fn fetch_small(&self, url: &Url, limit: u64) -> Result<Vec<u8>> {
        let mut req = self.http.get(url.clone());
        if let Some(tok) = &self.token {
            req = req.bearer_auth(tok);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| Error::HttpTimeout(format!("GET {url}: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(Error::HttpStatus {
                url: url.to_string(),
                status: status.as_u16(),
            });
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| Error::Http(format!("read {url}: {e}")))?;
        if bytes.len() as u64 > limit {
            return Err(Error::Hub(format!(
                "file at {url} is {} bytes, exceeds {limit} byte cap",
                bytes.len()
            )));
        }
        Ok(bytes.to_vec())
    }
}

/// File-listing response from `/api/v1/models/{repo}/repo/files`.
/// The API capitalises the wrapper and per-file keys; the aliases keep
/// us resilient to servers that emit lower-cased JSON.
#[derive(Debug, Deserialize)]
struct ModelScopeFilesResponse {
    #[serde(rename = "Code", alias = "code", default)]
    code: i64,
    #[serde(rename = "Data", alias = "data", default)]
    data: Option<ModelScopeFilesData>,
    #[serde(rename = "Message", alias = "message", default)]
    message: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelScopeFilesData {
    #[serde(rename = "Files", alias = "files", default)]
    files: Vec<ModelScopeFile>,
}

#[derive(Debug, Deserialize)]
struct ModelScopeFile {
    /// Basename, e.g. `model-Q4_0.gguf`.
    #[serde(rename = "Name", alias = "name", default)]
    name: String,
    /// Full repo-relative path, used to build the resolve URL.
    #[serde(rename = "Path", alias = "path", default)]
    path: String,
    #[serde(rename = "Size", alias = "size", default)]
    size: Option<u64>,
    /// `"blob"` for files, `"tree"` for directories.
    #[serde(rename = "Type", alias = "type", default)]
    file_type: String,
}

impl ModelScopeFile {
    fn is_file(&self) -> bool {
        !self.file_type.eq_ignore_ascii_case("tree")
    }
}

#[async_trait]
impl ModelSource for ModelScopeSource {
    async fn plan(&self) -> Result<Plan> {
        let api_url = self.api_url()?;
        let body = self.fetch_small(&api_url, MAX_MANIFEST_BYTES).await?;
        let parsed: ModelScopeFilesResponse = serde_json::from_slice(&body)?;

        if parsed.code != 0 && parsed.code != 200 {
            if parsed.code == 404 {
                return Err(Error::HubModelNotFound(self.repo.clone()));
            }
            return Err(Error::Hub(format!(
                "ModelScope repo {} error {}: {}",
                self.repo,
                parsed.code,
                parsed.message.unwrap_or_default()
            )));
        }
        let files = parsed.data.map(|d| d.files).unwrap_or_default();

        // basename -> (full path, size) so downloads address the real path
        // while the manifest and on-disk layout stay flat like HF.
        let mut by_basename: HashMap<String, (String, Option<u64>)> = HashMap::new();
        for f in files {
            if !f.is_file() || f.name.is_empty() {
                continue;
            }
            let path = if f.path.is_empty() {
                f.name.clone()
            } else {
                f.path.clone()
            };
            by_basename.insert(f.name.clone(), (path, f.size));
        }

        let file_names: Vec<String> = by_basename
            .keys()
            .filter(|n| n.as_str() != MANIFEST_FILE)
            .cloned()
            .collect();
        let file_sizes: HashMap<String, i64> = by_basename
            .iter()
            .map(|(name, (_, size))| (name.clone(), size.map(|s| s as i64).unwrap_or(0)))
            .collect();

        // Fetch `config.json` when the listing advertises one. Used by the
        // modality classifier; fetch failure is non-fatal (we degrade to
        // the mmproj filename heuristic).
        let config_bytes: Option<Vec<u8>> = match by_basename.get(CONFIG_FILE) {
            Some((path, _)) => match self.file_url(path) {
                Ok(url) => self.fetch_small(&url, MAX_MANIFEST_BYTES).await.ok(),
                Err(_) => None,
            },
            None => None,
        };
        let mut infer_hint = self.hint.clone();
        infer_hint.config_json_bytes = config_bytes;

        let mut manifest: ModelManifest;
        if let Some((path, _)) = by_basename.get(MANIFEST_FILE) {
            match self.file_url(path) {
                Ok(url) => match self.fetch_small(&url, MAX_MANIFEST_BYTES).await {
                    Ok(bytes) => match serde_json::from_slice(&bytes) {
                        Ok(m) => manifest = m,
                        Err(e) => {
                            crate::logging::warn(&format!(
                                "modelscope geniex.json for {} failed to parse ({e}); inferring",
                                self.repo
                            ));
                            manifest = infer_manifest_from_names(
                                &self.repo,
                                &file_names,
                                &file_sizes,
                                infer_hint,
                            )?;
                        }
                    },
                    Err(_) => {
                        manifest = infer_manifest_from_names(
                            &self.repo,
                            &file_names,
                            &file_sizes,
                            infer_hint,
                        )?;
                    }
                },
                Err(_) => {
                    manifest = infer_manifest_from_names(
                        &self.repo,
                        &file_names,
                        &file_sizes,
                        infer_hint,
                    )?;
                }
            }
        } else {
            manifest = infer_manifest_from_names(&self.repo, &file_names, &file_sizes, infer_hint)?;
        }
        manifest.name = self.repo.clone();

        // Only materialise files the manifest actually uses; readmes and
        // unused quantization shards in the listing are left behind.
        let mut files: Vec<FileSpec> = Vec::new();
        let mut push = |name: &str, size: Option<u64>| -> Result<()> {
            if name.is_empty() {
                return Ok(());
            }
            let path = by_basename
                .get(name)
                .map(|(p, _)| p.clone())
                .unwrap_or_else(|| name.to_string());
            let url = self.file_url(&path)?;
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
    use crate::source::basename;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Register a GET handler for a resolve URL (used for `geniex.json`,
    /// `config.json`, and existence probes).
    async fn mount_get(server: &MockServer, rel_path: &str, status: u16, body: &str) {
        Mock::given(method("GET"))
            .and(path(rel_path))
            .respond_with(ResponseTemplate::new(status).set_body_bytes(body.as_bytes()))
            .mount(server)
            .await;
    }

    /// Register the recursive file-listing endpoint, which carries query
    /// params that `path` alone cannot match.
    async fn mount_listing(server: &MockServer, repo: &str, body: &str) {
        let listing_path = format!("/api/v1/models/{repo}/repo/files");
        Mock::given(method("GET"))
            .and(path(&listing_path))
            .and(query_param("Recursive", "true"))
            .and(query_param("Revision", "master"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.as_bytes()))
            .mount(server)
            .await;
    }

    fn src(server: &MockServer, repo: &str) -> ModelScopeSource {
        ModelScopeSource::with_endpoint(
            repo.to_string(),
            &server.uri(),
            None,
            ManifestHint::default(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn plan_synthesises_manifest_when_repo_lacks_one() {
        let server = MockServer::start().await;
        let listing = r#"{
          "Code": 200,
          "Data": {
            "Files": [
              {"Name": "README.md", "Path": "README.md", "Size": 123, "Type": "blob"},
              {"Name": "model-Q4_K_M.gguf", "Path": "model-Q4_K_M.gguf", "Size": 1024, "Type": "blob"}
            ]
          },
          "Message": "success"
        }"#;
        mount_listing(&server, "org/Tiny-GGUF", listing).await;
        // GET for geniex.json → 404 means "repo does not ship one".
        mount_get(
            &server,
            "/api/v1/models/org/Tiny-GGUF/resolve/master/geniex.json",
            404,
            "",
        )
        .await;

        let src = src(&server, "org/Tiny-GGUF");
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.name, "org/Tiny-GGUF");
        assert!(plan.manifest.model_file.contains_key("Q4_K_M"));
        assert_eq!(plan.files.len(), 1);
        assert_eq!(plan.files[0].name, "model-Q4_K_M.gguf");
        match &plan.files[0].bytes {
            BytesSource::Http { url, .. } => {
                assert!(url.path().ends_with("model-Q4_K_M.gguf"));
            }
            _ => panic!("ModelScope file should be BytesSource::Http"),
        }
    }

    #[tokio::test]
    async fn plan_uses_ships_manifest_when_present() {
        let server = MockServer::start().await;
        let listing = r#"{
          "Code": 200,
          "Data": {
            "Files": [
              {"Name": "geniex.json", "Path": "geniex.json", "Size": 256, "Type": "blob"},
              {"Name": "model-Q8_0.gguf", "Path": "model-Q8_0.gguf", "Size": 2048, "Type": "blob"}
            ]
          },
          "Message": "success"
        }"#;
        mount_listing(&server, "org/Shipped", listing).await;
        mount_get(
            &server,
            "/api/v1/models/org/Shipped/resolve/master/geniex.json",
            200,
            r#"{"Name":"org/Shipped","ModelName":"Shipped","ModelType":"llm","PluginId":"llama_cpp","ModelFile":{"Q8_0":{"Name":"model-Q8_0.gguf","Downloaded":true,"Size":2048}},"MMProjFile":null,"TokenizerFile":null,"ExtraFiles":null}"#,
        )
        .await;

        let src = src(&server, "org/Shipped");
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.name, "org/Shipped");
        assert_eq!(plan.manifest.plugin_id, "llama_cpp");
        assert!(plan.manifest.model_file.contains_key("Q8_0"));
        assert_eq!(plan.files.len(), 1);
    }

    #[tokio::test]
    async fn plan_skips_tree_entries_and_nested_paths() {
        let server = MockServer::start().await;
        let listing = r#"{
          "Code": 200,
          "Data": {
            "Files": [
              {"Name": "sub", "Path": "sub", "Size": 0, "Type": "tree"},
              {"Name": "model-Q4_0.gguf", "Path": "sub/model-Q4_0.gguf", "Size": 512, "Type": "blob"}
            ]
          },
          "Message": "success"
        }"#;
        mount_listing(&server, "org/Nested", listing).await;
        mount_get(
            &server,
            "/api/v1/models/org/Nested/resolve/master/geniex.json",
            404,
            "",
        )
        .await;

        let src = src(&server, "org/Nested");
        let plan = src.plan().await.unwrap();
        assert!(plan.manifest.model_file.contains_key("Q4_0"));
        assert_eq!(plan.files.len(), 1);
        // Download addresses the real nested path; on-disk name is flat.
        assert_eq!(plan.files[0].name, "model-Q4_0.gguf");
        match &plan.files[0].bytes {
            BytesSource::Http { url, .. } => {
                assert!(url.path().ends_with("sub/model-Q4_0.gguf"));
            }
            _ => panic!("ModelScope file should be BytesSource::Http"),
        }
    }

    #[tokio::test]
    async fn plan_returns_hub_model_not_found_on_404() {
        let server = MockServer::start().await;
        let listing = r#"{"Code": 404, "Data": null, "Message": "model not found"}"#;
        mount_listing(&server, "org/Missing", listing).await;

        let src = src(&server, "org/Missing");
        let err = src.plan().await.unwrap_err();
        assert!(matches!(err, Error::HubModelNotFound(_)));
    }

    #[tokio::test]
    async fn plan_classifies_vlm_via_config_json() {
        let server = MockServer::start().await;
        let listing = r#"{
          "Code": 200,
          "Data": {
            "Files": [
              {"Name": "config.json", "Path": "config.json", "Size": 128, "Type": "blob"},
              {"Name": "model-Q4_K_M.gguf", "Path": "model-Q4_K_M.gguf", "Size": 1024, "Type": "blob"}
            ]
          },
          "Message": "success"
        }"#;
        mount_listing(&server, "org/VLM", listing).await;
        mount_get(
            &server,
            "/api/v1/models/org/VLM/resolve/master/config.json",
            200,
            r#"{"architectures":["Qwen2_5_VLForConditionalGeneration"],"vision_config":{}}"#,
        )
        .await;
        mount_get(
            &server,
            "/api/v1/models/org/VLM/resolve/master/geniex.json",
            404,
            "",
        )
        .await;

        let src = src(&server, "org/VLM");
        let plan = src.plan().await.unwrap();
        assert_eq!(plan.manifest.model_type, crate::manifest::ModelType::Vlm);
    }

    #[test]
    fn basename_helper_flattens_nested_paths() {
        assert_eq!(basename("sub/model.gguf"), "model.gguf");
        assert_eq!(basename("model.gguf"), "model.gguf");
    }
}
