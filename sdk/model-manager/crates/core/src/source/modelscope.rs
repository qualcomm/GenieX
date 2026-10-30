// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! ModelScope [`ModelSource`].
//!
//! ModelScope answers neither `HEAD` on its API (404) nor a usable one on
//! resolve (no `Content-Length`), so small documents use a plain GET and
//! weight sizes come from the listing.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use url::Url;

use crate::error::{Error, Result};
use crate::gguf;
use crate::manifest_builder::ManifestHint;
use crate::transport::{default_client, HttpTransport};

use super::{files_for_manifest, resolve_manifest, BytesSource, ModelSource, Plan, RepoFiles};

pub const DEFAULT_MODELSCOPE_ENDPOINT: &str = "https://modelscope.cn";
const DEFAULT_REVISION: &str = "master";
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

pub struct ModelScopeSource {
    repo: String,
    endpoint: Url,
    revision: String,
    token: Option<String>,
    http: Client,
    transport: Arc<dyn HttpTransport>,
    hint: ManifestHint,
    paths: OnceLock<HashMap<String, String>>,
}

impl ModelScopeSource {
    pub fn new(
        repo: String,
        token: Option<String>,
        transport: Arc<dyn HttpTransport>,
        hint: ManifestHint,
    ) -> Result<Self> {
        let endpoint = crate::config::StoreConfig::modelscope_endpoint();
        Self::with_endpoint(repo, &endpoint, token, transport, hint)
    }

    pub fn with_endpoint(
        repo: String,
        endpoint: &str,
        token: Option<String>,
        transport: Arc<dyn HttpTransport>,
        hint: ManifestHint,
    ) -> Result<Self> {
        let endpoint = Url::parse(endpoint)
            .map_err(|e| Error::invalid_url(format!("ModelScope endpoint {endpoint}"), e))?;
        Ok(Self {
            repo,
            endpoint,
            revision: DEFAULT_REVISION.to_string(),
            token,
            http: default_client()?,
            transport,
            hint,
            paths: OnceLock::new(),
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

    fn file_url(&self, name: &str) -> Result<Url> {
        let path = self
            .paths
            .get()
            .and_then(|p| p.get(name))
            .map_or(name, String::as_str);
        self.endpoint
            .join(&format!(
                "api/v1/models/{}/resolve/{}/{}",
                self.repo, self.revision, path
            ))
            .map_err(|e| Error::invalid_url(format!("ModelScope file {}/{path}", self.repo), e))
    }

    async fn fetch_small(&self, url: &Url) -> Result<Vec<u8>> {
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
        if bytes.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(Error::Hub(format!(
                "{url} is {} bytes, exceeds {MAX_MANIFEST_BYTES} byte cap",
                bytes.len()
            )));
        }
        Ok(bytes.to_vec())
    }
}

#[async_trait]
impl RepoFiles for ModelScopeSource {
    async fn read(&self, name: &str) -> Result<Vec<u8>> {
        self.fetch_small(&self.file_url(name)?).await
    }

    async fn header_quant(&self, name: &str, size: u64) -> Option<String> {
        let url = self.file_url(name).ok()?;
        gguf::probe_quant(self.transport.as_ref(), &url, self.token.as_deref(), size).await
    }
}

/// The API capitalises its keys; the aliases accept lower-cased JSON too.
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
    #[serde(rename = "Name", alias = "name", default)]
    name: String,
    #[serde(rename = "Path", alias = "path", default)]
    path: String,
    #[serde(rename = "Size", alias = "size", default)]
    size: Option<u64>,
    #[serde(rename = "Type", alias = "type", default)]
    file_type: String,
}

#[async_trait]
impl ModelSource for ModelScopeSource {
    async fn plan(&self) -> Result<Plan> {
        let body = self.fetch_small(&self.api_url()?).await?;
        let parsed: ModelScopeFilesResponse = serde_json::from_slice(&body)?;
        match parsed.code {
            0 | 200 => {}
            404 => return Err(Error::HubModelNotFound(self.repo.clone())),
            code => {
                return Err(Error::Hub(format!(
                    "ModelScope repo {} error {code}: {}",
                    self.repo,
                    parsed.message.unwrap_or_default()
                )))
            }
        }

        // Downloads address the real path; the manifest and disk stay flat, so
        // a basename repeated in subdirectories keeps only its last entry.
        let mut by_name: HashMap<String, (String, i64)> = HashMap::new();
        for f in parsed.data.map(|d| d.files).unwrap_or_default() {
            if f.file_type.eq_ignore_ascii_case("tree") || f.name.is_empty() {
                continue;
            }
            let path = if f.path.is_empty() { &f.name } else { &f.path };
            by_name.insert(f.name.clone(), (path.clone(), f.size.unwrap_or(0) as i64));
        }
        let mut listing: Vec<(String, i64)> =
            by_name.iter().map(|(n, (_, s))| (n.clone(), *s)).collect();
        listing.sort();
        let _ = self
            .paths
            .set(by_name.into_iter().map(|(n, (p, _))| (n, p)).collect());

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
    use crate::source::basename;
    use crate::test_support::{fast_transport, gguf, mount_ranged};
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn plan_for(repo: &str, files: &[(&str, &[u8])], dirs: &[&str]) -> Result<Plan> {
        let server = MockServer::start().await;
        let entry = |p: &str, size: usize, kind: &str| json!({"Name": basename(p), "Path": p, "Size": size, "Type": kind});
        let listing: Vec<_> = files
            .iter()
            .map(|(p, body)| entry(p, body.len(), "blob"))
            .chain(dirs.iter().map(|d| entry(d, 0, "tree")))
            .collect();
        let listing = json!({"Code": 200, "Data": {"Files": listing}, "Message": "success"});
        mount_listing(&server, repo, listing.to_string()).await;
        for (p, body) in files {
            let at = format!("/api/v1/models/{repo}/resolve/master/{p}");
            mount_ranged(&server, &at, body.to_vec()).await;
        }
        ModelScopeSource::with_endpoint(
            repo.to_string(),
            &server.uri(),
            None,
            fast_transport(),
            ManifestHint::default(),
        )?
        .plan()
        .await
    }

    async fn mount_listing(server: &MockServer, repo: &str, body: String) {
        Mock::given(method("GET"))
            .and(path(format!("/api/v1/models/{repo}/repo/files")))
            .and(query_param("Recursive", "true"))
            .and(query_param("Revision", "master"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(server)
            .await;
    }

    fn http_path(plan: &Plan, i: usize) -> &str {
        match &plan.files[i].bytes {
            BytesSource::Http { url, .. } => url.path(),
            other => panic!("ModelScope file should be Http, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn plan_synthesises_manifest_when_repo_lacks_one() {
        let plan = plan_for(
            "org/Tiny-GGUF",
            &[("README.md", b"hi"), ("model-Q4_K_M.gguf", b"w")],
            &[],
        )
        .await
        .unwrap();
        assert_eq!(plan.manifest.name, "org/Tiny-GGUF");
        assert!(plan.manifest.model_file.contains_key("Q4_K_M"));
        assert_eq!(plan.files.len(), 1);
        assert_eq!(plan.files[0].name, "model-Q4_K_M.gguf");
        assert!(http_path(&plan, 0).ends_with("model-Q4_K_M.gguf"));
    }

    #[tokio::test]
    async fn plan_uses_shipped_manifest_and_infers_past_a_broken_one() {
        let shipped: &[u8] = br#"{"Name":"x","ModelName":"Shipped","ModelType":"llm","PluginId":"llama_cpp","ModelFile":{"Q8_0":{"Name":"model-Q8_0.gguf","Downloaded":true,"Size":2048}},"MMProjFile":null,"TokenizerFile":null,"ExtraFiles":null}"#;
        for (manifest, want) in [(shipped, "Shipped"), (b"not json", "Repo")] {
            let plan = plan_for(
                "org/Repo",
                &[("geniex.json", manifest), ("model-Q8_0.gguf", b"w")],
                &[],
            )
            .await
            .unwrap();
            assert_eq!(plan.manifest.model_name, want);
            assert_eq!(plan.manifest.name, "org/Repo");
            assert!(plan.manifest.model_file.contains_key("Q8_0"));
            assert_eq!(plan.files.len(), 1);
        }
    }

    #[tokio::test]
    async fn plan_reads_quant_from_the_header_when_the_name_lacks_one() {
        let head = gguf::head(15, 0);
        let plan = plan_for("org/Untagged", &[("REAP-48-v2.gguf", &head)], &[])
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

    #[tokio::test]
    async fn plan_skips_tree_entries_and_flattens_nested_paths() {
        let plan = plan_for("org/Nested", &[("sub/model-Q4_0.gguf", b"w")], &["sub"])
            .await
            .unwrap();
        assert!(plan.manifest.model_file.contains_key("Q4_0"));
        assert_eq!(plan.files.len(), 1);
        assert_eq!(plan.files[0].name, "model-Q4_0.gguf");
        assert!(http_path(&plan, 0).ends_with("sub/model-Q4_0.gguf"));
    }

    #[tokio::test]
    async fn plan_keeps_one_entry_per_basename() {
        let plan = plan_for(
            "org/Dup",
            &[
                ("a/model-Q4_0.gguf", b"short"),
                ("b/model-Q4_0.gguf", b"longer body"),
                ("tokenizer.json", b"{}"),
                ("original/tokenizer.json", b"{}"),
            ],
            &[],
        )
        .await
        .unwrap();
        let gguf: Vec<_> = plan
            .files
            .iter()
            .filter(|f| f.name == "model-Q4_0.gguf")
            .collect();
        assert_eq!(gguf.len(), 1);
        assert_eq!(gguf[0].size, 11);
        assert!(
            matches!(&gguf[0].bytes, BytesSource::Http { url, .. } if url.path().ends_with("b/model-Q4_0.gguf"))
        );
    }

    #[tokio::test]
    async fn plan_classifies_vlm_via_config_json() {
        let config =
            br#"{"architectures":["Qwen2_5_VLForConditionalGeneration"],"vision_config":{}}"#;
        let plan = plan_for(
            "org/VLM",
            &[("config.json", config), ("model-Q4_K_M.gguf", b"w")],
            &[],
        )
        .await
        .unwrap();
        assert_eq!(plan.manifest.model_type, ModelType::Vlm);
    }

    #[tokio::test]
    async fn plan_returns_hub_model_not_found_on_404() {
        let server = MockServer::start().await;
        let listing = r#"{"Code": 404, "Data": null, "Message": "model not found"}"#;
        mount_listing(&server, "org/Missing", listing.to_string()).await;
        let src = ModelScopeSource::with_endpoint(
            "org/Missing".into(),
            &server.uri(),
            None,
            fast_transport(),
            ManifestHint::default(),
        );
        let err = src.unwrap().plan().await.unwrap_err();
        assert!(matches!(err, Error::HubModelNotFound(_)), "{err:?}");
    }
}
