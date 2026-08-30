// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! [`ModelSource`] that delegates OCI-registry model distribution to
//! [llmman](https://github.com/llmmanorg/llmman).
//!
//! GenieX does not speak the Docker Registry HTTP API V2 itself. Instead
//! it drives a running `llmman serve` daemon, which already implements
//! the whole acquisition side — anonymous/authenticated registry auth,
//! the Docker `application/vnd.docker.ai.*` layouts, CNCF ModelPack,
//! `hf.co` / `ms://` / `s3://` / `ngc://` sources, and content-addressed
//! resume — and keeps its own OCI store on disk.
//!
//! The split is:
//!
//! ```text
//! ┌──────────────────────────────┐   ┌────────────────────────────────┐
//! │ llmman serve  (acquisition)  │   │ GenieX model-manager (store)   │
//! │                              │   │                                │
//! │ POST /api/pull  ────────────▶│   │ plan() → FileSpec{LocalLink}   │
//! │   NDJSON progress            │──▶│ executor links/copies into     │
//! │ `llmman resolve` → abs path  │   │   <data-dir>/models/org/repo/  │
//! └──────────────────────────────┘   └────────────────────────────────┘
//! ```
//!
//! Two llmman surfaces are used, both documented and stable:
//!
//! - `POST /api/pull` on the daemon — newline-delimited JSON progress,
//!   errors reported **in band** at HTTP 200.
//! - `llmman resolve --no-pull <ref>` — one line of JSON naming the
//!   absolute path of the GGUF that the pull just materialised. The
//!   daemon deliberately exposes no path over HTTP (`/api/show` returns
//!   only a digest and size, `/props` hardcodes an empty `model_path`),
//!   so this is the only supported way to bridge from "llmman has it"
//!   to "GenieX can open it".
//!
//! Unlike every other source in this directory, the bulk transfer
//! happens inside [`ModelSource::plan_with_progress`] rather than in the
//! [`crate::executor`] — the bytes are moved by llmman, not by us. The
//! executor step that follows is a local hard-link (or copy) out of
//! llmman's store, which is why this source emits
//! [`BytesSource::LocalLink`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::{Error, Result};
use crate::executor::{FileProgress, ProgressCallback};
use crate::manifest::{ModelFileInfo, ModelManifest, ModelType};
use crate::manifest_builder::{extract_quant, ManifestHint};

use super::{BytesSource, FileSpec, ModelSource, Plan};

/// Where `llmman serve` listens when `LLMMAN_HOST` is unset. Mirrors
/// llmman's own `daemon::DEFAULT_HOST` / `DEFAULT_PORT`.
const DEFAULT_LLMMAN_HOST: &str = "127.0.0.1:17434";

/// llmman's default port, applied when `LLMMAN_HOST` names a host with
/// no `:port` and no scheme.
const DEFAULT_LLMMAN_PORT: u16 = 17434;

/// Default name of the llmman executable, overridable via
/// `GENIEX_LLMMAN_BIN` for non-`PATH` installs.
const DEFAULT_LLMMAN_BIN: &str = "llmman";

/// Liveness probe budget. `llmman serve` answers `/api/version` from
/// memory, so anything slower than this means "not actually there".
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// GenieX plugin that loads the GGUF this source produces.
const LLAMA_CPP_PLUGIN_ID: &str = "llama_cpp";

/// Quantization key used when the resolved filename carries no
/// recognisable GGUF quant tag — llmman stores HuggingFace-sourced GGUFs
/// as bare content-addressed blobs (`blobs/sha256/<hex>`, no extension),
/// so there is nothing to extract a tag from. Matches the "unknown"
/// sentinel the AI Hub path and the Go binding (`PrecisionNA`) use.
const PRECISION_NA: &str = "N/A";

/// Resolve the base URL of the `llmman serve` daemon.
///
/// Reads `LLMMAN_HOST` — llmman's own variable, deliberately not a
/// GenieX-specific one, so a user who points their llmman CLI at a
/// non-default daemon gets the same daemon here without configuring it
/// twice. Accepts `[scheme://]host[:port]`; a bare host defaults to
/// llmman's port, and the scheme defaults to `http` because `llmman
/// serve` terminates no TLS of its own.
pub fn endpoint_from_env() -> String {
    let raw = std::env::var("LLMMAN_HOST")
        .ok()
        .map(|v| v.trim().trim_matches(['"', '\'']).to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_LLMMAN_HOST.to_string());

    let (scheme, rest) = match raw.split_once("://") {
        Some((s, r)) => (s.to_ascii_lowercase(), r.to_string()),
        None => ("http".to_string(), raw),
    };
    // Drop any trailing path — callers give us an origin, not a route.
    let authority = rest.split('/').next().unwrap_or("").to_string();
    if authority.is_empty() {
        return format!("http://{DEFAULT_LLMMAN_HOST}");
    }

    // Already carries a port? `[::1]:8080` and `host:8080` both end in
    // ":<digits>"; a bare `[::1]` or `host` does not.
    let has_port = authority
        .rsplit_once(':')
        .is_some_and(|(_, p)| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()));
    if has_port {
        return format!("{scheme}://{authority}");
    }
    // Matches llmman's own `daemon::parse_host`: an explicit https://
    // means a real TLS endpoint on 443, anything else is the daemon's
    // own port.
    let port = if scheme == "https" {
        443
    } else {
        DEFAULT_LLMMAN_PORT
    };
    format!("{scheme}://{authority}:{port}")
}

/// Name/path of the `llmman` binary used for `llmman resolve`.
pub fn binary_from_env() -> String {
    std::env::var("GENIEX_LLMMAN_BIN")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_LLMMAN_BIN.to_string())
}

/// One line of `POST /api/pull`'s NDJSON body.
///
/// llmman reports *aggregate* byte counts for the whole pull rather than
/// Ollama's per-layer `digest` progress, so there is exactly one logical
/// file to report on. `error` is the in-band failure channel: the HTTP
/// status stays 200 once the stream has started.
#[derive(Debug, Default, Deserialize)]
struct ProgressLine {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    completed: Option<u64>,
}

/// The single JSON line `llmman resolve` prints on stdout.
#[derive(Debug, Deserialize)]
struct ResolvedModel {
    /// Absolute path of the GGUF (or, for `format == "safetensors"`, of
    /// the directory holding `config.json`).
    path: String,
    #[serde(default)]
    format: String,
    /// Multimodal projector companion, present only for VLMs.
    #[serde(default)]
    mmproj: Option<String>,
}

pub struct LlmmanSource {
    /// Reference exactly as llmman should see it, e.g.
    /// `docker.io/ai/gemma3:latest`.
    reference: String,
    /// GenieX store name (`org/repo`) the manifest is published under.
    model_name: String,
    /// Origin of the `llmman serve` daemon, e.g. `http://127.0.0.1:17434`.
    endpoint: String,
    /// `llmman` executable used for the path-resolution step.
    binary: String,
    hint: ManifestHint,
}

impl LlmmanSource {
    pub fn new(reference: String, model_name: String, hint: ManifestHint) -> Self {
        Self {
            reference,
            model_name,
            endpoint: endpoint_from_env(),
            binary: binary_from_env(),
            hint,
        }
    }

    /// Point at an explicit daemon origin / binary. Used by tests.
    pub fn with_endpoint_and_binary(mut self, endpoint: String, binary: String) -> Self {
        self.endpoint = endpoint;
        self.binary = binary;
        self
    }

    fn client(&self) -> Result<reqwest::Client> {
        reqwest::Client::builder()
            .user_agent(crate::transport::USER_AGENT)
            .use_preconfigured_tls(crate::transport::build_tls_config()?)
            // No overall request timeout: a pull legitimately runs for
            // as long as the model takes to download. Liveness is
            // established separately by `probe_daemon`, and the daemon
            // emits a progress line every 200 ms thereafter.
            .build()
            .map_err(|e| Error::Http(format!("build llmman client: {e}")))
    }

    /// Confirm a daemon is actually answering before we commit to a
    /// streaming pull, so "llmman isn't running" surfaces as a one-line
    /// actionable error instead of a connection reset mid-download.
    ///
    /// A non-2xx answer is treated the same as no answer at all: it means
    /// *something* holds the port but it isn't an llmman daemon, which
    /// the user fixes the same way. Reporting a bare "HTTP 404" here
    /// would send them looking for a missing model instead.
    async fn probe_daemon(&self, client: &reqwest::Client) -> Result<()> {
        let why = match client
            .get(self.version_url())
            .timeout(PROBE_TIMEOUT)
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => return Ok(()),
            Ok(r) => format!("it answered /api/version with HTTP {}", r.status().as_u16()),
            Err(e) => format!("{e}"),
        };
        Err(Error::Hub(format!(
            "llmman serve is not reachable at {} ({why}). Start it with `llmman serve`, \
             or set LLMMAN_HOST to an already-running daemon.",
            self.endpoint
        )))
    }

    fn version_url(&self) -> String {
        format!("{}/api/version", self.endpoint)
    }

    /// Drive `POST /api/pull` to completion, forwarding llmman's
    /// aggregate progress into `on_progress`.
    ///
    /// Returning `false` from the callback cancels: we drop the response
    /// (closing the connection) and return [`Error::Cancelled`]. llmman
    /// keeps its partial blobs and resumes on the next attempt, so this
    /// is a safe interruption point.
    async fn pull_via_daemon(
        &self,
        client: &reqwest::Client,
        on_progress: Option<&ProgressCallback>,
    ) -> Result<()> {
        let url = format!("{}/api/pull", self.endpoint);
        // Hand-rolled body rather than `RequestBuilder::json`, which would
        // pull in reqwest's `json` feature for this one call.
        let body = serde_json::to_vec(&serde_json::json!({ "model": self.reference }))?;
        let mut resp = client
            .post(&url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body)
            .send()
            .await
            .map_err(|e| Error::HttpTimeout(format!("POST {url}: {e}")))?;

        if !resp.status().is_success() {
            return Err(Error::HttpStatus {
                url,
                status: resp.status().as_u16(),
            });
        }

        let mut buf: Vec<u8> = Vec::new();
        let mut saw_success = false;
        loop {
            let chunk = resp
                .chunk()
                .await
                .map_err(|e| Error::HttpTimeout(format!("{url}: {e}")))?;
            let Some(chunk) = chunk else { break };
            buf.extend_from_slice(&chunk);
            while let Some(nl) = buf.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                if self.handle_line(&line[..nl], on_progress, &mut saw_success)? {
                    return Ok(());
                }
            }
        }
        // A final line without a trailing newline is still a line.
        if !buf.is_empty() && self.handle_line(&buf, on_progress, &mut saw_success)? {
            return Ok(());
        }

        if saw_success {
            Ok(())
        } else {
            // The daemon closed the stream without a terminal
            // {"status":"success"} — it died, or was restarted mid-pull.
            Err(Error::Hub(format!(
                "llmman serve closed the /api/pull stream for {} without reporting success; \
                 check its log (~/.local/share/llmman/serve.log)",
                self.reference
            )))
        }
    }

    /// Parse and act on one NDJSON line. Returns `Ok(true)` when the
    /// terminal success line was seen and the caller should stop reading.
    fn handle_line(
        &self,
        line: &[u8],
        on_progress: Option<&ProgressCallback>,
        saw_success: &mut bool,
    ) -> Result<bool> {
        if line.iter().all(u8::is_ascii_whitespace) {
            return Ok(false);
        }
        // Unparseable lines are not fatal: llmman's stream is additive,
        // and a future field or a stray log line must not abort a pull
        // that is otherwise succeeding.
        let Ok(ev) = serde_json::from_slice::<ProgressLine>(line) else {
            return Ok(false);
        };

        if let Some(msg) = ev.error.filter(|m| !m.is_empty()) {
            // llmman spells a missing reference "…: not found".
            if msg.to_ascii_lowercase().contains("not found") {
                return Err(Error::HubModelNotFound(self.reference.clone()));
            }
            return Err(Error::Hub(format!("llmman: {msg}")));
        }

        let status = ev.status.unwrap_or_default();
        if status == "success" {
            *saw_success = true;
            return Ok(true);
        }

        if let Some(cb) = on_progress {
            let total = ev.total.unwrap_or(0);
            let completed = ev.completed.unwrap_or(0);
            let snapshot = [FileProgress {
                file_name: self.reference.clone(),
                downloaded_bytes: completed.min(total.max(completed)) as i64,
                // -1 is the executor's "total unknown" sentinel, which is
                // exactly the state of a manifest-only progress line.
                total_bytes: if total == 0 { -1 } else { total as i64 },
            }];
            if !(cb)(&snapshot) {
                return Err(Error::Cancelled);
            }
        }
        Ok(false)
    }

    /// Ask the llmman CLI where the model it just pulled actually lives.
    ///
    /// `llmman resolve` runs entirely in-process (it does not touch the
    /// daemon) and `--no-pull` makes it fail rather than silently
    /// re-downloading, so by construction this only reports on bytes the
    /// `/api/pull` above already landed.
    async fn resolve_paths(&self) -> Result<ResolvedModel> {
        let binary = self.binary.clone();
        let reference = self.reference.clone();
        // std::process rather than tokio::process so the core crate
        // doesn't need tokio's `process` feature for one call.
        let out = tokio::task::spawn_blocking(move || {
            std::process::Command::new(&binary)
                .args(["resolve", "--no-pull", &reference])
                .output()
        })
        .await
        .map_err(|e| Error::Hub(format!("llmman resolve join: {e}")))?
        .map_err(|e| {
            Error::Hub(format!(
                "could not run `{} resolve` ({e}). Install llmman, or set GENIEX_LLMMAN_BIN \
                 to its full path.",
                self.binary
            ))
        })?;

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(Error::Hub(format!(
                "`{} resolve --no-pull {}` failed: {}",
                self.binary,
                self.reference,
                stderr.trim()
            )));
        }

        // Diagnostics go to stderr; stdout is exactly one JSON line.
        let stdout = String::from_utf8_lossy(&out.stdout);
        let line = stdout
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .next_back()
            .ok_or_else(|| {
                Error::Hub(format!(
                    "`{} resolve` printed nothing for {}",
                    self.binary, self.reference
                ))
            })?;
        crate::error::parse_manifest("llmman resolve output", line.as_bytes())
    }

    fn build_plan(&self, resolved: ResolvedModel) -> Result<Plan> {
        if !resolved.format.is_empty() && !resolved.format.eq_ignore_ascii_case("gguf") {
            return Err(Error::Hub(format!(
                "{} resolves to a {} model; GenieX plugins load GGUF (llama_cpp) or \
                 QAIRT bundles, so this reference is not usable here",
                self.reference, resolved.format
            )));
        }

        let model_path = PathBuf::from(&resolved.path);
        let model_file_name = gguf_file_name(&model_path, &self.model_name, "model");
        let quant = extract_quant(&model_file_name).unwrap_or_else(|| PRECISION_NA.to_string());
        let model_size = file_size(&model_path)?;

        let mut files = vec![FileSpec {
            name: model_file_name.clone(),
            size: model_size,
            bytes: BytesSource::LocalLink {
                path: model_path.clone(),
            },
        }];

        let mut model_file = HashMap::new();
        model_file.insert(
            quant,
            ModelFileInfo {
                name: model_file_name,
                downloaded: true,
                size: model_size as i64,
            },
        );

        // An mmproj companion is llmman's own signal that the reference is
        // multimodal — the same conclusion the Docker Registry path drew
        // from an `mmproj` layer.
        let mut mmproj_file = ModelFileInfo::default();
        if let Some(p) = resolved.mmproj.filter(|p| !p.is_empty()) {
            let path = PathBuf::from(&p);
            let name = gguf_file_name(&path, &self.model_name, "mmproj");
            let size = file_size(&path)?;
            mmproj_file = ModelFileInfo {
                name: name.clone(),
                downloaded: true,
                size: size as i64,
            };
            files.push(FileSpec {
                name,
                size,
                bytes: BytesSource::LocalLink { path },
            });
        }

        let model_type = if mmproj_file.downloaded {
            ModelType::Vlm
        } else {
            self.hint.model_type.clone().unwrap_or(ModelType::Llm)
        };

        let manifest = ModelManifest {
            name: self.model_name.clone(),
            model_name: derive_model_name(&self.model_name),
            model_type,
            plugin_id: LLAMA_CPP_PLUGIN_ID.to_string(),
            precision: String::new(),
            model_file,
            mmproj_file,
            tokenizer_file: ModelFileInfo::default(),
            extra_files: Vec::new(),
        };

        Ok(Plan { manifest, files })
    }
}

#[async_trait]
impl ModelSource for LlmmanSource {
    async fn plan(&self) -> Result<Plan> {
        self.plan_with_progress(None).await
    }

    async fn plan_with_progress(&self, on_progress: Option<&ProgressCallback>) -> Result<Plan> {
        let client = self.client()?;
        self.probe_daemon(&client).await?;
        self.pull_via_daemon(&client, on_progress).await?;
        let resolved = self.resolve_paths().await?;
        self.build_plan(resolved)
    }
}

/// Pick the basename this file should have inside GenieX's store.
///
/// llmman materialises a tar-layer model as `<cache>/<hex>/<name>.gguf`,
/// where `<name>` usually carries the quant tag — keep that verbatim so
/// [`extract_quant`] can read it. A HuggingFace-sourced GGUF, however, is
/// used in place as a bare content-addressed blob
/// (`<store>/blobs/sha256/<hex>`, no extension); copying that name into
/// our store would leave a file no plugin recognises as a GGUF, so
/// synthesize `<repo>[-<role>].gguf` instead.
fn gguf_file_name(path: &Path, model_name: &str, role: &str) -> String {
    let base = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();
    if base.to_ascii_lowercase().ends_with(".gguf") {
        return base;
    }
    let repo = derive_model_name(model_name);
    let repo = if repo.is_empty() { "model" } else { &repo };
    if role == "model" {
        format!("{repo}.gguf")
    } else {
        format!("{role}-{repo}.gguf")
    }
}

fn file_size(path: &Path) -> Result<u64> {
    std::fs::metadata(path)
        .map(|m| m.len())
        .map_err(|e| Error::Hub(format!("llmman reported {} but {e}", path.display())))
}

/// Last path component of an `org/repo` name — mirrors
/// `infer_manifest_from_names` so `model_name` is stable across hubs.
fn derive_model_name(name: &str) -> String {
    name.rsplit('/').next().unwrap_or(name).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, prev }
        }
        fn unset(key: &'static str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn src(reference: &str, model_name: &str) -> LlmmanSource {
        LlmmanSource::new(
            reference.to_string(),
            model_name.to_string(),
            ManifestHint::default(),
        )
    }

    #[test]
    fn endpoint_defaults_to_llmman_loopback_port() {
        let _g = EnvGuard::unset("LLMMAN_HOST");
        assert_eq!(endpoint_from_env(), "http://127.0.0.1:17434");
    }

    #[test]
    fn endpoint_accepts_host_port_scheme_and_quotes() {
        {
            let _g = EnvGuard::set("LLMMAN_HOST", "192.168.1.9:9000");
            assert_eq!(endpoint_from_env(), "http://192.168.1.9:9000");
        }
        {
            // Bare host: llmman's own default port is appended.
            let _g = EnvGuard::set("LLMMAN_HOST", "llmman-box");
            assert_eq!(endpoint_from_env(), "http://llmman-box:17434");
        }
        {
            let _g = EnvGuard::set("LLMMAN_HOST", "https://llmman.internal");
            assert_eq!(endpoint_from_env(), "https://llmman.internal:443");
        }
        {
            // Trailing path is not part of the origin.
            let _g = EnvGuard::set("LLMMAN_HOST", "http://127.0.0.1:17434/api");
            assert_eq!(endpoint_from_env(), "http://127.0.0.1:17434");
        }
        {
            let _g = EnvGuard::set("LLMMAN_HOST", "  \"127.0.0.1:1234\"  ");
            assert_eq!(endpoint_from_env(), "http://127.0.0.1:1234");
        }
        {
            let _g = EnvGuard::set("LLMMAN_HOST", "");
            assert_eq!(endpoint_from_env(), "http://127.0.0.1:17434");
        }
    }

    #[test]
    fn in_band_error_line_fails_the_pull() {
        let s = src("docker.io/ai/gemma3:latest", "ai/gemma3");
        let mut ok = false;
        let err = s
            .handle_line(br#"{"error":"pulling x: boom"}"#, None, &mut ok)
            .unwrap_err();
        assert!(format!("{err}").contains("boom"), "{err}");
        assert!(!ok);
    }

    #[test]
    fn not_found_error_maps_to_hub_model_not_found() {
        let s = src("docker.io/ai/nope:latest", "ai/nope");
        let mut ok = false;
        let err = s
            .handle_line(
                br#"{"error":"pulling docker.io/ai/nope:latest: not found"}"#,
                None,
                &mut ok,
            )
            .unwrap_err();
        assert!(matches!(err, Error::HubModelNotFound(_)), "{err:?}");
    }

    #[test]
    fn success_line_terminates_the_stream() {
        let s = src("docker.io/ai/gemma3:latest", "ai/gemma3");
        let mut ok = false;
        assert!(s
            .handle_line(br#"{"status":"success"}"#, None, &mut ok)
            .unwrap());
        assert!(ok);
    }

    #[test]
    fn blank_and_unparseable_lines_are_ignored() {
        let s = src("docker.io/ai/gemma3:latest", "ai/gemma3");
        let mut ok = false;
        assert!(!s.handle_line(b"   ", None, &mut ok).unwrap());
        assert!(!s.handle_line(b"not json at all", None, &mut ok).unwrap());
        assert!(!ok);
    }

    #[test]
    fn progress_line_forwards_aggregate_counts() {
        let s = src("docker.io/ai/gemma3:latest", "ai/gemma3");
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(i64, i64)>::new()));
        let sink = seen.clone();
        let cb: ProgressCallback = Box::new(move |files: &[FileProgress]| {
            let mut g = sink.lock().unwrap();
            for f in files {
                g.push((f.downloaded_bytes, f.total_bytes));
            }
            true
        });
        let mut ok = false;
        s.handle_line(
            br#"{"status":"pulling","total":100,"completed":40}"#,
            Some(&cb),
            &mut ok,
        )
        .unwrap();
        // A manifest-only line has no byte counts: total is reported as
        // the executor's -1 "unknown" sentinel rather than a bogus 0.
        s.handle_line(br#"{"status":"pulling manifest"}"#, Some(&cb), &mut ok)
            .unwrap();
        assert_eq!(*seen.lock().unwrap(), vec![(40, 100), (0, -1)]);
    }

    #[test]
    fn callback_returning_false_cancels() {
        let s = src("docker.io/ai/gemma3:latest", "ai/gemma3");
        let cb: ProgressCallback = Box::new(|_: &[FileProgress]| false);
        let mut ok = false;
        let err = s
            .handle_line(
                br#"{"status":"pulling","total":10,"completed":1}"#,
                Some(&cb),
                &mut ok,
            )
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled), "{err:?}");
    }

    #[test]
    fn gguf_name_kept_when_llmman_extracted_a_named_layer() {
        let p = PathBuf::from("/c/ab12/gemma-3-4b-it-Q4_K_M.gguf");
        assert_eq!(
            gguf_file_name(&p, "ai/gemma3", "model"),
            "gemma-3-4b-it-Q4_K_M.gguf"
        );
        assert_eq!(
            extract_quant("gemma-3-4b-it-Q4_K_M.gguf").as_deref(),
            Some("Q4_K_M")
        );
    }

    #[test]
    fn bare_blob_path_gets_a_synthesized_gguf_name() {
        // HuggingFace-sourced GGUFs are used in place as blobs, so the
        // basename is a hex digest with no extension.
        let p = PathBuf::from("/s/blobs/sha256/9b7f3c1e2d");
        assert_eq!(
            gguf_file_name(&p, "unsloth/Qwen3-GGUF", "model"),
            "Qwen3-GGUF.gguf"
        );
        assert_eq!(
            gguf_file_name(&p, "unsloth/Qwen3-GGUF", "mmproj"),
            "mmproj-Qwen3-GGUF.gguf"
        );
    }

    #[test]
    fn safetensors_reference_is_rejected_with_an_actionable_message() {
        let s = src("hf.co/org/repo:latest", "org/repo");
        let err = s
            .build_plan(ResolvedModel {
                path: "/c/ab12".to_string(),
                format: "safetensors".to_string(),
                mmproj: None,
            })
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("safetensors"), "{msg}");
        assert!(msg.contains("GGUF"), "{msg}");
    }

    #[test]
    fn plan_emits_local_link_and_marks_mmproj_models_as_vlm() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("gemma-3-4b-it-Q4_K_M.gguf");
        let mmproj = tmp.path().join("mmproj-F16.gguf");
        std::fs::write(&model, b"weights").unwrap();
        std::fs::write(&mmproj, b"proj").unwrap();

        let s = src("docker.io/ai/gemma3:latest", "ai/gemma3");
        let plan = s
            .build_plan(ResolvedModel {
                path: model.display().to_string(),
                format: "gguf".to_string(),
                mmproj: Some(mmproj.display().to_string()),
            })
            .unwrap();

        assert_eq!(plan.manifest.model_type, ModelType::Vlm);
        assert_eq!(plan.manifest.plugin_id, LLAMA_CPP_PLUGIN_ID);
        assert_eq!(plan.manifest.name, "ai/gemma3");
        assert_eq!(plan.manifest.model_name, "gemma3");
        assert!(plan.manifest.model_file.contains_key("Q4_K_M"));
        assert_eq!(plan.manifest.mmproj_file.name, "mmproj-F16.gguf");
        assert_eq!(plan.files.len(), 2);
        for f in &plan.files {
            assert!(
                matches!(f.bytes, BytesSource::LocalLink { .. }),
                "expected LocalLink, got {:?}",
                f.bytes
            );
        }
    }

    #[test]
    fn plan_without_mmproj_defaults_to_llm_and_na_precision() {
        let tmp = tempfile::tempdir().unwrap();
        // No extension: the blob layout, so no quant tag to extract.
        let model = tmp.path().join("9b7f3c1e2d");
        std::fs::write(&model, b"weights").unwrap();

        let s = src("hf.co/unsloth/Qwen3-GGUF:latest", "unsloth/Qwen3-GGUF");
        let plan = s
            .build_plan(ResolvedModel {
                path: model.display().to_string(),
                format: "gguf".to_string(),
                mmproj: None,
            })
            .unwrap();

        assert_eq!(plan.manifest.model_type, ModelType::Llm);
        assert!(plan.manifest.model_file.contains_key(PRECISION_NA));
        assert_eq!(
            plan.manifest.model_file[PRECISION_NA].name,
            "Qwen3-GGUF.gguf"
        );
        assert!(!plan.manifest.mmproj_file.downloaded);
    }

    #[test]
    fn missing_resolved_file_is_reported_against_its_path() {
        let s = src("docker.io/ai/gemma3:latest", "ai/gemma3");
        let err = s
            .build_plan(ResolvedModel {
                path: "/nonexistent/xyz-Q4_0.gguf".to_string(),
                format: "gguf".to_string(),
                mmproj: None,
            })
            .unwrap_err();
        assert!(format!("{err}").contains("/nonexistent/xyz-Q4_0.gguf"));
    }
}
