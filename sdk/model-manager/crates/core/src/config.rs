// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::path::PathBuf;

/// Persistent configuration for the model Store.
///
/// HuggingFace tokens are **not** part of this struct: they are a
/// per-pull concern, passed into [`crate::pull::PullRequest`] each time
/// a download is requested. This keeps long-lived tokens out of process
/// state and lets callers rotate credentials on each pull.
#[derive(Debug, Clone)]
pub struct StoreConfig {
    pub data_dir: PathBuf,
}

impl StoreConfig {
    /// Read the data directory from `GENIEX_DATADIR`, falling back to
    /// `~/.cache/geniex` when unset.
    pub fn from_env() -> Self {
        let data_dir = if let Ok(d) = std::env::var("GENIEX_DATADIR") {
            PathBuf::from(d)
        } else {
            default_data_dir()
        };
        Self { data_dir }
    }

    pub fn new(data_dir: PathBuf) -> Self {
        Self { data_dir }
    }

    /// Read the HuggingFace token from the `GENIEX_HFTOKEN` environment
    /// variable. Callers of [`crate::pull::pull`] may use this as a
    /// fallback when the user did not supply an explicit token.
    pub fn hf_token_from_env() -> Option<String> {
        std::env::var("GENIEX_HFTOKEN").ok()
    }

    pub fn hf_endpoint() -> String {
        match std::env::var("HF_ENDPOINT") {
            Ok(v) if !v.trim().is_empty() => v.trim().trim_end_matches('/').to_string(),
            _ => crate::source::hf::DEFAULT_HF_ENDPOINT.to_string(),
        }
    }

    /// AI Hub public assets base URL. Mirrors the Go CLI's
    /// `DefaultAIHubBaseURL`; override via `GENIEX_AIHUBBASEURL`.
    pub fn ai_hub_base_url() -> String {
        std::env::var("GENIEX_AIHUBBASEURL").unwrap_or_else(|_| DEFAULT_AI_HUB_BASE_URL.to_string())
    }

    /// Pinned aihm release version the SDK consumes. The public bucket has
    /// no `latest` alias; override via `GENIEX_AIHUBVERSION`.
    pub fn ai_hub_version() -> String {
        std::env::var("GENIEX_AIHUBVERSION").unwrap_or_else(|_| DEFAULT_AI_HUB_VERSION.to_string())
    }

    /// Cache directory for AI Hub index JSONs. Matches the Go CLI layout
    /// (`<data-dir>/aihub`) so both agents can share the same cache.
    pub fn ai_hub_cache_dir(&self) -> PathBuf {
        self.data_dir.join("aihub")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    pub fn model_dir(&self, name: &str) -> PathBuf {
        self.models_dir().join(name)
    }

    pub fn model_file_path(&self, name: &str, file: &str) -> PathBuf {
        self.model_dir(name).join(file)
    }
}

/// Mirrors `cli/internal/config/config.go:DefaultAIHubBaseURL`.
const DEFAULT_AI_HUB_BASE_URL: &str =
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models";

/// Mirrors `cli/internal/config/config.go:DefaultAIHubVersion`.
const DEFAULT_AI_HUB_VERSION: &str = "v0.57.0";

fn default_data_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("geniex")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::hf::DEFAULT_HF_ENDPOINT;

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

    #[test]
    fn hf_endpoint_covers_unset_custom_and_trailing_slash() {
        {
            let _g = EnvGuard::unset("HF_ENDPOINT");
            assert_eq!(StoreConfig::hf_endpoint(), DEFAULT_HF_ENDPOINT);
        }
        {
            let _g = EnvGuard::set("HF_ENDPOINT", "https://hf-mirror.com");
            assert_eq!(StoreConfig::hf_endpoint(), "https://hf-mirror.com");
        }
        {
            let _g = EnvGuard::set("HF_ENDPOINT", "https://hf-mirror.com/");
            assert_eq!(StoreConfig::hf_endpoint(), "https://hf-mirror.com");
        }
        {
            let _g = EnvGuard::set("HF_ENDPOINT", "  ");
            assert_eq!(StoreConfig::hf_endpoint(), DEFAULT_HF_ENDPOINT);
        }
    }
}
