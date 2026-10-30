// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

use std::path::PathBuf;

/// HF tokens are deliberately per-pull ([`crate::pull::PullRequest`]), not stored here.
#[derive(Debug, Clone)]
pub struct StoreConfig {
    pub data_dir: PathBuf,
}

impl StoreConfig {
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

    /// Same layout as the Go CLI so both share the cache.
    pub fn ai_hub_cache_dir(&self) -> PathBuf {
        self.data_dir.join("aihub")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    pub fn model_dir(&self, name: &str) -> PathBuf {
        self.models_dir().join(name)
    }
}

/// Mirrors `cli/internal/config/config.go:DefaultAIHubBaseURL`.
const DEFAULT_AI_HUB_BASE_URL: &str =
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models";

const DEFAULT_AI_HUB_VERSION: &str = "v0.60.0";

pub fn hf_token_from_env() -> Option<String> {
    std::env::var("GENIEX_HFTOKEN").ok()
}

pub(crate) fn hf_endpoint() -> String {
    env_endpoint("HF_ENDPOINT", crate::source::hf::DEFAULT_HF_ENDPOINT)
}

/// Honours the ModelScope SDK's own `MODELSCOPE_ENDPOINT`.
pub(crate) fn modelscope_endpoint() -> String {
    env_endpoint(
        "MODELSCOPE_ENDPOINT",
        crate::source::modelscope::DEFAULT_MODELSCOPE_ENDPOINT,
    )
}

pub(crate) fn ai_hub_base_url() -> String {
    std::env::var("GENIEX_AIHUBBASEURL").unwrap_or_else(|_| DEFAULT_AI_HUB_BASE_URL.to_string())
}

/// An explicit `GENIEX_AIHUBVERSION` pin, which also skips `releases/latest/`.
pub(crate) fn ai_hub_version_override() -> Option<String> {
    std::env::var("GENIEX_AIHUBVERSION")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// The pinned version, else the release used when `releases/latest/` is unavailable.
pub(crate) fn ai_hub_version() -> String {
    ai_hub_version_override().unwrap_or_else(|| DEFAULT_AI_HUB_VERSION.to_string())
}

fn env_endpoint(var: &str, default: &str) -> String {
    match std::env::var(var) {
        Ok(v) if !v.trim().is_empty() => v.trim().trim_end_matches('/').to_string(),
        _ => default.to_string(),
    }
}

pub(crate) fn env_positive<T: std::str::FromStr + PartialOrd + Default>(var: &str) -> Option<T> {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse::<T>().ok())
        .filter(|v| *v > T::default())
}

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
    use crate::source::modelscope::DEFAULT_MODELSCOPE_ENDPOINT;
    use crate::test_support::EnvGuard;

    #[test]
    fn hub_endpoints_cover_unset_custom_and_trailing_slash() {
        type Getter = fn() -> String;
        let hubs: [(&str, Getter, &str, &str); 2] = [
            (
                "HF_ENDPOINT",
                hf_endpoint,
                DEFAULT_HF_ENDPOINT,
                "https://hf-mirror.com",
            ),
            (
                "MODELSCOPE_ENDPOINT",
                modelscope_endpoint,
                DEFAULT_MODELSCOPE_ENDPOINT,
                "https://modelscope.example.com",
            ),
        ];
        for (var, get, default, custom) in hubs {
            let with_slash = format!("{custom}/");
            for (value, want) in [
                (None, default),
                (Some(custom), custom),
                (Some(with_slash.as_str()), custom),
                (Some("  "), default),
            ] {
                let env = EnvGuard::lock();
                let _env = match value {
                    Some(v) => env.set(var, v),
                    None => env.unset(var),
                };
                assert_eq!(get(), want, "{var}={value:?}");
            }
        }
    }
}
