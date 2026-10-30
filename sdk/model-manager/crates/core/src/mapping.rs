// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

/// `qwen3` stays so the Python bindings and CLI smoke tests have an alias that resolves.
pub fn resolve_alias(alias: &str) -> Option<String> {
    match alias {
        "qwen3" => Some("ggml-org/Qwen3-1.7B-GGUF:Q4_K_M".to_string()),
        _ => None,
    }
}

/// Lowercased AI Hub orgs; `ai-hub-models` is an alias for `qualcomm`.
const AI_HUB_ORGS: &[&str] = &["qualcomm", "ai-hub-models"];

/// Call before `pull` / `get_paths` so every alias maps to one cache entry.
pub fn canonicalize_model_name(name: &str) -> String {
    let name = name
        .strip_prefix("https://huggingface.co/")
        .or_else(|| name.strip_prefix("http://huggingface.co/"))
        .or_else(|| strip_modelscope_url_prefix(name))
        .unwrap_or(name);
    match name.split_once('/') {
        None => format!("qualcomm/{name}"),
        Some((org, repo)) if org.eq_ignore_ascii_case("ai-hub-models") => {
            format!("qualcomm/{repo}")
        }
        Some(_) => name.to_string(),
    }
}

/// The AI Hub `display_name` for an AI Hub org's model, ignoring any `:quant`.
pub fn aihub_display_name_from_repo(model_name: &str) -> Option<&str> {
    let without_quant = model_name.split_once(':').map_or(model_name, |(n, _)| n);
    let (org, repo) = without_quant.split_once('/')?;
    if org.is_empty() || repo.is_empty() {
        return None;
    }
    let org_lower = org.to_ascii_lowercase();
    if AI_HUB_ORGS.iter().any(|o| *o == org_lower) {
        Some(repo)
    } else {
        None
    }
}

/// A bare "ai/x" is never treated as Docker Hub: an HF org could be named `ai`.
const DOCKER_HUB_PREFIXES: &[&str] = &[
    "docker.io/",
    "index.docker.io/",
    "registry-1.docker.io/",
    "https://hub.docker.com/r/",
    "http://hub.docker.com/r/",
    "hub.docker.com/r/",
];

const MODELSCOPE_URL_PREFIXES: &[&str] = &[
    "https://modelscope.cn/models/",
    "http://modelscope.cn/models/",
];

fn strip_modelscope_url_prefix(name: &str) -> Option<&str> {
    MODELSCOPE_URL_PREFIXES
        .iter()
        .find_map(|p| name.strip_prefix(p))
}

pub fn is_modelscope_reference(name: &str) -> bool {
    strip_modelscope_url_prefix(name).is_some()
}

pub fn is_docker_hub_reference(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    DOCKER_HUB_PREFIXES.iter().any(|p| lower.starts_with(p))
}

pub fn docker_hub_repo_from_name(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    for p in DOCKER_HUB_PREFIXES {
        if lower.starts_with(p) {
            return name[p.len()..].to_string();
        }
    }
    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_aliases() {
        assert_eq!(
            resolve_alias("qwen3").as_deref(),
            Some("ggml-org/Qwen3-1.7B-GGUF:Q4_K_M")
        );
        assert!(resolve_alias("nonexistent_model_xyz").is_none());
    }

    #[test]
    fn aihub_display_name_from_repo_cases() {
        for (repo, want) in [
            ("qualcomm/Qwen3-4B", Some("Qwen3-4B")),
            (
                "ai-hub-models/Llama-v3.2-3B-Chat",
                Some("Llama-v3.2-3B-Chat"),
            ),
            ("Qualcomm/Qwen3-4B", Some("Qwen3-4B")),
            (
                "AI-Hub-Models/Llama-v3.2-3B-Chat",
                Some("Llama-v3.2-3B-Chat"),
            ),
            ("qualcomm/Qwen3-4B:Q4_K_M", Some("Qwen3-4B")),
            ("aihub/llama_v3_2_3b_instruct", None),
            ("ggml-org/Qwen3-0.6B-GGUF", None),
            ("bartowski/Foo", None),
            ("qualcomm", None),
            ("qualcomm/", None),
            ("/Qwen3-4B", None),
            ("ai-hub-models/", None),
        ] {
            assert_eq!(aihub_display_name_from_repo(repo), want, "{repo}");
        }
    }

    #[test]
    fn canonicalize_model_name_cases() {
        for (name, want) in [
            ("llama_v3_2_3b_instruct", "qualcomm/llama_v3_2_3b_instruct"),
            ("qualcomm/Qwen3-4B", "qualcomm/Qwen3-4B"),
            ("ggml-org/Qwen3-1.7B-GGUF", "ggml-org/Qwen3-1.7B-GGUF"),
            (
                "https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF",
                "ggml-org/Qwen3-1.7B-GGUF",
            ),
            ("http://huggingface.co/bartowski/Foo", "bartowski/Foo"),
            (
                "https://modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF",
                "Qwen/Qwen3-0.6B-GGUF",
            ),
            (
                "http://modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF",
                "Qwen/Qwen3-0.6B-GGUF",
            ),
            (
                "ai-hub-models/Llama-v3.2-3B-Chat",
                "qualcomm/Llama-v3.2-3B-Chat",
            ),
            ("AI-Hub-Models/Qwen3-4B", "qualcomm/Qwen3-4B"),
            ("ai-hub-models/Qwen3-4B:Q4_K_M", "qualcomm/Qwen3-4B:Q4_K_M"),
        ] {
            assert_eq!(canonicalize_model_name(name), want, "{name}");
        }
    }

    #[test]
    fn hub_reference_detection_needs_an_explicit_prefix() {
        for (name, modelscope, docker) in [
            (
                "https://modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF",
                true,
                false,
            ),
            (
                "http://modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF",
                true,
                false,
            ),
            ("https://huggingface.co/Qwen/Qwen3-0.6B-GGUF", false, false),
            ("Qwen/Qwen3-0.6B-GGUF", false, false),
            ("docker.io/ai/gemma3", false, true),
            ("DOCKER.IO/ai/gemma3", false, true),
            ("index.docker.io/ai/gemma3", false, true),
            ("https://hub.docker.com/r/ai/gemma3", false, true),
            ("ai/gemma3", false, false),
            ("ggml-org/Qwen3-1.7B-GGUF", false, false),
        ] {
            assert_eq!(
                is_modelscope_reference(name),
                modelscope,
                "modelscope {name}"
            );
            assert_eq!(is_docker_hub_reference(name), docker, "docker {name}");
        }
    }

    #[test]
    fn docker_hub_repo_from_name_strips_known_prefixes() {
        for name in [
            "docker.io/ai/gemma3",
            "index.docker.io/ai/gemma3",
            "https://hub.docker.com/r/ai/gemma3",
            "ai/gemma3",
        ] {
            assert_eq!(docker_hub_repo_from_name(name), "ai/gemma3", "{name}");
        }
    }
}
