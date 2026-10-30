// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

/// Resolve a short model alias to its canonical "org/repo" name.
///
/// Currently single-entry — `qwen3` is kept as the canonical example so
/// Python bindings and CLI smoke tests have something that resolves.
/// Add more aliases inline as needed.
pub fn resolve_alias(alias: &str) -> Option<String> {
    match alias {
        "qwen3" => Some("ggml-org/Qwen3-1.7B-GGUF:Q4_K_M".to_string()),
        _ => None,
    }
}

/// Orgs whose "<org>/<repo>" names are published on AI Hub rather than HF.
/// Lowercased; lookup is case-insensitive. `qualcomm` is the canonical
/// org; `ai-hub-models` is a user-convenience alias that
/// [`canonicalize_model_name`] rewrites to `qualcomm`.
const AI_HUB_ORGS: &[&str] = &["qualcomm", "ai-hub-models"];

/// Canonicalise a user-supplied model name into the "org/repo" shape
/// that [`crate::validation::validate_model_name`] expects.
///
/// A name without '/' is assumed to be a bare AI Hub model id
/// (e.g. `llama_v3_2_3b_instruct`) and is rewritten to `qualcomm/<name>`.
/// The `ai-hub-models/<repo>` alias (case-insensitive) is rewritten to
/// the canonical `qualcomm/<repo>` so both prefixes share one cache
/// entry. Anything else with a '/' is returned unchanged.
///
/// This is the single entry point callers should use before handing a
/// name to `pull` / `get_paths` so the Store layout stays consistent.
pub fn canonicalize_model_name(name: &str) -> String {
    // A pasted model-page URL carries a scheme + host the rest of the
    // pipeline can't parse; strip it down to "org/repo" first.
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

/// If `model_name` is "<org>/<repo>" where `<org>` belongs to an AI Hub
/// org, return `<repo>` — the value a caller should pass as AI Hub
/// `display_name`. Returns `None` when the model is not AI Hub.
///
/// The split discards anything after a colon (`":<quant>"`) since AI Hub
/// models don't use the HF-style quant suffix; the storage name keeps
/// the suffix untouched on the caller side.
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

/// URL-ish prefixes that unambiguously mark a reference as Docker Hub
/// rather than a HuggingFace "org/repo" name that happens to share the
/// same shape (an HF org could, in principle, be named `ai`). Checked
/// against the *original* user input, mirroring the
/// `https://huggingface.co/` strip above — never inferred from a bare
/// "org/repo" name, since unlike the AI Hub orgs there's no dedicated
/// namespace we can claim outright.
const DOCKER_HUB_PREFIXES: &[&str] = &[
    "docker.io/",
    "index.docker.io/",
    "registry-1.docker.io/",
    "https://hub.docker.com/r/",
    "http://hub.docker.com/r/",
    "hub.docker.com/r/",
];

/// Model-page URL prefixes that identify a ModelScope reference.
const MODELSCOPE_URL_PREFIXES: &[&str] = &[
    "https://modelscope.cn/models/",
    "http://modelscope.cn/models/",
];

/// Strip a recognised ModelScope model-page prefix, yielding "org/repo".
/// `None` when `name` carries no such prefix.
fn strip_modelscope_url_prefix(name: &str) -> Option<&str> {
    MODELSCOPE_URL_PREFIXES
        .iter()
        .find_map(|p| name.strip_prefix(p))
}

/// True when `name` is a pasted ModelScope model-page URL. Used to route
/// `--model-hub auto` pulls to ModelScope without the caller passing
/// `--model-hub modelscope`; a bare "org/repo" is never inferred, since
/// ModelScope shares its namespace shape with HuggingFace.
pub fn is_modelscope_reference(name: &str) -> bool {
    strip_modelscope_url_prefix(name).is_some()
}

/// True when `name` carries one of [`DOCKER_HUB_PREFIXES`]. Used to
/// route `--model-hub auto` pulls to Docker Hub without the caller
/// passing `--model-hub docker` explicitly.
pub fn is_docker_hub_reference(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    DOCKER_HUB_PREFIXES.iter().any(|p| lower.starts_with(p))
}

/// Strip a recognised Docker Hub / registry prefix, returning the bare
/// "org/repo" Docker reference (e.g. `"docker.io/ai/gemma3"` ->
/// `"ai/gemma3"`). A name with no recognised prefix is returned
/// unchanged, so this is also safe to call when the hub was selected
/// explicitly via `--model-hub docker` and the user passed a bare
/// `"ai/gemma3"`.
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
            // Bare "org/repo" is indistinguishable from a HuggingFace repo.
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
