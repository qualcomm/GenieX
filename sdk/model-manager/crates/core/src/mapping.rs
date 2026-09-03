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
    // A pasted HuggingFace URL ("https://huggingface.co/org/repo") carries a
    // scheme + host the rest of the pipeline can't parse; strip it down to
    // "org/repo" first.
    let name = name
        .strip_prefix("https://huggingface.co/")
        .or_else(|| name.strip_prefix("http://huggingface.co/"))
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

/// URL-ish prefixes that unambiguously mark a reference as an OCI
/// registry artifact, which GenieX pulls via `llmman serve` rather than
/// natively. Checked against the *original* user input, mirroring the
/// `https://huggingface.co/` strip above.
///
/// A bare "org/repo" is never on this list: that shape is
/// indistinguishable from a HuggingFace repo, and unlike the AI Hub orgs
/// there's no namespace we can claim outright. Users who want a bare
/// name routed to llmman pass `--model-hub llmman` explicitly.
///
/// `hub.docker.com/r/` is a *web UI* URL rather than a registry host, so
/// it needs the rewrite in [`llmman_reference_from_name`]; every other
/// entry is already a real registry host that llmman resolves as-is.
const LLMMAN_PREFIXES: &[&str] = &[
    "oci://",
    "docker.io/",
    "index.docker.io/",
    "registry-1.docker.io/",
    "ghcr.io/",
    "quay.io/",
    "gcr.io/",
    "mcr.microsoft.com/",
    "public.ecr.aws/",
    "https://hub.docker.com/r/",
    "http://hub.docker.com/r/",
    "hub.docker.com/r/",
];

/// Docker Hub web-UI prefixes, which must be rewritten to the registry
/// host llmman actually talks to.
const DOCKER_HUB_WEB_PREFIXES: &[&str] = &[
    "https://hub.docker.com/r/",
    "http://hub.docker.com/r/",
    "hub.docker.com/r/",
];

/// True when `name` carries one of [`LLMMAN_PREFIXES`]. Used to route
/// `--model-hub auto` pulls to llmman without the caller passing
/// `--model-hub llmman` explicitly.
pub fn is_llmman_reference(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    LLMMAN_PREFIXES.iter().any(|p| lower.starts_with(p))
}

/// Normalise a user-supplied name into the reference string to hand
/// llmman verbatim.
///
/// The registry host is deliberately **kept** — llmman's own
/// short-name resolution would otherwise expand a bare `ai/gemma3` to
/// `hf.co/ai/gemma3` (it reads a `/` with no host as a HuggingFace repo).
/// Only two rewrites happen: an `oci://` scheme is dropped, and a
/// `hub.docker.com/r/` web URL becomes `docker.io/`.
///
/// A hostless name (reached only via an explicit `--model-hub llmman`)
/// is passed through untouched so llmman's own defaulting and
/// `shortnames.conf` aliases still apply.
pub fn llmman_reference_from_name(name: &str) -> String {
    const OCI_SCHEME: &str = "oci://";
    let lower = name.to_ascii_lowercase();
    for p in DOCKER_HUB_WEB_PREFIXES {
        if lower.starts_with(p) {
            return format!("docker.io/{}", &name[p.len()..]);
        }
    }
    if lower.starts_with(OCI_SCHEME) {
        return name[OCI_SCHEME.len()..].to_string();
    }
    name.to_string()
}

/// True when `reference` already pins a tag or a digest, so a
/// caller-supplied one must not be appended on top of it.
///
/// Only the final path component can carry a tag; a `:` in the first
/// component is a registry port (`localhost:5000/org/model`).
pub fn reference_has_tag(reference: &str) -> bool {
    if reference.contains('@') {
        return true;
    }
    let tail = reference.rsplit('/').next().unwrap_or(reference);
    tail.contains(':')
}

/// Derive the "org/repo" GenieX stores an llmman reference under.
///
/// Drops the registry host and any `:<tag>` / `@sha256:<hex>` suffix, so
/// `docker.io/ai/gemma3:latest` and `ghcr.io/org/model:v1` land at
/// `ai/gemma3` and `org/model`. Keeping the host out of the path means
/// the same model pulled through two mirrors shares one cache entry, and
/// keeps the name inside the "org/repo" shape
/// [`crate::validation::validate_model_name`] enforces.
///
/// A deeper path (`public.ecr.aws/team/sub/model`) keeps only its last
/// two components, since the store layout is exactly two levels deep.
///
/// A single-component name gets the `ai/` org rather than the
/// `qualcomm/` that [`canonicalize_model_name`] would apply — llmman
/// expands a bare name to Docker Hub's `ai/` namespace (the same rule
/// `docker model pull` uses), and borrowing the AI Hub org here would
/// collide `geniex pull gemma3 --model-hub llmman` with the genuine AI
/// Hub model of that name. (A user alias in llmman's `shortnames.conf`
/// can redirect the reference elsewhere; the store name stays `ai/` and
/// remains stable, it just no longer describes the origin.)
pub fn llmman_store_name_from_reference(reference: &str) -> String {
    // Strip a digest first: '@' can't appear in a tag, and a digest
    // contains the ':' that the tag strip below would otherwise find.
    let without_digest = reference.split_once('@').map_or(reference, |(n, _)| n);

    let mut parts: Vec<&str> = without_digest
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    // Drop a leading registry host — the only component allowed to carry
    // a '.' or a ':' (a port). A single-component reference has no host.
    if parts.len() > 1 && (parts[0].contains('.') || parts[0].contains(':')) {
        parts.remove(0);
    }
    if parts.len() > 2 {
        parts.drain(..parts.len() - 2);
    }

    // The tag rides on the final component.
    if let Some(last) = parts.last_mut() {
        *last = last.split_once(':').map_or(*last, |(n, _)| n);
    }
    if parts.len() == 1 {
        return format!("{DOCKER_HUB_AI_ORG}/{}", parts[0]);
    }
    parts.join("/")
}

/// Docker Hub namespace llmman expands a bare model name into.
const DOCKER_HUB_AI_ORG: &str = "ai";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_alias_resolves() {
        assert_eq!(
            resolve_alias("qwen3").as_deref(),
            Some("ggml-org/Qwen3-1.7B-GGUF:Q4_K_M")
        );
    }

    #[test]
    fn unknown_alias_returns_none() {
        assert!(resolve_alias("nonexistent_model_xyz").is_none());
    }

    #[test]
    fn aihub_display_name_matches_qualcomm_org() {
        assert_eq!(
            aihub_display_name_from_repo("qualcomm/Qwen3-4B"),
            Some("Qwen3-4B")
        );
    }

    #[test]
    fn aihub_display_name_matches_ai_hub_models_org() {
        assert_eq!(
            aihub_display_name_from_repo("ai-hub-models/Llama-v3.2-3B-Chat"),
            Some("Llama-v3.2-3B-Chat")
        );
    }

    #[test]
    fn aihub_display_name_rejects_retired_aihub_prefix() {
        assert!(aihub_display_name_from_repo("aihub/llama_v3_2_3b_instruct").is_none());
    }

    #[test]
    fn aihub_display_name_case_insensitive_on_org() {
        assert_eq!(
            aihub_display_name_from_repo("Qualcomm/Qwen3-4B"),
            Some("Qwen3-4B")
        );
        assert_eq!(
            aihub_display_name_from_repo("AI-Hub-Models/Llama-v3.2-3B-Chat"),
            Some("Llama-v3.2-3B-Chat")
        );
    }

    #[test]
    fn aihub_display_name_strips_quant_suffix() {
        assert_eq!(
            aihub_display_name_from_repo("qualcomm/Qwen3-4B:Q4_K_M"),
            Some("Qwen3-4B")
        );
    }

    #[test]
    fn aihub_display_name_rejects_hf_orgs() {
        assert!(aihub_display_name_from_repo("ggml-org/Qwen3-0.6B-GGUF").is_none());
        assert!(aihub_display_name_from_repo("bartowski/Foo").is_none());
    }

    #[test]
    fn aihub_display_name_rejects_non_org_repo() {
        assert!(aihub_display_name_from_repo("qualcomm").is_none());
        assert!(aihub_display_name_from_repo("qualcomm/").is_none());
        assert!(aihub_display_name_from_repo("/Qwen3-4B").is_none());
        assert!(aihub_display_name_from_repo("ai-hub-models/").is_none());
    }

    #[test]
    fn canonicalize_bare_name_routes_to_qualcomm() {
        assert_eq!(
            canonicalize_model_name("llama_v3_2_3b_instruct"),
            "qualcomm/llama_v3_2_3b_instruct"
        );
    }

    #[test]
    fn canonicalize_preserves_org_repo() {
        assert_eq!(
            canonicalize_model_name("qualcomm/Qwen3-4B"),
            "qualcomm/Qwen3-4B"
        );
        assert_eq!(
            canonicalize_model_name("ggml-org/Qwen3-1.7B-GGUF"),
            "ggml-org/Qwen3-1.7B-GGUF"
        );
    }

    #[test]
    fn canonicalize_strips_huggingface_url_prefix() {
        assert_eq!(
            canonicalize_model_name("https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF"),
            "ggml-org/Qwen3-1.7B-GGUF"
        );
        assert_eq!(
            canonicalize_model_name("http://huggingface.co/bartowski/Foo"),
            "bartowski/Foo"
        );
    }

    #[test]
    fn llmman_reference_requires_an_explicit_registry_prefix() {
        assert!(is_llmman_reference("docker.io/ai/gemma3"));
        assert!(is_llmman_reference("DOCKER.IO/ai/gemma3"));
        assert!(is_llmman_reference("index.docker.io/ai/gemma3"));
        assert!(is_llmman_reference("ghcr.io/org/model:v1"));
        assert!(is_llmman_reference("quay.io/org/model"));
        assert!(is_llmman_reference("public.ecr.aws/team/model"));
        assert!(is_llmman_reference("oci://registry.internal/org/model"));
        assert!(is_llmman_reference("https://hub.docker.com/r/ai/gemma3"));
        // Bare "org/repo" is never auto-routed — it's indistinguishable
        // from a HuggingFace repo of the same shape.
        assert!(!is_llmman_reference("ai/gemma3"));
        assert!(!is_llmman_reference("ggml-org/Qwen3-1.7B-GGUF"));
        // HuggingFace stays on GenieX's own native puller.
        assert!(!is_llmman_reference("https://huggingface.co/org/repo"));
    }

    #[test]
    fn llmman_reference_keeps_the_registry_host() {
        // Kept verbatim: stripping the host would make llmman's own
        // short-name rule expand "ai/gemma3" to "hf.co/ai/gemma3".
        assert_eq!(
            llmman_reference_from_name("docker.io/ai/gemma3"),
            "docker.io/ai/gemma3"
        );
        assert_eq!(
            llmman_reference_from_name("ghcr.io/org/model:v1"),
            "ghcr.io/org/model:v1"
        );
        // Web-UI URL → the registry host llmman actually talks to.
        assert_eq!(
            llmman_reference_from_name("https://hub.docker.com/r/ai/gemma3"),
            "docker.io/ai/gemma3"
        );
        assert_eq!(
            llmman_reference_from_name("hub.docker.com/r/ai/gemma3"),
            "docker.io/ai/gemma3"
        );
        // An oci:// scheme is a GenieX-side marker, not part of the ref.
        assert_eq!(
            llmman_reference_from_name("oci://registry.internal/org/model:v1"),
            "registry.internal/org/model:v1"
        );
        // Explicit --model-hub llmman with a bare name: passed through
        // so llmman's own defaulting applies.
        assert_eq!(llmman_reference_from_name("ai/gemma3"), "ai/gemma3");
        assert_eq!(llmman_reference_from_name("gemma3"), "gemma3");
    }

    #[test]
    fn reference_has_tag_ignores_a_registry_port() {
        assert!(reference_has_tag("docker.io/ai/gemma3:latest"));
        assert!(reference_has_tag("docker.io/ai/gemma3@sha256:abc"));
        assert!(!reference_has_tag("docker.io/ai/gemma3"));
        // A port in the host is not a tag.
        assert!(!reference_has_tag("localhost:5000/org/model"));
        assert!(reference_has_tag("localhost:5000/org/model:dev"));
    }

    #[test]
    fn llmman_store_name_drops_host_and_tag() {
        assert_eq!(
            llmman_store_name_from_reference("docker.io/ai/gemma3:latest"),
            "ai/gemma3"
        );
        assert_eq!(
            llmman_store_name_from_reference("ghcr.io/org/model:v1.2.3"),
            "org/model"
        );
        // A registry with an explicit port is still a host.
        assert_eq!(
            llmman_store_name_from_reference("localhost:5000/org/model:dev"),
            "org/model"
        );
        // Digest references: '@sha256:…' must not be mistaken for a tag.
        assert_eq!(
            llmman_store_name_from_reference("docker.io/ai/gemma3@sha256:abc123"),
            "ai/gemma3"
        );
        // Deeper paths keep only the last two components, matching the
        // two-level store layout.
        assert_eq!(
            llmman_store_name_from_reference("public.ecr.aws/team/sub/model:1"),
            "sub/model"
        );
        // No host at all (explicit --model-hub llmman).
        assert_eq!(llmman_store_name_from_reference("ai/gemma3"), "ai/gemma3");
    }

    #[test]
    fn llmman_bare_name_uses_the_ai_org_not_qualcomm() {
        // llmman expands a bare name to Docker Hub's ai/ namespace, so
        // `gemma3` and `docker.io/ai/gemma3` share one store entry...
        assert_eq!(
            llmman_store_name_from_reference("gemma3:latest"),
            "ai/gemma3"
        );
        assert_eq!(
            llmman_store_name_from_reference("docker.io/ai/gemma3:latest"),
            "ai/gemma3"
        );
        // ...and, critically, it must not land in the AI Hub org, which
        // would collide with the genuine `qualcomm/gemma3`.
        assert_ne!(
            llmman_store_name_from_reference("gemma3"),
            canonicalize_model_name("gemma3")
        );
    }

    #[test]
    fn canonicalize_rewrites_ai_hub_models_to_qualcomm() {
        assert_eq!(
            canonicalize_model_name("ai-hub-models/Llama-v3.2-3B-Chat"),
            "qualcomm/Llama-v3.2-3B-Chat"
        );
        // Case-insensitive on the org segment.
        assert_eq!(
            canonicalize_model_name("AI-Hub-Models/Qwen3-4B"),
            "qualcomm/Qwen3-4B"
        );
        // The ":quant" suffix rides along on the repo untouched.
        assert_eq!(
            canonicalize_model_name("ai-hub-models/Qwen3-4B:Q4_K_M"),
            "qualcomm/Qwen3-4B:Q4_K_M"
        );
    }
}
