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
/// Lowercased; lookup is case-insensitive. Mirrors the Go CLI's
/// `aiHubOrgs` — keep the two lists in sync when adding an org.
///
/// `aihub` is the canonical prefix we use to store bare AI Hub model
/// names after [`canonicalize_model_name`] normalises them.
const AI_HUB_ORGS: &[&str] = &["qualcomm", "qai-hub-models", "aihub"];

/// Canonicalise a user-supplied model name into the "org/repo" shape
/// that [`crate::validation::validate_model_name`] expects.
///
/// A name without '/' is assumed to be a bare AI Hub model id
/// (e.g. `llama_v3_2_3b_instruct`) and is rewritten to `aihub/<name>`.
/// Anything that already contains '/' is returned unchanged.
///
/// This is the single entry point callers should use before handing a
/// name to `pull` / `get_paths` so the Store layout stays consistent.
pub fn canonicalize_model_name(name: &str) -> String {
    if name.contains('/') {
        name.to_string()
    } else {
        format!("aihub/{name}")
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
    if AI_HUB_ORGS.contains(&org_lower.as_str()) {
        Some(repo)
    } else {
        None
    }
}

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
    fn aihub_display_name_matches_qai_hub_models_org() {
        assert_eq!(
            aihub_display_name_from_repo("qai-hub-models/Phi-3.5-Mini-Instruct"),
            Some("Phi-3.5-Mini-Instruct")
        );
    }

    #[test]
    fn aihub_display_name_case_insensitive_on_org() {
        assert_eq!(
            aihub_display_name_from_repo("Qualcomm/Qwen3-4B"),
            Some("Qwen3-4B")
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
    }

    #[test]
    fn aihub_display_name_matches_aihub_prefix() {
        assert_eq!(
            aihub_display_name_from_repo("aihub/llama_v3_2_3b_instruct"),
            Some("llama_v3_2_3b_instruct")
        );
    }

    #[test]
    fn aihub_display_name_matches_aihub_prefix_case_insensitive() {
        assert_eq!(aihub_display_name_from_repo("AIHub/Foo"), Some("Foo"));
    }

    #[test]
    fn aihub_display_name_strips_quant_on_aihub_prefix() {
        assert_eq!(
            aihub_display_name_from_repo("aihub/Foo:Q4_K_M"),
            Some("Foo")
        );
    }

    #[test]
    fn canonicalize_bare_name_routes_to_aihub() {
        assert_eq!(
            canonicalize_model_name("llama_v3_2_3b_instruct"),
            "aihub/llama_v3_2_3b_instruct"
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
    fn canonicalize_preserves_existing_aihub_prefix() {
        assert_eq!(canonicalize_model_name("aihub/foo"), "aihub/foo");
    }
}
