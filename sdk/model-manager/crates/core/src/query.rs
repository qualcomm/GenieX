//! Resolve a model's remote candidate quantizations without downloading.
//!
//! Reuses [`crate::source::ModelSource::plan`] — the same "what does this
//! model consist of" discovery that drives [`crate::pull`] — but stops
//! after planning, so the caller can present a precision picker before
//! committing to a download.

use std::sync::Arc;

use crate::error::Result;
use crate::manifest::ModelType;
use crate::mapping::canonicalize_model_name;
use crate::paths::pick_default_quant;
use crate::pull::{build_source, PullRequest};
use crate::source::Plan;
use crate::store::Store;
use crate::transport::{HttpTransport, ReqwestTransport};
use crate::validation::validate_model_name;

/// One quantization the source advertises for a model.
#[derive(Debug, Clone)]
pub struct QuantCandidate {
    pub quant: String,
    pub size: i64,
    pub is_default: bool,
}

/// Result of a plan-only query against a hub.
#[derive(Debug, Clone)]
pub struct ModelQuery {
    pub model_name: String,
    pub model_type: ModelType,
    pub plugin_id: String,
    pub candidates: Vec<QuantCandidate>,
}

/// Plan a model against its source and return the candidate quants. No
/// bytes are fetched.
pub async fn query(store: &Store, mut req: PullRequest) -> Result<ModelQuery> {
    req.model_name = canonicalize_model_name(&req.model_name);
    validate_model_name(&req.model_name)?;

    let transport: Arc<dyn HttpTransport> = Arc::new(ReqwestTransport::new()?);
    let source = build_source(&req, store, transport)?;
    let plan = source.plan().await?;
    Ok(model_query_from_plan(req.model_name, plan))
}

/// Sync entry point for callers without a runtime; drives [`query`] on a
/// caller-supplied tokio handle (the FFI crate owns the runtime).
pub fn query_blocking(
    handle: &tokio::runtime::Handle,
    store: &Store,
    req: PullRequest,
) -> Result<ModelQuery> {
    handle.block_on(query(store, req))
}

fn model_query_from_plan(model_name: String, plan: Plan) -> ModelQuery {
    let quants: Vec<&str> = plan
        .manifest
        .model_file
        .keys()
        .map(String::as_str)
        .collect();
    let default = (!quants.is_empty()).then(|| pick_default_quant(&quants).to_string());

    let mut candidates: Vec<QuantCandidate> = plan
        .manifest
        .model_file
        .iter()
        .map(|(quant, fi)| QuantCandidate {
            quant: quant.clone(),
            size: fi.size,
            is_default: Some(quant) == default.as_ref(),
        })
        .collect();
    candidates.sort_by(|a, b| b.size.cmp(&a.size).then_with(|| a.quant.cmp(&b.quant)));

    ModelQuery {
        model_name,
        model_type: plan.manifest.model_type,
        plugin_id: plan.manifest.plugin_id,
        candidates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelFileInfo, ModelManifest};
    use crate::source::Plan;
    use std::collections::HashMap;

    fn plan_with(quants: &[(&str, i64)]) -> Plan {
        let mut model_file = HashMap::new();
        for (q, size) in quants {
            model_file.insert(
                (*q).to_string(),
                ModelFileInfo {
                    name: format!("model-{q}.gguf"),
                    downloaded: true,
                    size: *size,
                },
            );
        }
        Plan {
            manifest: ModelManifest {
                name: "Org/Repo".to_string(),
                model_name: "Repo".to_string(),
                model_type: ModelType::Llm,
                plugin_id: "llama_cpp".to_string(),
                precision: String::new(),
                model_file,
                mmproj_file: ModelFileInfo::default(),
                tokenizer_file: ModelFileInfo::default(),
                extra_files: Vec::new(),
            },
            files: Vec::new(),
        }
    }

    #[test]
    fn marks_priority_default_and_sorts_by_size() {
        let q = model_query_from_plan(
            "Org/Repo".to_string(),
            plan_with(&[("Q4_0", 900), ("Q4_K_M", 1_000), ("Q8_0", 1_800)]),
        );
        // Sorted size-desc.
        assert_eq!(q.candidates[0].quant, "Q8_0");
        assert_eq!(q.candidates[2].quant, "Q4_0");
        // Q4_0 is the priority default even though it is the smallest.
        let default: Vec<&str> = q
            .candidates
            .iter()
            .filter(|c| c.is_default)
            .map(|c| c.quant.as_str())
            .collect();
        assert_eq!(default, vec!["Q4_0"]);
    }

    #[test]
    fn empty_model_file_yields_no_candidates() {
        let q = model_query_from_plan("Org/Repo".to_string(), plan_with(&[]));
        assert!(q.candidates.is_empty());
    }
}
