// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

//! Resolve a model's remote candidate quantizations without downloading.
//!
//! Reuses [`crate::source::ModelSource::plan`] — the same "what does this
//! model consist of" discovery that drives [`crate::pull`] — but stops
//! after planning, so the caller can present a precision picker before
//! committing to a download.

use std::sync::Arc;

use crate::error::Result;
use crate::manifest::ModelType;
use crate::manifest_builder::quant_sort_key;
use crate::mapping::canonicalize_model_name;
use crate::pull::{build_source, PullRequest};
use crate::source::Plan;
use crate::store::Store;
use crate::transport::{HttpTransport, ReqwestTransport};
use crate::validation::validate_model_name;

/// One quantization the source advertises for a model. Candidates in
/// [`ModelQuery::candidates`] are sorted so callers can grab the head as
/// the recommended pick.
#[derive(Debug, Clone)]
pub struct QuantCandidate {
    pub quant: String,
    pub size: i64,
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
    let source = build_source(&req, store, transport).await?;
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
    let mut candidates: Vec<QuantCandidate> = plan
        .manifest
        .model_file
        .iter()
        .map(|(quant, fi)| QuantCandidate {
            quant: quant.clone(),
            size: fi.size,
        })
        .collect();
    // Rank only: a remote listing's size, not the tag, breaks ties here.
    candidates.sort_by(|a, b| {
        quant_sort_key(&a.quant)
            .0
            .cmp(&quant_sort_key(&b.quant).0)
            .then_with(|| a.size.cmp(&b.size))
            .then_with(|| a.quant.cmp(&b.quant))
    });

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
    use crate::source::Plan;
    use crate::test_support::{file_info, manifest};

    fn candidates(quants: &[(&str, i64)]) -> Vec<String> {
        let plan = Plan {
            manifest: manifest(
                "Org/Repo",
                quants
                    .iter()
                    .map(|(q, size)| (*q, file_info(&format!("model-{q}.gguf"), *size))),
            ),
            files: Vec::new(),
        };
        let q = model_query_from_plan("Org/Repo".to_string(), plan);
        q.candidates.into_iter().map(|c| c.quant).collect()
    }

    #[test]
    fn sorts_priority_first_then_size() {
        let got = candidates(&[
            ("Q8_0", 1_800),
            ("Q5_K_M", 1_200),
            ("Q4_K_M", 1_000),
            ("Q4_0", 900),
        ]);
        assert_eq!(got, ["Q4_0", "Q4_K_M", "Q8_0", "Q5_K_M"]);
        assert!(candidates(&[]).is_empty());
    }
}
