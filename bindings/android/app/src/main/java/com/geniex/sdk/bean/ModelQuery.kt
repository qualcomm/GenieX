package com.geniex.sdk.bean

/** Mirrors `geniex_ModelQueryOutput`. Result of a plan-only query. */
data class ModelQuery(
    val model_name: String,
    val plugin_id: String,
    val model_type: ModelType,
    val candidates: Array<QuantCandidate>,
)
