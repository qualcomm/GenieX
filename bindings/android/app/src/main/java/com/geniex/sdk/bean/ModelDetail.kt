package com.geniex.sdk.bean

/** Mirrors `geniex_ModelDetail`. Full metadata for one cached model. */
data class ModelDetail(
    val name: String,
    val model_name: String,
    val plugin_id: String,
    val model_type: ModelType,
    val total_size: Long,
    val precisions: Array<String>,
)
