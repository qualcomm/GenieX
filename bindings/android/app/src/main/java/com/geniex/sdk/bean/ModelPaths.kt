package com.geniex.sdk.bean

/** Mirrors `geniex_ModelPaths`. NULL char* fields in C become `null` here. */
data class ModelPaths(
    val model_path: String,
    val model_dir: String,
    val model_name: String,
    val plugin_id: String,
    val model_type: ModelType,
    val mmproj_path: String? = null,
    val tokenizer_path: String? = null,
)
