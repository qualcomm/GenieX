package com.geniex.sdk.bean

/**
 * Mirrors `geniex_ModelPullInput` from geniex_model.h.
 *
 * Field order is intentional — the JNI bridge caches field IDs by name,
 * not offset, so order is purely documentary. But keeping it aligned
 * with the C struct makes the bridge easy to audit.
 */
data class ModelPullInput(
    /** "org/repo" or a short alias (see [resolveAlias]). */
    val model_name: String,
    /** Quantization hint (e.g. "Q4_K_M"). `null` lets the SDK pick. */
    val quant: String? = null,
    /** Source hub. [HubSource.AUTO] routes by model name. */
    val hub: HubSource = HubSource.AUTO,
    /** Required only when [hub] is [HubSource.LOCALFS]. */
    val local_path: String? = null,
    /**
     * HuggingFace bearer token. `null` falls back to the `GENIEX_HFTOKEN`
     * env var, then anonymous. Only consulted when [hub] is
     * [HubSource.HUGGINGFACE].
     */
    val hf_token: String? = null,
    /**
     * Target chipset for AI Hub pulls (e.g. "SM8650"). On Android there
     * is no auto-detect — callers must pass an explicit chipset when
     * [hub] is [HubSource.AIHUB].
     */
    val chipset: String? = null,
    /** AI Hub `display_name`. Required when [hub] is [HubSource.AIHUB]. */
    val display_name: String? = null,
    /**
     * Optional model-type override written into the manifest during the
     * pull. `null` auto-detects (the default); [ModelType.LLM] / [ModelType.VLM]
     * force the stored type.
     */
    val model_type: ModelType? = null,
)
