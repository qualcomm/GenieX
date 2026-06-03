package com.geniex.demo.bean

import android.annotation.SuppressLint
import kotlinx.serialization.Serializable

/**
 * Demo-side description of a model that the Rust model manager can
 * download and resolve. Only the fields needed by `geniex_model_pull`
 * and the UI live here — file paths come from `ModelManagerWrapper
 * .getPaths(modelName)` after the pull completes.
 */
@SuppressLint("UnsafeOptInUsageError")
@Serializable
data class ModelData(
    /** Stable UI key (spinner selection, SharedPreferences). */
    val id: String,
    /** Human-readable label shown in the UI. */
    val displayName: String,
    /**
     * `org/repo` as the Rust hub expects. Not an alias — keep the
     * alias layer (`resolveAlias`) out of the demo for simplicity.
     */
    val modelName: String,
    /** "chat" / "llm" / "vlm". */
    val type: String? = null,
    /**
     * Bitmask of supported plugins (see [getSupportPluginIds]):
     * 0b001=cpu, 0b010=gpu, 0b100=npu.
     */
    val pluginIds: Int? = 0,
    /** Quantization hint passed to `geniex_model_pull`. */
    val quant: String? = null,
    /**
     * Hub name — matches [com.geniex.sdk.bean.HubSource] by enum name.
     * Default `AUTO` lets Rust pick based on `modelName`.
     */
    val hub: String? = "AUTO",
    /** AI Hub display_name; required when [hub] is `AIHUB`. */
    val aiHubDisplayName: String? = null,
    /**
     * Target chipset for AI Hub pulls (e.g. "SM8650"). On Android the
     * Rust side has no auto-detect, so `aiHubDisplayName` entries must
     * pair with an explicit `chipset`.
     */
    val chipset: String? = null,
) {
    var isSupport = true
}

/**
 * Expands the [ModelData.pluginIds] bitmask into plugin ids the UI
 * shows in the plugin-picker dialog.
 */
fun ModelData.getSupportPluginIds(): ArrayList<String> {
    val out = arrayListOf<String>()
    val mask = pluginIds ?: 0
    if (mask == 0) {
        out.add("cpu")
    } else {
        if (mask and 0b100 == 0b100) out.add("npu")
        if (mask and 0b010 == 0b010) out.add("gpu")
        if (mask and 0b001 == 0b001) out.add("cpu")
    }
    return out
}

/** True when the model is NPU-first (bitmask has the npu bit set). */
fun ModelData.isNpuModel(): Boolean = ((pluginIds ?: 0) and 0b100) == 0b100
