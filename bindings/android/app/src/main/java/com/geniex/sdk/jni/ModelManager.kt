package com.geniex.sdk.jni

import com.geniex.sdk.bean.ChipsetInfo
import com.geniex.sdk.bean.HubModel
import com.geniex.sdk.bean.ModelDetail
import com.geniex.sdk.bean.ModelPaths
import com.geniex.sdk.bean.ModelPullInput
import com.geniex.sdk.bean.ModelQuery
import com.geniex.sdk.callback.DownloadProgressCallback

/**
 * Thin Kotlin shim over the `geniex_model_*` C FFI. Functions are
 * instance methods but operate on a process-global store (matches the
 * FFI's singleton). Prefer `ModelManagerWrapper` from app code.
 */
internal class ModelManager {
    /**
     * @return 0 on success, the FFI error code otherwise.
     *   `GENIEX_ERROR_COMMON_ALREADY_INITIALIZED` (-100008) if the
     *   store has already been initialized.
     */
    external fun init(dataDir: String): Int

    external fun deinit(): Int

    /**
     * Human-readable message for the most recent failing `geniex_model_*`
     * call on the calling thread, or `null` if none recorded yet. Valid
     * until the next FFI call on this thread.
     */
    external fun lastErrorMessage(): String?

    /**
     * Blocking. Invokes [callback] periodically (~100 ms) from a tokio
     * worker thread that the bridge attaches to the JVM. Returning
     * `false` from the callback cancels; the FFI returns
     * `GENIEX_ERROR_COMMON_CANCELLED` (-100006).
     */
    external fun pull(input: ModelPullInput, callback: DownloadProgressCallback?): Int

    external fun list(): Array<String>

    /** Cached models with full metadata (size, plugin, type, precisions). */
    external fun listDetailed(): Array<ModelDetail>

    /**
     * One cached model's metadata, without listing the whole store.
     * [modelName] takes the same loose forms as [getPaths]: a bare AI Hub id is
     * canonicalized to `qualcomm/<id>`, and any `:<precision>` suffix is
     * ignored since the detail covers every downloaded precision.
     * @return `null` if the model is not cached.
     */
    external fun getDetailed(modelName: String): ModelDetail?

    /**
     * Resolve a model's remote candidate quantizations without downloading.
     * @return `null` if the model cannot be planned.
     */
    external fun query(input: ModelPullInput): ModelQuery?

    /** @return `null` if the model is not cached or paths cannot be resolved. */
    external fun getPaths(modelName: String): ModelPaths?

    external fun remove(modelName: String): Int

    external fun clean(): Int

    /**
     * @return `geniex_ModelType` as an ordinal (0 = LLM, 1 = VLM), or a
     *   negative FFI error code.
     */
    external fun getType(modelName: String): Int

    /**
     * Override the stored model type.
     * @param modelType `geniex_ModelType` ordinal (0 = LLM, 1 = VLM).
     * @return 0 on success, the FFI error code otherwise.
     */
    external fun setType(modelName: String, modelType: Int): Int

    external fun resolveAlias(alias: String): String?

    /** Chipsets Qualcomm AI Hub supports, with aliases (sourced from platform.json). */
    external fun listChipsets(): Array<ChipsetInfo>

    /** Detect the host chipset. [offline] stays local; else may hit the network. @return `null` if not probeable. */
    external fun detectChipset(offline: Boolean): String?

    /**
     * AI Hub models with a qairt (NPU) build, sorted by name (from manifest.json).
     * @param chipset canonical chipset id to filter by, or null to list every model.
     */
    external fun listHubModels(chipset: String?): Array<HubModel>
}
