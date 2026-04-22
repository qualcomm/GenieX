package com.geniex.sdk.bean

/**
 * Input structure for creating a reranker instance.
 * Maps to ml_RerankerCreateInput from ml.h
 */
data class RerankerCreateInput(
    val model_name: String,                    // Name of the model
    override val model_path: String,           // Path to the model file
    val tokenizer_path: String? = null,        // Path to the tokenizer file (optional)
    override val config: ModelConfig,          // Model configuration (includes NPU paths)
    /**
     * [PluginIdValue] to use for the model
     */
    override val plugin_id: String? = null,    // Plugin to use for the model
    /**
     * Device to use for the model, NULL for default device.
     * When using the [PluginIdValue.LLAMA_CPP] plugin, The default value is [DeviceIdValue.CPU],
     * you can use either the [DeviceIdValue.GPU] or the [DeviceIdValue.NPU].
     */
    override val device_id: String? = null,    // Device to use for the model
) : CreateInputBase

