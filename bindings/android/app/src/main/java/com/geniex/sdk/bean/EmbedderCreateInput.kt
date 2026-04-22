package com.geniex.sdk.bean

/**
 * Input structure for creating an embedder instance.
 * Maps to ml_EmbedderCreateInput from ml.h
 */
data class EmbedderCreateInput(
    val model_name: String,                    // Name of the model
    /**
     * Path to the model file
     */
    override val model_path: String,
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

