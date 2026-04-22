package com.geniex.sdk.bean

data class LlmCreateInput(
    val model_name: String,
    override val model_path: String,
    val tokenizer_path: String? = null,
    override val config: ModelConfig,
    /**
     * [PluginIdValue] to use for the model
     */
    override val plugin_id: String? = null,
    /**
     * Device to use for the model, NULL for default device.
     * When using the [PluginIdValue.LLAMA_CPP] plugin, The default value is [DeviceIdValue.CPU],
     * you can use either the [DeviceIdValue.GPU] or the [DeviceIdValue.NPU].
     */
    override val device_id: String?= null,
) : CreateInputBase
