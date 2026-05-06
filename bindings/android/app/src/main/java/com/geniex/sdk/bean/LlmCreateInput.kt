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
     * Device alias. `null` selects the plugin default ([DeviceIdValue.HYBRID]
     * for `llama_cpp`, [DeviceIdValue.NPU] for `qairt`). Use
     * [DeviceIdValue.CPU] / [DeviceIdValue.GPU] / [DeviceIdValue.NPU] /
     * [DeviceIdValue.HYBRID] to pin a specific backend.
     */
    override val device_id: String?= null,
) : CreateInputBase
