package com.geniex.sdk.bean

interface CreateInputBase : InputPluginBase {
    val model_path: String?
    val config: ModelConfig
    /**
     * [PluginIdValue] to use for the model
     */
    override val plugin_id: String?
    /**
     * Device alias. `null` selects the plugin default ([DeviceIdValue.HYBRID]
     * for `llama_cpp`, [DeviceIdValue.NPU] for `qairt`).
     */
    override val device_id: String?
}