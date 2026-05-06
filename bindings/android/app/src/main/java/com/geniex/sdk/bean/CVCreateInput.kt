package com.geniex.sdk.bean

data class CVCreateInput(
    val model_name: String,
    val config: CVModelConfig,
    /**
     * [PluginIdValue] to use for the model
     */
    override val plugin_id: String,
    /**
     * Device alias. `null` selects the plugin default ([DeviceIdValue.HYBRID]
     * for `llama_cpp`, [DeviceIdValue.NPU] for `qairt`).
     */
    override val device_id: String? = null,
    val license_id: String? = null,
    val license_key: String? = null
): InputPluginBase