package com.geniex.sdk.bean

data class CVCreateInput(
    val model_name: String,
    val config: CVModelConfig,
    /**
     * [PluginIdValue] to use for the model
     */
    override val plugin_id: String,
    /**
     * Device to use for the model, NULL for default device.
     * When using the [PluginIdValue.LLAMA_CPP] plugin, The default value is [DeviceIdValue.CPU],
     * you can use either the [DeviceIdValue.GPU] or the [DeviceIdValue.NPU].
     */
    override val device_id: String? = null,
    val license_id: String? = null,
    val license_key: String? = null
): InputPluginBase