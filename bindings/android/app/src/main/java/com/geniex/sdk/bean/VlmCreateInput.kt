package com.geniex.sdk.bean

data class VlmCreateInput(
    /**
     * Optional model identifier. The QAIRT plugin reads `metadata.json` from
     * the bundle directory directly; llama_cpp uses this only for the
     * gpt-oss `geniex_resolve_device` override.
     */
    val model_name: String? = null,
    override val model_path: String,
    val mmproj_path: String? = null,
    override val config: ModelConfig,
    /**
     * [PluginIdValue] to use for the model
     */
    override val plugin_id: String? = null,
    /**
     * Device alias. `null` selects the plugin default ([DeviceIdValue.HYBRID]
     * for `llama_cpp`, [DeviceIdValue.NPU] for `qairt`).
     */
    override val device_id: String? = null,
) : CreateInputBase
