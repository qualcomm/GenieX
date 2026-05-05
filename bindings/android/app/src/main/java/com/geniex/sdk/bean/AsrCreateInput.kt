package com.geniex.sdk.bean

data class AsrCreateInput(
    /** Name of the model */
    val model_name: String?,
    /** Path to the model file */
    override val model_path: String,
    /** Path to the tokenizer file (may be NULL) */
    val tokenizer_path: String? = null,
    /** Model configuration */
    override val config: ModelConfig,
    /** Language code (ISO 639-1 or NULL) */
    val language: String? = null,
    /**
     * [PluginIdValue] to use for the model
     */
    override val plugin_id: String?,
    /**
     * Device alias. `null` selects the plugin default ([DeviceIdValue.HYBRID]
     * for `llama_cpp`, [DeviceIdValue.NPU] for `qairt`).
     */
    override val device_id: String? = null,
    /** licence id for loading NPU models, must be provided upon the first use of the license
    key. null terminated string */
    val license_id: String? = null,
    /** licence key for loading NPU models, null terminated string */
    val license_key: String? = null
): CreateInputBase

