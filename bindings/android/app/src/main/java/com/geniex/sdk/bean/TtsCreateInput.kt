package com.geniex.sdk.bean

data class TtsCreateInput(
    /** Name of the model */
    val model_name: String?,
    /** Path to the TTS model file */
    override val model_path: String,
    /** Path to the vocoder file (optional) */
    val vocoder_path: String? = null,
    /** Model configuration */
    override val config: ModelConfig,
    /**
     * Plugin ID to use for the model
     */
    override val plugin_id: String?,
    /**
     * Device to use for the model, NULL for default device.
     */
    override val device_id: String? = null
): CreateInputBase
