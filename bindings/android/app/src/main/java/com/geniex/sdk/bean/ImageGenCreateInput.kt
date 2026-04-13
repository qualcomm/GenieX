package com.geniex.sdk.bean

data class ImageGenCreateInput(
    /** Name of the model */
    val model_name: String,
    /** Path to the model file */
    val model_path: String,
    /** Model configuration */
    val config: ModelConfig? = ModelConfig(),
    /** Path to the scheduler config file */
    val scheduler_config_path: String? = null,
    /** Plugin to use for the model */
    val plugin_id: String? = null,
    /** Device to use for the model, NULL for default device */
    val device_id: String? = null
)
