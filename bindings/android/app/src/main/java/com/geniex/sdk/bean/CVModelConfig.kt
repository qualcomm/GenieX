package com.geniex.sdk.bean

data class CVModelConfig(
    val capabilities: CVCapability,
    /** Detection model path */
    val det_model_path: String,
    /** Recognition model path */
    val rec_model_path: String,
    /** Character dictionary path */
    val char_dict_path: String,
)

enum class CVCapability {
    OCR,
    CLASSIFICATION,
    SEGMENTATION,
    CUSTOM
}
