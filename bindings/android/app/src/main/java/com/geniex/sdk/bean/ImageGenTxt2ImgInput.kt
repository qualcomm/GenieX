package com.geniex.sdk.bean

data class ImageGenTxt2ImgInput(
    /** Text prompt in UTF-8 encoding */
    val prompt_utf8: String,
    /** Image generation configuration */
    val config: ImageGenerationConfig,
    /** Optional output file path (NULL for auto-generated) */
    val output_path: String?
)
