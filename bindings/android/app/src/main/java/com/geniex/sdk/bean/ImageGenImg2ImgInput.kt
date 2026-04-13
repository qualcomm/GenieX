package com.geniex.sdk.bean

data class ImageGenImg2ImgInput(
    /** Path to initial image file for img2img */
    val init_image_path: String,
    /** Text prompt in UTF-8 encoding */
    val prompt_utf8: String,
    /** Image generation configuration */
    val config: ImageGenerationConfig,
    /** Optional output file path (NULL for auto-generated) */
    val output_path: String
)
