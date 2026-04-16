package com.geniex.sdk.bean

/**
 * Input structure for TTS synthesis
 */
data class TtsSynthesizeInput(
    /** Text to synthesize in UTF-8 encoding */
    val textUtf8: String,
    /** TTS configuration (optional) */
    val config: TtsConfig? = null,
    /** Optional output file path (NULL for auto-generated) */
    val outputPath: String? = null
)
