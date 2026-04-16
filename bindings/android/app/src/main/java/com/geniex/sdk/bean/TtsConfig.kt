package com.geniex.sdk.bean

/**
 * TTS synthesis configuration
 */
data class TtsConfig(
    /** Voice identifier */
    val voice: String? = null,
    /** Speech speed (1.0 = normal) */
    val speed: Float = 1.0f,
    /** Random seed (-1 for random) */
    val seed: Int = -1,
    /** Output sample rate in Hz */
    val sampleRate: Int = 22050
)
