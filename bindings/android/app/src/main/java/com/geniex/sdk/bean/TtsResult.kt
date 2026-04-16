package com.geniex.sdk.bean

/**
 * TTS synthesis result
 */
data class TtsResult(
    /** Path where the synthesized audio is saved */
    val audioPath: String,
    /** Audio duration in seconds */
    val durationSeconds: Float,
    /** Audio sample rate in Hz */
    val sampleRate: Int,
    /** Number of audio channels (default: 1) */
    val channels: Int,
    /** Number of audio samples */
    val numSamples: Int
)
