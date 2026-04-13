package com.geniex.sdk.bean

/**
 * Output structure for TTS synthesis
 */
data class TtsSynthesizeOutput(
    /** Synthesis result with audio saved to filesystem */
    val result: TtsResult,
    /** Profiling data for the synthesis operation */
    val profileData: ProfilingData
)
