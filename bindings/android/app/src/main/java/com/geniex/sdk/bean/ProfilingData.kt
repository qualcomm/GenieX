package com.geniex.sdk.bean

data class ProfilingData(
    val ttftMs: Double,  /* Time to first token (ms) */
    val promptTimeMs: Double,  /* Prompt processing time (ms) */
    val decodeTimeMs: Double,   /* Token generation time (ms) */
    val promptTokens: Long,    /* Number of prompt tokens */
    val generatedTokens: Long,  /* Number of generated tokens */
    val audioDurationMs: Long,   /* Audio duration (ms) */
    val prefillSpeed: Double,   /* Prefill speed (tokens/sec) */
    val decodingSpeed: Double,  /* Decoding speed (tokens/sec) */
    val realTimeFactor: Double,  /* Real-Time Factor(RTF) (1.0 = real-time, >1.0 = faster, <1.0 = slower) */
    val stopReason: String  /* Stop reason: "eos", "length", "user", "stop_sequence" */
)
