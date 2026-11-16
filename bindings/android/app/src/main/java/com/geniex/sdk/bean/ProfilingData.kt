package com.geniex.sdk.bean

data class ProfilingData(
    val ttftMs: Double,  /* Time to first token (ms) */
    val mediaMs: Double,  /* Image/audio encoder time (ms); 0 for text-only runs */
    val promptTimeMs: Double,  /* Prefill time (ms); includes media-token prefill, excludes encoder */
    val decodeTimeMs: Double,   /* Token generation time (ms) */
    val promptTokens: Long,    /* Prompt tokens evaluated in prefill (text + media tokens) */
    val totalPromptTokens: Long,  /* Full requested prompt tokens, including those reused from the KV cache */
    val generatedTokens: Long,  /* Number of generated tokens */
    val prefillSpeed: Double,   /* Prefill speed (tokens/sec) */
    val decodingSpeed: Double,  /* Decoding speed (tokens/sec) */
    val draftNTotal: Long,       /* Speculative decoding: draft tokens generated (0 when disabled) */
    val draftNAccepted: Long,    /* Speculative decoding: draft tokens accepted by the target model */
    val stopReason: String  /* Stop reason: "eos", "length", "user", "stop_sequence" */
)
