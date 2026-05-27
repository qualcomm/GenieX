package com.geniex.sdk.bean;

/**
 * Configuration for sampling strategy when generating responses.
 * Matches the C++ struct `ml_SamplerConfig`.
 *
 * Numeric fields default to 0, which means "defer to the bundle/plugin
 * default". Pass a non-zero value to override.
 */
data class SamplerConfig(
    /** Sampling temperature (0.0 - 2.0). Higher → more random. */
    var temperature: Float = 0.0f,

    /** Top-p nucleus sampling (0.0 - 1.0). */
    var topP: Float = 0.0f,

    /** Top-k sampling. Limits to the k most probable tokens. */
    var topK: Int = 0,

    /** Minimum probability for nucleus sampling (0.0 - 1.0). */
    var minP: Float = 0.0f,

    /** Repetition penalty (>1.0 discourages repetitions). */
    var repetitionPenalty: Float = 0.0f,

    /** Presence penalty (encourages new tokens). */
    var presencePenalty: Float = 0.0f,

    /** Frequency penalty (discourages frequent tokens). */
    var frequencyPenalty: Float = 0.0f,

    /** Random seed. Set the same non-zero seed for reproducible outputs. */
    var seed: Int = 0,

    /** Grammar file path (optional). BNF-like grammar to constrain outputs. */
    var grammarPath: String? = null,

    /** Grammar content string (optional). Takes priority over `grammarPath`. */
    var grammarString: String? = null
)
