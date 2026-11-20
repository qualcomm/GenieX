package com.geniex.sdk.bean

/**
 * Configuration for sampling strategy when generating responses.
 * Matches the C++ struct `ml_SamplerConfig`.
 *
 * Numeric fields default to null, which means "defer to the bundle/plugin
 * default". Any non-null value, 0 included, is used as given (temperature 0 = greedy).
 */
data class SamplerConfig(
    /** Sampling temperature (0.0 - 2.0). Higher → more random. */
    var temperature: Float? = null,

    /** Top-p nucleus sampling (0.0 - 1.0). */
    var topP: Float? = null,

    /** Top-k sampling. Limits to the k most probable tokens. */
    var topK: Int? = null,

    /** Minimum probability for nucleus sampling (0.0 - 1.0). */
    var minP: Float? = null,

    /** Repetition penalty (>1.0 discourages repetitions). */
    var repetitionPenalty: Float? = null,

    /** Presence penalty (encourages new tokens). */
    var presencePenalty: Float? = null,

    /** Frequency penalty (discourages frequent tokens). */
    var frequencyPenalty: Float? = null,

    /** Random seed. Set the same seed for reproducible outputs. */
    var seed: Int? = null,

    /** Grammar file path (optional). BNF-like grammar to constrain outputs. */
    var grammarPath: String? = null,

    /** Grammar content string (optional). Takes priority over `grammarPath`. */
    var grammarString: String? = null,

    /** Grammar metadata returned by `applyChatTemplate` (optional). */
    var chatTemplateGrammar: ChatTemplateGrammar? = null
)
