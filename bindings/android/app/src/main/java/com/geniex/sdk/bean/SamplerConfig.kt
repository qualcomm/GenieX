package com.geniex.sdk.bean;

/**
 * Configuration for sampling strategy when generating responses.
 * Matches the C++ struct `ml_SamplerConfig`.
 */
data class SamplerConfig(
    /** Sampling temperature (0.0 - 2.0).
     * Higher values → more randomness; lower values → more deterministic.
     * Typical range: 0.7 ~ 1.0 */
    var temperature: Float = 1.0f,

    /** Top-p nucleus sampling (0.0 - 1.0).
     * Selects tokens from the smallest possible set whose cumulative probability ≥ p.
     * Example: topP = 0.9 → choose from the top 90% probable tokens. */
    var topP: Float = 1.0f,

    /** Top-k sampling.
     * Limits sampling to the top-k most probable tokens.
     * Example: topK = 50 → only consider the 50 most probable tokens. */
    var topK: Int = 0,

    /** Minimum probability for nucleus sampling (0.0 - 1.0).
     * Tokens with probability lower than this threshold will be excluded. */
    var minP: Float = 0.0f,

    /** Repetition penalty (>1.0 discourages repetitions).
     * Larger values penalize tokens that were already generated,
     * preventing repetitive outputs. */
    var repetitionPenalty: Float = 1.0f,

    /** Presence penalty.
     * Increases the likelihood of generating tokens that haven't appeared yet.
     * Higher values encourage introducing new topics. */
    var presencePenalty: Float = 0.0f,

    /** Frequency penalty.
     * Penalizes tokens based on how frequently they've already appeared.
     * Helps avoid repeating words too often. */
    var frequencyPenalty: Float = 0.0f,

    /** Random seed (-1 = random).
     * Set the same seed for reproducible outputs. */
    var seed: Int = -1,

    /** Grammar file path (optional).
     * Path to an external BNF-like grammar file.
     * If provided, the model will generate outputs conforming to this grammar. */
    var grammarPath: String? = null,

    /** Grammar content string (optional).
     * Directly pass a BNF-like grammar as a string.
     * If set, this field takes priority over `grammarPath`. */
    var grammarString: String? = null
)
