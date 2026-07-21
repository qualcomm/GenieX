package com.geniex.sdk.bean

/**
 * Model configuration corresponding to the native `geniex_ModelConfig` struct.
 */
data class ModelConfig(
    /** Text context size, 0 = use model default */
    var nCtx: Int = 2048,

    /** Number of threads used for text generation */
    var nThreads: Int = 8,

    /** Number of threads used for batch processing */
    var nThreadsBatch: Int = 8,

    /** Maximum logical batch size submitted to llama_decode */
    var nBatch: Int = 2048,

    /** Maximum physical batch size supported by backend */
    var nUBatch: Int = 512,

    /** Maximum number of distinct sequences (states) */
    var nSeqMax: Int = 1,

    /**
     * Number of layers to offload to GPU / NPU; -1 (the default) means all
     * layers, which llama.cpp interprets natively. The JNI layer forces 0
     * when [InputPluginBase.compute_unit] is [ComputeUnitValue.CPU] (and for
     * qairt, which ignores it). For [ComputeUnitValue.GPU] / [ComputeUnitValue.NPU]
     * / [ComputeUnitValue.HYBRID] the caller's value passes through.
     */
    var nGpuLayers: Int = -1,

    /** Path to the chat template file (optional) */
    val chat_template_path: String = "",

    /** Content of the chat template file (optional) */
    val chat_template_content: String = "",

    /** Maximum number of tokens to generate */
    val max_tokens: Int = 2048,

    /** Enable "thinking" mode for more detailed reasoning */
    val enable_thinking: Boolean = false,

    val verbose: Boolean = false,

    /** Speculative decoding type(s), comma-separated (llama_cpp only; "" / "none" = disabled).
     *  Draft models: "draft-mtp","draft-eagle3","draft-simple". Self-speculative: "ngram-*". */
    val spec_type: String = "",

    /** Draft GGUF for draft-* spec types ("" for ngram-* or when disabled) */
    val spec_draft_model: String = "",

    /** Max draft tokens per step (0 = plugin default of 3) */
    val spec_n_max: Int = 0,

    /** Min draft tokens per step (0 = llama.cpp default) */
    val spec_n_min: Int = 0,

    /** Min greedy draft probability (0 = llama.cpp default) */
    val spec_p_min: Float = 0.0f,
)
