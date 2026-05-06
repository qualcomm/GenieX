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
     * Number of layers to offload to GPU / NPU. The JNI layer overrides
     * this when [InputPluginBase.device_id] is [DeviceIdValue.CPU] (forces
     * 0) or [DeviceIdValue.HYBRID] (forces 999). For [DeviceIdValue.GPU]
     * and [DeviceIdValue.NPU] the caller's value is used as-is; set it
     * to 999 to offload every layer.
     */
    var nGpuLayers: Int = 0,

    /** Path to the chat template file (optional) */
    val chat_template_path: String = "",

    /** Content of the chat template file (optional) */
    val chat_template_content: String = "",

    /** Maximum number of tokens to generate */
    val max_tokens: Int = 2048,

    /** Enable "thinking" mode for more detailed reasoning */
    val enable_thinking: Boolean = false,

    val verbose: Boolean = false,
)
