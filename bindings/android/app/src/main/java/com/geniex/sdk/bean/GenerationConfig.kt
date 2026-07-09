package com.geniex.sdk.bean

data class GenerationConfig(
    var maxTokens: Int = 32,
    var stopWords: Array<String>? = null,
    var stopCount: Int = 0,
    var nPast: Int = 0,
    var samplerConfig: SamplerConfig? = null,

    var imagePaths: Array<String>? = null,
    var imageCount: Int = 0,
    var audioPaths: Array<String>? = null,
    var audioCount: Int = 0,

    // Opt-in ring-buffer context eviction (qairt only).
    var slidingWindow: Boolean = false,
    var slidingWindowNKeep: Int = 0 // 0 = plugin default (4)
)
