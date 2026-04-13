package com.geniex.sdk.bean

data class EmbeddingConfig(
    var batchSize: Int = 32,
    var normalize: Boolean = true,
    var normalizeMethod: String = "l2"
)