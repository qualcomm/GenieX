package com.geniex.sdk.bean

data class AsrTranscribeInput(
    val audioPath: String, // 音频文件路径
    val language: String? = null, // 语言（可选）
    val config: AsrConfig? = null // 推理配置（可选）
)

data class AsrConfig(
    val timestamps: String? = null, // Timestamp mode: "none", "segment", "word"
    val beamSize: Int? = null, // Beam size for decoding
    val stream: Boolean = false // Whether to use streaming mode
)
