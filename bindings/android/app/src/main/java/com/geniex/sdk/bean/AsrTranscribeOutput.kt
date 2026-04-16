package com.geniex.sdk.bean

data class AsrTranscribeOutput(
    val result: AsrResult, // 识别结果
    val profileData: ProfilingData, // 性能数据
)

data class AsrResult(
    val transcript: String?, // 识别文本
    val confidenceScores: List<Float>? = null, // 置信度数组
    val timestamps: List<Float>? = null, // 时间戳数组
) {
    constructor(transcript: String) : this(transcript, null, null)
}

