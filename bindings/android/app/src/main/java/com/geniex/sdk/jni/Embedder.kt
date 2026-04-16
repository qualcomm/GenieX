package com.geniex.sdk.jni

import com.geniex.sdk.bean.EmbedResult
import com.geniex.sdk.bean.EmbedderCreateInput
import com.geniex.sdk.bean.EmbeddingConfig
import com.geniex.sdk.bean.ProfilingData

internal class Embedder {
    // Create embedder using EmbedderCreateInput (matches ml_EmbedderCreateInput)
    external fun create(embedderCreateInput: EmbedderCreateInput): Long
    external fun destroy(handle: Long): Int
    external fun embed(handle: Long, texts: Array<String>, config: EmbeddingConfig): EmbedResult
    external fun embeddingDim(handle: Long): Int
    external fun setLora(handle: Long, loraId: Int)
    external fun addLora(handle: Long, loraPath: String): Int
    external fun removeLora(handle: Long, loraId: Int)
    external fun listLoras(handle: Long): IntArray
    external fun getProfilingData(handle: Long, outData: ProfilingData): Int
}