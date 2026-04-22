package com.geniex.sdk

import com.geniex.sdk.bean.EmbedResult
import com.geniex.sdk.bean.EmbedderCreateInput
import com.geniex.sdk.bean.EmbeddingConfig
import com.geniex.sdk.jni.Embedder
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable

// EmbedderWrapper - provides high-level API for embedder operations with coroutine support
class EmbedderWrapper private constructor(
    private val embedderCreateInput: EmbedderCreateInput,
    private val dispatcher: CoroutineDispatcher
) : Closeable {

    private val embedder = Embedder()
    private var handle: Long = 0

    companion object {
        @JvmStatic
        fun builder() = Builder()
    }

    class Builder {
        private var embedderCreateInput: EmbedderCreateInput? = null
        private var dispatcher: CoroutineDispatcher = Dispatchers.IO

        fun embedderCreateInput(embedderCreateInput: EmbedderCreateInput) =
            apply { this.embedderCreateInput = embedderCreateInput }

        fun dispatcher(dispatcher: CoroutineDispatcher) = apply { this.dispatcher = dispatcher }

        // Build the EmbedderWrapper instance and initialize the native handle
        suspend fun build(): Result<EmbedderWrapper> = withContext(dispatcher) {
            try {
                val input = embedderCreateInput
                    ?: throw IllegalArgumentException("embedderCreateInput required")
                val wrapper = EmbedderWrapper(input, dispatcher)
                wrapper.handle = wrapper.embedder.create(input)
                Result.success(wrapper)
            } catch (e: Exception) {
                Result.failure(e)
            }
        }
    }

    suspend fun embed(texts: Array<String>, config: EmbeddingConfig): Result<EmbedResult> =
        withContext(dispatcher) {
            runCatching { embedder.embed(handle, texts, config) }
        }

    suspend fun embeddingDim(): Result<Int> = withContext(dispatcher) {
        runCatching { embedder.embeddingDim(handle) }
    }

    suspend fun setLora(loraId: Int): Result<Unit> = withContext(dispatcher) {
        runCatching { embedder.setLora(handle, loraId) }
    }

    suspend fun addLora(loraPath: String): Result<Int> = withContext(dispatcher) {
        runCatching { embedder.addLora(handle, loraPath) }
    }

    suspend fun removeLora(loraId: Int): Result<Unit> = withContext(dispatcher) {
        runCatching { embedder.removeLora(handle, loraId) }
    }

    suspend fun listLoras(): Result<IntArray> = withContext(dispatcher) {
        runCatching { embedder.listLoras(handle) }
    }

    override fun close() {
        destroy()
    }

    fun destroy(): Int {
        var result = 0
        if (handle != 0L) {
            result = embedder.destroy(handle)
            handle = 0L
        }
        return result
    }
}