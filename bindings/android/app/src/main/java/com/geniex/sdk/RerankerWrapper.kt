package com.geniex.sdk

import com.geniex.sdk.bean.RerankConfig
import com.geniex.sdk.bean.RerankerCreateInput
import com.geniex.sdk.bean.RerankerOutputResult
import com.geniex.sdk.jni.Reranker
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable

// RerankerWrapper - provides high-level API for reranker operations with coroutine support
class RerankerWrapper private constructor(
    private val rerankerCreateInput: RerankerCreateInput,
    private val dispatcher: CoroutineDispatcher
) : Closeable {

    private val reranker = Reranker()
    private var handle: Long = 0

    companion object {
        @JvmStatic
        fun builder() = Builder()
    }

    class Builder {
        private var rerankerCreateInput: RerankerCreateInput? = null
        private var dispatcher: CoroutineDispatcher = Dispatchers.IO

        fun rerankerCreateInput(rerankerCreateInput: RerankerCreateInput) =
            apply { this.rerankerCreateInput = rerankerCreateInput }

        fun dispatcher(dispatcher: CoroutineDispatcher) = apply { this.dispatcher = dispatcher }

        // Build the RerankerWrapper instance and initialize the native handle
        suspend fun build(): Result<RerankerWrapper> = withContext(dispatcher) {
            try {
                val input = rerankerCreateInput ?: throw IllegalArgumentException("rerankerCreateInput required")
                val wrapper = RerankerWrapper(input, dispatcher)
                wrapper.handle = wrapper.reranker.create(input)
                Result.success(wrapper)
            } catch (e: Exception) {
                Result.failure(e)
            }
        }
    }

    /**
     * @param documents Array of document texts path
     */
    suspend fun rerank(
        query: String,
        documents: Array<String>,
        config: RerankConfig
    ): Result<RerankerOutputResult> = withContext(dispatcher) {
        try {
            Result.success(reranker.rerank(handle, query, documents, config))
        }catch (e: Exception) {
            Result.failure(e)
        } as Result<RerankerOutputResult>
    }

    override fun close() {
        destroy()
    }

    /**
     * @return 0: success, else error code
     */
    fun destroy(): Int {
        var result = 0
        if (handle != 0L) {
            result = reranker.destroy(handle)
            handle = 0L
        }
        return result
    }
}