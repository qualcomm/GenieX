package com.geniex.sdk

import android.util.Log
import com.geniex.sdk.bean.ChatMessage
import com.geniex.sdk.bean.GenerationConfig
import com.geniex.sdk.bean.LLMTokenCallback
import com.geniex.sdk.bean.LlmApplyChatTemplateOutput
import com.geniex.sdk.bean.LlmCreateInput
import com.geniex.sdk.bean.LlmGenerateResult
import com.geniex.sdk.bean.LlmStreamResult
import com.geniex.sdk.jni.Llm
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import java.io.Closeable

// LlmWrapper - provides high-level API for LLM operations with coroutine support
class LlmWrapper private constructor(
    private val llmCreateInput: LlmCreateInput,
    private val dispatcher: CoroutineDispatcher
) : Closeable {

    // Native LLM bridge instance
    private val llm = Llm()
    private var handle: Long = 0

    companion object {
        @JvmStatic
        fun builder() = Builder()
    }

    class Builder {
        private var llmCreateInput: LlmCreateInput? = null
        private var dispatcher: CoroutineDispatcher = Dispatchers.IO

        fun llmCreateInput(llmCreateInput: LlmCreateInput) =
            apply { this.llmCreateInput = llmCreateInput }

        fun dispatcher(dispatcher: CoroutineDispatcher) = apply { this.dispatcher = dispatcher }

        // Build the LlmWrapper instance and initialize the native handle
        suspend fun build(): Result<LlmWrapper> = withContext(dispatcher) {
            try {
                val input = llmCreateInput ?: throw IllegalArgumentException("modelPath required")
                val wrapper = LlmWrapper(input, dispatcher)
                wrapper.handle = wrapper.llm.create(input)
                Result.success(wrapper)
            } catch (e: Exception) {
                Result.failure(e)
            }
        }
    }

    /**
     * Formats a chat prompt using the model's chat template.
     *
     * @param messages Chat message array expected by the native side.
     * @param tools Optional tool JSON; pass null for no tools.
     * @param enableThinking Enables “thinking” mode if supported by the backend.
     */
    suspend fun applyChatTemplate(
        messages: Array<ChatMessage>,
        tools: String?,
        enableThinking: Boolean,
        addGenerationPrompt: Boolean = true
    ): Result<LlmApplyChatTemplateOutput> =
        withContext(dispatcher) {
            if (handle == 0L) {
                return@withContext Result.failure(IllegalStateException("Llm not initialized"))
            }
            runCatching { llm.applyChatTemplate(handle, messages, tools, enableThinking, addGenerationPrompt) }
        }

    // Generate tokens with streaming using Kotlin Flow
    fun generateStreamFlow(
        prompt: String,
        config: GenerationConfig
    ): Flow<LlmStreamResult> = callbackFlow {
        withContext(dispatcher) {
            val callback = object : LLMTokenCallback {
                override fun onToken(token: String): Boolean {
                    trySend(LlmStreamResult.Token(token))
                    return true
                }

                override fun onComplete(result: LlmGenerateResult) {
                    trySend(LlmStreamResult.Completed(result.profileData))
                    close()
                }
            }
            try {
                val result = llm.generate(handle, prompt, config, callback)
                // FIXME: 当有 callback 时这里返回 result = null
                Log.d("nfl", "llm result:$result")
            } catch (e: Exception) {
                // generate() failed
                // TODO: 这里应该处理返回的 code
                Log.d("nfl", "llm generate failed code:${e.message}")
                trySend(LlmStreamResult.Error(e))
                close()
            }
        }
        awaitClose {
            close()
        }
    }

    override fun close() {
        destroy()
    }

    /** Resets the LLM state. */
    suspend fun reset(): Int = withContext(dispatcher) { llm.reset(handle) }

    // Stop streaming generation
    suspend fun stopStream() = withContext(dispatcher) {
        runCatching { llm.stopStream(handle) }
    }

    // Clean up native resources
    fun destroy(): Int {
        var result = 0
        if (handle != 0L) {
            result = llm.destroy(handle)
            handle = 0L
        }
        return result
    }
}