package com.geniex.sdk

import android.system.Os
import android.util.Log
import com.geniex.sdk.GeniexSdk.Companion.KEY_NPU_LIB_FOLDER_PATH
import com.geniex.sdk.bean.*
import com.geniex.sdk.jni.Vlm
import com.geniex.sdk.utils.ModeConfigUtil
import java.io.Closeable
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext

/**
 * Wrapper class around the native VLM (Vision-Language Model) interface.
 *
 * Manages native VLM lifecycle, generation requests, streaming output, and chat template
 * application.
 */
class VlmWrapper
private constructor(
        private val vlmCreateInput: VlmCreateInput,
        private val dispatcher: CoroutineDispatcher
) : Closeable {

    private val vlm = Vlm()
    private var handle: Long = 0

    companion object {
        fun builder() = Builder()
    }

    /**
     * Builder for creating a [VlmWrapper] instance.
     *
     * Example:
     * ```
     * val wrapper = VlmWrapper.builder()
     *     .vlmCreateInput(input)
     *     .dispatcher(Dispatchers.IO)
     *     .build()
     * ```
     */
    class Builder {
        private var vlmCreateInput: VlmCreateInput? = null
        private var dispatcher: CoroutineDispatcher = Dispatchers.IO

        /** Sets the VLM creation input (required). */
        fun vlmCreateInput(vlmCreateInput: VlmCreateInput) = apply {
            val npu_model_folder_path = ModeConfigUtil.getNpuModelFolderPath(
                vlmCreateInput.plugin_id,
                vlmCreateInput.model_path,
                vlmCreateInput.config
            )
            val npu_lib_folder_path = ModeConfigUtil.getNpuLibFolderPath(vlmCreateInput.config)
            this.vlmCreateInput = vlmCreateInput.copy(
                config = vlmCreateInput.config.copy(
                    npu_lib_folder_path = npu_lib_folder_path,
                    npu_model_folder_path = npu_model_folder_path
                )
            )
        }

        /** Sets the coroutine dispatcher (default = [Dispatchers.IO]). */
        fun dispatcher(dispatcher: CoroutineDispatcher) = apply { this.dispatcher = dispatcher }

        /**
         * Builds a [VlmWrapper] instance asynchronously.
         *
         * @return [Result] containing either a successfully created [VlmWrapper]
         * ```
         *         or an exception if creation fails.
         * ```
         */
        suspend fun build(): Result<VlmWrapper> =
                withContext(dispatcher) {
                    try {
                        val input =
                                vlmCreateInput
                                        ?: throw IllegalArgumentException("modelPath required")
                        val wrapper = VlmWrapper(input, dispatcher)
                        wrapper.handle = wrapper.vlm.create(input)
                        Result.success(wrapper)
                    } catch (e: Exception) {
                        Result.failure(e)
                    }
                }
    }

    /**
     * Extracts image and audio paths from VlmChatMessage contents and injects them into the config.
     * This is useful when you have images/audio embedded in chat messages.
     *
     * @param messages The chat messages containing media references
     * @param config The generation config to update
     * @return Updated GenerationConfig with media paths populated
     */
    fun injectMediaPathsToConfig(
            messages: Array<VlmChatMessage>,
            config: GenerationConfig
    ): GenerationConfig {
        val (imagePaths, audioPaths) = vlm.extractMediaPaths(messages)
        return config.copy(
                imagePaths = if (imagePaths.isNotEmpty()) imagePaths else null,
                imageCount = imagePaths.size,
                audioPaths = if (audioPaths.isNotEmpty()) audioPaths else null,
                audioCount = audioPaths.size
        )
    }

    /**
     * Generates tokens in a streaming fashion as a [Flow].
     *
     * - Uses [callbackFlow] to receive tokens from native VLM callbacks.
     * - Sends [LlmStreamResult.Token] events for each new token.
     * - Emits [LlmStreamResult.Completed] when generation finishes.
     * - Emits [LlmStreamResult.Error] if an exception occurs.
     *
     * @param prompt Input text prompt.
     * @param config Generation configuration.
     */
    fun generateStreamFlow(prompt: String, config: GenerationConfig): Flow<LlmStreamResult> =
            callbackFlow {
                withContext(dispatcher) {
                    val callback =
                            object : LLMTokenCallback {
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
                        val result = vlm.generate(handle, prompt, config, callback)
                        // FIXME: 当有 callback 时这里返回 result = null
                        Log.d("nfl", "vlm result:$result")
                    } catch (e: Exception) {
                        // generate() failed
                        // TODO: 这里应该处理返回的 code
                        Log.d("nfl", "vlm generate failed code:${e.message}")
                        trySend(LlmStreamResult.Error(e))
                        close()
                    }
                }
                awaitClose { close() }
            }

    /**
     * Applies a chat template on a set of chat messages.
     *
     * @param messages Array of [VlmChatMessage].
     * @param tools JSON string describing tools, nullable.
     * @param enableThinking Whether to enable "thinking" mode.
     *
     * @return [Result] containing [LlmApplyChatTemplateOutput].
     */
    suspend fun applyChatTemplate(
            messages: Array<VlmChatMessage>,
            tools: String?,
            enableThinking: Boolean
    ): Result<LlmApplyChatTemplateOutput> =
            withContext(dispatcher) {
                runCatching { vlm.applyChatTemplate(handle, messages, tools, enableThinking) }
            }

    /** Resets the VLM state. */
    suspend fun reset(): Int = withContext(dispatcher) { vlm.reset(handle) }

    /**
     * Stops an ongoing streaming generation if any.
     *
     * @return [Result] indicating success or failure.
     */
    suspend fun stopStream() = withContext(dispatcher) { runCatching { vlm.stopStream(handle) } }

    /** Releases native resources when closed. */
    override fun close() {
        destroy()
    }

    /** Destroys the native VLM instance and clears the handle. */
    fun destroy(): Int {
        var result = 0
        if (handle != 0L) {
            result = vlm.destroy(handle)
            handle = 0L
        }
        return result
    }
}