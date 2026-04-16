package com.geniex.sdk

import com.geniex.sdk.bean.TtsCreateInput
import com.geniex.sdk.bean.TtsSynthesizeInput
import com.geniex.sdk.bean.TtsSynthesizeOutput
import com.geniex.sdk.jni.Tts
import com.geniex.sdk.utils.ModeConfigUtil
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable

/**
 * TtsWrapper - High-level API for Text-to-Speech (TTS) operations
 * Provides coroutine-based API for speech synthesis on Android
 */
class TtsWrapper private constructor(
    private val ttsCreateInput: TtsCreateInput,
    private val dispatcher: CoroutineDispatcher
) : Closeable {

    // Native TTS bridge instance
    private val tts = Tts()
    private var handle: Long = 0

    companion object {
        @JvmStatic
        fun builder() = Builder()
    }

    /**
     * Builder for creating TtsWrapper instances
     */
    class Builder {
        var ttsCreateInput: TtsCreateInput? = null
        var dispatcher: CoroutineDispatcher = Dispatchers.IO

        fun ttsCreateInput(input: TtsCreateInput) =
            apply {
                val npu_model_folder_path = ModeConfigUtil.getNpuModelFolderPath(
                    input.plugin_id,
                    input.model_path,
                    input.config
                )
                val npu_lib_folder_path = ModeConfigUtil.getNpuLibFolderPath(input.config)
                this.ttsCreateInput = input.copy(
                    config = input.config.copy(
                        npu_lib_folder_path = npu_lib_folder_path,
                        npu_model_folder_path = npu_model_folder_path
                    )
                )
            }

        fun dispatcher(dispatcher: CoroutineDispatcher) = apply { this.dispatcher = dispatcher }

        /**
         * Build the TtsWrapper instance and initialize the native handle
         * @return Result containing the wrapper or error
         */
        suspend fun build(): Result<TtsWrapper> = withContext(dispatcher) {
            try {
                val input = ttsCreateInput
                    ?: throw IllegalArgumentException("ttsCreateInput required")
                val wrapper = TtsWrapper(input, dispatcher)
                wrapper.handle = wrapper.tts.create(input)
                Result.success(wrapper)
            } catch (e: Exception) {
                Result.failure(e)
            }
        }
    }

    /**
     * Synthesize speech from text
     * @param input Synthesis input containing text and optional configuration
     * @return Result containing synthesis output with audio file path
     */
    suspend fun synthesize(input: TtsSynthesizeInput): Result<TtsSynthesizeOutput> =
        withContext(dispatcher) {
            if (handle == 0L) {
                return@withContext Result.failure(IllegalStateException("TTS not initialized"))
            }
            runCatching {
                tts.synthesize(handle, input)
                    ?: throw IllegalStateException("Synthesis failed or returned null")
            }
        }

    /**
     * Get list of available voices for this TTS model
     * @return List of voice identifiers, or null if failed
     */
    suspend fun listAvailableVoices(): List<String>? =
        withContext(dispatcher) {
            tts.listAvailableVoices(handle)
        }

    override fun close() {
        destroy()
    }

    /**
     * Clean up native resources
     * Should be called when done with the TTS model
     */
    fun destroy(): Int {
        var result = 0
        if (handle != 0L) {
            result = tts.destroy(handle)
            handle = 0L
        }
        return result
    }
}
