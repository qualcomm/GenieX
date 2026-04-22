package com.geniex.sdk

import com.geniex.sdk.bean.AsrCreateInput
import com.geniex.sdk.bean.AsrStreamBeginInput
import com.geniex.sdk.bean.AsrStreamBeginOutput
import com.geniex.sdk.bean.AsrTranscribeInput
import com.geniex.sdk.bean.AsrTranscribeOutput
import com.geniex.sdk.callback.AsrTranscriptionCallback
import com.geniex.sdk.jni.Asr
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable

/**
 * AsrWrapper - High-level API for Automatic Speech Recognition (ASR) operations
 * Provides coroutine-based API for audio transcription on Android
 * 
 * Supports both:
 * - File-based transcription via [transcribe]
 * - Real-time streaming transcription via [streamBegin], [streamPushAudio], [streamStop]
 */
class AsrWrapper private constructor(
    private val asrCreateInput: AsrCreateInput,
    private val dispatcher: CoroutineDispatcher
) : Closeable {

    // Native ASR bridge instance
    private val asr = Asr()
    private var handle: Long = 0
    
    // Track streaming state
    @Volatile
    private var isStreaming: Boolean = false

    companion object {
        @JvmStatic
        fun builder() = Builder()
    }

    /**
     * Builder for creating AsrWrapper instances
     */
    class Builder {
        var asrCreateInput: AsrCreateInput? = null
        var dispatcher: CoroutineDispatcher = Dispatchers.IO

        fun asrCreateInput(input: AsrCreateInput) =
            apply { this.asrCreateInput = input }

        fun dispatcher(dispatcher: CoroutineDispatcher) = apply { this.dispatcher = dispatcher }


        /**
         * Build the AsrWrapper instance and initialize the native handle
         * @return Result containing the wrapper or error
         */
        suspend fun build(): Result<AsrWrapper> = withContext(dispatcher) {
            try {
                val input = asrCreateInput
                    ?: throw IllegalArgumentException("asrCreateInput required")
                val wrapper = AsrWrapper(input, dispatcher)
                wrapper.handle = wrapper.asr.create(input)
                Result.success(wrapper)
            } catch (e: Exception) {
                Result.failure(e)
            }
        }
    }

    /**
     * Transcribe audio file to text
     * @param input Input containing audio file path and optional language/config
     * @return Result containing transcript output
     */
    suspend fun transcribe(input: AsrTranscribeInput): Result<AsrTranscribeOutput> =
        withContext(dispatcher) {
            if (handle == 0L) {
                return@withContext Result.failure(IllegalStateException("ASR not initialized"))
            }
            runCatching {
                asr.transcribe(handle, input)
                    ?: throw IllegalStateException("Transcription failed or returned null")
            }
        }

    suspend fun listSupportedLanguages(): List<String>? {
        return asr.listSupportedLanguages(handle)
    }

    // ==========================================================================
    // Streaming ASR API
    // ==========================================================================

    /**
     * Begin a streaming ASR session.
     * 
     * After calling this, use [streamPushAudio] to send audio data and receive
     * transcription updates via the callback. Call [streamStop] when done.
     * 
     * @param input Streaming configuration including language and callback
     * @return Result containing AsrStreamBeginOutput with status
     * 
     * @throws IllegalStateException if ASR is not initialized or already streaming
     * 
     * Example:
     * ```kotlin
     * val callback = object : AsrTranscriptionCallback {
     *     override fun onTranscription(text: String) {
     *         println("Transcription: $text")
     *     }
     * }
     * 
     * val input = AsrStreamBeginInput(
     *     language = "en",
     *     callback = callback
     * )
     * 
     * asrWrapper.streamBegin(input).onSuccess { output ->
     *     // Start pushing audio data
     * }
     * ```
     */
    suspend fun streamBegin(input: AsrStreamBeginInput): Result<AsrStreamBeginOutput> =
        withContext(dispatcher) {
            if (handle == 0L) {
                return@withContext Result.failure(IllegalStateException("ASR not initialized"))
            }
            if (isStreaming) {
                return@withContext Result.failure(IllegalStateException("Streaming already active"))
            }
            runCatching {
                val result = asr.streamBegin(handle, input, input.callback)
                isStreaming = true
                result
            }
        }

    /**
     * Push audio data to the streaming ASR for processing.
     * 
     * Audio should be float32 samples normalized to [-1.0, 1.0].
     * Typically 16kHz mono audio. Call this repeatedly as audio becomes available.
     * 
     * @param audioData Float array of audio samples
     * @return Result with status code (0 = success)
     * 
     * Example:
     * ```kotlin
     * // Convert 16-bit PCM to float32
     * val floatData = pcmData.map { it.toFloat() / 32768.0f }.toFloatArray()
     * asrWrapper.streamPushAudio(floatData)
     * ```
     */
    suspend fun streamPushAudio(audioData: FloatArray): Result<Int> =
        withContext(dispatcher) {
            if (handle == 0L) {
                return@withContext Result.failure(IllegalStateException("ASR not initialized"))
            }
            if (!isStreaming) {
                return@withContext Result.failure(IllegalStateException("Streaming not started"))
            }
            runCatching {
                asr.streamPushAudio(handle, audioData)
            }
        }

    /**
     * Stop the streaming ASR session.
     * 
     * @param graceful If true (default), processes remaining audio before stopping.
     *                 If false, stops immediately discarding any buffered audio.
     * @return Result with status code (0 = success)
     */
    suspend fun streamStop(graceful: Boolean = true): Result<Int> =
        withContext(dispatcher) {
            if (handle == 0L) {
                return@withContext Result.failure(IllegalStateException("ASR not initialized"))
            }
            if (!isStreaming) {
                return@withContext Result.failure(IllegalStateException("Streaming not started"))
            }
            runCatching {
                val result = asr.streamStop(handle, graceful)
                isStreaming = false
                result
            }
        }

    /**
     * Check if streaming is currently active.
     */
    fun isStreamingActive(): Boolean = isStreaming

    override fun close() {
        destroy()
    }

    /**
     * Clean up native resources
     * Should be called when done with the ASR model
     */
    fun destroy(): Int {
        var result = 0
        if (handle != 0L) {
            // Stop streaming if active
            if (isStreaming) {
                try {
                    asr.streamStop(handle, false)
                } catch (e: Exception) {
                    // Ignore errors during cleanup
                }
                isStreaming = false
            }
            result = asr.destroy(handle)
            handle = 0L
        }
        return result
    }
}

