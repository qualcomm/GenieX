package com.geniex.sdk.jni

import com.geniex.sdk.bean.AsrCreateInput
import com.geniex.sdk.bean.AsrStreamBeginInput
import com.geniex.sdk.bean.AsrStreamBeginOutput
import com.geniex.sdk.bean.AsrStreamPushAudioInput
import com.geniex.sdk.bean.AsrStreamStopInput
import com.geniex.sdk.bean.AsrTranscribeInput
import com.geniex.sdk.bean.AsrTranscribeOutput
import com.geniex.sdk.callback.AsrTranscriptionCallback

internal class Asr {
    external fun create(
        asrCreateInput: AsrCreateInput
    ): Long

    external fun destroy(handle: Long): Int

    external fun transcribe(
        handle: Long,
        input: AsrTranscribeInput
    ): AsrTranscribeOutput

    external fun listSupportedLanguages(
        handle: Long
    ): List<String>?

    /**
     * Begin ASR streaming session with the given configuration.
     * 
     * @param handle The ASR handle from create()
     * @param input Streaming configuration including language and callback
     * @param callback Callback for receiving transcription updates
     * @return AsrStreamBeginOutput with status (0 = success)
     */
    external fun streamBegin(
        handle: Long,
        input: AsrStreamBeginInput,
        callback: AsrTranscriptionCallback
    ): AsrStreamBeginOutput

    /**
     * Push audio data to the streaming ASR for processing.
     * 
     * @param handle The ASR handle from create()
     * @param audioData Audio samples as float32 array (normalized to [-1.0, 1.0])
     * @return Status code (0 = success, negative = error)
     */
    external fun streamPushAudio(
        handle: Long,
        audioData: FloatArray
    ): Int

    /**
     * Stop the streaming ASR session.
     * 
     * @param handle The ASR handle from create()
     * @param graceful If true, process remaining audio before stopping
     * @return Status code (0 = success, negative = error)
     */
    external fun streamStop(
        handle: Long,
        graceful: Boolean
    ): Int
}

