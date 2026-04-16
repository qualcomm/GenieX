package com.geniex.sdk.bean

import com.geniex.sdk.callback.AsrTranscriptionCallback

/**
 * Input configuration for beginning ASR streaming.
 * 
 * @param language Language code (ISO 639-1, e.g., "en", "zh") or null for auto-detect
 * @param streamConfig Streaming configuration parameters (optional)
 * @param callback Required callback for receiving transcription updates
 */
data class AsrStreamBeginInput(
    val language: String? = null,
    val streamConfig: AsrStreamConfig? = null,
    val callback: AsrTranscriptionCallback
)

/**
 * Configuration for ASR streaming behavior.
 * 
 * @param chunkDuration Duration in seconds for each audio chunk (default: 4.0)
 * @param overlapDuration Overlap between chunks in seconds (default: 3.0)
 * @param sampleRate Audio sample rate in Hz (default: 16000)
 * @param maxQueueSize Maximum chunks in processing queue (default: 10)
 * @param bufferSize Audio buffer size for input (default: 512)
 * @param timestamps Timestamp mode: "none", "segment", "word"
 * @param beamSize Beam search size for decoding
 */
data class AsrStreamConfig(
    val chunkDuration: Float = 4.0f,
    val overlapDuration: Float = 3.0f,
    val sampleRate: Int = 16000,
    val maxQueueSize: Int = 10,
    val bufferSize: Int = 512,
    val timestamps: String? = null,
    val beamSize: Int? = null
)

