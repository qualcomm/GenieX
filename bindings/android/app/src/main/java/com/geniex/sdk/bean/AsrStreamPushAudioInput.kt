package com.geniex.sdk.bean

/**
 * Input for pushing audio data to the streaming ASR.
 * 
 * @param audioData Audio samples as float32 array (values normalized to [-1.0, 1.0])
 *                  Typically 16kHz mono audio converted to float format
 */
data class AsrStreamPushAudioInput(
    val audioData: FloatArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as AsrStreamPushAudioInput
        return audioData.contentEquals(other.audioData)
    }

    override fun hashCode(): Int {
        return audioData.contentHashCode()
    }
}

