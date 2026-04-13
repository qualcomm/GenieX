package com.geniex.sdk.callback

/**
 * Callback interface for ASR streaming transcription updates.
 * Called whenever new transcription text is available during streaming.
 */
interface AsrTranscriptionCallback {
    /**
     * Called when a new transcription update is available.
     * 
     * @param text The transcribed text (partial or complete)
     */
    fun onTranscription(text: String)
}
