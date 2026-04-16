package com.geniex.sdk.bean

/**
 * Input for stopping the streaming ASR session.
 * 
 * @param graceful If true, processes remaining audio before stopping.
 *                 If false, stops immediately discarding any buffered audio.
 */
data class AsrStreamStopInput(
    val graceful: Boolean = true
)

