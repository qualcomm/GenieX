package com.geniex.sdk.bean

/**
 * Output from beginning an ASR streaming session.
 * 
 * @param status Status code (0 = success, negative = error)
 */
data class AsrStreamBeginOutput(
    val status: Int = 0
)

