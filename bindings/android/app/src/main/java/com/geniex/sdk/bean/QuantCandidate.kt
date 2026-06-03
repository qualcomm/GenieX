package com.geniex.sdk.bean

/** Mirrors `geniex_QuantCandidate`. One quantization advertised by a hub. */
data class QuantCandidate(
    val quant: String,
    val size: Long,
    val is_default: Boolean,
)
