package com.geniex.sdk.bean

/** Mirrors `geniex_QuantCandidate`. One precision advertised by a hub. */
data class PrecisionCandidate(
    val precision: String,
    val size: Long,
)
