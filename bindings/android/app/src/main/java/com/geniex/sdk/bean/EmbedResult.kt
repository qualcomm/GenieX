package com.geniex.sdk.bean

data class EmbedResult(val embeddings: FloatArray, val profileData: ProfilingData) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as EmbedResult

        if (!embeddings.contentEquals(other.embeddings)) return false
        if (profileData != other.profileData) return false

        return true
    }

    override fun hashCode(): Int {
        var result = embeddings.contentHashCode()
        result = 31 * result + profileData.hashCode()
        return result
    }
}
