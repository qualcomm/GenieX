package com.geniex.sdk.bean

data class RerankerOutputResult(
    val scores: FloatArray?,
    val scoreCount: Int,
    val profileData: ProfilingData?
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as RerankerOutputResult

        if (scoreCount != other.scoreCount) return false
        if (!scores.contentEquals(other.scores)) return false
        if (profileData != other.profileData) return false

        return true
    }

    override fun hashCode(): Int {
        var result = scoreCount
        result = 31 * result + (scores?.contentHashCode() ?: 0)
        result = 31 * result + (profileData?.hashCode() ?: 0)
        return result
    }
}