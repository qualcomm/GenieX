package com.geniex.sdk.bean

sealed class LlmStreamResult {
    data class Token(val text: String) : LlmStreamResult()
    data class Completed(val profile: ProfilingData) : LlmStreamResult()
    data class Error(val throwable: Throwable) : LlmStreamResult()
}