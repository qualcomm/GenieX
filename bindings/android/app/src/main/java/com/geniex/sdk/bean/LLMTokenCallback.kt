package com.geniex.sdk.bean;

interface LLMTokenCallback {
    fun onToken(token: String): Boolean
    fun onComplete(result: LlmGenerateResult) {}
}
