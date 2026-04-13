package com.geniex.sdk.jni

import com.geniex.sdk.bean.ChatMessage
import com.geniex.sdk.bean.GenerationConfig
import com.geniex.sdk.bean.LLMTokenCallback
import com.geniex.sdk.bean.LlmApplyChatTemplateOutput
import com.geniex.sdk.bean.LlmCreateInput
import com.geniex.sdk.bean.LlmGenerateResult

internal class Llm {

    external fun create(llmCreateInputObj: LlmCreateInput): Long
    external fun reset(handle: Long): Int
    external fun destroy(handle: Long): Int
    external fun stopStream(handle: Long)
    external fun applyChatTemplate(
        handle: Long,
        messages: Array<ChatMessage>,
        tools: String?,
        enableThinking: Boolean,
        addGenerationPrompt: Boolean = true
    ): LlmApplyChatTemplateOutput

    external fun generate(
        handle: Long,
        prompt: String,
        config: GenerationConfig,
        cb: LLMTokenCallback
    ): LlmGenerateResult

}