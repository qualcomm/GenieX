package com.geniex.sdk.jni

import com.geniex.sdk.bean.GenerationConfig
import com.geniex.sdk.bean.LLMTokenCallback
import com.geniex.sdk.bean.LlmApplyChatTemplateOutput
import com.geniex.sdk.bean.LlmGenerateResult
import com.geniex.sdk.bean.VlmCapabilities
import com.geniex.sdk.bean.VlmChatMessage
import com.geniex.sdk.bean.VlmCreateInput

internal class Vlm {
    external fun create(vlmCreateInput: VlmCreateInput): Long

    external fun destroy(handle: Long): Int
    external fun reset(handle: Long): Int

    external fun getCapabilities(handle: Long): VlmCapabilities

    external fun generate(
            handle: Long,
            prompt: String,
            config: GenerationConfig,
            cb: LLMTokenCallback
    ): LlmGenerateResult

    external fun applyChatTemplate(
            handle: Long,
            messages: Array<VlmChatMessage>,
            tools: String?,
            enableThinking: Boolean
    ): LlmApplyChatTemplateOutput

    external fun stopStream(handle: Long)

    /**
     * Extract image and audio paths from VlmChatMessage contents. Returns a Pair: first = image
     * paths, second = audio paths
     */
    external fun extractMediaPaths(
            messages: Array<VlmChatMessage>
    ): Pair<Array<String>, Array<String>>

    //    external fun embed(handle: Long, texts: Array<String>): FloatArray?
    //    external fun setSampler(handle: Long, cfg: SamplerConfig)
    //    external fun resetSampler(handle: Long)
    //    external fun encode(handle: Long, text: String, outTokens: IntArray): Int
    //    external fun decode(handle: Long, tokens: IntArray, length: Int): String?

}