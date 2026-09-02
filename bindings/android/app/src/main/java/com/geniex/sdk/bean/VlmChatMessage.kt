package com.geniex.sdk.bean

data class VlmChatMessage(
    val role: String?,
    val contents: List<VlmContent>,
    // Assistant turns carry toolCalls; the matching tool response carries
    // toolCallId and toolName. All unset for a plain chat turn.
    val toolCalls: List<ToolCall> = emptyList(),
    val toolCallId: String? = null,
    val toolName: String? = null
)
