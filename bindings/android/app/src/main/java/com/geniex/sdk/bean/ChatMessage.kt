package com.geniex.sdk.bean

data class ChatMessage(
    var role: String,
    var content: String,
    // Assistant turns carry toolCalls; the matching tool response carries
    // toolCallId and toolName. All unset for a plain chat turn.
    var toolCalls: List<ToolCall> = emptyList(),
    var toolCallId: String? = null,
    var toolName: String? = null,
    // The thinking an assistant turn generated before content. Templates that
    // keep thinking in the history render it from here.
    var reasoningContent: String? = null
)
