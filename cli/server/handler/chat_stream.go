// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// Local types so finish_reason serializes as null on intermediate chunks and a
// value on the terminal chunk; openai-go's plain-string field can't (#1243).
type streamDelta struct {
	Role             string                                          `json:"role,omitempty"`
	Content          string                                          `json:"content,omitempty"`
	ReasoningContent string                                          `json:"reasoning_content,omitempty"`
	ToolCalls        []openai.ChatCompletionChunkChoiceDeltaToolCall `json:"tool_calls,omitempty"`
}

type streamChoice struct {
	Index        int64       `json:"index"`
	Delta        streamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"` // null until the final chunk
}

type streamChunk struct {
	Object  string                  `json:"object"`
	Choices []streamChoice          `json:"choices"`
	Usage   *openai.CompletionUsage `json:"usage,omitempty"`
}

const streamChunkObject = "chat.completion.chunk"

func contentChunk(content string) streamChunk {
	return streamChunk{
		Object: streamChunkObject,
		Choices: []streamChoice{{
			Delta: streamDelta{Role: string(openai.MessageRoleAssistant), Content: content},
		}},
	}
}

func tokenChunk(text string, reasoning bool) streamChunk {
	delta := streamDelta{Role: string(openai.MessageRoleAssistant)}
	if reasoning {
		delta.ReasoningContent = text
	} else {
		delta.Content = text
	}
	return streamChunk{Object: streamChunkObject, Choices: []streamChoice{{Delta: delta}}}
}

func finishChunk(reason string) streamChunk {
	return streamChunk{
		Object:  streamChunkObject,
		Choices: []streamChoice{{FinishReason: &reason}},
	}
}

func usageChunk(u openai.CompletionUsage) streamChunk {
	return streamChunk{Object: streamChunkObject, Choices: []streamChoice{}, Usage: &u}
}

func toolCallChunk(call openai.ChatCompletionMessageFunctionToolCallFunction) streamChunk {
	return streamChunk{
		Object: streamChunkObject,
		Choices: []streamChoice{{
			Delta: streamDelta{
				ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{{
					ID:   fmt.Sprintf("call_%d", rand.Uint32()),
					Type: "function",
					Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
						Name:      call.Name,
						Arguments: call.Arguments,
					},
				}},
			},
		}},
	}
}

// profile is read only after wait() returns, when generation has filled it.
func streamPlainText(c *gin.Context, dataCh <-chan string, wait func() error, includeUsage bool, profile *geniex_sdk.ProfileData, render tokenRender) {
	c.Stream(func(w io.Writer) bool {
		r, ok := <-dataCh
		if ok {
			if chunk, emit := render(r); emit {
				c.SSEvent("", chunk)
			}
			return true
		}
		if err := wait(); err != nil {
			c.SSEvent("", map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return false
		}
		c.SSEvent("", finishChunk(mapFinishReason(profile.StopReason)))
		if includeUsage {
			c.SSEvent("", usageChunk(profile2Usage(*profile)))
		}
		c.SSEvent("", "[DONE]")
		return false
	})
}

// Buffers the whole stream, then emits one tool-call chunk (or a content chunk
// on parse failure).
func streamToolCall(c *gin.Context, dataCh <-chan string, wait func() error, includeUsage bool, profile *geniex_sdk.ProfileData, policy toolSelectionPolicy) {
	buffer := strings.Builder{}
	c.Stream(func(w io.Writer) bool {
		r, ok := <-dataCh
		if ok {
			buffer.WriteString(r)
			return true
		}
		if err := wait(); err != nil {
			slog.Error("Generation error", "error", err)
			c.SSEvent("", map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return false
		}
		finishReason := "tool_calls"
		toolCall, err := parseSelectedToolCall(buffer.String(), policy)
		if err != nil {
			if policy.explicit() {
				slog.Warn("Required tool call parse failed", "error", err, "tool_choice", policy.mode)
				c.SSEvent("", requiredToolError())
				c.SSEvent("", "[DONE]")
				return false
			}
			slog.Warn("Tool call parse error, fallback to text", "error", err)
			finishReason = mapFinishReason(profile.StopReason)
			c.SSEvent("", contentChunk(buffer.String()))
		} else {
			c.SSEvent("", toolCallChunk(toolCall))
		}
		c.SSEvent("", finishChunk(finishReason))
		if includeUsage {
			c.SSEvent("", usageChunk(profile2Usage(*profile)))
		}
		c.SSEvent("", "[DONE]")
		return false
	})
}
