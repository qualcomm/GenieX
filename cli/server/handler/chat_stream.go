// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/server/utils"
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
	Timings *timings                `json:"timings,omitempty"`
}

const streamChunkObject = "chat.completion.chunk"

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

func usageChunk(u openai.CompletionUsage, t timings) streamChunk {
	return streamChunk{Object: streamChunkObject, Choices: []streamChoice{}, Usage: &u, Timings: &t}
}

func toolCallChunk(index int, call openai.ChatCompletionMessageFunctionToolCallFunction) streamChunk {
	return streamChunk{
		Object: streamChunkObject,
		Choices: []streamChoice{{
			Delta: streamDelta{
				ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{{
					Index: int64(index),
					ID:    fmt.Sprintf("call_%d", rand.Uint32()),
					Type:  "function",
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
		// A context window exhausted mid-stream is a normal truncated completion
		// (finish_reason=length): fall through to the finish chunk. Other errors
		// (including a too-long prompt) are surfaced as an error event.
		if err := wait(); err != nil && !errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) {
			c.SSEvent("", map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return false
		}
		logProfile(*profile)
		c.SSEvent("", finishChunk(mapFinishReason(profile.StopReason)))
		if includeUsage {
			c.SSEvent("", usageChunk(profile2Usage(*profile), profile2Timings(*profile)))
		}
		c.SSEvent("", "[DONE]")
		return false
	})
}

// Streams the text that cannot be part of a tool call as it arrives, and each
// tool call as soon as it is complete.
func streamToolCall(c *gin.Context, dataCh <-chan string, wait func() error, includeUsage bool, profile *geniex_sdk.ProfileData, class tokenClass, tools string) {
	scanner := utils.NewToolCallScanner(utils.ToolParameterTypesFromTools(tools))
	sent := 0 // the delta index, which has to keep rising across chunks
	c.Stream(func(w io.Writer) bool {
		r, ok := <-dataCh
		if ok {
			token, isReasoning, emit := class(r)
			if !emit {
				return true
			}
			// A tool call never lives in the thinking block, which goes out as it is.
			if isReasoning {
				c.SSEvent("", tokenChunk(token, true))
				return true
			}
			text, calls := scanner.Push(token)
			if text != "" {
				c.SSEvent("", tokenChunk(text, false))
			}
			for _, call := range calls {
				c.SSEvent("", toolCallChunk(sent, call))
				sent++
			}
			return true
		}
		// A context window exhausted mid-stream is a normal truncated completion:
		// fall through and emit what was buffered. Other errors (including a
		// too-long prompt) are surfaced as an error event.
		if err := wait(); err != nil && !errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) {
			slog.Error("Generation error", "error", err)
			c.SSEvent("", map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return false
		}
		// The held tail may still hold calls the model stopped short of closing.
		tail, calls := scanner.Tail()
		if tail != "" {
			slog.Warn("Tool call not matched, streaming the held text instead")
			c.SSEvent("", tokenChunk(tail, false))
		}
		for _, call := range calls {
			c.SSEvent("", toolCallChunk(sent, call))
			sent++
		}
		logProfile(*profile)
		finishReason := mapFinishReason(profile.StopReason)
		if sent > 0 {
			finishReason = "tool_calls"
		}
		c.SSEvent("", finishChunk(finishReason))
		if includeUsage {
			c.SSEvent("", usageChunk(profile2Usage(*profile), profile2Timings(*profile)))
		}
		c.SSEvent("", "[DONE]")
		return false
	})
}
