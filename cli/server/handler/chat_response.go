// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"errors"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared/constant"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/server/utils"
)

// Returns true when the error was written, so the caller should return.
func writeKeepAliveError(c *gin.Context, err error) bool {
	if err == nil {
		return false
	}
	switch {
	case errors.Is(err, os.ErrNotExist):
		c.JSON(http.StatusNotFound, map[string]any{"error": "model not found"})
	case errors.Is(err, geniex_sdk.ErrCommonParamNotSupported):
		c.JSON(http.StatusBadRequest, map[string]any{
			"error": "a parameter in the request is not supported by the runtime",
			"code":  geniex_sdk.SDKErrorCode(err),
		})
	default:
		c.JSON(http.StatusInternalServerError, map[string]any{
			"error": err.Error(),
			"code":  geniex_sdk.SDKErrorCode(err),
		})
	}
	return true
}

// OpenAI's 400 body for a prompt that is longer than the context window. No
// partial output exists (nothing was generated), so this returns only the error.
func writePromptTooLong(c *gin.Context, profile geniex_sdk.ProfileData) {
	c.JSON(http.StatusBadRequest, map[string]any{
		"error": map[string]any{
			"message": "prompt is longer than the model's context window",
			"type":    "invalid_request_error",
			"code":    "context_length_exceeded",
		},
		"usage": profile2Usage(profile),
	})
}

// Adds reasoning_content, which openai-go's ChatCompletionMessage lacks; used
// only when thinking is separated so the inline default stays byte-identical.
type blockingMessage struct {
	Role             string `json:"role"`
	Content          string `json:"content"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type blockingChoice struct {
	Index        int64           `json:"index"`
	Message      blockingMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

type blockingResponse struct {
	Object  string                 `json:"object"`
	Choices []blockingChoice       `json:"choices"`
	Usage   openai.CompletionUsage `json:"usage"`
}

func writeBlockingResponse(c *gin.Context, content, reasoning string, profile geniex_sdk.ProfileData, parseTool bool) {
	if parseTool {
		toolCall, err := utils.ParseToolCalls(content)
		if err == nil {
			choice := openai.ChatCompletionChoice{
				FinishReason: "tool_calls",
				Message: openai.ChatCompletionMessage{
					Role: constant.Assistant(openai.MessageRoleAssistant),
					ToolCalls: []openai.ChatCompletionMessageToolCallUnion{{
						ID:       fmt.Sprintf("call_%d", rand.Uint32()),
						Type:     "function",
						Function: toolCall,
					}},
				},
			}
			c.JSON(http.StatusOK, openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{choice},
				Usage:   profile2Usage(profile),
			})
			return
		}
		slog.Warn("Tool call parse error, fallback to text", "error", err)
	}

	if reasoning == "" {
		choice := openai.ChatCompletionChoice{
			FinishReason: mapFinishReason(profile.StopReason),
			Message: openai.ChatCompletionMessage{
				Role:    constant.Assistant(openai.MessageRoleAssistant),
				Content: content,
			},
		}
		c.JSON(http.StatusOK, openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{choice},
			Usage:   profile2Usage(profile),
		})
		return
	}

	c.JSON(http.StatusOK, blockingResponse{
		Object: "chat.completion",
		Choices: []blockingChoice{{
			Message: blockingMessage{
				Role:             string(openai.MessageRoleAssistant),
				Content:          content,
				ReasoningContent: reasoning,
			},
			FinishReason: mapFinishReason(profile.StopReason),
		}},
		Usage: profile2Usage(profile),
	})
}

func profile2Usage(p geniex_sdk.ProfileData) openai.CompletionUsage {
	return openai.CompletionUsage{
		CompletionTokens: p.GeneratedTokens,
		PromptTokens:     p.PromptTokens,
		TotalTokens:      p.TotalTokens(),
		CompletionTokensDetails: openai.CompletionUsageCompletionTokensDetails{
			AcceptedPredictionTokens: p.DraftNAccepted,
			RejectedPredictionTokens: p.DraftNTotal - p.DraftNAccepted,
		},
	}
}

func mapFinishReason(stopReason string) string {
	switch stopReason {
	case "length":
		return "length"
	case "user":
		return "stop"
	case "eos", "stop_sequence", "":
		return "stop"
	default:
		return "stop"
	}
}
