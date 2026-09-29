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
	Role             string                                      `json:"role"`
	Content          string                                      `json:"content"`
	ReasoningContent string                                      `json:"reasoning_content,omitempty"`
	ToolCalls        []openai.ChatCompletionMessageToolCallUnion `json:"tool_calls,omitempty"`
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
	Timings timings                `json:"timings"`
}

// chatCompletionWithTimings adds timings to the openai-go response type via
// embedding, keeping the reasoning=="" branch's fields byte-identical.
type chatCompletionWithTimings struct {
	openai.ChatCompletion
	Timings timings `json:"timings"`
}

func writeBlockingResponse(c *gin.Context, content, reasoning string, profile geniex_sdk.ProfileData, parseTool bool, tools string) {
	finishReason := mapFinishReason(profile.StopReason)
	var toolCalls []openai.ChatCompletionMessageToolCallUnion
	if parseTool {
		// Parse keeps the text around a call: that is content, not part of it.
		text, calls := utils.NewToolCallScanner(utils.ToolParameterTypesFromTools(tools)).Parse(content)
		content = text
		if len(calls) > 0 {
			finishReason = "tool_calls"
		} else {
			slog.Debug("No tool call in the response")
		}
		for _, call := range calls {
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnion{
				ID:       fmt.Sprintf("call_%d", rand.Uint32()),
				Type:     "function",
				Function: call,
			})
		}
	}

	if reasoning == "" {
		choice := openai.ChatCompletionChoice{
			FinishReason: finishReason,
			Message: openai.ChatCompletionMessage{
				Role:      constant.Assistant(openai.MessageRoleAssistant),
				Content:   content,
				ToolCalls: toolCalls,
			},
		}
		c.JSON(http.StatusOK, chatCompletionWithTimings{
			ChatCompletion: openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{choice},
				Usage:   profile2Usage(profile),
			},
			Timings: profile2Timings(profile),
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
				ToolCalls:        toolCalls,
			},
			FinishReason: finishReason,
		}},
		Usage:   profile2Usage(profile),
		Timings: profile2Timings(profile),
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

// timings mirrors llama-server's per-request timing object (prompt_n,
// predicted_per_second, ...) so clients familiar with that shape can read
// prefill/decode speed the same way.
type timings struct {
	PromptN            int64   `json:"prompt_n"`
	PromptMs           float64 `json:"prompt_ms"`
	PromptPerSecond    float64 `json:"prompt_per_second"`
	PredictedN         int64   `json:"predicted_n"`
	PredictedMs        float64 `json:"predicted_ms"`
	PredictedPerSecond float64 `json:"predicted_per_second"`
	DraftN             int64   `json:"draft_n,omitempty"`
	DraftNAccepted     int64   `json:"draft_n_accepted,omitempty"`
}

func profile2Timings(p geniex_sdk.ProfileData) timings {
	return timings{
		PromptN:            p.PromptTokens,
		PromptMs:           float64(p.PromptTime) / 1e3,
		PromptPerSecond:    p.PrefillSpeed,
		PredictedN:         p.GeneratedTokens,
		PredictedMs:        float64(p.DecodeTime) / 1e3,
		PredictedPerSecond: p.DecodingSpeed,
		DraftN:             p.DraftNTotal,
		DraftNAccepted:     p.DraftNAccepted,
	}
}

// logProfile emits the same prefill/decode numbers llama-server logs per
// request, since geniex serve's request log otherwise ends at "param" and
// never reports how generation went.
func logProfile(profile geniex_sdk.ProfileData) {
	slog.Info("Generation complete",
		"prompt_tokens", profile.PromptTokens,
		"prompt_tok_s", profile.PrefillSpeed,
		"decode_tokens", profile.GeneratedTokens,
		"decode_tok_s", profile.DecodingSpeed,
		"ttft_s", float64(profile.TTFT)/1e6,
	)
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
