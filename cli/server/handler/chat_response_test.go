// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// Covers both bodies writeBlockingResponse can produce: openai's own struct, and the
// local one that carries reasoning_content.
type blockingBody struct {
	GenieXCache *managedCacheMetadata `json:"geniex_cache"`
	Choices     []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
			ToolCalls        []struct {
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
}

// A tool call must not cost the text around it, nor the reasoning: both used to be
// dropped whenever one was found.
func TestWriteBlockingResponse(t *testing.T) {
	gin.SetMode(gin.TestMode)
	const call = `{"name":"get_weather","arguments":{"city":"Beijing"}}`
	weather := [2]string{"get_weather", `{"city":"Beijing"}`}
	cache := &managedCacheMetadata{
		Mode:     "managed",
		Status:   "reused",
		Revision: "sha256:0123456789abcdef",
		Reason:   "exact_extension",
	}

	tests := []struct {
		name          string
		content       string
		reasoning     string
		parseTool     bool
		wantContent   string
		wantReasoning string
		wantFinish    string
		wantCalls     [][2]string
		cache         *managedCacheMetadata
	}{
		{
			name: "text only", content: "hello",
			wantContent: "hello", wantFinish: "stop",
		},
		{
			name: "tools on but nothing to match", content: "hello", parseTool: true,
			wantContent: "hello", wantFinish: "stop",
		},
		{
			name: "a call keeps the prose around it", content: "sure " + call + " done", parseTool: true,
			wantContent: "sure  done", wantFinish: "tool_calls", wantCalls: [][2]string{weather},
		},
		{
			name: "parallel calls", content: call + call, parseTool: true,
			wantFinish: "tool_calls", wantCalls: [][2]string{weather, weather},
		},
		{
			name: "gemma4 syntax", content: `x <|tool_call>call:f{a:1}<tool_call|>`, parseTool: true,
			wantContent: "x ", wantFinish: "tool_calls", wantCalls: [][2]string{{"f", `{"a":1}`}},
		},
		{
			name: "reasoning with a call", content: "sure " + call, reasoning: "let me check",
			parseTool: true, wantContent: "sure ", wantReasoning: "let me check",
			wantFinish: "tool_calls", wantCalls: [][2]string{weather},
		},
		{
			name: "reasoning without a call", content: "hello", reasoning: "let me check",
			parseTool: true, wantContent: "hello", wantReasoning: "let me check", wantFinish: "stop",
		},
		{
			name: "managed metadata with reasoning", content: "hello", reasoning: "let me check",
			wantContent: "hello", wantReasoning: "let me check", wantFinish: "stop", cache: cache,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			profile := geniex_sdk.ProfileData{StopReason: "eos"}
			writeBlockingResponse(c, tt.content, tt.reasoning, profile, tt.parseTool, tt.cache)

			var got blockingBody
			if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
				t.Fatalf("body %s: %v", w.Body.Bytes(), err)
			}
			if len(got.Choices) != 1 {
				t.Fatalf("choices = %d, want 1", len(got.Choices))
			}
			if (got.GenieXCache == nil) != (tt.cache == nil) {
				t.Fatalf("geniex_cache = %+v, want %+v", got.GenieXCache, tt.cache)
			}
			if tt.cache != nil && *got.GenieXCache != *tt.cache {
				t.Errorf("geniex_cache = %+v, want %+v", got.GenieXCache, tt.cache)
			}
			choice := got.Choices[0]
			if choice.FinishReason != tt.wantFinish {
				t.Errorf("finish_reason = %q, want %q", choice.FinishReason, tt.wantFinish)
			}
			if choice.Message.Content != tt.wantContent {
				t.Errorf("content = %q, want %q", choice.Message.Content, tt.wantContent)
			}
			if choice.Message.ReasoningContent != tt.wantReasoning {
				t.Errorf("reasoning_content = %q, want %q", choice.Message.ReasoningContent, tt.wantReasoning)
			}
			if len(choice.Message.ToolCalls) != len(tt.wantCalls) {
				t.Fatalf("tool_calls = %+v, want %v", choice.Message.ToolCalls, tt.wantCalls)
			}
			for i, want := range tt.wantCalls {
				fn := choice.Message.ToolCalls[i].Function
				if fn.Name != want[0] || fn.Arguments != want[1] {
					t.Errorf("tool_calls[%d] = (%q, %q), want (%q, %q)",
						i, fn.Name, fn.Arguments, want[0], want[1])
				}
			}
		})
	}
}
