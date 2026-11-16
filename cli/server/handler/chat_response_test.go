// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/thinkfsm"
	"github.com/qualcomm/GenieX/cli/server/types"
)

func TestProfile2UsageCachedTokens(t *testing.T) {
	u := profile2Usage(geniex_sdk.ProfileData{PromptTokens: 1, TotalPromptTokens: 51, GeneratedTokens: 5})
	if u.PromptTokens != 51 || u.PromptTokensDetails.CachedTokens != 50 || u.TotalTokens != 56 {
		t.Errorf("usage=%+v", u)
	}
}

func checkResponseMetadata(t *testing.T, body []byte, prefix, model string) types.ResponseMetadata {
	t.Helper()
	var meta types.ResponseMetadata
	if err := json.Unmarshal(body, &meta); err != nil {
		t.Fatal(err)
	}
	now := time.Now().Unix()
	if !strings.HasPrefix(meta.ID, prefix+"-") || meta.Model != model || meta.Created <= 0 || meta.Created > now || now-meta.Created > 5 {
		t.Fatalf("incomplete response metadata: %s", body)
	}
	return meta
}

type blockingBody struct {
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Role             string `json:"role"`
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
			ToolCalls        []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Timings timings `json:"timings"`
}

func TestWriteBlockingResponse(t *testing.T) {
	gin.SetMode(gin.TestMode)
	call := openai.ChatCompletionMessageFunctionToolCallFunction{Name: "get_weather", Arguments: `{"city":"Beijing"}`}
	cases := []struct {
		name               string
		content, reasoning string
		calls              []openai.ChatCompletionMessageFunctionToolCallFunction
		stop, finish       string
	}{
		{name: "text only", content: "hello", stop: "eos", finish: "stop"},
		{name: "empty response", stop: "eos", finish: "stop"},
		{name: "empty display with a call", calls: []openai.ChatCompletionMessageFunctionToolCallFunction{call}, stop: "eos", finish: "tool_calls"},
		{name: "prose with a call", content: "sure  done", calls: []openai.ChatCompletionMessageFunctionToolCallFunction{call}, stop: "eos", finish: "tool_calls"},
		{name: "parallel calls", calls: []openai.ChatCompletionMessageFunctionToolCallFunction{call, call}, stop: "eos", finish: "tool_calls"},
		{name: "reasoning with a call", content: "sure ", reasoning: "let me check", calls: []openai.ChatCompletionMessageFunctionToolCallFunction{call}, stop: "eos", finish: "tool_calls"},
		{name: "reasoning without a call", content: "hello", reasoning: "let me check", stop: "eos", finish: "stop"},
		{name: "truncated text", content: "partial", stop: "length", finish: "length"},
	}
	seen := map[string]bool{}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			profile := geniex_sdk.ProfileData{StopReason: tc.stop, PromptTokens: 5, PrefillSpeed: 10, GeneratedTokens: 3, DecodingSpeed: 20}
			var content, reasoning strings.Builder
			finished := false
			finish := func() []openai.ChatCompletionMessageFunctionToolCallFunction {
				if finished {
					t.Error("finish called more than once")
				}
				finished = true
				content.WriteString(tc.content)
				reasoning.WriteString(tc.reasoning)
				return tc.calls
			}
			writeBlockingResponse(c, &content, &reasoning, profile, finish, "test-model")
			meta := checkResponseMetadata(t, w.Body.Bytes(), "chatcmpl", "test-model")
			if seen[meta.ID] {
				t.Fatalf("reused response ID %q", meta.ID)
			}
			seen[meta.ID] = true
			if !finished {
				t.Error("response did not finalize the parser")
			}
			var got blockingBody
			if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
				t.Fatal(err)
			}
			if w.Code != 200 || len(got.Choices) != 1 {
				t.Fatalf("status=%d, choices=%+v", w.Code, got.Choices)
			}
			if got.Timings.PromptPerSecond != 10 || got.Timings.PredictedPerSecond != 20 {
				t.Errorf("timings=%+v", got.Timings)
			}
			choice := got.Choices[0]
			if choice.Message.Role != "assistant" || choice.Message.Content != tc.content || choice.Message.ReasoningContent != tc.reasoning || choice.FinishReason != tc.finish {
				t.Errorf("choice=%+v", choice)
			}
			if len(choice.Message.ToolCalls) != len(tc.calls) {
				t.Fatalf("calls=%+v, want %+v", choice.Message.ToolCalls, tc.calls)
			}
			for i, call := range choice.Message.ToolCalls {
				if call.Type != "function" || !strings.HasPrefix(call.ID, "call_") || call.Function.Name != tc.calls[i].Name || call.Function.Arguments != tc.calls[i].Arguments {
					t.Errorf("call %d=%+v, want %+v", i, call, tc.calls[i])
				}
			}
		})
	}
}

func TestChatResponsesFinalizeParserAtEOF(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, reasoning := range []string{"", "reason"} {
		for _, tail := range []struct {
			name   string
			tokens []string
			text   string
			calls  []openai.ChatCompletionMessageFunctionToolCallFunction
		}{
			{name: "open object", tokens: []string{"{"}, text: "{"},
			{name: "Ministral3 call", tokens: []string{`[TOOL_CALLS]f[ARGS]{}`}, calls: []openai.ChatCompletionMessageFunctionToolCallFunction{{Name: "f", Arguments: `{}`}}},
			{name: "Ministral3 partial arguments", tokens: []string{`[TOOL_CALLS]f[ARGS]{"x":`}, text: `[TOOL_CALLS]f[ARGS]{"x":`},
			{name: "partial wrapper", tokens: []string{"answer", " <", "tool", "_c"}, text: "answer <tool_c"},
			{name: "recovered call", tokens: []string{`<tool_call>{"name":"f","arguments":{}}`}, calls: []openai.ChatCompletionMessageFunctionToolCallFunction{{Name: "f", Arguments: `{}`}}},
		} {
			for _, stop := range []struct {
				name, reason string
				err          error
				finish       string
			}{
				{"EOF", "eos", nil, "stop"},
				{"context limit", "length", geniex_sdk.ErrLlmTokenizationContextLength, "length"},
			} {
				t.Run(reasoning+"/"+tail.name+"/"+stop.name, func(t *testing.T) {
					class := tokenClass(func(token string) thinkfsm.Event {
						kind := thinkfsm.Content
						if token == "reason" {
							kind = thinkfsm.Thinking
						}
						return thinkfsm.Event{Kind: kind, Text: token}
					})
					tokens := append([]string(nil), tail.tokens...)
					if reasoning != "" {
						tokens = append([]string{reasoning}, tokens...)
					}
					wantFinish := stop.finish
					if len(tail.calls) > 0 {
						wantFinish = "tool_calls"
					}
					var content, thought strings.Builder
					onToken, finish := sinkWithToolCalls(class, &content, &thought, "")
					for _, token := range tokens {
						onToken(token)
					}
					w := httptest.NewRecorder()
					c, _ := gin.CreateTestContext(w)
					profile := geniex_sdk.ProfileData{StopReason: stop.reason}
					writeBlockingResponse(c, &content, &thought, profile, finish, "test-model")
					checkResponseMetadata(t, w.Body.Bytes(), "chatcmpl", "test-model")
					var got blockingBody
					if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
						t.Fatal(err)
					}
					if len(got.Choices) != 1 {
						t.Fatalf("choices=%+v", got.Choices)
					}
					choice := got.Choices[0]
					if choice.Message.Content != tail.text || choice.Message.ReasoningContent != reasoning || choice.FinishReason != wantFinish || len(choice.Message.ToolCalls) != len(tail.calls) {
						t.Errorf("blocking choice=%+v", choice)
					}
					for i, call := range choice.Message.ToolCalls {
						if i < len(tail.calls) && (call.Function.Name != tail.calls[i].Name || call.Function.Arguments != tail.calls[i].Arguments) {
							t.Errorf("blocking call=%+v, want %+v", call, tail.calls[i])
						}
					}
					sw := &streamRecorder{httptest.NewRecorder(), make(chan bool)}
					sc, _ := gin.CreateTestContext(sw)
					sc.Request = httptest.NewRequest("POST", "/v1/chat/completions", nil)
					dataCh := make(chan string, len(tokens))
					for _, token := range tokens {
						dataCh <- token
					}
					close(dataCh)
					streamToolCall(sc, dataCh, func() error { return stop.err }, false, &profile, class, "", "test-model")
					deltas := sseDeltas(t, sw.Body.String())
					var streamedContent, streamedReasoning strings.Builder
					var calls []toolCallDelta
					for _, delta := range deltas {
						streamedContent.WriteString(delta.Delta.Content)
						streamedReasoning.WriteString(delta.Delta.ReasoningContent)
						for _, call := range delta.Delta.ToolCalls {
							calls = append(calls, toolCallDelta{int(call.Index), call.Function.Name, call.Function.Arguments})
						}
					}
					if streamedContent.String() != tail.text || streamedReasoning.String() != reasoning || len(calls) != len(tail.calls) {
						t.Errorf("streamed=(%q, %q, %+v)", streamedContent.String(), streamedReasoning.String(), calls)
					}
					for i, call := range calls {
						if i < len(tail.calls) && (call.index != i || call.name != tail.calls[i].Name || call.arguments != tail.calls[i].Arguments) {
							t.Errorf("streamed call=%+v, want %+v", call, tail.calls[i])
						}
					}
					if len(deltas) == 0 {
						t.Fatal("stream emitted no terminal chunk")
					}
					if reason := deltas[len(deltas)-1].FinishReason; reason == nil || *reason != wantFinish {
						t.Errorf("streamed finish=%v, want %q", reason, wantFinish)
					}
				})
			}
		}
	}
}
