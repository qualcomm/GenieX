// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

func TestReasoningSeparated(t *testing.T) {
	cases := map[string]bool{
		"":                false,
		"none":            false,
		"None":            false,
		"  none  ":        false,
		"deepseek":        true,
		"deepseek-legacy": true,
		"auto":            true,
	}
	for format, want := range cases {
		if got := reasoningSeparated(format); got != want {
			t.Errorf("reasoningSeparated(%q) = %v, want %v", format, got, want)
		}
	}
}

func TestSinks(t *testing.T) {
	tokens := []string{"<think>", "reason", "</think>", "answer"}

	t.Run("reasoningClass splits think block", func(t *testing.T) {
		var content, reasoning strings.Builder
		s := sink(reasoningClass(), &content, &reasoning)
		for _, tok := range tokens {
			s(tok)
		}
		if got := content.String(); got != "answer" {
			t.Errorf("content = %q, want %q", got, "answer")
		}
		if got := reasoning.String(); got != "reason" {
			t.Errorf("reasoning = %q, want %q", got, "reason")
		}
	})

	t.Run("plainClass keeps everything inline", func(t *testing.T) {
		var content, reasoning strings.Builder
		s := sink(plainClass, &content, &reasoning)
		for _, tok := range tokens {
			s(tok)
		}
		if got := content.String(); got != "<think>reason</think>answer" {
			t.Errorf("content = %q, want raw inline", got)
		}
	})
}

const testKnowledgeTool = `{
  "type":"function",
  "function":{
    "name":"search_knowledge",
    "description":"Search the local knowledge source.",
    "parameters":{
      "type":"object",
      "properties":{"query":{"type":"string"}},
      "required":["query"],
      "additionalProperties":false
    }
  }
}`

func toolRequest(t *testing.T, choice string, withTools bool) ChatCompletionRequest {
	t.Helper()
	tools := "[]"
	if withTools {
		tools = "[" + testKnowledgeTool + "]"
	}
	input := `{"messages":[{"role":"user","content":"Search coffee."}],"tools":` + tools
	if choice != "" {
		input += `,"tool_choice":` + choice
	}
	input += `}`
	var request ChatCompletionRequest
	if err := json.Unmarshal([]byte(input), &request); err != nil {
		t.Fatal(err)
	}
	return request
}

func namedToolRequest(t *testing.T, name string) ChatCompletionRequest {
	t.Helper()
	return toolRequest(t, `{"type":"function","function":{"name":"`+name+`"}}`, true)
}

func TestParseToolsTranslatesOpenAIToolChoice(t *testing.T) {
	tests := []struct {
		name       string
		choice     string
		named      string
		withTools  bool
		wantMode   toolSelectionMode
		wantName   string
		wantChoice string
		wantError  bool
	}{
		{name: "omitted remains auto", withTools: true, wantMode: toolSelectionAuto, wantChoice: "auto"},
		{name: "auto remains auto", choice: `"auto"`, withTools: true, wantMode: toolSelectionAuto, wantChoice: "auto"},
		{name: "required", choice: `"required"`, withTools: true, wantMode: toolSelectionRequired, wantChoice: "required"},
		{name: "none", choice: `"none"`, withTools: true, wantMode: toolSelectionNone, wantChoice: "none"},
		{name: "named function", withTools: true, named: "search_knowledge", wantMode: toolSelectionNamed, wantName: "search_knowledge", wantChoice: "required"},
		{name: "unknown named function", withTools: true, named: "unknown", wantError: true},
		{name: "required without tools", choice: `"required"`, wantError: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := toolRequest(t, tt.choice, tt.withTools)
			if tt.named != "" {
				request = namedToolRequest(t, tt.named)
			}
			policy, tools, err := parseTools(request)
			if tt.wantError {
				if err == nil {
					t.Fatalf("expected error, got policy=%+v tools=%s", policy, tools)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if policy.mode != tt.wantMode || policy.namedTool != tt.wantName {
				t.Fatalf("policy=%+v, want mode=%s name=%q", policy, tt.wantMode, tt.wantName)
			}
			if got := policy.templateToolChoice(); got != tt.wantChoice {
				t.Fatalf("template tool choice=%q, want %q", got, tt.wantChoice)
			}
			if tt.wantMode == toolSelectionNone && tools != "" {
				t.Fatalf("none selection still rendered tools: %s", tools)
			}
			if tt.wantMode != toolSelectionNone && tt.withTools && !strings.Contains(tools, "search_knowledge") {
				t.Fatalf("selected function missing from tools: %s", tools)
			}
		})
	}
}

func TestExplicitToolSelectionAddsOneTemplateDirective(t *testing.T) {
	for _, request := range []ChatCompletionRequest{
		toolRequest(t, `"required"`, true),
		namedToolRequest(t, "search_knowledge"),
	} {
		policy, _, err := parseTools(request)
		if err != nil {
			t.Fatal(err)
		}
		messages := applyLlmToolSelectionDirective([]geniex_sdk.LlmChatMessage{{Role: "user", Content: "Search coffee."}}, policy)
		if len(messages) != 2 || messages[0].Role != "system" || !strings.Contains(messages[0].Content, "Emit only") || messages[1].Content != "Search coffee." {
			t.Fatalf("unexpected translated messages: %+v", messages)
		}
	}

	auto, _, err := parseTools(toolRequest(t, `"auto"`, true))
	if err != nil {
		t.Fatal(err)
	}
	original := []geniex_sdk.LlmChatMessage{{Role: "user", Content: "ordinary"}}
	if got := applyLlmToolSelectionDirective(original, auto); len(got) != 1 || got[0].Content != "ordinary" {
		t.Fatalf("auto request was modified: %+v", got)
	}
}

func TestStrictRequiredToolGrammarRemovesProsePrefix(t *testing.T) {
	grammar := "message ::= thought content\nroot ::= start message* scan-to-toolcall tool-call\ntool-call ::= \"call\"\n"
	strict, err := strictRequiredToolGrammar(grammar)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strict, "root ::= start tool-call") || strings.Contains(strict, "message* scan-to-toolcall") {
		t.Fatalf("required grammar still permits a prose prefix: %s", strict)
	}
	if _, err := strictRequiredToolGrammar("root ::= content"); err == nil {
		t.Fatal("unexpected grammar shape did not fail closed")
	}
}

type closeNotifyRecorder struct{ *httptest.ResponseRecorder }

func (r closeNotifyRecorder) CloseNotify() <-chan bool { return make(chan bool) }

func TestExplicitToolCallValidation(t *testing.T) {
	policy, _, err := parseTools(toolRequest(t, `"required"`, true))
	if err != nil {
		t.Fatal(err)
	}
	call, err := parseSelectedToolCall(`<|tool_call>call:search_knowledge{query:<|"|>coffee<|"|>}<tool_call|>`, policy)
	if err != nil || call.Name != "search_knowledge" || call.Arguments != `{"query":"coffee"}` {
		t.Fatalf("unexpected selected call: call=%+v err=%v", call, err)
	}
	if _, err := parseSelectedToolCall(`{"name":"unknown","arguments":{}}`, policy); err == nil {
		t.Fatal("unavailable model-selected tool was accepted")
	}
}

func TestRequiredBlockingResponseNormalizesCall(t *testing.T) {
	gin.SetMode(gin.TestMode)
	policy, _, err := parseTools(toolRequest(t, `"required"`, true))
	if err != nil {
		t.Fatal(err)
	}
	recorder := httptest.NewRecorder()
	context, _ := gin.CreateTestContext(recorder)
	writeBlockingResponse(context, `<|tool_call>call:search_knowledge{query:<|"|>coffee<|"|>}<tool_call|>`, "", geniex_sdk.ProfileData{}, policy)
	if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), `"finish_reason":"tool_calls"`) || !strings.Contains(recorder.Body.String(), "search_knowledge") {
		t.Fatalf("required call not normalized: status=%d body=%s", recorder.Code, recorder.Body.String())
	}
}

func TestRequiredBlockingResponseFailsClosed(t *testing.T) {
	gin.SetMode(gin.TestMode)
	policy, _, err := parseTools(toolRequest(t, `"required"`, true))
	if err != nil {
		t.Fatal(err)
	}
	recorder := closeNotifyRecorder{httptest.NewRecorder()}
	context, _ := gin.CreateTestContext(recorder)
	writeBlockingResponse(context, "I need to search first.", "", geniex_sdk.ProfileData{}, policy)
	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if strings.Contains(recorder.Body.String(), "I need to search") || !strings.Contains(recorder.Body.String(), requiredToolCallParseFailedCode) {
		t.Fatalf("required failure leaked prose or omitted code: %s", recorder.Body.String())
	}
}

func TestAutoBlockingResponseStillAllowsOrdinaryText(t *testing.T) {
	gin.SetMode(gin.TestMode)
	policy, _, err := parseTools(toolRequest(t, `"auto"`, true))
	if err != nil {
		t.Fatal(err)
	}
	recorder := httptest.NewRecorder()
	context, _ := gin.CreateTestContext(recorder)
	writeBlockingResponse(context, "ordinary answer", "", geniex_sdk.ProfileData{}, policy)
	if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), "ordinary answer") {
		t.Fatalf("auto text response changed: status=%d body=%s", recorder.Code, recorder.Body.String())
	}
}

func TestRequiredStreamingCallIsBufferedAndNormalized(t *testing.T) {
	gin.SetMode(gin.TestMode)
	policy, _, err := parseTools(toolRequest(t, `"required"`, true))
	if err != nil {
		t.Fatal(err)
	}
	data := make(chan string, 5)
	for _, piece := range []string{"<|tool_", "call>call:search_", "knowledge{query:<|\"|>", "coffee<|\"|>}<tool_", "call|>"} {
		data <- piece
	}
	close(data)
	recorder := closeNotifyRecorder{httptest.NewRecorder()}
	context, _ := gin.CreateTestContext(recorder)
	profile := geniex_sdk.ProfileData{}
	streamToolCall(context, data, func() error { return nil }, false, &profile, policy)
	body, err := io.ReadAll(recorder.Result().Body)
	if err != nil {
		t.Fatal(err)
	}
	text := string(body)
	if strings.Contains(text, "<|tool_call>") || !strings.Contains(text, "search_knowledge") || !strings.Contains(text, `"finish_reason":"tool_calls"`) {
		t.Fatalf("stream did not normalize selected call: %s", text)
	}
}

func TestRequiredStreamingFailureDoesNotFallbackToContent(t *testing.T) {
	gin.SetMode(gin.TestMode)
	policy, _, err := parseTools(toolRequest(t, `"required"`, true))
	if err != nil {
		t.Fatal(err)
	}
	data := make(chan string, 1)
	data <- "I need to search first."
	close(data)
	recorder := closeNotifyRecorder{httptest.NewRecorder()}
	context, _ := gin.CreateTestContext(recorder)
	profile := geniex_sdk.ProfileData{}
	streamToolCall(context, data, func() error { return nil }, false, &profile, policy)
	text := recorder.Body.String()
	if strings.Contains(text, "I need to search") || !strings.Contains(text, requiredToolCallParseFailedCode) || strings.Contains(text, `"finish_reason":"stop"`) {
		t.Fatalf("stream required failure fell back to text: %s", text)
	}
}
