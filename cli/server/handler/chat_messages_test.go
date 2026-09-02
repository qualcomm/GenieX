// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// One round of an agentic loop: the assistant issues a tool call, the harness
// attaches the result as a `tool` message.
const toolLoopBody = `{
  "model": "m",
  "messages": [
    {"role": "user", "content": "what is in this folder?"},
    {"role": "assistant", "content": null,
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "numerics-summary.txt"}
  ]
}`

func bindRequest(t *testing.T, body string) (*gin.Context, ChatCompletionRequest) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	param := defaultChatCompletionRequest()
	if err := c.ShouldBindJSON(&param); err != nil {
		t.Fatalf("bind: %v", err)
	}
	return c, param
}

// Regression: tool calls used to be flattened into assistant text, which left
// the following `tool` message invisible to templates that render a tool
// response only from the tool_calls of the turn before it.
func TestBuildLLMMessagesKeepsToolRoundTrip(t *testing.T) {
	c, param := bindRequest(t, toolLoopBody)
	messages, ok := buildLLMMessages(c, param)
	if !ok {
		t.Fatalf("buildLLMMessages failed, status %d", c.Writer.Status())
	}
	if len(messages) != 3 {
		t.Fatalf("len(messages) = %d, want 3: %+v", len(messages), messages)
	}

	assistant := messages[1]
	if assistant.Role != geniex_sdk.LlmRoleAssistant {
		t.Errorf("assistant role = %q", assistant.Role)
	}
	if assistant.Content != "" {
		t.Errorf("assistant content = %q, want empty (no flattened tool_call text)", assistant.Content)
	}
	want := geniex_sdk.ToolCall{ID: "call_1", Name: "bash", Arguments: `{"cmd":"ls"}`}
	if len(assistant.ToolCalls) != 1 || assistant.ToolCalls[0] != want {
		t.Errorf("assistant tool calls = %+v, want [%+v]", assistant.ToolCalls, want)
	}

	tool := messages[2]
	if tool.Role != geniex_sdk.LlmRoleTool {
		t.Errorf("tool role = %q", tool.Role)
	}
	if tool.ToolCallID != "call_1" {
		t.Errorf("tool call id = %q, want call_1", tool.ToolCallID)
	}
	if tool.Content != "numerics-summary.txt" {
		t.Errorf("tool content = %q", tool.Content)
	}
}

func TestBuildVLMMessagesKeepsToolRoundTrip(t *testing.T) {
	c, param := bindRequest(t, toolLoopBody)
	messages, tempFiles, ok := buildVLMMessages(c, param)
	if !ok {
		t.Fatalf("buildVLMMessages failed, status %d", c.Writer.Status())
	}
	if len(tempFiles) != 0 {
		t.Errorf("tempFiles = %v, want none", tempFiles)
	}
	if len(messages) != 3 {
		t.Fatalf("len(messages) = %d, want 3: %+v", len(messages), messages)
	}

	assistant := messages[1]
	if len(assistant.Contents) != 0 {
		t.Errorf("assistant contents = %+v, want none", assistant.Contents)
	}
	want := geniex_sdk.ToolCall{ID: "call_1", Name: "bash", Arguments: `{"cmd":"ls"}`}
	if len(assistant.ToolCalls) != 1 || assistant.ToolCalls[0] != want {
		t.Errorf("assistant tool calls = %+v, want [%+v]", assistant.ToolCalls, want)
	}

	tool := messages[2]
	if tool.ToolCallID != "call_1" {
		t.Errorf("tool call id = %q, want call_1", tool.ToolCallID)
	}
	if len(tool.Contents) != 1 || tool.Contents[0].Text != "numerics-summary.txt" {
		t.Errorf("tool contents = %+v", tool.Contents)
	}
}

// An assistant turn may carry prose alongside its tool calls; both must survive.
func TestBuildLLMMessagesKeepsAssistantTextWithToolCalls(t *testing.T) {
	body := `{
	  "model": "m",
	  "messages": [
	    {"role": "assistant", "content": "let me look",
	     "tool_calls": [{"id": "c1", "type": "function",
	                     "function": {"name": "ls", "arguments": "{}"}}]}
	  ]
	}`
	c, param := bindRequest(t, body)
	messages, ok := buildLLMMessages(c, param)
	if !ok {
		t.Fatalf("buildLLMMessages failed, status %d", c.Writer.Status())
	}
	if len(messages) != 1 {
		t.Fatalf("len(messages) = %d, want 1", len(messages))
	}
	if messages[0].Content != "let me look" {
		t.Errorf("content = %q, want %q", messages[0].Content, "let me look")
	}
	if len(messages[0].ToolCalls) != 1 {
		t.Errorf("tool calls = %+v, want 1", messages[0].ToolCalls)
	}
}

// Tool results may arrive as a content-part array instead of a bare string.
func TestBuildLLMMessagesToolContentParts(t *testing.T) {
	body := `{
	  "model": "m",
	  "messages": [
	    {"role": "tool", "tool_call_id": "c1",
	     "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
	  ]
	}`
	c, param := bindRequest(t, body)
	messages, ok := buildLLMMessages(c, param)
	if !ok {
		t.Fatalf("buildLLMMessages failed, status %d", c.Writer.Status())
	}
	if len(messages) != 1 || messages[0].Content != "ab" {
		t.Fatalf("messages = %+v, want one message with content \"ab\"", messages)
	}
}
