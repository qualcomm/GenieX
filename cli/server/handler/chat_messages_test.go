// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/server/utils"
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

func TestChatRequestPreservesReasoningContent(t *testing.T) {
	_, request := bindRequest(t, `{"messages":[{"role":"assistant","content":null,
		"reasoning_content":"Check the weather in Paris.",
		"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]}]}`)
	assistant := request.Messages[0].OfAssistant
	if assistant == nil {
		t.Fatal("assistant message was lost")
	}
	if got := assistant.ExtraFields()["reasoning_content"]; got != "Check the weather in Paris." {
		t.Fatalf("reasoning_content = %v, want the original assistant reasoning", got)
	}
}

func TestBuildMessagesReasoningContent(t *testing.T) {
	for _, tc := range []struct {
		name, message, reasoning, content string
	}{
		{"text", `"content":"answer","reasoning_content":"think first"`, "think first", "answer"},
		{"parts", `"content":[{"type":"text","text":"answer"}],"reasoning_content":"think first"`, "think first", "answer"},
		{"null content", `"content":null,"reasoning_content":"考える\nthen act"`, "考える\nthen act", ""},
		{"missing content", `"reasoning_content":"think first"`, "think first", ""},
		{"no reasoning", `"content":"answer"`, "", "answer"},
		{"null reasoning", `"content":"answer","reasoning_content":null`, "", "answer"},
		{"empty reasoning", `"content":"answer","reasoning_content":""`, "", "answer"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, request := bindRequest(t, `{"messages":[{"role":"assistant",`+tc.message+`}]}`)
			llm, ok := buildLLMMessages(c, request)
			if !ok || len(llm) != 1 {
				t.Fatalf("LLM message conversion failed: %+v", llm)
			}
			if llm[0].ReasoningContent != tc.reasoning || llm[0].Content != tc.content {
				t.Errorf("LLM message = %+v", llm[0])
			}
			vlm, files, ok := buildVLMMessages(c, request)
			if !ok || len(vlm) != 1 || len(files) != 0 {
				t.Fatalf("VLM message conversion failed: %+v, files=%v", vlm, files)
			}
			var content string
			for _, part := range vlm[0].Contents {
				content += part.Text
			}
			if vlm[0].ReasoningContent != tc.reasoning || content != tc.content {
				t.Errorf("VLM message = %+v", vlm[0])
			}
		})
	}
}

func TestReasoningToolRoundTripAndSessionKey(t *testing.T) {
	body := strings.Replace(toolLoopBody, `"content": null`, `"content": null, "reasoning_content": "List the folder."`, 1)
	c, request := bindRequest(t, body)
	llm, ok := buildLLMMessages(c, request)
	if !ok || len(llm) != 3 {
		t.Fatalf("LLM conversion failed: %+v", llm)
	}
	vlm, _, ok := buildVLMMessages(c, request)
	if !ok || len(vlm) != 3 {
		t.Fatalf("VLM conversion failed: %+v", vlm)
	}
	wantCall := geniex_sdk.ToolCall{ID: "call_1", Name: "bash", Arguments: `{"cmd":"ls"}`}
	if llm[1].ReasoningContent != "List the folder." || vlm[1].ReasoningContent != "List the folder." ||
		!reflect.DeepEqual(llm[1].ToolCalls, []geniex_sdk.ToolCall{wantCall}) ||
		!reflect.DeepEqual(vlm[1].ToolCalls, llm[1].ToolCalls) ||
		llm[2].ToolCallID != "call_1" || vlm[2].ToolCallID != "call_1" {
		t.Fatalf("reasoning or tool round trip lost: LLM=%+v VLM=%+v", llm, vlm)
	}
	llmKey := utils.SessionKeyOf(llm)
	vlmKey := utils.SessionKeyOfVLMRequest(request.Messages)
	llm[1].ReasoningContent = "Changed reasoning."
	request.Messages[1].OfAssistant.SetExtraFields(map[string]any{"reasoning_content": "Changed reasoning."})
	if reflect.DeepEqual(llmKey, utils.SessionKeyOf(llm)) ||
		reflect.DeepEqual(vlmKey, utils.SessionKeyOfVLMRequest(request.Messages)) {
		t.Fatal("changed reasoning must change the session key")
	}
}

func TestParallelToolCallsRequest(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		want       bool
	}{
		{name: "default", body: `{}`, want: true},
		{name: "true", body: `{"parallel_tool_calls":true}`, want: true},
		{name: "false", body: `{"parallel_tool_calls":false}`, want: false},
		{name: "null", body: `{"parallel_tool_calls":null}`, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, request := bindRequest(t, tc.body)
			if got := request.ParallelToolCalls.Or(true); got != tc.want {
				t.Errorf("parallel_tool_calls = %v, want %v", got, tc.want)
			}
		})
	}
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

func TestBuildVLMMessagesRejectsBadMediaAndCleansPartialFiles(t *testing.T) {
	tmp := t.TempDir()
	for _, name := range []string{"TMPDIR", "TMP", "TEMP"} {
		t.Setenv(name, tmp)
	}
	imagePath := filepath.Join(tmp, "image.png")
	if err := os.WriteFile(imagePath, []byte("image bytes"), 0o600); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct{ name, content string }{
		{"missing local image", fmt.Sprintf(`{"type":"image_url","image_url":{"url":%q}}`, filepath.Join(tmp, "missing.png"))},
		{"invalid base64 audio", `{"type":"input_audio","input_audio":{"data":"not-base64!","format":"wav"}}`},
		{"later input error after a saved image", fmt.Sprintf(`{"type":"image_url","image_url":{"url":%q}},{"type":"input_audio","input_audio":{"data":"bad!","format":"wav"}}`, imagePath)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, param := bindRequest(t, `{"model":"m","messages":[{"role":"user","content":[`+tc.content+`]}]}`)
			messages, tempFiles, ok := buildVLMMessages(c, param)
			if ok || messages != nil || tempFiles != nil {
				t.Fatalf("buildVLMMessages = %+v, %v, %v; want nil, nil, false", messages, tempFiles, ok)
			}
			if got := c.Writer.Status(); got != http.StatusBadRequest {
				t.Errorf("HTTP status = %d, want 400", got)
			}
			entries, err := os.ReadDir(tmp)
			if err != nil {
				t.Fatal(err)
			}
			if len(entries) != 1 || entries[0].Name() != "image.png" {
				t.Errorf("temp dir entries = %v, want only the source image", entries)
			}
		})
	}
}

func TestBuildVLMMessagesDecodesBase64Audio(t *testing.T) {
	audio := append([]byte("RIFF\x24\x00\x00\x00WAVEfmt "), make([]byte, 256)...)
	body := fmt.Sprintf(`{
	  "model": "m",
	  "messages": [{
	    "role": "user",
	    "content": [{
	      "type": "input_audio",
	      "input_audio": {"data": %q, "format": "wav"}
	    }]
	  }]
	}`, base64.StdEncoding.EncodeToString(audio))

	c, param := bindRequest(t, body)
	messages, tempFiles, ok := buildVLMMessages(c, param)
	for _, file := range tempFiles {
		defer os.Remove(file)
	}
	if !ok {
		t.Fatalf("buildVLMMessages failed, status %d", c.Writer.Status())
	}
	if len(messages) != 1 || len(messages[0].Contents) != 1 {
		t.Fatalf("messages = %+v, want one message with one audio part", messages)
	}
	if len(tempFiles) != 1 {
		t.Fatalf("tempFiles = %v, want one file", tempFiles)
	}

	got, err := os.ReadFile(tempFiles[0])
	if err != nil {
		t.Fatalf("read temporary audio: %v", err)
	}
	if string(got) != string(audio) {
		t.Errorf("temporary audio does not match decoded input")
	}
	if messages[0].Contents[0].Type != geniex_sdk.VlmContentTypeAudio {
		t.Errorf("content type = %q, want audio", messages[0].Contents[0].Type)
	}
	if messages[0].Contents[0].Text != tempFiles[0] {
		t.Errorf("content path = %q, want %q", messages[0].Contents[0].Text, tempFiles[0])
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
	want := geniex_sdk.ToolCall{ID: "c1", Name: "ls", Arguments: "{}"}
	if len(messages[0].ToolCalls) != 1 || messages[0].ToolCalls[0] != want {
		t.Errorf("tool calls = %+v, want [%+v]", messages[0].ToolCalls, want)
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
