// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// The deltas of one SSE stream, in order, with [DONE] dropped.
func sseDeltas(t *testing.T, body string) []streamChoice {
	t.Helper()
	var out []streamChoice
	for _, line := range strings.Split(body, "\n") {
		data, ok := strings.CutPrefix(line, "data:")
		if !ok || strings.TrimSpace(data) == "[DONE]" {
			continue
		}
		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("chunk %s: %v", data, err)
		}
		out = append(out, chunk.Choices...)
	}
	return out
}

// gin.Context.Stream waits on CloseNotify, which the plain recorder lacks.
type streamRecorder struct {
	*httptest.ResponseRecorder
	gone chan bool
}

func (r *streamRecorder) CloseNotify() <-chan bool { return r.gone }

func runStreamToolCall(t *testing.T, class tokenClass, tokens ...string) []streamChoice {
	t.Helper()
	gin.SetMode(gin.TestMode)
	w := &streamRecorder{httptest.NewRecorder(), make(chan bool)}
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/v1/chat/completions", nil)

	dataCh := make(chan string, len(tokens))
	for _, tok := range tokens {
		dataCh <- tok
	}
	close(dataCh)
	profile := geniex_sdk.ProfileData{StopReason: "eos"}
	streamToolCall(c, dataCh, func() error { return nil }, false, &profile, class)
	return sseDeltas(t, w.Body.String())
}

// One tool_calls delta, flattened for comparison.
type toolCallDelta struct {
	index     int
	name      string
	arguments string
}

// Thinking must not reach the scanner, and must not leak into content: with tools
// present it used to stream inline, and a call then dropped it along with the prose.
func TestStreamToolCallSeparatesReasoning(t *testing.T) {
	got := runStreamToolCall(t, reasoningClass(),
		"<think>", "the city ", "is Beijing", "</think>", "sure ",
		`{"name":"f","arg`, `uments":{"city":"Beijing"}}`, " done")

	var content, reasoning strings.Builder
	var calls []toolCallDelta
	for _, ch := range got {
		content.WriteString(ch.Delta.Content)
		reasoning.WriteString(ch.Delta.ReasoningContent)
		for _, tc := range ch.Delta.ToolCalls {
			calls = append(calls, toolCallDelta{int(tc.Index), tc.Function.Name, tc.Function.Arguments})
		}
	}
	if reasoning.String() != "the city is Beijing" {
		t.Errorf("reasoning = %q", reasoning.String())
	}
	if content.String() != "sure  done" {
		t.Errorf("content = %q", content.String())
	}
	if len(calls) != 1 || calls[0] != (toolCallDelta{0, "f", `{"city":"Beijing"}`}) {
		t.Errorf("calls = %+v", calls)
	}
	if reason := got[len(got)-1].FinishReason; reason == nil || *reason != "tool_calls" {
		t.Errorf("finish_reason = %v, want tool_calls", reason)
	}
}

// Parallel calls need a rising index, and plain text needs no tool_calls finish.
func TestStreamToolCallIndexes(t *testing.T) {
	got := runStreamToolCall(t, plainClass,
		`{"name":"a","arguments":{}}`, `{"name":"b","arguments":{}}`)
	var indexes []int64
	for _, ch := range got {
		for _, tc := range ch.Delta.ToolCalls {
			indexes = append(indexes, tc.Index)
		}
	}
	if len(indexes) != 2 || indexes[0] != 0 || indexes[1] != 1 {
		t.Errorf("indexes = %v, want [0 1]", indexes)
	}

	got = runStreamToolCall(t, plainClass, "no ", "call ", "here")
	last := got[len(got)-1]
	if last.FinishReason == nil || *last.FinishReason != "stop" {
		t.Errorf("finish_reason = %v, want stop", last.FinishReason)
	}
}
