// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/thinkfsm"
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

	t.Run("splitter separates think block", func(t *testing.T) {
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

func TestPlainClassEvent(t *testing.T) {
	for _, token := range []string{"", "text", "<think>", "</think>", "<|message|>", "<|end|>", "<|call|>"} {
		want := thinkfsm.Event{Kind: thinkfsm.Content, Text: token}
		if got := plainClass(token); got != want {
			t.Errorf("plainClass(%q) = %+v, want %+v", token, got, want)
		}
	}
}

func TestReasoningClassInstancesAreIndependent(t *testing.T) {
	a, b := reasoningClass(), reasoningClass()
	a("<think>")
	if got := b("answer"); got != (thinkfsm.Event{Kind: thinkfsm.Content, Text: "answer"}) {
		t.Errorf("new classifier=%+v, want independent content state", got)
	}
	if got := a("reason"); got != (thinkfsm.Event{Kind: thinkfsm.Thinking, Text: "reason"}) {
		t.Errorf("first classifier=%+v, want preserved thinking state", got)
	}
}

func TestTokenEventConsumers(t *testing.T) {
	cases := []struct {
		event thinkfsm.Event
		emit  bool
	}{
		{thinkfsm.Event{Kind: thinkfsm.EnterThinking}, false},
		{thinkfsm.Event{Kind: thinkfsm.Thinking, Text: "reason"}, true},
		{thinkfsm.Event{Kind: thinkfsm.ToolCall, Text: " to=functions.f"}, false},
		{thinkfsm.Event{Kind: thinkfsm.ToolCallStart}, false},
		{thinkfsm.Event{Kind: thinkfsm.ToolCallEnd}, false},
		{thinkfsm.Event{Kind: thinkfsm.Consume, Text: "ignored"}, false},
		{thinkfsm.Event{Kind: thinkfsm.LeaveThinking}, false},
		{thinkfsm.Event{Kind: thinkfsm.Content, Text: "answer"}, true},
		{thinkfsm.Event{Kind: thinkfsm.Content}, true},
	}
	var content, reasoning strings.Builder
	for _, tc := range cases {
		class := tokenClass(func(string) thinkfsm.Event { return tc.event })
		sink(class, &content, &reasoning)("original token")
		gin.SetMode(gin.TestMode)
		w := &streamRecorder{httptest.NewRecorder(), make(chan bool)}
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest("POST", "/v1/chat/completions", nil)
		dataCh := make(chan string, 1)
		dataCh <- "original token"
		close(dataCh)
		profile := geniex_sdk.ProfileData{StopReason: "eos"}
		streamPlainText(c, dataCh, func() error { return nil }, false, &profile, class, "test-model")
		choices := sseDeltas(t, w.Body.String())
		wantChoices := 1 // terminal chunk only for internal events
		if tc.emit {
			wantChoices++
		}
		if len(choices) != wantChoices {
			t.Fatalf("streamPlainText(%+v) choices = %+v, want %d", tc.event, choices, wantChoices)
		}
		if reason := choices[len(choices)-1].FinishReason; reason == nil || *reason != "stop" {
			t.Errorf("streamPlainText(%+v) finish_reason = %v, want stop", tc.event, reason)
		}
		if !tc.emit {
			continue
		}
		want := streamDelta{Role: "assistant"}
		if tc.event.Kind == thinkfsm.Thinking {
			want.ReasoningContent = tc.event.Text
		} else {
			want.Content = tc.event.Text
		}
		got := choices[0].Delta
		if got.Role != want.Role || got.Content != want.Content || got.ReasoningContent != want.ReasoningContent || len(got.ToolCalls) != 0 {
			t.Errorf("streamPlainText(%+v) delta = %+v, want %+v", tc.event, got, want)
		}
	}
	if content.String() != "answer" || reasoning.String() != "reason" {
		t.Errorf("sink content = %q, reasoning = %q", content.String(), reasoning.String())
	}
}
