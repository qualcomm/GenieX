// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"strings"
	"testing"
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

// Mirrors the alias collapse ChatCompletions applies after binding.
func TestMaxCompletionTokensAlias(t *testing.T) {
	cases := []struct {
		name string
		body string
		want int64
	}{
		{"no cap in body falls back to default", `{}`, 2048},
		{"max_completion_tokens honoured", `{"max_completion_tokens":512}`, 512},
		{"deprecated max_tokens honoured", `{"max_tokens":512}`, 512},
		{"max_completion_tokens wins over max_tokens", `{"max_tokens":512,"max_completion_tokens":128}`, 128},
		{"explicit null falls back to default", `{"max_tokens":null}`, 2048},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := defaultChatCompletionRequest()
			if err := json.Unmarshal([]byte(tc.body), &p); err != nil {
				t.Fatalf("unmarshal %s: %v", tc.body, err)
			}
			if got := p.MaxCompletionTokens.Or(p.MaxTokens.Value); got != tc.want {
				t.Errorf("cap = %d, want %d", got, tc.want)
			}
		})
	}
}
