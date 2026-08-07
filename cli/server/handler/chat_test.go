// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"strings"
	"testing"

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

func TestReasoningSink(t *testing.T) {
	tokens := []string{"<think>", "reason", "</think>", "answer"}

	t.Run("separation splits think block", func(t *testing.T) {
		var content, reasoning strings.Builder
		sink := reasoningSink(thinkfsm.New(), &content, &reasoning)
		for _, tok := range tokens {
			sink(tok)
		}
		if got := content.String(); got != "answer" {
			t.Errorf("content = %q, want %q", got, "answer")
		}
		if got := reasoning.String(); got != "reason" {
			t.Errorf("reasoning = %q, want %q", got, "reason")
		}
	})

	t.Run("nil splitter keeps everything inline", func(t *testing.T) {
		var content, reasoning strings.Builder
		sink := reasoningSink(nil, &content, &reasoning)
		for _, tok := range tokens {
			sink(tok)
		}
		if got := content.String(); got != "<think>reason</think>answer" {
			t.Errorf("content = %q, want raw inline", got)
		}
		if got := reasoning.String(); got != "" {
			t.Errorf("reasoning = %q, want empty", got)
		}
	})
}
