// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"strings"
	"testing"
)

// feedAll drives a stopMatcher token by token and returns the concatenated
// safe text plus whether a stop sequence matched, stopping early (as a real
// caller would) once feed reports a match.
func feedAll(m *stopMatcher, tokens []string) (string, bool) {
	var out strings.Builder
	for _, tok := range tokens {
		safe, stopped := m.feed(tok)
		out.WriteString(safe)
		if stopped {
			return out.String(), true
		}
	}
	return out.String(), false
}

func TestStopMatcher(t *testing.T) {
	tests := []struct {
		name     string
		stops    []string
		tokens   []string
		wantText string
		wantStop bool
	}{
		{"no stops configured", nil, []string{"hello", " world"}, "hello world", false},
		{"no match", []string{"STOP"}, []string{"hello", " world"}, "hello world", false},
		{"match within a single token", []string{"STOP"}, []string{"abc", "defSTOPghi"}, "abcdef", true},
		{
			// The FIM repro from #1341: an editor's stop list, split across
			// streamed token boundaries the way real generation delivers it.
			"stop sequence split across tokens",
			[]string{"<fim_prefix>", "<fim_suffix>", "<fim_middle>", "<|endoftext|>"},
			[]string{"def main():\n    print(", "<fim", "_suffix", ">", ")\n"},
			"def main():\n    print(",
			true,
		},
		{"earliest of several stops wins", []string{"<fim_suffix>", "<fim_prefix>"}, []string{"abc<fim_prefix>def<fim_suffix>ghi"}, "abc", true},
		{"empty stop string is ignored", []string{""}, []string{"hello"}, "hello", false},
		{"match at the very first token", []string{"STOP"}, []string{"STOPtrailing"}, "", true},
		{"partial match that never completes is still flushed", []string{"<fim_prefix>"}, []string{"abc<fim", "_notit"}, "abc<fim_notit", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, stopped := feedAll(newStopMatcher(tt.stops), tt.tokens)
			if got != tt.wantText || stopped != tt.wantStop {
				t.Errorf("feedAll() = (%q, %v), want (%q, %v)", got, stopped, tt.wantText, tt.wantStop)
			}
		})
	}
}

func TestStopMatcherHoldsBackAmbiguousSuffix(t *testing.T) {
	// After a token ending in a proper prefix of the stop sequence, that
	// prefix must not be emitted yet — the next token could complete it.
	m := newStopMatcher([]string{"<fim_prefix>"})
	safe, stopped := m.feed("abc<fim")
	if safe != "abc" || stopped {
		t.Fatalf("feed() = (%q, %v), want (\"abc\", false)", safe, stopped)
	}
}
