// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"strings"

	"github.com/qualcomm/GenieX/cli/internal/thinkfsm"
)

// "" / "none" keep thinking inline in content (default); others separate it.
func reasoningSeparated(format string) bool {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "", "none":
		return false
	default:
		return true
	}
}

// tokenClass classifies a token; emit is false for a consumed marker (<think>).
type tokenClass func(token string) (text string, reasoning, emit bool)

// plainClass passes tokens through verbatim, keeping the default byte-identical.
func plainClass(token string) (string, bool, bool) {
	return token, false, true
}

func reasoningClass() tokenClass {
	s := thinkfsm.New()
	return func(token string) (string, bool, bool) {
		ev := s.Feed(token)
		if ev.Consumed {
			return "", false, false
		}
		return ev.Text, ev.Reasoning, true
	}
}

func sink(class tokenClass, content, reasoning *strings.Builder) func(string) bool {
	return func(token string) bool {
		if text, isReasoning, emit := class(token); emit {
			if isReasoning {
				reasoning.WriteString(text)
			} else {
				content.WriteString(text)
			}
		}
		return true
	}
}

type tokenRender func(token string) (chunk streamChunk, emit bool)

func render(class tokenClass) tokenRender {
	return func(token string) (streamChunk, bool) {
		text, reasoning, emit := class(token)
		if !emit {
			return streamChunk{}, false
		}
		return tokenChunk(text, reasoning), true
	}
}
