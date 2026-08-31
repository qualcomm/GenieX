// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"strings"

	"github.com/bytedance/sonic"
)

// Gemma 4 wraps each call in `<|tool_call>call:NAME{key:value,...}<tool_call|>`.
// The body is a custom dict, not JSON: strings are wrapped in `<|"|>...<|"|>`, keys
// are bare up to ':', and scalars are verbatim. Parallel calls are separate
// wrappers, so a region still holds exactly one. Grammar: llama.cpp's
// common_chat_params_init_gemma4.
const (
	gemma4Open   = "<|tool_call>call:"
	gemma4Close  = "<tool_call|>"
	gemma4String = `<|"|>`
)

type gemma4ToolCall struct {
	markerFormat
}

func newGemma4ToolCall() *gemma4ToolCall {
	return &gemma4ToolCall{newMarkerFormat(gemma4Open, gemma4Close)}
}

func (t *gemma4ToolCall) parse(s string) []toolCallFn { return parseGemma4ToolCalls(s) }

// parseGemma4ToolCalls returns every call in s. The end marker is not required: a
// call whose dict closed is complete, so Tail can still recover a truncated one.
func parseGemma4ToolCalls(s string) []toolCallFn {
	var calls []toolCallFn
	for {
		i := strings.Index(s, gemma4Open)
		if i < 0 {
			return calls
		}
		s = s[i+len(gemma4Open):] // a malformed call resumes at the next marker
		brace := strings.IndexByte(s, '{')
		if brace < 0 {
			return calls
		}
		name := strings.TrimSpace(s[:brace])
		args, next := parseGemma4Seq(s, brace, '{', '}', parseGemma4Member)
		if name == "" || next < 0 {
			continue
		}
		calls = append(calls, toolCallFn{Name: name, Arguments: args})
		s = s[next:]
	}
}

// parseGemma4Value reads one value at s[i] as JSON plus the index past it, or a
// negative index when s does not match the grammar. The parsers below all do.
func parseGemma4Value(s string, i int) (string, int) {
	switch {
	case i >= len(s):
		return "", -1

	case strings.HasPrefix(s[i:], gemma4String):
		i += len(gemma4String)
		end := strings.Index(s[i:], gemma4String)
		if end < 0 {
			return "", -1
		}
		quoted, _ := sonic.MarshalString(s[i : i+end])
		return quoted, i + end + len(gemma4String)

	case s[i] == '{':
		return parseGemma4Seq(s, i, '{', '}', parseGemma4Member)
	case s[i] == '[':
		return parseGemma4Seq(s, i, '[', ']', parseGemma4Value)

	default: // a number, bool or null runs to the next ',', '}' or ']'
		start := i
		for i < len(s) && s[i] != ',' && s[i] != '}' && s[i] != ']' {
			i++
		}
		tok := strings.TrimSpace(s[start:i])
		var v any
		if tok == "" || sonic.UnmarshalString(tok, &v) != nil {
			return "", -1
		}
		return tok, i
	}
}

// parseGemma4Member reads one `key:value` pair as `"key":value`. The key grammar is
// `[^:}]+`, so a member missing its value fails instead of swallowing later text.
func parseGemma4Member(s string, i int) (string, int) {
	colon := strings.IndexByte(s[i:], ':')
	if colon < 0 || strings.IndexByte(s[i:i+colon], '}') >= 0 {
		return "", -1
	}
	key := strings.TrimSpace(s[i : i+colon])
	if key == "" {
		return "", -1
	}
	val, next := parseGemma4Value(s, skipGemma4Space(s, i+colon+1))
	if next < 0 {
		return "", -1
	}
	quoted, _ := sonic.MarshalString(key)
	return quoted + ":" + val, next
}

// parseGemma4Seq reads a comma-separated `open ... shut` sequence of elem. s[i] is
// open, which the caller has checked.
func parseGemma4Seq(s string, i int, open, shut byte,
	elem func(string, int) (string, int)) (string, int) {
	var b strings.Builder
	b.WriteByte(open)

	i = skipGemma4Space(s, i+1)
	if i < len(s) && s[i] == shut {
		b.WriteByte(shut)
		return b.String(), i + 1
	}
	for {
		enc, next := elem(s, skipGemma4Space(s, i))
		if next < 0 {
			return "", -1
		}
		b.WriteString(enc)
		i = skipGemma4Space(s, next)
		if i >= len(s) {
			return "", -1
		}
		switch s[i] {
		case ',':
			i++
			b.WriteByte(',')
		case shut:
			b.WriteByte(shut)
			return b.String(), i + 1
		default:
			return "", -1
		}
	}
}

func skipGemma4Space(s string, i int) int {
	for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r') {
		i++
	}
	return i
}
