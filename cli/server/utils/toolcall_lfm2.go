// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"strings"

	"github.com/bytedance/sonic"
)

// LFM2 and LFM2.5 write all their calls as one Python-style list wrapped in a
// pair of control tokens:
//
//	<|tool_call_start|>[get_time(city="Paris"), toggle(enabled=True)]<|tool_call_end|>
//
// Both are grammar-only tokens the model has no other way to produce, so seeing
// one is already the whole signal — the boundary is the marker pair, not the
// brackets, same as Qwen3's `<tool_call>` wrapping plain JSON. Values are Python
// literals — quoted strings, numbers, True / False / None — and JSON's true /
// false / null. Grammar: llama.cpp's common_chat_params_init_lfm2.
const (
	lfm2ToolCallStart = "<|tool_call_start|>"
	lfm2ToolCallEnd   = "<|tool_call_end|>"
)

type lfm2ToolCall struct {
	markerFormat
}

func newLFM2ToolCall() *lfm2ToolCall {
	return &lfm2ToolCall{newMarkerFormat(lfm2ToolCallStart, lfm2ToolCallEnd)}
}

func (t *lfm2ToolCall) parse(s string) []toolCallFn {
	s = strings.TrimPrefix(s, lfm2ToolCallStart)
	s = strings.TrimSuffix(s, lfm2ToolCallEnd)
	return parseLFM2ToolCalls(s)
}

// parseLFM2ToolCalls returns every call in the list s holds, nothing unless the
// whole list matches the grammar: one call read out of a malformed list is more
// likely a misread than a call the model meant.
func parseLFM2ToolCalls(s string) []toolCallFn {
	i := skipSpace(s, 0)
	if i >= len(s) || s[i] != '[' {
		return nil
	}

	var calls []toolCallFn
	for i = skipSpace(s, i+1); ; {
		call, next := parseLFM2Call(s, i)
		if next < 0 {
			return nil
		}
		calls = append(calls, call)
		i = skipSpace(s, next)
		if i >= len(s) {
			return nil
		}
		switch s[i] {
		case ',':
			i = skipSpace(s, i+1)
		case ']':
			return calls
		default:
			return nil
		}
	}
}

// parseLFM2Call reads one `NAME(key=value, ...)` call plus the index past it.
func parseLFM2Call(s string, i int) (toolCallFn, int) {
	name := i + lfm2NameEnd(s[i:])
	if name == i || name >= len(s) || s[name] != '(' {
		return toolCallFn{}, -1
	}
	args, next := parseSeq(s, name, '(', ')', func(s string, i int) (string, int) {
		return parseLFM2Pair(s, i, true)
	})
	if next < 0 {
		return toolCallFn{}, -1
	}
	// parseSeq gives the list back parenthesised; arguments are a JSON object.
	return toolCallFn{Name: s[i:name], Arguments: "{" + args[1:len(args)-1] + "}"}, next
}

// parseLFM2Pair reads one key/value pair as `"key":value`: a call's key=value
// argument if bare, a dict's "key": value member otherwise.
func parseLFM2Pair(s string, i int, bare bool) (string, int) {
	key, sep, next := "", byte('='), 0
	if bare {
		if next = i + lfm2NameEnd(s[i:]); next == i {
			return "", -1
		}
		key = s[i:next]
	} else {
		sep = ':'
		if key, next = parseLFM2String(s, i); next < 0 {
			return "", -1
		}
		next = skipSpace(s, next) // the dict grammar allows space before ':', unlike bare key=value
	}
	if next >= len(s) || s[next] != sep {
		return "", -1
	}
	val, end := parseLFM2Value(s, skipSpace(s, next+1))
	if end < 0 {
		return "", -1
	}
	quoted, _ := sonic.MarshalString(key)
	return quoted + ":" + val, end
}

// parseLFM2Value reads one Python literal at s[i] as JSON plus the index past it.
func parseLFM2Value(s string, i int) (string, int) {
	switch {
	case i >= len(s):
		return "", -1

	case s[i] == '"' || s[i] == '\'':
		val, next := parseLFM2String(s, i)
		if next < 0 {
			return "", -1
		}
		quoted, _ := sonic.MarshalString(val)
		return quoted, next

	case s[i] == '{':
		return parseSeq(s, i, '{', '}', func(s string, i int) (string, int) {
			return parseLFM2Pair(s, i, false)
		})
	case s[i] == '[':
		return parseSeq(s, i, '[', ']', parseLFM2Value)

	default: // a number or a bool / null in either casing
		start := i
		for i < len(s) && s[i] != ',' && s[i] != ')' && s[i] != '}' && s[i] != ']' {
			i++
		}
		switch tok := strings.TrimSpace(s[start:i]); tok {
		case "True":
			return "true", i
		case "False":
			return "false", i
		case "None":
			return "null", i
		default:
			var v any
			if tok == "" || sonic.UnmarshalString(tok, &v) != nil {
				return "", -1
			}
			return tok, i
		}
	}
}

// parseLFM2String decodes one quoted literal, either quoting style, plus the index
// past it. Escapes decode to the characters they name, so the JSON encoding of the
// result carries a real newline as \n rather than a doubled backslash.
func parseLFM2String(s string, i int) (string, int) {
	if i >= len(s) || (s[i] != '"' && s[i] != '\'') {
		return "", -1
	}
	quote := s[i]
	var b strings.Builder
	for i++; i < len(s); i++ {
		switch c := s[i]; {
		case c == quote:
			return b.String(), i + 1
		case c != '\\':
			b.WriteByte(c)
		case i+1 < len(s):
			i++
			switch e := s[i]; e {
			case 'n':
				b.WriteByte('\n')
			case 't':
				b.WriteByte('\t')
			case 'r':
				b.WriteByte('\r')
			default: // \\ , \" , \' and anything else stands for itself
				b.WriteByte(e)
			}
		}
	}
	return "", -1
}

// lfm2NameEnd is the length of the function or argument name at the head of s.
// Dotted names such as Calendar.create_event are one name.
func lfm2NameEnd(s string) int {
	i := 0
	for i < len(s) {
		c := s[i]
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' ||
			c == '_' || c == '.') {
			break
		}
		i++
	}
	return i
}
