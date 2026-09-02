// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"strings"

	"github.com/bytedance/sonic"
)

// LFM2 and LFM2.5 write all their calls as one Python-style list:
//
//	<|tool_call_start|>[get_time(city="Paris"), toggle(enabled=True)]<|tool_call_end|>
//
// Both markers are control tokens, which the runtime detokenizes to nothing, so
// only the bare list arrives and `[NAME(` is what identifies it — a marker that
// does reach here goes out as text. Values are Python literals — quoted strings,
// numbers, True / False / None — and JSON's true / false / null. Grammar:
// llama.cpp's common_chat_params_init_lfm2.
type lfm2ToolCall struct {
	begin markerScan
	held  int // the offset last reported, so a stale scan is noticed
	at    int // where the list began; -1 while no list is open
	end   int // where it closed; -1 until it does
}

func newLFM2ToolCall() *lfm2ToolCall {
	t := &lfm2ToolCall{begin: markerScan{marker: "["}}
	t.reset(0)
	return t
}

func (t *lfm2ToolCall) reset(from int) {
	t.begin.reset(from)
	t.held, t.at, t.end = from, -1, -1
}

func (t *lfm2ToolCall) parse(s string) []toolCallFn { return parseLFM2ToolCalls(s) }

func (t *lfm2ToolCall) feed(all string, from int) (int, int) {
	if from > t.held { // the region was consumed or bypassed: start over
		t.reset(from)
	}
	at, end := t.scan(all)
	if at < 0 {
		t.held = len(all)
	} else {
		t.held = at
	}
	return at, end
}

func (t *lfm2ToolCall) scan(all string) (int, int) {
	for t.at < 0 {
		t.begin.feed(all)
		if t.begin.done == 0 {
			if t.begin.start < len(all) {
				return t.begin.start, -1 // a '[' that has not arrived whole
			}
			return -1, -1
		}
		// A '[' opens far too much prose to claim on its own: the list has to name
		// a function before this is taken for a call.
		at := t.begin.done - 1
		ok, settled := isCallList(all, at)
		if !settled {
			return at, -1
		}
		if !ok {
			t.begin.reset(at + 1) // prose: keep looking past it
			continue
		}
		t.at = at
	}
	if t.end < 0 {
		n := listEnd(all[t.at:])
		if n < 0 {
			return t.at, -1 // the list is still open
		}
		t.end = t.at + n
	}
	return t.at, t.end
}

// isCallList reports whether s[i:] opens a list of Python calls. settled is false
// while too little has arrived to tell, so the caller has to wait for more input.
func isCallList(s string, i int) (ok, settled bool) {
	if i >= len(s) {
		return false, false
	}
	if s[i] != '[' {
		return false, true
	}
	i = skipSpace(s, i+1)
	name := i + lfm2NameEnd(s[i:])
	if name >= len(s) {
		return false, false
	}
	return name > i && s[name] == '(', true
}

// listEnd is the offset past the ']' closing the list s opens, or -1 while it is
// still open. Brackets inside a string argument do not count.
func listEnd(s string) int {
	depth, inStr, quote, escaped := 0, false, byte(0), false
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case inStr && escaped:
			escaped = false
		case inStr && c == '\\':
			escaped = true
		case inStr:
			inStr = c != quote
		case c == '"' || c == '\'':
			inStr, quote = true, c
		case c == '[' || c == '{' || c == '(':
			depth++
		case c == ']' || c == '}' || c == ')':
			depth--
			if depth == 0 {
				return i + 1
			}
		}
	}
	return -1
}

// parseLFM2ToolCalls returns every call in the list s holds, nothing unless the
// whole list matches the grammar: one call read out of a malformed list is more
// likely a misread than a call the model meant.
func parseLFM2ToolCalls(s string) []toolCallFn {
	i := skipSpace(s, 0)
	if ok, _ := isCallList(s, i); !ok {
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
	args, next := parseSeq(s, name, '(', ')', parseLFM2Arg)
	if next < 0 {
		return toolCallFn{}, -1
	}
	// parseSeq gives the list back parenthesised; arguments are a JSON object.
	return toolCallFn{Name: s[i:name], Arguments: "{" + args[1:len(args)-1] + "}"}, next
}

// parseLFM2Arg reads one `key=value` argument as `"key":value`.
func parseLFM2Arg(s string, i int) (string, int) {
	key := i + lfm2NameEnd(s[i:])
	if key == i || key >= len(s) || s[key] != '=' {
		return "", -1
	}
	val, next := parseLFM2Value(s, skipSpace(s, key+1))
	if next < 0 {
		return "", -1
	}
	quoted, _ := sonic.MarshalString(s[i:key])
	return quoted + ":" + val, next
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
		return parseSeq(s, i, '{', '}', parseLFM2Member)
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

// parseLFM2Member reads one `"key": value` pair of a dict.
func parseLFM2Member(s string, i int) (string, int) {
	key, next := parseLFM2String(s, i)
	if next < 0 {
		return "", -1
	}
	i = skipSpace(s, next)
	if i >= len(s) || s[i] != ':' {
		return "", -1
	}
	val, next := parseLFM2Value(s, skipSpace(s, i+1))
	if next < 0 {
		return "", -1
	}
	quoted, _ := sonic.MarshalString(key)
	return quoted + ":" + val, next
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
