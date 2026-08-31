// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"strings"

	"github.com/bytedance/sonic"
	"github.com/bytedance/sonic/ast"
)

// jsonToolCall is a bare JSON object, with nothing marking where it ends: the
// object it opens is the whole candidate, and parse decides whether it is a call.
type jsonToolCall struct {
	pos  int // bytes of all consumed
	obj  int // where the candidate object began; == pos when none is open
	end  int // where that object closed; 0 until it does
	walk braceWalk
}

func (t *jsonToolCall) parse(s string) []toolCallFn { return parseJSONToolCalls(s) }

func (t *jsonToolCall) feed(all string, from int) (int, int) {
	if from > t.obj { // the region was consumed or bypassed: start over
		t.pos, t.obj, t.end = from, from, 0
	}
	if t.end > 0 {
		return t.obj, t.end
	}
	if t.obj == t.pos {
		i := strings.IndexByte(all[t.pos:], '{')
		if i < 0 {
			t.obj, t.pos = len(all), len(all)
			return -1, -1
		}
		t.obj, t.pos, t.walk = t.pos+i, t.pos+i, braceWalk{}
	}
	n := t.walk.feed(all[t.pos:])
	if n < 0 {
		t.pos = len(all)
		return t.obj, -1 // still open: it may yet close into a call
	}
	t.pos += n
	t.end = t.pos
	return t.obj, t.end
}

// braceWalk tracks nesting depth across chunk boundaries, ignoring braces inside
// string literals. feed returns the offset past the '}' closing the first object.
type braceWalk struct {
	depth   int
	inStr   bool
	escaped bool
}

func (w *braceWalk) feed(s string) int {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if w.inStr {
			switch {
			case w.escaped:
				w.escaped = false
			case c == '\\':
				w.escaped = true
			case c == '"':
				w.inStr = false
			}
			continue
		}
		switch c {
		case '"':
			w.inStr = true
		case '{':
			w.depth++
		case '}':
			w.depth--
			if w.depth == 0 {
				return i + 1
			}
		}
	}
	return -1
}

// parseToolCallObject decodes one object, nil unless "name" is a string and
// "arguments" an object or a string.
func parseToolCallObject(obj string) *toolCallFn {
	name, err := sonic.GetFromString(obj, "name")
	if err != nil || name.TypeSafe() != ast.V_STRING {
		return nil
	}
	args, err := sonic.GetFromString(obj, "arguments")
	if err != nil {
		return nil
	}
	var raw string
	switch args.TypeSafe() {
	case ast.V_OBJECT:
		raw, _ = args.Raw()
	case ast.V_STRING:
		raw, _ = args.String()
	default:
		return nil
	}
	call := &toolCallFn{Arguments: raw}
	call.Name, _ = name.String()
	return call
}

// parseJSONToolCalls returns every object in s that is a tool call. Wrappers such
// as `<tool_call>` tags or code fences are transparent: only the objects matter.
func parseJSONToolCalls(s string) []toolCallFn {
	var calls []toolCallFn
	for i := 0; i < len(s); {
		j := strings.IndexByte(s[i:], '{')
		if j < 0 {
			break
		}
		i += j
		// s is complete here, so an object left open holds nothing: look inside.
		var w braceWalk
		end := w.feed(s[i:])
		if end < 0 {
			i++
			continue
		}
		if call := parseToolCallObject(s[i : i+end]); call != nil {
			calls = append(calls, *call)
		}
		i += end
	}
	return calls
}
