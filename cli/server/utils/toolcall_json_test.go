// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import "testing"

func TestParseJSONToolCalls(t *testing.T) {
	tests := []struct {
		name string
		resp string
		want []toolCallFn // nil: no tool call
	}{
		{
			name: "bare json object",
			resp: `{"name": "get_weather", "arguments": {"city": "Beijing"}}`,
			want: []toolCallFn{{Name: "get_weather", Arguments: `{"city": "Beijing"}`}},
		},
		{
			name: "json with surrounding prose",
			resp: `Sure, let me call it. {"name": "get_weather", "arguments": {"city": "Beijing"}} Done.`,
			want: []toolCallFn{{Name: "get_weather", Arguments: `{"city": "Beijing"}`}},
		},
		{
			name: "string arguments",
			resp: `{"name": "echo", "arguments": "hello"}`,
			want: []toolCallFn{{Name: "echo", Arguments: "hello"}},
		},
		{
			name: "nested braces in arguments",
			resp: `{"name": "f", "arguments": {"a": {"b": {"c": 1}}}}`,
			want: []toolCallFn{{Name: "f", Arguments: `{"a": {"b": {"c": 1}}}`}},
		},
		{
			name: "deeply nested arguments",
			resp: `prose {"name": "deep", "arguments": {"l1": {"l2": {"l3": {"l4": {"l5": {"v": [1, {"x": 2}]}}}}}}} tail`,
			want: []toolCallFn{{Name: "deep", Arguments: `{"l1": {"l2": {"l3": {"l4": {"l5": {"v": [1, {"x": 2}]}}}}}}`}},
		},
		{
			name: "nested objects with braces inside strings",
			resp: `{"name": "g", "arguments": {"outer": {"inner": {"note": "close } here { and {} there"}}}}`,
			want: []toolCallFn{{Name: "g", Arguments: `{"outer": {"inner": {"note": "close } here { and {} there"}}}`}},
		},
		{
			name: "array of nested objects as arguments",
			resp: `{"name": "batch", "arguments": {"items": [{"a": {"b": 1}}, {"c": {"d": 2}}]}}`,
			want: []toolCallFn{{Name: "batch", Arguments: `{"items": [{"a": {"b": 1}}, {"c": {"d": 2}}]}`}},
		},
		{
			name: "braces inside string literal ignored",
			resp: `{"name": "say", "arguments": {"text": "a } b { c"}}`,
			want: []toolCallFn{{Name: "say", Arguments: `{"text": "a } b { c"}`}},
		},
		{
			name: "escaped quote inside string",
			resp: `{"name": "say", "arguments": {"text": "quote \" and } brace"}}`,
			want: []toolCallFn{{Name: "say", Arguments: `{"text": "quote \" and } brace"}`}},
		},
		{
			name: "skips leading object without name",
			resp: `{"thinking": "hmm"} then {"name": "go", "arguments": {}}`,
			want: []toolCallFn{{Name: "go", Arguments: `{}`}},
		},
		{
			name: "skips object with name but no arguments",
			resp: `{"name": "no_args"} {"name": "real", "arguments": {"x": 1}}`,
			want: []toolCallFn{{Name: "real", Arguments: `{"x": 1}`}},
		},
		{
			name: "adjacent tool calls",
			resp: `{"name": "first", "arguments": {"a": 1}}{"name": "second", "arguments": {"b": 2}}`,
			want: []toolCallFn{{Name: "first", Arguments: `{"a": 1}`}, {Name: "second", Arguments: `{"b": 2}`}},
		},
		{
			name: "skips candidate with array arguments",
			resp: `{"name": "bad", "arguments": [1, 2, 3]} {"name": "good", "arguments": {"ok": 1}}`,
			want: []toolCallFn{{Name: "good", Arguments: `{"ok": 1}`}},
		},
		{
			name: "skips candidate with numeric arguments",
			resp: `{"name": "bad", "arguments": 42} {"name": "good", "arguments": "text"}`,
			want: []toolCallFn{{Name: "good", Arguments: "text"}},
		},
		{
			name: "skips candidate with non-string name",
			resp: `{"name": 7, "arguments": {}} {"name": "good", "arguments": {}}`,
			want: []toolCallFn{{Name: "good", Arguments: `{}`}},
		},
		{
			name: "all candidates invalid falls through to error",
			resp: `{"name": "a", "arguments": [1]} {"name": 2, "arguments": {}} {"name": "c"}`,
		},
		{
			name: "tool calls separated by prose",
			resp: `Call one: {"name": "first", "arguments": {}}. Then: {"name": "second", "arguments": {}}`,
			want: []toolCallFn{{Name: "first", Arguments: `{}`}, {Name: "second", Arguments: `{}`}},
		},
		{
			name: "leading unterminated brace does not swallow later call",
			resp: `reasoning { partial and never closed ... {"name": "go", "arguments": {"ok": true}}`,
			want: []toolCallFn{{Name: "go", Arguments: `{"ok": true}`}},
		},
		{
			name: "stray closing braces before real call",
			resp: `garbage } } more } {"name": "go", "arguments": {}}`,
			want: []toolCallFn{{Name: "go", Arguments: `{}`}},
		},
		{
			name: "name with escaped chars",
			resp: `{"name": "ns\\tool", "arguments": {"path": "C:\\tmp"}}`,
			want: []toolCallFn{{Name: `ns\tool`, Arguments: `{"path": "C:\\tmp"}`}},
		},
		{
			name: "empty object skipped then real call",
			resp: `{} {"name": "go", "arguments": {}}`,
			want: []toolCallFn{{Name: "go", Arguments: `{}`}},
		},
		{
			// a wrapper is transparent: only the object inside it matters
			name: "json inside code fence",
			resp: "here you go:\n```json\n{\"name\": \"fenced\", \"arguments\": {\"q\": \"x\"}}\n```",
			want: []toolCallFn{{Name: "fenced", Arguments: `{"q": "x"}`}},
		},
		{
			name: "no json object",
			resp: "just some plain text without any call",
		},
		{
			name: "unterminated object",
			resp: `{"name": "go", "arguments":`,
		},
		{
			name: "only non-tool-call objects",
			resp: `{"foo": 1} {"bar": 2} {"name": 42}`,
		},
		{
			name: "empty string",
			resp: "",
		},
		{
			name: "braces only inside a string literal",
			resp: `the model said "{ not json }" and stopped`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parseJSONToolCalls(tt.resp); !callsEqual(got, tt.want) {
				t.Errorf("parsed %+v, want %+v", got, tt.want)
			}
		})
	}
}

// A bare object is held while it is open and reported as a region once it closes;
// whether that region is a call is parse's business, not feed's.
func TestJSONToolCallFeed(t *testing.T) {
	tests := []struct {
		name            string
		s               string
		wantAt, wantEnd int
	}{
		{"prose", "no braces here", -1, -1},
		{"open object", `prose {"na`, 6, -1},
		{"tool call", `hi {"name": "f", "arguments": {}}`, 3, 33},
		{"a non-call object is still a region", `{"foo": 1} ok`, 0, 10},
		{"stops at the first close", "a { b } c { d", 2, 7},
		{"first of two calls", `{"name":"a","arguments":{}}{"name":"b","arguments":{}}`, 0, 27},
		{"braces inside a string", `{"name":"f","arguments":{"s":"}}"}}`, 0, 35},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			at, end := (&jsonToolCall{}).feed(tt.s, 0)
			if at != tt.wantAt || end != tt.wantEnd {
				t.Errorf("feed(%q) = (%d, %d), want (%d, %d)", tt.s, at, end, tt.wantAt, tt.wantEnd)
			}
		})
	}
}
