// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import "testing"

func TestParseToolCalls(t *testing.T) {
	tests := []struct {
		name      string
		resp      string
		wantName  string
		wantArgs  string
		wantError bool
	}{
		{
			name:     "bare json object",
			resp:     `{"name": "get_weather", "arguments": {"city": "Beijing"}}`,
			wantName: "get_weather",
			wantArgs: `{"city": "Beijing"}`,
		},
		{
			name:     "json with surrounding prose",
			resp:     `Sure, let me call it. {"name": "get_weather", "arguments": {"city": "Beijing"}} Done.`,
			wantName: "get_weather",
			wantArgs: `{"city": "Beijing"}`,
		},
		{
			name:     "string arguments",
			resp:     `{"name": "echo", "arguments": "hello"}`,
			wantName: "echo",
			wantArgs: "hello",
		},
		{
			name:     "nested braces in arguments",
			resp:     `{"name": "f", "arguments": {"a": {"b": {"c": 1}}}}`,
			wantName: "f",
			wantArgs: `{"a": {"b": {"c": 1}}}`,
		},
		{
			name:     "deeply nested arguments",
			resp:     `prose {"name": "deep", "arguments": {"l1": {"l2": {"l3": {"l4": {"l5": {"v": [1, {"x": 2}]}}}}}}} tail`,
			wantName: "deep",
			wantArgs: `{"l1": {"l2": {"l3": {"l4": {"l5": {"v": [1, {"x": 2}]}}}}}}`,
		},
		{
			name:     "nested objects with braces inside strings",
			resp:     `{"name": "g", "arguments": {"outer": {"inner": {"note": "close } here { and {} there"}}}}`,
			wantName: "g",
			wantArgs: `{"outer": {"inner": {"note": "close } here { and {} there"}}}`,
		},
		{
			name:     "array of nested objects as arguments",
			resp:     `{"name": "batch", "arguments": {"items": [{"a": {"b": 1}}, {"c": {"d": 2}}]}}`,
			wantName: "batch",
			wantArgs: `{"items": [{"a": {"b": 1}}, {"c": {"d": 2}}]}`,
		},
		{
			name:     "braces inside string literal ignored",
			resp:     `{"name": "say", "arguments": {"text": "a } b { c"}}`,
			wantName: "say",
			wantArgs: `{"text": "a } b { c"}`,
		},
		{
			name:     "escaped quote inside string",
			resp:     `{"name": "say", "arguments": {"text": "quote \" and } brace"}}`,
			wantName: "say",
			wantArgs: `{"text": "quote \" and } brace"}`,
		},
		{
			name:     "skips leading object without name",
			resp:     `{"thinking": "hmm"} then {"name": "go", "arguments": {}}`,
			wantName: "go",
			wantArgs: `{}`,
		},
		{
			name:     "skips object with name but no arguments",
			resp:     `{"name": "no_args"} {"name": "real", "arguments": {"x": 1}}`,
			wantName: "real",
			wantArgs: `{"x": 1}`,
		},
		{
			name:     "picks first of multiple tool calls",
			resp:     `{"name": "first", "arguments": {"a": 1}}{"name": "second", "arguments": {"b": 2}}`,
			wantName: "first",
			wantArgs: `{"a": 1}`,
		},
		{
			name:     "skips candidate with array arguments",
			resp:     `{"name": "bad", "arguments": [1, 2, 3]} {"name": "good", "arguments": {"ok": 1}}`,
			wantName: "good",
			wantArgs: `{"ok": 1}`,
		},
		{
			name:     "skips candidate with numeric arguments",
			resp:     `{"name": "bad", "arguments": 42} {"name": "good", "arguments": "text"}`,
			wantName: "good",
			wantArgs: "text",
		},
		{
			name:     "skips candidate with non-string name",
			resp:     `{"name": 7, "arguments": {}} {"name": "good", "arguments": {}}`,
			wantName: "good",
			wantArgs: `{}`,
		},
		{
			name:      "all candidates invalid falls through to error",
			resp:      `{"name": "a", "arguments": [1]} {"name": 2, "arguments": {}} {"name": "c"}`,
			wantError: true,
		},
		{
			name:     "multiple tool calls separated by prose",
			resp:     `Call one: {"name": "first", "arguments": {}}. Then: {"name": "second", "arguments": {}}`,
			wantName: "first",
			wantArgs: `{}`,
		},
		{
			name:     "leading unterminated brace does not swallow later call",
			resp:     `reasoning { partial and never closed ... {"name": "go", "arguments": {"ok": true}}`,
			wantName: "go",
			wantArgs: `{"ok": true}`,
		},
		{
			name:     "stray closing braces before real call",
			resp:     `garbage } } more } {"name": "go", "arguments": {}}`,
			wantName: "go",
			wantArgs: `{}`,
		},
		{
			name:     "name with escaped chars",
			resp:     `{"name": "ns\\tool", "arguments": {"path": "C:\\tmp"}}`,
			wantName: `ns\tool`,
			wantArgs: `{"path": "C:\\tmp"}`,
		},
		{
			name:     "empty object skipped then real call",
			resp:     `{} {"name": "go", "arguments": {}}`,
			wantName: "go",
			wantArgs: `{}`,
		},
		{
			// tag/fence wrappers are transparent: the inner object is extracted
			name:     "json inside tool_call tag",
			resp:     "<tool_call>\n{\"name\": \"tagged\", \"arguments\": {\"k\": 1}}\n</tool_call>",
			wantName: "tagged",
			wantArgs: `{"k": 1}`,
		},
		{
			name:     "json inside code fence",
			resp:     "here you go:\n```json\n{\"name\": \"fenced\", \"arguments\": {\"q\": \"x\"}}\n```",
			wantName: "fenced",
			wantArgs: `{"q": "x"}`,
		},
		{
			name:      "no json object",
			resp:      "just some plain text without any call",
			wantError: true,
		},
		{
			name:      "unterminated object",
			resp:      `{"name": "go", "arguments":`,
			wantError: true,
		},
		{
			name:      "only non-tool-call objects",
			resp:      `{"foo": 1} {"bar": 2} {"name": 42}`,
			wantError: true,
		},
		{
			name:      "empty string",
			resp:      "",
			wantError: true,
		},
		{
			name:      "braces only inside a string literal",
			resp:      `the model said "{ not json }" and stopped`,
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseToolCalls(tt.resp)
			if tt.wantError {
				if err == nil {
					t.Fatalf("expected error, got tool call %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got.Name != tt.wantName {
				t.Errorf("name = %q, want %q", got.Name, tt.wantName)
			}
			if got.Arguments != tt.wantArgs {
				t.Errorf("arguments = %q, want %q", got.Arguments, tt.wantArgs)
			}
		})
	}
}
