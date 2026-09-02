// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import "testing"

// The headers llama.cpp's gpt-oss tests cover, plus the one a reasoning splitter
// upstream leaves behind once it has eaten `<|channel|>`.
func TestParseGptOssToolCalls(t *testing.T) {
	tests := []struct {
		name string
		resp string
		want []toolCallFn // nil: no tool call
	}{
		{
			name: "recipient in the channel",
			resp: `<|channel|>commentary to=functions.get_weather<|message|>{"city":"Beijing"}<|call|>`,
			want: []toolCallFn{{Name: "get_weather", Arguments: `{"city":"Beijing"}`}},
		},
		{
			name: "recipient in the role",
			resp: `<|start|>assistant to=functions.f<|channel|>analysis<|message|>{"x":1}<|call|>`,
			want: []toolCallFn{{Name: "f", Arguments: `{"x":1}`}},
		},
		{
			name: "constrain between the recipient and the message",
			resp: `<|channel|>commentary to=functions.f <|constrain|>json<|message|>{"x":1}<|call|>`,
			want: []toolCallFn{{Name: "f", Arguments: `{"x":1}`}},
		},
		{
			name: "channel already consumed upstream",
			resp: `commentary to=functions.f<|message|>{"x":1}<|call|>`,
			want: []toolCallFn{{Name: "f", Arguments: `{"x":1}`}},
		},
		{
			name: "no closing call marker",
			resp: `<|channel|>commentary to=functions.f<|message|>{"x":1}`,
			want: []toolCallFn{{Name: "f", Arguments: `{"x":1}`}},
		},
		{
			name: "reasoning before the call",
			resp: "<|channel|>analysis<|message|>let me check<|end|>" +
				`<|start|>assistant to=functions.f<|channel|>commentary<|message|>{"x":1}<|call|>`,
			want: []toolCallFn{{Name: "f", Arguments: `{"x":1}`}},
		},
		{
			name: "nested arguments",
			resp: `<|channel|>commentary to=functions.f<|message|>{"a":{"b":[1,2]},"s":"}"}<|call|>`,
			want: []toolCallFn{{Name: "f", Arguments: `{"a":{"b":[1,2]},"s":"}"}`}},
		},
		{
			name: "two calls",
			resp: `<|channel|>commentary to=functions.a<|message|>{}<|call|>` +
				`<|start|>assistant<|channel|>commentary to=functions.b<|message|>{"x":1}<|call|>`,
			want: []toolCallFn{{Name: "a", Arguments: `{}`}, {Name: "b", Arguments: `{"x":1}`}},
		},
		{
			name: "builtin recipients are not tool calls",
			resp: `<|channel|>commentary to=container.exec<|message|>{"cmd":"ls"}<|call|>`,
		},
		{
			name: "final channel",
			resp: "<|channel|>final<|message|>Beijing is sunny.",
		},
		{
			name: "arguments never closed",
			resp: `<|channel|>commentary to=functions.f<|message|>{"x":`,
		},
		{
			name: "arguments are not an object",
			resp: `<|channel|>commentary to=functions.f<|message|>oops<|call|>`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parseGptOssToolCalls(tt.resp); !callsEqual(got, tt.want) {
				t.Errorf("parseGptOssToolCalls() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

// A stream emits the header before the recipient that marks it as a call arrives,
// so the header goes out as content — the same way the analysis channel's markers
// already do. What follows the recipient, the closing marker included, is the
// call's own and never reaches the content.
func TestGptOssToolCallStream(t *testing.T) {
	tests := []struct {
		name      string
		resp      string
		wantText  string
		wantCalls []toolCallFn
	}{
		{
			name:      "recipient in the channel",
			resp:      `<|channel|>commentary to=functions.f<|message|>{"x":1}<|call|>`,
			wantText:  "<|channel|>commentary",
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{"x":1}`}},
		},
		{
			name:      "recipient in the role",
			resp:      `<|start|>assistant to=functions.f<|channel|>commentary<|message|>{"x":1}<|call|>`,
			wantText:  "<|start|>assistant", // what sits after the recipient is the call's
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{"x":1}`}},
		},
		{
			name:      "channel already consumed upstream",
			resp:      `commentary to=functions.f<|message|>{"x":1}<|call|>`,
			wantText:  "commentary",
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{"x":1}`}},
		},
		{
			name:      "call after a preamble",
			resp:      `Let me check.<|channel|>commentary to=functions.f<|message|>{"x":1}<|call|>`,
			wantText:  "Let me check.<|channel|>commentary",
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{"x":1}`}},
		},
		{
			name:     "final channel streams as content",
			resp:     "<|channel|>final<|message|>Beijing is sunny.",
			wantText: "<|channel|>final<|message|>Beijing is sunny.",
		},
		{
			name:     "a recipient that is not a tool",
			resp:     `<|channel|>commentary to=python<|message|>print(1)`,
			wantText: `<|channel|>commentary to=python<|message|>print(1)`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for size := 1; size <= 8; size++ {
				text, calls := stream(tt.resp, size)
				if text != tt.wantText {
					t.Errorf("size %d: text = %q, want %q", size, text, tt.wantText)
				}
				if !callsEqual(calls, tt.wantCalls) {
					t.Errorf("size %d: calls = %+v, want %+v", size, calls, tt.wantCalls)
				}
			}
		})
	}
}

// Whole-response parsing sees the header and the recipient together, so there the
// header is the call's and the content comes out empty.
func TestGptOssHeaderInOneChunk(t *testing.T) {
	want := []toolCallFn{{Name: "f", Arguments: `{"x":1}`}}
	for _, resp := range []string{
		`<|channel|>commentary to=functions.f<|message|>{"x":1}<|call|>`,
		`<|start|>assistant to=functions.f<|channel|>commentary<|message|>{"x":1}<|call|>`,
		// what gpt-oss-20b actually emits: the role and the channel both
		`<|start|>assistant<|channel|>commentary to=functions.f<|message|>{"x":1}<|call|>`,
		`commentary to=functions.f<|message|>{"x":1}<|call|>`,
	} {
		t.Run(resp, func(t *testing.T) {
			text, calls := NewToolCallScanner().Parse(resp)
			if text != "" {
				t.Errorf("text = %q, want empty", text)
			}
			if !callsEqual(calls, want) {
				t.Errorf("calls = %+v, want %+v", calls, want)
			}
		})
	}
}
