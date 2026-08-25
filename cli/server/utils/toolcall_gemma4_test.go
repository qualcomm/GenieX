// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import "testing"

func TestParseToolCallsGemma4(t *testing.T) {
	tests := []struct {
		name      string
		resp      string
		wantName  string
		wantArgs  string
		wantError bool
	}{
		{
			name:     "single string argument",
			resp:     `<|tool_call>call:get_weather{city:<|"|>Beijing<|"|>}<tool_call|>`,
			wantName: "get_weather",
			wantArgs: `{"city":"Beijing"}`,
		},
		{
			name:     "multiple arguments mixed types",
			resp:     `<|tool_call>call:get_weather{city:<|"|>Beijing<|"|>,days:3,metric:true}<tool_call|>`,
			wantName: "get_weather",
			wantArgs: `{"city":"Beijing","days":3,"metric":true}`,
		},
		{
			name:     "empty arguments",
			resp:     `<|tool_call>call:now{}<tool_call|>`,
			wantName: "now",
			wantArgs: `{}`,
		},
		{
			name:     "nested dict",
			resp:     `<|tool_call>call:f{loc:{lat:1.5,lng:<|"|>x<|"|>}}<tool_call|>`,
			wantName: "f",
			wantArgs: `{"loc":{"lat":1.5,"lng":"x"}}`,
		},
		{
			name:     "array of strings",
			resp:     `<|tool_call>call:pick{items:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>`,
			wantName: "pick",
			wantArgs: `{"items":["a","b"]}`,
		},
		{
			name:     "array of dicts",
			resp:     `<|tool_call>call:batch{items:[{a:1},{b:2}]}<tool_call|>`,
			wantName: "batch",
			wantArgs: `{"items":[{"a":1},{"b":2}]}`,
		},
		{
			name:     "null and negative number",
			resp:     `<|tool_call>call:f{x:null,y:-4.2}<tool_call|>`,
			wantName: "f",
			wantArgs: `{"x":null,"y":-4.2}`,
		},
		{
			name:     "string with special chars escaped",
			resp:     `<|tool_call>call:say{text:<|"|>a "quote" and }brace{<|"|>}<tool_call|>`,
			wantName: "say",
			wantArgs: `{"text":"a \"quote\" and }brace{"}`,
		},
		{
			name:     "preceded by reasoning and content",
			resp:     "<|channel>thought\nlet me check\n<channel|>Sure! <|tool_call>call:go{x:1}<tool_call|>",
			wantName: "go",
			wantArgs: `{"x":1}`,
		},
		{
			name:     "picks first of parallel tool calls",
			resp:     `<|tool_call>call:first{a:1}<tool_call|><|tool_call>call:second{b:2}<tool_call|>`,
			wantName: "first",
			wantArgs: `{"a":1}`,
		},
		{
			name:     "whitespace around members",
			resp:     `<|tool_call>call:f{ a : 1 , b : <|"|>x<|"|> }<tool_call|>`,
			wantName: "f",
			wantArgs: `{"a":1,"b":"x"}`,
		},
		{
			name:      "no tool call marker",
			resp:      "just some plain text",
			wantError: true,
		},
		{
			name:      "marker but no brace",
			resp:      `<|tool_call>call:broken`,
			wantError: true,
		},
		{
			name:      "unterminated dict",
			resp:      `<|tool_call>call:f{x:1`,
			wantError: true,
		},
		{
			name:      "unterminated string value",
			resp:      `<|tool_call>call:f{x:<|"|>abc}<tool_call|>`,
			wantError: true,
		},
		{
			name:      "empty function name",
			resp:      `<|tool_call>call:{x:1}<tool_call|>`,
			wantError: true,
		},
		{
			name:      "empty key rejected",
			resp:      `<|tool_call>call:f{:1}<tool_call|>`,
			wantError: true,
		},
		{
			// key scan must stop at '}' rather than swallowing the next member's colon
			name:      "member without value does not swallow later key",
			resp:      `<|tool_call>call:f{k}{x:1}<tool_call|>`,
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseToolCallsGemma4(tt.resp)
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

// ParseToolCalls must route gemma4-marked output to the gemma4 parser while
// leaving JSON output on the generic path.
func TestParseToolCallsDispatch(t *testing.T) {
	got, err := ParseToolCalls(`<|tool_call>call:get_weather{city:<|"|>Beijing<|"|>}<tool_call|>`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Name != "get_weather" || got.Arguments != `{"city":"Beijing"}` {
		t.Errorf("gemma4 dispatch failed: %+v", got)
	}
}
