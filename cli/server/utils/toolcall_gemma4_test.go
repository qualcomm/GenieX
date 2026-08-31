// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import "testing"

func TestParseGemma4ToolCalls(t *testing.T) {
	tests := []struct {
		name string
		resp string
		want []toolCallFn // nil: no tool call
	}{
		{
			name: "single string argument",
			resp: `<|tool_call>call:get_weather{city:<|"|>Beijing<|"|>}<tool_call|>`,
			want: []toolCallFn{{Name: "get_weather", Arguments: `{"city":"Beijing"}`}},
		},
		{
			name: "multiple arguments mixed types",
			resp: `<|tool_call>call:get_weather{city:<|"|>Beijing<|"|>,days:3,metric:true}<tool_call|>`,
			want: []toolCallFn{{Name: "get_weather", Arguments: `{"city":"Beijing","days":3,"metric":true}`}},
		},
		{
			name: "empty arguments",
			resp: `<|tool_call>call:now{}<tool_call|>`,
			want: []toolCallFn{{Name: "now", Arguments: `{}`}},
		},
		{
			name: "nested dict",
			resp: `<|tool_call>call:f{loc:{lat:1.5,lng:<|"|>x<|"|>}}<tool_call|>`,
			want: []toolCallFn{{Name: "f", Arguments: `{"loc":{"lat":1.5,"lng":"x"}}`}},
		},
		{
			name: "array of strings",
			resp: `<|tool_call>call:pick{items:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>`,
			want: []toolCallFn{{Name: "pick", Arguments: `{"items":["a","b"]}`}},
		},
		{
			name: "array of dicts",
			resp: `<|tool_call>call:batch{items:[{a:1},{b:2}]}<tool_call|>`,
			want: []toolCallFn{{Name: "batch", Arguments: `{"items":[{"a":1},{"b":2}]}`}},
		},
		{
			name: "null and negative number",
			resp: `<|tool_call>call:f{x:null,y:-4.2}<tool_call|>`,
			want: []toolCallFn{{Name: "f", Arguments: `{"x":null,"y":-4.2}`}},
		},
		{
			name: "string with special chars escaped",
			resp: `<|tool_call>call:say{text:<|"|>a "quote" and }brace{<|"|>}<tool_call|>`,
			want: []toolCallFn{{Name: "say", Arguments: `{"text":"a \"quote\" and }brace{"}`}},
		},
		{
			name: "preceded by reasoning and content",
			resp: "<|channel>thought\nlet me check\n<channel|>Sure! <|tool_call>call:go{x:1}<tool_call|>",
			want: []toolCallFn{{Name: "go", Arguments: `{"x":1}`}},
		},
		{
			name: "whitespace around members",
			resp: `<|tool_call>call:f{ a : 1 , b : <|"|>x<|"|> }<tool_call|>`,
			want: []toolCallFn{{Name: "f", Arguments: `{"a":1,"b":"x"}`}},
		},
		{
			// p.repeat in the grammar: parallel calls are separate wrappers
			name: "parallel tool calls",
			resp: `<|tool_call>call:first{a:1}<tool_call|><|tool_call>call:second{b:2}<tool_call|>`,
			want: []toolCallFn{{Name: "first", Arguments: `{"a":1}`}, {Name: "second", Arguments: `{"b":2}`}},
		},
		{
			name: "three parallel calls with text between",
			resp: `<|tool_call>call:a{}<tool_call|> then <|tool_call>call:b{}<tool_call|>ok<|tool_call>call:c{}<tool_call|>`,
			want: []toolCallFn{{Name: "a", Arguments: `{}`}, {Name: "b", Arguments: `{}`}, {Name: "c", Arguments: `{}`}},
		},
		{
			name: "a malformed call does not hide the next one",
			resp: `<|tool_call>call:bad{k}<tool_call|><|tool_call>call:good{x:1}<tool_call|>`,
			want: []toolCallFn{{Name: "good", Arguments: `{"x":1}`}},
		},
		{
			// the end marker is not part of the grammar this parser needs
			name: "unclosed wrapper still parses",
			resp: `<|tool_call>call:f{x:1}`,
			want: []toolCallFn{{Name: "f", Arguments: `{"x":1}`}},
		},
		{
			name: "no tool call marker",
			resp: "just some plain text",
		},
		{
			name: "marker but no brace",
			resp: `<|tool_call>call:broken`,
		},
		{
			name: "unterminated dict",
			resp: `<|tool_call>call:f{x:1`,
		},
		{
			name: "unterminated string value",
			resp: `<|tool_call>call:f{x:<|"|>abc}<tool_call|>`,
		},
		{
			name: "empty function name",
			resp: `<|tool_call>call:{x:1}<tool_call|>`,
		},
		{
			name: "empty key rejected",
			resp: `<|tool_call>call:f{:1}<tool_call|>`,
		},
		{
			// key scan must stop at '}' rather than swallowing the next member's colon
			name: "member without value does not swallow later key",
			resp: `<|tool_call>call:f{k}{x:1}<tool_call|>`,
		},
		{
			name: "json syntax is not a gemma4 call",
			resp: `{"name": "f", "arguments": {}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parseGemma4ToolCalls(tt.resp); !callsEqual(got, tt.want) {
				t.Errorf("parsed %+v, want %+v", got, tt.want)
			}
		})
	}
}

// Each wrapper is its own region, so parallel calls come out one per Push with the
// text between them streamed in order.
func TestGemma4Stream(t *testing.T) {
	tests := []struct {
		name      string
		resp      string
		wantText  string
		wantCalls []toolCallFn
	}{
		{
			name:      "one call",
			resp:      `<|tool_call>call:f{a:1}<tool_call|>`,
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{"a":1}`}},
		},
		{
			name:      "prose around a call",
			resp:      `Sure! <|tool_call>call:f{a:1}<tool_call|> done`,
			wantText:  "Sure!  done",
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{"a":1}`}},
		},
		{
			name:      "two parallel calls",
			resp:      `<|tool_call>call:first{a:1}<tool_call|><|tool_call>call:second{b:2}<tool_call|>`,
			wantCalls: []toolCallFn{{Name: "first", Arguments: `{"a":1}`}, {Name: "second", Arguments: `{"b":2}`}},
		},
		{
			name:      "parallel calls separated by a newline",
			resp:      "<|tool_call>call:a{}<tool_call|>\n<|tool_call>call:b{}<tool_call|>",
			wantText:  "\n",
			wantCalls: []toolCallFn{{Name: "a", Arguments: `{}`}, {Name: "b", Arguments: `{}`}},
		},
		{
			name: "three parallel calls",
			resp: `<|tool_call>call:a{x:<|"|>1<|"|>}<tool_call|><|tool_call>call:b{}<tool_call|><|tool_call>call:c{y:[1,2]}<tool_call|>`,
			wantCalls: []toolCallFn{
				{Name: "a", Arguments: `{"x":"1"}`},
				{Name: "b", Arguments: `{}`},
				{Name: "c", Arguments: `{"y":[1,2]}`},
			},
		},
		{
			name:      "a malformed call streams as text between two good ones",
			resp:      `<|tool_call>call:a{}<tool_call|><|tool_call>call:bad{k}<tool_call|><|tool_call>call:c{}<tool_call|>`,
			wantText:  `<|tool_call>call:bad{k}<tool_call|>`,
			wantCalls: []toolCallFn{{Name: "a", Arguments: `{}`}, {Name: "c", Arguments: `{}`}},
		},
		{
			// the model stopped before the end marker: Tail still finds the call
			name:      "unclosed wrapper",
			resp:      `<|tool_call>call:f{a:1}`,
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{"a":1}`}},
		},
		{
			name:      "braces and quotes inside a gemma4 string",
			resp:      `<|tool_call>call:say{t:<|"|>a "q" }{ b<|"|>}<tool_call|>!`,
			wantText:  "!",
			wantCalls: []toolCallFn{{Name: "say", Arguments: `{"t":"a \"q\" }{ b"}`}},
		},
		{
			name:     "refuted open marker",
			resp:     "<|tool_call>calX nothing here",
			wantText: "<|tool_call>calX nothing here",
		},
		{
			name:     "ends on a partial open marker",
			resp:     "stops at <|tool_call>",
			wantText: "stops at <|tool_call>",
		},
		{
			name:     "a stray end marker is prose",
			resp:     "<tool_call|> stray",
			wantText: "<tool_call|> stray",
		},
		{
			// gemma4 arguments are not JSON, so no format claims them
			name:     "wrapper holding no call is prose",
			resp:     `<|tool_call>call:f{not a dict}<tool_call|>`,
			wantText: `<|tool_call>call:f{not a dict}<tool_call|>`,
		},
		{
			// markerFormat's BUG: the quoted end marker cuts the region short, so this
			// parses as nothing and streams as text. Parse on the whole text finds it.
			name:     "an end marker inside a gemma4 string loses the call",
			resp:     `<|tool_call>call:f{t:<|"|>a<tool_call|>b<|"|>}<tool_call|>`,
			wantText: `<|tool_call>call:f{t:<|"|>a<tool_call|>b<|"|>}<tool_call|>`,
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

// Streaming and the whole-text scan are separate implementations of the parallel case
// too, so they check each other.
func TestGemma4StreamAgreesWithParse(t *testing.T) {
	responses := []string{
		`<|tool_call>call:f{a:1}<tool_call|>`,
		`<|tool_call>call:a{}<tool_call|><|tool_call>call:b{}<tool_call|>`,
		`hi <|tool_call>call:a{}<tool_call|> mid <|tool_call>call:b{x:[1,{y:2}]}<tool_call|> tail`,
		`<|tool_call>call:f{a:1}`,
		"<|channel>thought\nhmm\n<channel|><|tool_call>call:go{}<tool_call|>",
		// a gemma4 string holding JSON: the bare-JSON scan must not claim it
		`<|tool_call>call:f{body:<|"|>{"name":"x","arguments":{}}<|"|>}<tool_call|>`,
	}

	for _, resp := range responses {
		t.Run(resp, func(t *testing.T) {
			_, got := stream(resp, 1)
			if want := parseGemma4ToolCalls(resp); !callsEqual(got, want) {
				t.Errorf("streamed %+v, parsed %+v", got, want)
			}
		})
	}
}
