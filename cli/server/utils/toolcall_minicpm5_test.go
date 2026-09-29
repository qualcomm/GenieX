// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"strings"
	"testing"
)

func TestParseMiniCPM5ToolCalls(t *testing.T) {
	tests := []struct {
		name string
		resp string
		want []toolCallFn
	}{
		{
			name: "single call",
			resp: `<function name="get_weather"><param name="city">Beijing</param><param name="units">celsius</param></function>`,
			want: []toolCallFn{{Name: "get_weather", Arguments: `{"city":"Beijing","units":"celsius"}`}},
		},
		{
			name: "empty arguments",
			resp: `<function name="now"></function>`,
			want: []toolCallFn{{Name: "now", Arguments: `{}`}},
		},
		{
			name: "CDATA multiline and entities",
			resp: `<function name="write"><param name="body"><![CDATA[line one
line <two> & three]]></param><param name="label">A &amp; B</param></function>`,
			want: []toolCallFn{{Name: "write", Arguments: `{"body":"line one\nline <two> & three","label":"A & B"}`}},
		},
		{
			name: "parallel calls",
			resp: `<function name="first"><param name="a">1</param></function><function name="second"><param name="b">2</param></function>`,
			want: []toolCallFn{{Name: "first", Arguments: `{"a":"1"}`}, {Name: "second", Arguments: `{"b":"2"}`}},
		},
		{
			name: "invalid elements are ignored",
			resp: `<functionality name="not_a_call"></functionality><function name=""><param name="x">1</param></function><function name="good"><param>ignored</param></function>`,
			want: []toolCallFn{{Name: "good", Arguments: `{}`}},
		},
		{
			name: "malformed XML does not become a call",
			resp: `<function name="bad"><param name="x">unescaped & value</param></function>`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parseMiniCPM5ToolCalls(tt.resp, nil); !callsEqual(got, tt.want) {
				t.Errorf("parsed %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestToolParameterTypesFromToolsWithMiniCPM5(t *testing.T) {
	const tools = `[
		{"type":"function","function":{"name":"typed","parameters":{"type":"object","properties":{
			"count":{"type":"integer"},"price":{"type":"number"},"flag":{"type":"boolean"},
			"object":{"type":"object"},"list":{"type":"array"},"nothing":{"type":"null"},
			"label":{"type":"string"},"badCount":{"type":"integer"},"badKind":{"type":"integer"},
			"badFraction":{"type":"integer"},
			"badFlag":{"type":"boolean"},"unknown":{"type":["integer","string"]}
		}}}},
		{"type":"function","function":{"name":"other","parameters":{"type":"object","properties":{
			"count":{"type":"string"}
		}}}}
	]`
	const resp = `<function name="typed">` +
		`<param name="count">1e0</param><param name="price">1.5</param><param name="flag">true</param>` +
		`<param name="object">{"x":1}</param><param name="list">[1,2]</param><param name="nothing">null</param>` +
		`<param name="label">5</param><param name="badCount">abc</param><param name="badKind">true</param>` +
		`<param name="badFraction">1.5</param>` +
		`<param name="badFlag">yes</param><param name="unknown">5</param><param name="noSchema">5</param>` +
		`</function><function name="other"><param name="count">5</param></function>`
	want := []toolCallFn{
		{Name: "typed", Arguments: `{"count":1e0,"price":1.5,"flag":true,"object":{"x":1},"list":[1,2],"nothing":null,"label":"5","badCount":"abc","badKind":"true","badFraction":"1.5","badFlag":"yes","unknown":"5","noSchema":"5"}`},
		{Name: "other", Arguments: `{"count":"5"}`},
	}
	types := ToolParameterTypesFromTools(tools)
	if types["typed"]["count"] != "integer" || types["other"]["count"] != "string" {
		t.Fatalf("incorrect per-function schema types: %+v", types)
	}
	if text, got := NewToolCallScanner(types).Parse(resp); text != "" || !callsEqual(got, want) {
		t.Errorf("blocking: text=%q, calls=%+v, want %+v", text, got, want)
	}
	for size := 1; size <= 8; size++ {
		s := NewToolCallScanner(types)
		var text string
		var calls []toolCallFn
		for i := 0; i < len(resp); i += size {
			out, got := s.Push(resp[i:min(i+size, len(resp))])
			text += out
			calls = append(calls, got...)
		}
		tail, got := s.Tail()
		text += tail
		calls = append(calls, got...)
		if text != "" || !callsEqual(calls, want) {
			t.Errorf("size %d: text=%q, calls=%+v, want %+v", size, text, calls, want)
		}
	}
	if got := ToolParameterTypesFromTools("not JSON"); got != nil {
		t.Errorf("invalid tools types = %+v, want nil", got)
	}
}

func TestToolParameterTypesFromToolsIgnoresUnsupportedProperties(t *testing.T) {
	for _, tt := range []struct {
		name  string
		tools string
	}{
		{
			name: "same tool",
			tools: `[{"type":"function","function":{"name":"typed","parameters":{"type":"object","properties":{
				"count":{"type":"integer"},"disabled":false,"optional":true
			}}}}]`,
		},
		{
			name: "another tool",
			tools: `[{"type":"function","function":{"name":"typed","parameters":{"type":"object","properties":{"count":{"type":"integer"}}}}},
				{"type":"function","function":{"name":"other","parameters":{"type":"object","properties":{"disabled":false}}}}]`,
		},
		{
			name: "another tool with unsupported parameters",
			tools: `[{"type":"function","function":{"name":"typed","parameters":{"type":"object","properties":{"count":{"type":"integer"}}}}},
				{"type":"function","function":{"name":"other","parameters":false}}]`,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			types := ToolParameterTypesFromTools(tt.tools)
			if types["typed"]["count"] != "integer" {
				t.Fatalf("lost recognized type after unsupported schema: %+v", types)
			}
			const resp = `<function name="typed"><param name="count">5</param><param name="disabled">false</param></function>`
			want := []toolCallFn{{Name: "typed", Arguments: `{"count":5,"disabled":"false"}`}}
			if text, got := NewToolCallScanner(types).Parse(resp); text != "" || !callsEqual(got, want) {
				t.Errorf("text=%q, calls=%+v, want %+v", text, got, want)
			}
		})
	}
}

func TestMiniCPM5Stream(t *testing.T) {
	tests := []struct {
		name      string
		resp      string
		wantText  string
		wantCalls []toolCallFn
	}{
		{
			name:      "prose around a call",
			resp:      `Checking. <function name="lookup"><param name="q">weather</param></function> Done.`,
			wantText:  "Checking.  Done.",
			wantCalls: []toolCallFn{{Name: "lookup", Arguments: `{"q":"weather"}`}},
		},
		{
			name:      "self-closing call",
			resp:      `<function name="now"/>`,
			wantCalls: []toolCallFn{{Name: "now", Arguments: `{}`}},
		},
		{
			name:      "self-closing param is not the end of the call",
			resp:      `<function name="now"><param name="unused"/></function>`,
			wantCalls: []toolCallFn{{Name: "now", Arguments: `{"unused":""}`}},
		},
		{
			name: "parallel calls with CDATA closing tag",
			resp: `<function name="write"><param name="body"><![CDATA[a </function> b]]></param></function>
<function name="done"></function>`,
			wantText: "\n",
			wantCalls: []toolCallFn{
				{Name: "write", Arguments: `{"body":"a </function> b"}`},
				{Name: "done", Arguments: `{}`},
			},
		},
		{
			name:     "incomplete call is text",
			resp:     `<function name="unfinished"><param name="x">value</param>`,
			wantText: `<function name="unfinished"><param name="x">value</param>`,
		},
		{
			name:     "near miss is text",
			resp:     `<functionality name="explain">text</functionality>`,
			wantText: `<functionality name="explain">text</functionality>`,
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

func TestMiniCPM5InvalidXMLDoesNotHoldLaterOutput(t *testing.T) {
	const invalid = `<function name="bad"><param name="x">A & B</param></function>`
	for _, tt := range []struct {
		name, after string
		wantCalls   []toolCallFn
	}{
		{name: "prose", after: " later text"},
		{name: "MiniCPM5 call", after: ` later <function name="good"></function>`,
			wantCalls: []toolCallFn{{Name: "good", Arguments: `{}`}}},
		{name: "Gemma4 call", after: ` later <|tool_call>call:good{}<tool_call|>`,
			wantCalls: []toolCallFn{{Name: "good", Arguments: `{}`}}},
		{name: "both formats in order", after: ` later <function name="first"></function><|tool_call>call:second{}<tool_call|>`,
			wantCalls: []toolCallFn{{Name: "first", Arguments: `{}`}, {Name: "second", Arguments: `{}`}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			for size := 1; size <= 8; size++ {
				s := NewToolCallScanner(nil)
				var text string
				var calls []toolCallFn
				resp := invalid + tt.after
				for i := 0; i < len(resp); i += size {
					out, got := s.Push(resp[i:min(i+size, len(resp))])
					text += out
					calls = append(calls, got...)
				}
				wantText := invalid + tt.after
				if len(tt.wantCalls) > 0 {
					wantText = invalid + " later "
				}
				if text != wantText || !callsEqual(calls, tt.wantCalls) {
					t.Errorf("size %d: before Tail: text=%q, calls=%+v, want %q, %+v", size, text, calls, wantText, tt.wantCalls)
				}
				if tail, got := s.Tail(); tail != "" || len(got) != 0 {
					t.Errorf("size %d: Tail = %q, %+v; want empty", size, tail, got)
				}
			}
		})
	}
}

func TestMiniCPM5PartialEntityWaitsForDecision(t *testing.T) {
	s := NewToolCallScanner(nil)
	const prefix = `<function name="bad"><param name="x">A &`
	if text, calls := s.Push(prefix); text != "" || len(calls) != 0 {
		t.Fatalf("partial entity: text=%q, calls=%+v; want buffered", text, calls)
	}
	if text, calls := s.Push(" B"); text != "" || len(calls) != 0 {
		t.Errorf("unfinished XML: text=%q, calls=%+v; want buffered until a possible close", text, calls)
	}
	if text, calls := s.Push(`</param></function> later`); text != prefix+` B</param></function> later` || len(calls) != 0 {
		t.Errorf("invalid completed XML: text=%q, calls=%+v; want released text", text, calls)
	}
}

func TestMiniCPM5IncompleteXMLWaitsForCompletion(t *testing.T) {
	const prefix = `<function name="good"><param name="x"><![CDATA[A </function> & B`
	const suffix = `]]></param></function>`
	for size := 1; size <= 8; size++ {
		s := NewToolCallScanner(nil)
		var calls []toolCallFn
		for i := 0; i < len(prefix); i += size {
			out, got := s.Push(prefix[i:min(i+size, len(prefix))])
			if out != "" || len(got) != 0 {
				t.Errorf("size %d: incomplete XML emitted text=%q calls=%+v", size, out, got)
			}
		}
		for i := 0; i < len(suffix); i += size {
			out, got := s.Push(suffix[i:min(i+size, len(suffix))])
			if out != "" {
				t.Errorf("size %d: completed XML leaked text=%q", size, out)
			}
			calls = append(calls, got...)
		}
		if !callsEqual(calls, []toolCallFn{{Name: "good", Arguments: `{"x":"A </function> & B"}`}}) {
			t.Errorf("size %d: calls=%+v", size, calls)
		}
		if tail, got := s.Tail(); tail != "" || len(got) != 0 {
			t.Errorf("size %d: Tail = %q, %+v; want empty", size, tail, got)
		}
	}
}

func BenchmarkMiniCPM5LongXMLStream(b *testing.B) {
	for _, tt := range []struct {
		name, value string
		cdata       bool
	}{
		{name: "512-byte-value", value: strings.Repeat("x", 512)},
		{name: "4096-byte-value", value: strings.Repeat("x", 4096)},
		{name: "CDATA-false-closers", value: strings.Repeat("</function>", 400), cdata: true},
	} {
		b.Run(tt.name, func(b *testing.B) {
			value := tt.value
			resp := `<function name="write"><param name="body">` + value + `</param></function>`
			if tt.cdata {
				resp = `<function name="write"><param name="body"><![CDATA[` + value + `]]></param></function>`
			}
			want := []toolCallFn{{Name: "write", Arguments: `{"body":"` + value + `"}`}}
			b.SetBytes(int64(len(resp)))
			for b.Loop() {
				text, calls := stream(resp, 4)
				if text != "" || !callsEqual(calls, want) {
					b.Fatalf("text = %q, calls = %+v", text, calls)
				}
			}
		})
	}
}

func TestMiniCPM5StreamAgreesWithParse(t *testing.T) {
	for _, resp := range []string{
		`<function name="f"><param name="x">1</param></function>`,
		`prose <function name="f"><param name="text"><![CDATA[a </function> b]]></param></function> tail`,
		`<function name="a"></function><function name="b"><param name="y">2</param></function>`,
	} {
		t.Run(resp, func(t *testing.T) {
			_, got := stream(resp, 1)
			if want := parseMiniCPM5ToolCalls(resp, nil); !callsEqual(got, want) {
				t.Errorf("streamed %+v, parsed %+v", got, want)
			}
		})
	}
}
