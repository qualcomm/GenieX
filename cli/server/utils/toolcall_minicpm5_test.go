// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import "testing"

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
			if got := parseMiniCPM5ToolCalls(tt.resp); !callsEqual(got, tt.want) {
				t.Errorf("parsed %+v, want %+v", got, tt.want)
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

func TestMiniCPM5StreamAgreesWithParse(t *testing.T) {
	for _, resp := range []string{
		`<function name="f"><param name="x">1</param></function>`,
		`prose <function name="f"><param name="text"><![CDATA[a </function> b]]></param></function> tail`,
		`<function name="a"></function><function name="b"><param name="y">2</param></function>`,
	} {
		t.Run(resp, func(t *testing.T) {
			_, got := stream(resp, 1)
			if want := parseMiniCPM5ToolCalls(resp); !callsEqual(got, want) {
				t.Errorf("streamed %+v, parsed %+v", got, want)
			}
		})
	}
}
