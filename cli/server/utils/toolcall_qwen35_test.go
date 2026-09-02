// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import "testing"

// The cases llama.cpp's Qwen3-Coder tests cover, minus the ones that need the tool
// schema to type an argument.
func TestParseQwen35ToolCalls(t *testing.T) {
	tests := []struct {
		name string
		resp string
		want []toolCallFn // nil: no tool call
	}{
		{
			name: "one parameter",
			resp: "<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>",
			want: []toolCallFn{{Name: "special_function", Arguments: `{"arg1":1}`}},
		},
		{
			name: "string parameter",
			resp: "<function=get_weather>\n<parameter=city>\nBeijing\n</parameter>\n</function>",
			want: []toolCallFn{{Name: "get_weather", Arguments: `{"city":"Beijing"}`}},
		},
		{
			name: "no parameters",
			resp: "<function=now>\n</function>",
			want: []toolCallFn{{Name: "now", Arguments: `{}`}},
		},
		{
			name: "several parameters keep their order",
			resp: "<function=edit>\n<parameter=file>\na.c\n</parameter>\n" +
				"<parameter=line>\n42\n</parameter>\n</function>",
			want: []toolCallFn{{Name: "edit", Arguments: `{"file":"a.c","line":42}`}},
		},
		{
			name: "one line per tag is not required",
			resp: "<function=get_weather><parameter=city>Beijing</parameter></function>",
			want: []toolCallFn{{Name: "get_weather", Arguments: `{"city":"Beijing"}`}},
		},
		{
			name: "multiline code keeps its line breaks",
			resp: "<function=python>\n<parameter=code>\ndef hello():\n    print(\"hi\")\n\nhello()\n" +
				"</parameter>\n</function>",
			want: []toolCallFn{{Name: "python",
				Arguments: `{"code":"def hello():\n    print(\"hi\")\n\nhello()"}`}},
		},
		{
			name: "markup value is not read as tags",
			resp: "<function=html>\n<parameter=markup>\n<html>\n <title>Hello!</title>\n</html>\n" +
				"</parameter>\n</function>",
			want: []toolCallFn{{Name: "html",
				Arguments: `{"markup":"<html>\n <title>Hello!</title>\n</html>"}`}},
		},
		{
			name: "json value goes through as it is",
			resp: "<function=todo>\n<parameter=items>\n[{\"item\": \"a\", \"done\": false}]\n" +
				"</parameter>\n</function>",
			want: []toolCallFn{{Name: "todo", Arguments: `{"items":[{"item": "a", "done": false}]}`}},
		},
		{
			name: "unicode value",
			resp: "<function=python>\n<parameter=code>\n格\n</parameter>\n</function>",
			want: []toolCallFn{{Name: "python", Arguments: `{"code":"格"}`}},
		},
		{
			name: "two calls in one region",
			resp: "<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n" +
				"<function=b>\n<parameter=y>\n2\n</parameter>\n</function>",
			want: []toolCallFn{{Name: "a", Arguments: `{"x":1}`}, {Name: "b", Arguments: `{"y":2}`}},
		},
		{
			// Tail's last chance: the arguments that did arrive are still a call
			name: "function tag left open",
			resp: "<function=f>\n<parameter=x>\n1\n</parameter>\n",
			want: []toolCallFn{{Name: "f", Arguments: `{"x":1}`}},
		},
		{
			// nothing closed: prose opening with the tag reads exactly the same
			name: "parameter cut short is not a call",
			resp: "<function=f>\n<parameter=x>\n1",
		},
		{
			name: "no function tag",
			resp: "<parameter=x>\n1\n</parameter>",
		},
		{
			name: "unnamed function",
			resp: "<function=>\n</function>",
		},
		{
			name: "prose that mentions the tag",
			resp: "call it with <function= to open the call",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parseQwen35ToolCalls(tt.resp); !callsEqual(got, tt.want) {
				t.Errorf("parseQwen35ToolCalls() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

// The repro of #1525: Qwen3.5 answers in tags inside Qwen3's wrapper, so the two
// formats hold the same region and the call still has to come out of the stream
// whole, with no text left behind.
func TestQwen35ToolCallStream(t *testing.T) {
	call := "<function=get_weather>\n<parameter=city>\nBeijing\n</parameter>\n</function>\n"
	want := []toolCallFn{{Name: "get_weather", Arguments: `{"city":"Beijing"}`}}

	tests := []struct {
		name      string
		resp      string
		wantText  string
		wantCalls []toolCallFn
	}{
		{
			name:      "wrapped call",
			resp:      "<tool_call>\n" + call + "</tool_call>",
			wantCalls: want,
		},
		{
			name:      "after content",
			resp:      "Let me check.\n<tool_call>\n" + call + "</tool_call>",
			wantText:  "Let me check.\n",
			wantCalls: want,
		},
		{
			name:      "content after the call",
			resp:      "<tool_call>\n" + call + "</tool_call> done",
			wantText:  " done",
			wantCalls: want,
		},
		{
			name:      "parallel calls",
			resp:      "<tool_call>\n" + call + "</tool_call>\n<tool_call>\n" + call + "</tool_call>",
			wantText:  "\n", // the line break between two wrappers is content
			wantCalls: append(want, want...),
		},
		{
			// the wrapper is Qwen3's, and a JSON body still parses as one
			name:      "json body in the same wrapper",
			resp:      `<tool_call>{"name":"get_weather","arguments":{"city":"Beijing"}}</tool_call>`,
			wantCalls: []toolCallFn{{Name: "get_weather", Arguments: `{"city":"Beijing"}`}},
		},
		{
			name:     "tags in prose stay prose",
			resp:     "the tag is <function=name> in this template",
			wantText: "the tag is <function=name> in this template",
		},
		{
			name:     "wrapper holding neither syntax is prose",
			resp:     "<tool_call>not a call at all</tool_call> ok",
			wantText: "<tool_call>not a call at all</tool_call> ok",
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
