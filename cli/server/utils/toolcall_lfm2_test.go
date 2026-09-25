// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import "testing"

// The cases llama.cpp's LFM2 tests cover, against the bare call-list body: parse
// strips the surrounding control-token markers before parseLFM2ToolCalls ever sees it.
func TestParseLFM2ToolCalls(t *testing.T) {
	tests := []struct {
		name string
		resp string
		want []toolCallFn // nil: no tool call
	}{
		{
			name: "one argument",
			resp: `[special_function(arg1=1)]`,
			want: []toolCallFn{{Name: "special_function", Arguments: `{"arg1":1}`}},
		},
		{
			name: "string argument",
			resp: `[get_time(city="XYZCITY")]`,
			want: []toolCallFn{{Name: "get_time", Arguments: `{"city":"XYZCITY"}`}},
		},
		{
			name: "single quoted string",
			resp: `[get_time(city='Paris')]`,
			want: []toolCallFn{{Name: "get_time", Arguments: `{"city":"Paris"}`}},
		},
		{
			name: "python literals become json",
			resp: `[toggle(enabled=True, off=False, value=None)]`,
			want: []toolCallFn{{Name: "toggle", Arguments: `{"enabled":true,"off":false,"value":null}`}},
		},
		{
			name: "json literals are accepted too",
			resp: `[set_config(config={"enabled": true, "note": null})]`,
			want: []toolCallFn{{Name: "set_config", Arguments: `{"config":{"enabled":true,"note":null}}`}},
		},
		{
			name: "nested python dict",
			resp: `[set_config(config={"enabled": True, "count": 3})]`,
			want: []toolCallFn{{Name: "set_config", Arguments: `{"config":{"enabled":true,"count":3}}`}},
		},
		{
			// the grammar's dict member allows space on both sides of ':', unlike a
			// bare key=value call argument
			name: "space before the colon in a nested dict",
			resp: `[configure(opts={"key" : "value"})]`,
			want: []toolCallFn{{Name: "configure", Arguments: `{"opts":{"key":"value"}}`}},
		},
		{
			name: "dotted name and an array",
			resp: `[Calendar.create_event(title="demo", participants=["Alice", "Bob"])]`,
			want: []toolCallFn{{Name: "Calendar.create_event",
				Arguments: `{"title":"demo","participants":["Alice","Bob"]}`}},
		},
		{
			name: "no arguments",
			resp: `[empty_args()]`,
			want: []toolCallFn{{Name: "empty_args", Arguments: `{}`}},
		},
		{
			name: "escapes decode to the characters they name",
			resp: `[python(code="def hello():\n    print('hey')")]`,
			want: []toolCallFn{{Name: "python", Arguments: `{"code":"def hello():\n    print('hey')"}`}},
		},
		{
			name: "escaped quotes survive",
			resp: `[python(code="print(\"hi\")")]`,
			want: []toolCallFn{{Name: "python", Arguments: `{"code":"print(\"hi\")"}`}},
		},
		{
			name: "brackets inside a string",
			resp: `[python(code="a[0], b(1)")]`,
			want: []toolCallFn{{Name: "python", Arguments: `{"code":"a[0], b(1)"}`}},
		},
		{
			name: "parallel calls",
			resp: `[a(arg1=1), b(arg1=1, arg2=2)]`,
			want: []toolCallFn{{Name: "a", Arguments: `{"arg1":1}`},
				{Name: "b", Arguments: `{"arg1":1,"arg2":2}`}},
		},
		{
			name: "negative and float numbers",
			resp: `[f(x=-4.2, y=1e3)]`,
			want: []toolCallFn{{Name: "f", Arguments: `{"x":-4.2,"y":1e3}`}},
		},
		{
			name: "markdown links stay content",
			resp: `Use this format: [link text](url). Example: [Wikipedia](https://www.wikipedia.org).`,
		},
		{
			name: "a json array of objects is not a call list",
			resp: `[{"name":"f","arguments":{}}]`,
		},
		{
			name: "positional arguments are not the grammar",
			resp: `[f(1)]`,
		},
		{
			name: "one malformed call drops the whole list",
			resp: `[a(x=1), b(oops)]`,
		},
		{
			name: "call cut short",
			resp: `[special_function(arg1=`,
		},
		{
			// the markers are the model's, but the region handed here starts at '['
			name: "a marker left in front is not the grammar",
			resp: `<|tool_call_start|>[special_function(arg1=1)]`,
		},
		{
			name: "empty list",
			resp: `[]`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parseLFM2ToolCalls(tt.resp); !callsEqual(got, tt.want) {
				t.Errorf("parseLFM2ToolCalls() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

// The boundary is the marker pair, not the brackets: a list with neither marker is
// just content, byte for byte, same as any other bracket a model might write.
func TestLFM2ToolCallStream(t *testing.T) {
	tests := []struct {
		name      string
		resp      string
		wantText  string
		wantCalls []toolCallFn
	}{
		{
			name:      "markers around the list are consumed with the call",
			resp:      `<|tool_call_start|>[get_time(city="Paris")]<|tool_call_end|>`,
			wantCalls: []toolCallFn{{Name: "get_time", Arguments: `{"city":"Paris"}`}},
		},
		{
			name:      "content before a marker-wrapped call",
			resp:      `Let me check.<|tool_call_start|>[get_time(city="Paris")]<|tool_call_end|>`,
			wantText:  "Let me check.",
			wantCalls: []toolCallFn{{Name: "get_time", Arguments: `{"city":"Paris"}`}},
		},
		{
			name:      "content after a marker-wrapped call",
			resp:      `<|tool_call_start|>[get_time(city="Paris")]<|tool_call_end|> done`,
			wantText:  " done",
			wantCalls: []toolCallFn{{Name: "get_time", Arguments: `{"city":"Paris"}`}},
		},
		{
			name:      "parallel calls, both markers",
			resp:      `<|tool_call_start|>[a(x=1), b(y=2)]<|tool_call_end|>`,
			wantCalls: []toolCallFn{{Name: "a", Arguments: `{"x":1}`}, {Name: "b", Arguments: `{"y":2}`}},
		},
		{
			// an unrelated bracket ahead of the real call is never even looked at:
			// nothing here scans for '[' on its own anymore
			name:      "an unrelated bracket before a marker-wrapped call",
			resp:      `See [1] for details.<|tool_call_start|>[get_time(city="Paris")]<|tool_call_end|>`,
			wantText:  "See [1] for details.",
			wantCalls: []toolCallFn{{Name: "get_time", Arguments: `{"city":"Paris"}`}},
		},
		{
			// no start marker at all: the list is not even a candidate, just content
			name:     "a bare list with no markers is content",
			resp:     `[get_time(city="Paris")]`,
			wantText: `[get_time(city="Paris")]`,
		},
		{
			// an end marker with nothing to open it is content too
			name:     "an end marker with no start marker is content",
			resp:     `[get_time(city="Paris")]<|tool_call_end|>`,
			wantText: `[get_time(city="Paris")]<|tool_call_end|>`,
		},
		{
			// the model stopped before the closing marker: Tail still finds the call,
			// since the list itself is a complete, well-formed one
			name:      "only the start marker: recovered at the tail if the list is complete",
			resp:      `<|tool_call_start|>[get_time(city="Paris")]`,
			wantCalls: []toolCallFn{{Name: "get_time", Arguments: `{"city":"Paris"}`}},
		},
		{
			// no list ever follows the marker, and no end marker arrives either
			name:     "a start marker with no list after it is content",
			resp:     `<|tool_call_start|>not a list after all`,
			wantText: `<|tool_call_start|>not a list after all`,
		},
		{
			// both markers close the region, but the body inside is not the grammar
			name:     "a start marker around something that is not a call",
			resp:     `<|tool_call_start|>[1, 2, 3]<|tool_call_end|>`,
			wantText: `<|tool_call_start|>[1, 2, 3]<|tool_call_end|>`,
		},
		{
			name:     "markdown links stay content",
			resp:     "see [Wikipedia](https://www.wikipedia.org) and [a list] of things",
			wantText: "see [Wikipedia](https://www.wikipedia.org) and [a list] of things",
		},
		{
			name:     "a list that is not a call is content",
			resp:     "steps: [1, 2, 3] and then [f(x=1] broken",
			wantText: "steps: [1, 2, 3] and then [f(x=1] broken",
		},
		{
			name:     "an unclosed bracket is content, not held",
			resp:     "an [ that never closes",
			wantText: "an [ that never closes",
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
