// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"strings"
	"testing"
)

func callsEqual(got, want []toolCallFn) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i].Name != want[i].Name || got[i].Arguments != want[i].Arguments {
			return false
		}
	}
	return true
}

// stream feeds resp in chunks of size and returns everything emitted as content
// plus every call reported, in order.
func stream(resp string, size int) (string, []toolCallFn) {
	s := NewToolCallScanner()
	text := ""
	var calls []toolCallFn
	for i := 0; i < len(resp); i += size {
		out, got := s.Push(resp[i:min(i+size, len(resp))])
		text += out
		calls = append(calls, got...)
	}
	tail, got := s.Tail()
	return text + tail, append(calls, got...)
}

// Every byte is either streamed as content or consumed by a tool call, and the
// split must not depend on where the token boundaries fall.
func TestScannerStream(t *testing.T) {
	tests := []struct {
		name      string
		resp      string
		wantText  string
		wantCalls []toolCallFn
	}{
		{
			name:     "prose only",
			resp:     "Sure, here is the answer.",
			wantText: "Sure, here is the answer.",
		},
		{
			name:      "bare call after prose",
			resp:      `sure {"name": "f", "arguments": {}}`,
			wantText:  "sure ",
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{}`}},
		},
		{
			// the point of an end marker: the stream resumes after the call
			name:      "prose after a call",
			resp:      `{"name": "f", "arguments": {}} and then some prose`,
			wantText:  " and then some prose",
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{}`}},
		},
		{
			name:      "two calls with prose between",
			resp:      `a {"name":"f","arguments":{}} b {"name":"g","arguments":{"x":1}} c`,
			wantText:  "a  b  c",
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{}`}, {Name: "g", Arguments: `{"x":1}`}},
		},
		{
			name:      "qwen3 wrapper",
			resp:      "<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>",
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{}`}},
		},
		{
			name:      "qwen3 wrapper then prose",
			resp:      "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call> bye",
			wantText:  " bye",
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{}`}},
		},
		{
			name:      "two qwen3 wrappers",
			resp:      `<tool_call>{"name":"f","arguments":{}}</tool_call><tool_call>{"name":"g","arguments":{}}</tool_call>`,
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{}`}, {Name: "g", Arguments: `{}`}},
		},
		{
			// the model stopped before the closing tag: Tail still finds the call
			name:      "unclosed wrapper",
			resp:      `<tool_call>{"name": "f", "arguments": {}}`,
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{}`}},
		},
		{
			// The brace holds the buffer for JSON, but the call in it is gemma4's.
			name:      "a gemma4 call held by a stray brace",
			resp:      `code { <|tool_call>call:f{a:1}<tool_call|>`,
			wantText:  "code ",
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{"a":1}`}},
		},
		{
			name:      "a gemma4 call held by a qwen3 marker",
			resp:      `<tool_call>oops <|tool_call>call:f{a:1}<tool_call|>`,
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{"a":1}`}},
		},
		{
			// one region, two calls: both come out on the Push that closed it
			name:      "two calls in one wrapper",
			resp:      `<tool_call>{"name":"a","arguments":{}}{"name":"b","arguments":{}}</tool_call>`,
			wantCalls: []toolCallFn{{Name: "a", Arguments: `{}`}, {Name: "b", Arguments: `{}`}},
		},
		{
			// what a Mistral-style array syntax looks like inside one region
			name:      "a json array of calls in one wrapper",
			resp:      `<tool_call>[{"name":"a","arguments":{}},{"name":"b","arguments":{"x":1}}]</tool_call>`,
			wantCalls: []toolCallFn{{Name: "a", Arguments: `{}`}, {Name: "b", Arguments: `{"x":1}`}},
		},
		{
			name:     "wrapper holding no call is prose",
			resp:     "<tool_call>not a call at all</tool_call> ok",
			wantText: "<tool_call>not a call at all</tool_call> ok",
		},
		{
			name:     "refuted marker prefix",
			resp:     "<tool_callaaa and more",
			wantText: "<tool_callaaa and more",
		},
		{
			name:     "non-call object is released",
			resp:     `{"foo": 1} hi`,
			wantText: `{"foo": 1} hi`,
		},
		{
			name:     "unclosed brace held to the end",
			resp:     "here is code {\n\tf(a, b)",
			wantText: "here is code {\n\tf(a, b)",
		},
		{
			name:      "braces and escapes inside strings",
			resp:      `{"name":"f","arguments":{"s":"}{ \" }"}} tail`,
			wantText:  " tail",
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{"s":"}{ \" }"}`}},
		},
		{
			name:      "arguments given as a string",
			resp:      `{"name":"echo","arguments":"hi"} bye`,
			wantText:  " bye",
			wantCalls: []toolCallFn{{Name: "echo", Arguments: "hi"}},
		},
		{
			name:      "two wrappers with text between",
			resp:      `<tool_call>{"name":"f","arguments":{}}</tool_call>tail<tool_call>{"name":"g","arguments":{}}</tool_call>`,
			wantText:  "tail",
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{}`}, {Name: "g", Arguments: `{}`}},
		},
		{
			name:     "stray closing marker",
			resp:     "</tool_call> stray close",
			wantText: "</tool_call> stray close",
		},
		{
			name:     "empty wrapper",
			resp:     "<tool_call></tool_call> ok",
			wantText: "<tool_call></tool_call> ok",
		},
		{
			name:      "a marker inside the call's own arguments",
			resp:      `{"name":"f","arguments":{"s":"<tool_call>"}}`,
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{"s":"<tool_call>"}`}},
		},
		{
			name:     "a marker quoted in prose is held, then released",
			resp:     `talk about "<tool_call>" in prose`,
			wantText: `talk about "<tool_call>" in prose`,
		},
		{
			// markerFormat's BUG: the quoted end tag cuts the region short, so this
			// parses as nothing and streams as text. Parse on the whole text finds it.
			name:     "a closing marker inside the arguments loses the call",
			resp:     `<tool_call>{"name":"f","arguments":{"s":"</tool_call>"}}</tool_call>`,
			wantText: `<tool_call>{"name":"f","arguments":{"s":"</tool_call>"}}</tool_call>`,
		},
		{
			name:      "a quoted brace before the call",
			resp:      `note: "{" then {"name":"f","arguments":{}}`,
			wantText:  `note: "`,
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{}`}},
		},
		{
			name:      "braces inside the arguments string",
			resp:      `{"name":"f","arguments":{"s":"}}{{"}}`,
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{"s":"}}{{"}`}},
		},
		{
			name:      "an escaped backslash ends the string",
			resp:      `{"name":"f","arguments":{"p":"C:\\"}} x`,
			wantText:  " x",
			wantCalls: []toolCallFn{{Name: "f", Arguments: `{"p":"C:\\"}`}},
		},
		{
			name:     "a marker that never opens",
			resp:     "a <b <c <too <tool_ nothing opens here",
			wantText: "a <b <c <too <tool_ nothing opens here",
		},
		{
			name:     "ends on a bare marker prefix",
			resp:     "stops mid marker <tool_c",
			wantText: "stops mid marker <tool_c",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 8 bytes is well above any real token; TestParseMatchesStream covers the
			// other end, a whole response in one chunk.
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

// The streaming path and the whole-text scan are separate implementations, so they
// check each other.
func TestStreamAgreesWithJSONParse(t *testing.T) {
	responses := []string{
		"just prose",
		`{"name":"f","arguments":{"a":1}}`,
		`prose {"name":"f","arguments":{}} more`,
		`{"name":"a","arguments":{}} x {"name":"b","arguments":{}}`,
		"<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>",
		`<tool_call>{"name":"f","arguments":{}}`,
		`{"foo":1} noise {"name":"f","arguments":{}}`,
		"a } b { c unbalanced",
	}

	for _, resp := range responses {
		t.Run(resp, func(t *testing.T) {
			_, got := stream(resp, 1)
			if want := parseJSONToolCalls(resp); !callsEqual(got, want) {
				t.Errorf("streamed %+v, parsed %+v", got, want)
			}
		})
	}
}

// Parse is the non-streaming twin of Push, so it must keep and drop the same bytes.
func TestParseMatchesStream(t *testing.T) {
	responses := []string{
		"just prose",
		`sure {"name":"f","arguments":{}} and then prose`,
		`{"name":"a","arguments":{}} x {"name":"b","arguments":{}} y`,
		`<tool_call>{"name":"f","arguments":{}}</tool_call>tail`,
		`{"foo":1} then {"name":"f","arguments":{}}`,
		`hi <|tool_call>call:a{}<tool_call|>mid<|tool_call>call:b{x:1}<tool_call|>bye`,
		`<tool_call>{"name":"f","arguments":{}}`,
		"here is code {\n\tf(a, b)",
	}

	for _, resp := range responses {
		t.Run(resp, func(t *testing.T) {
			wantText, wantCalls := stream(resp, 1)
			text, calls := NewToolCallScanner().Parse(resp)
			if text != wantText {
				t.Errorf("text = %q, want %q", text, wantText)
			}
			if !callsEqual(calls, wantCalls) {
				t.Errorf("calls = %+v, want %+v", calls, wantCalls)
			}
		})
	}
}

// Tail ends the stream, so what it drained is consumed: calling it again is empty.
func TestTailIsTerminal(t *testing.T) {
	for _, resp := range []string{
		"here is code {\n\tf(a, b)",
		`<tool_call>{"name":"f","arguments":{}}`,
	} {
		s := NewToolCallScanner()
		s.Push(resp)
		s.Tail()
		if text, calls := s.Tail(); text != "" || len(calls) > 0 {
			t.Errorf("resp %q: second Tail = (%q, %+v), want empty", resp, text, calls)
		}
	}
}

// A format reporting an offset behind from panics Push on a negative-length slice.
// Only the interface pins that, so run the formats the way Push runs them.
func TestFeedOffsetsStayAhead(t *testing.T) {
	// Near-miss adjacency is what breaks a scan: try every triple of these.
	pieces := []string{"<|tool_call>call:", "<tool_call>", "</tool_call>", "<tool_call|>",
		"{", "}", `"`, `<|"|>`, "a:1", " "}
	var corpus []string
	for _, a := range pieces {
		for _, b := range pieces {
			for _, c := range pieces {
				corpus = append(corpus, a+b+c)
			}
		}
	}
	corpus = append(corpus,
		`{"name":"f","arguments":{}} and {"name":"g","arguments":{}}`,
		`<tool_call>{"name":"f","arguments":{"s":"</tool_call>"}}</tool_call>`,
		`<|tool_call>call:a{}<tool_call|>mid<|tool_call>call:b{x:<|"|>}{<|"|>}<tool_call|>`,
		`{"a":"<tool_call>"} more`,
		"prose { unclosed <tool_call> forever")

	for _, resp := range corpus {
		for size := 1; size <= 3; size++ {
			formats := NewToolCallScanner().formats
			from := 0
			for j := min(size, len(resp)); ; j = min(j+size, len(resp)) {
				for { // Push steps until nothing closes, so from can move several times
					hold, end := j, -1
					for i, f := range formats {
						at, e := f.feed(resp[:j], from)
						if at >= 0 && at < from {
							t.Fatalf("format %d: feed(%q, %d) = (%d, %d), start behind from",
								i, resp[:j], from, at, e)
						}
						if e >= 0 && e < at {
							t.Fatalf("format %d: feed(%q, %d) = (%d, %d), end behind start",
								i, resp[:j], from, at, e)
						}
						if at >= 0 && at < hold {
							hold, end = at, e
						}
					}
					from = hold // what a step emits up to
					if end < 0 {
						break
					}
					from = end // or consumes through
				}
				if j == len(resp) {
					break
				}
			}
		}
	}
}

// The first whole marker, else the earliest tail still a prefix of one.
func TestMarkerScan(t *testing.T) {
	tests := []struct {
		name        string
		marker      string
		s           string
		start, done int
	}{
		{"empty", "<ab>", "", 0, 0},
		{"no candidate", "<ab>", "plain prose", 11, 0},
		{"whole marker", "<ab>", "hi <ab> there", 3, 7},
		{"partial at end", "<ab>", "hi <a", 3, 0},
		{"partial refuted", "<ab>", "hi <xy", 6, 0},
		{"restart inside a candidate", "<ab>", "<<ab>", 1, 5},
		{"overlapping restart", "aab", "aaab", 1, 4},
		{"first of two", "<ab>", "<ab> x <ab>", 0, 4},
		{"stops at the first match", "<ab>", "<ab>zzz", 0, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for size := 1; size <= max(len(tt.s), 1); size++ {
				m := &markerScan{marker: tt.marker}
				for i := 0; i < len(tt.s); i += size {
					m.feed(tt.s[:min(i+size, len(tt.s))])
				}
				if m.start != tt.start || m.done != tt.done {
					t.Errorf("size %d: (start, done) = (%d, %d), want (%d, %d)",
						size, m.start, m.done, tt.start, tt.done)
				}
			}
		})
	}
}

// A region is reported only once both markers are in, and repeated until from
// moves past it.
func TestMarkerFormatFeed(t *testing.T) {
	tests := []struct {
		name            string
		s               string
		wantAt, wantEnd int
	}{
		{"prose", "nothing here", -1, -1},
		{"partial open", "prose <tool_c", 6, -1},
		{"open only", "prose <tool_call>{}", 6, -1},
		{"both markers", "hi <tool_call>x</tool_call>!", 3, 27},
		{"refuted prefix", "prose <tool_callx", -1, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := newMarkerFormat("<tool_call>", "</tool_call>")
			at, end := m.feed(tt.s, 0)
			if at != tt.wantAt || end != tt.wantEnd {
				t.Errorf("feed(%q) = (%d, %d), want (%d, %d)", tt.s, at, end, tt.wantAt, tt.wantEnd)
			}
		})
	}
}

// Each format added puts another marker scan on the prose case, which is the one
// that has to stay fast. 4 bytes per Push stands in for a model token.
func BenchmarkToolCallScanner(b *testing.B) {
	prose := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 100)
	call := `{"name": "write_file", "arguments": {"path": "a.go", "body": "` +
		strings.Repeat("x", 4096) + `"}}`
	// A stray brace in prose holds everything after it: the worst case.
	held := "here is the code {\n" + strings.Repeat("\tdoSomething(a, b)\n", 1500)

	for _, bb := range []struct{ name, resp string }{
		{"prose", prose},
		{"call", prose[:200] + call},
		{"two_calls", prose[:200] + call + " and then " + call},
		{"wrapped", prose[:200] + "<tool_call>" + call + "</tool_call>" + prose[:200]},
		{"held", held},
	} {
		b.Run(bb.name, func(b *testing.B) {
			b.SetBytes(int64(len(bb.resp)))
			for b.Loop() {
				s := NewToolCallScanner()
				for i := 0; i < len(bb.resp); i += 4 {
					s.Push(bb.resp[i:min(i+4, len(bb.resp))])
				}
				s.Tail()
			}
		})
	}
}
