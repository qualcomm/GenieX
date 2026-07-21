// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestFormatToolCallBlockValidJSON guards the Round-2 assistant-turn
// concatenation path: the previous fmt.Sprintf-based implementation embedded
// the OpenAI-schema `arguments` (already a JSON string) via %s, producing
// invalid JSON that crashed the C-side chat-template applier.
func TestFormatToolCallBlockValidJSON(t *testing.T) {
	cases := []struct {
		name      string
		toolName  string
		arguments string
	}{
		{"simple", "get_current_weather", `{"city":"San Diego"}`},
		{"nested", "search", `{"q":"quoted \"word\"","limit":10}`},
		{"empty_object", "list_files", `{}`},
		{"array", "batch", `[1,2,3]`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := formatToolCallBlock(tc.toolName, tc.arguments)
			if !strings.HasPrefix(out, "<tool_call>") || !strings.HasSuffix(out, "</tool_call>") {
				t.Fatalf("missing tool_call tags: %q", out)
			}
			inner := strings.TrimSuffix(strings.TrimPrefix(out, "<tool_call>"), "</tool_call>")

			// The inner payload must be parseable JSON — this is the
			// invariant the CGO crash violates when %s splices a raw
			// JSON string into "arguments":"%s".
			var parsed struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}
			if err := json.Unmarshal([]byte(inner), &parsed); err != nil {
				t.Fatalf("inner payload is not valid JSON: %v (payload=%q)", err, inner)
			}
			if parsed.Name != tc.toolName {
				t.Errorf("name mismatch: got %q, want %q", parsed.Name, tc.toolName)
			}
			// Arguments should round-trip as the original JSON structure —
			// compare parsed forms so key ordering / whitespace differences
			// from the JSON encoder don't spuriously fail the check.
			var gotArgs, wantArgs any
			if err := json.Unmarshal(parsed.Arguments, &gotArgs); err != nil {
				t.Fatalf("arguments not valid JSON: %v", err)
			}
			if err := json.Unmarshal([]byte(tc.arguments), &wantArgs); err != nil {
				t.Fatalf("test-case arguments not valid JSON: %v", err)
			}
			gotJSON, _ := json.Marshal(gotArgs)
			wantJSON, _ := json.Marshal(wantArgs)
			if string(gotJSON) != string(wantJSON) {
				t.Errorf("arguments mismatch: got %s, want %s", gotJSON, wantJSON)
			}
		})
	}
}

// TestFormatToolCallBlockInvalidArgumentsFallback guards the graceful path
// when a model emits invalid JSON as `arguments`: the block must still be
// well-formed JSON so the chat template doesn't crash.
func TestFormatToolCallBlockInvalidArgumentsFallback(t *testing.T) {
	out := formatToolCallBlock("weird", "not valid json {")
	inner := strings.TrimSuffix(strings.TrimPrefix(out, "<tool_call>"), "</tool_call>")

	var parsed map[string]any
	if err := json.Unmarshal([]byte(inner), &parsed); err != nil {
		t.Fatalf("fallback payload is not valid JSON: %v (payload=%q)", err, inner)
	}
	if parsed["name"] != "weird" {
		t.Errorf("name mismatch: got %v", parsed["name"])
	}
	// Malformed arguments get wrapped as a JSON string so the payload stays
	// well-formed rather than corrupting downstream parsers.
	if _, ok := parsed["arguments"].(string); !ok {
		t.Errorf("expected arguments to be encoded as string, got %T", parsed["arguments"])
	}
}
