// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

// qwen3ToolCall is the `<tool_call>{json}</tool_call>` wrapper of Qwen3 and
// Hermes-style templates. Its body is plain JSON, so it shares that parser.
type qwen3ToolCall struct {
	markerFormat
}

func newQwen3ToolCall() *qwen3ToolCall {
	return &qwen3ToolCall{newMarkerFormat("<tool_call>", "</tool_call>")}
}

func (t *qwen3ToolCall) parse(s string) []toolCallFn { return parseJSONToolCalls(s) }
