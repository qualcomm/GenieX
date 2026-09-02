// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"strings"

	"github.com/bytedance/sonic"
)

// Qwen3.5, Qwen3-Coder, Nemotron Nano 3 and StepFun-3.5 keep Qwen3's `<tool_call>`
// wrapper but spell the body as tags, one per line:
//
//	<tool_call>
//	<function=NAME>
//	<parameter=KEY>
//	VALUE
//	</parameter>
//	</function>
//	</tool_call>
//
// The wrapper is the same as qwen3ToolCall's, so both formats claim the region and
// whichever reads a call out of it wins. Grammar: llama.cpp's
// common_chat_params_init_qwen3_coder.
const (
	qwen35FuncOpen   = "<function="
	qwen35FuncClose  = "</function>"
	qwen35ParamOpen  = "<parameter="
	qwen35ParamClose = "</parameter>"
)

type qwen35ToolCall struct {
	markerFormat
}

func newQwen35ToolCall() *qwen35ToolCall {
	return &qwen35ToolCall{newMarkerFormat("<tool_call>", "</tool_call>")}
}

func (t *qwen35ToolCall) parse(s string) []toolCallFn { return parseQwen35ToolCalls(s) }

// parseQwen35ToolCalls returns every function tag in s. A tag left open counts only
// once an argument has closed, so a call the model cut short is still recovered
// without reading prose that mentions the tag as one.
func parseQwen35ToolCalls(s string) []toolCallFn {
	var calls []toolCallFn
	for {
		i := strings.Index(s, qwen35FuncOpen)
		if i < 0 {
			return calls
		}
		s = s[i+len(qwen35FuncOpen):]
		gt := strings.IndexByte(s, '>')
		if gt < 0 {
			return calls
		}
		name, body, closed := strings.TrimSpace(s[:gt]), s[gt+1:], true
		if end := strings.Index(body, qwen35FuncClose); end >= 0 {
			body, s = body[:end], body[end+len(qwen35FuncClose):]
		} else {
			closed, s = false, ""
		}
		args := parseQwen35Params(body)
		if name == "" || (!closed && args == "{}") {
			continue
		}
		calls = append(calls, toolCallFn{Name: name, Arguments: args})
	}
}

// parseQwen35Params encodes the parameter tags of one call as a JSON object. A tag
// that never closes is dropped, so a truncated call keeps the arguments before it.
func parseQwen35Params(body string) string {
	var b strings.Builder
	b.WriteByte('{')
	for {
		i := strings.Index(body, qwen35ParamOpen)
		if i < 0 {
			break
		}
		body = body[i+len(qwen35ParamOpen):]
		gt := strings.IndexByte(body, '>')
		if gt < 0 {
			break
		}
		end := strings.Index(body[gt+1:], qwen35ParamClose)
		if end < 0 {
			break
		}
		key, val := strings.TrimSpace(body[:gt]), body[gt+1:gt+1+end]
		body = body[gt+1+end+len(qwen35ParamClose):]
		if key == "" {
			continue
		}
		if b.Len() > 1 {
			b.WriteByte(',')
		}
		quoted, _ := sonic.MarshalString(key)
		b.WriteString(quoted)
		b.WriteByte(':')
		b.WriteString(parseQwen35Value(val))
	}
	b.WriteByte('}')
	return b.String()
}

// parseQwen35Value encodes one parameter body as JSON. The syntax carries no types:
// llama.cpp reads them off the tool schema, which the parser here does not have, so
// a value that is already JSON goes through as it is and anything else becomes a
// string. A string argument that looks like a number is the case this gets wrong.
// The tags stand on their own lines, so one line break on each side is syntax.
func parseQwen35Value(v string) string {
	v = strings.TrimSuffix(strings.TrimPrefix(v, "\n"), "\n")
	if trimmed := strings.TrimSpace(v); trimmed != "" && sonic.ValidString(trimmed) {
		return trimmed
	}
	quoted, _ := sonic.MarshalString(v)
	return quoted
}
