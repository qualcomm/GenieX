// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"encoding/xml"
	"strings"

	"github.com/bytedance/sonic"
)

// MiniCPM5 emits one XML element for each call, without an outer wrapper:
//
//	<function name="get_weather"><param name="city">Beijing</param></function>
//
// Parameter values can be multiline and, for special characters, wrapped in a
// CDATA block. encoding/xml handles both forms and decodes XML entities before
// the values are re-encoded as the JSON object OpenAI-compatible clients expect.
const miniCPM5FunctionOpen = "<function name="

type miniCPM5ToolCall struct {
	pos int // bytes searched, except a possible opener prefix at the tail
}

func newMiniCPM5ToolCall() *miniCPM5ToolCall { return &miniCPM5ToolCall{} }

func (t *miniCPM5ToolCall) parse(s string) []toolCallFn { return parseMiniCPM5ToolCalls(s) }

func (t *miniCPM5ToolCall) feed(all string, from int) (int, int) {
	at, openEnd := findMarker(all, miniCPM5FunctionOpen, max(from, t.pos))
	if at < 0 {
		t.pos = max(from, len(all)-len(miniCPM5FunctionOpen)+1)
		return -1, -1
	}
	if openEnd == 0 {
		t.pos = at
		return at, -1
	}
	if _, end, ok := decodeMiniCPM5Function(all[at:]); ok {
		t.pos = at // report the same completed region until the scanner consumes it
		return at, at + end
	}
	t.pos = at
	return at, -1
}

type miniCPM5Function struct {
	XMLName xml.Name        `xml:"function"`
	Name    string          `xml:"name,attr"`
	Params  []miniCPM5Param `xml:"param"`
}

type miniCPM5Param struct {
	Name  string `xml:"name,attr"`
	Value string `xml:",chardata"`
}

func decodeMiniCPM5Function(s string) (miniCPM5Function, int, bool) {
	var fn miniCPM5Function
	decoder := xml.NewDecoder(strings.NewReader(s))
	if err := decoder.Decode(&fn); err != nil {
		return fn, 0, false
	}
	return fn, int(decoder.InputOffset()), true
}

// parseMiniCPM5ToolCalls returns every complete, well-formed MiniCPM5 function
// element in s. Invalid XML is deliberately not repaired: treating malformed
// model text as an executable call would be unsafe.
func parseMiniCPM5ToolCalls(s string) []toolCallFn {
	var calls []toolCallFn
	for {
		i := strings.Index(s, miniCPM5FunctionOpen)
		if i < 0 {
			return calls
		}
		s = s[i:]

		fn, end, ok := decodeMiniCPM5Function(s)
		if !ok {
			s = s[len(miniCPM5FunctionOpen):]
			continue
		}
		s = s[end:]
		if fn.Name == "" {
			continue
		}

		args := miniCPM5Arguments(fn.Params)
		calls = append(calls, toolCallFn{Name: fn.Name, Arguments: args})
	}
}

func miniCPM5Arguments(params []miniCPM5Param) string {
	var b strings.Builder
	b.WriteByte('{')
	for _, param := range params {
		if param.Name == "" {
			continue
		}
		if b.Len() > 1 {
			b.WriteByte(',')
		}
		name, _ := sonic.MarshalString(param.Name)
		value, _ := sonic.MarshalString(param.Value)
		b.WriteString(name)
		b.WriteByte(':')
		b.WriteString(value)
	}
	b.WriteByte('}')
	return b.String()
}
