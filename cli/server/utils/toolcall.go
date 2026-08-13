// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/bytedance/sonic/ast"
	"github.com/openai/openai-go/v3"
)

const (
	gemma4ToolCallStart = "<|tool_call>"
	gemma4ToolCallEnd   = "<tool_call|>"
	gemma4StringMarker  = "<|\"|>"
)

// gemma4ValueParser converts Gemma 4's native argument notation into JSON.
// Gemma 4 uses unquoted object keys and <|"|> delimiters for string values:
//
//	{query:<|"|>coffee latte<|"|>}
//
// This parser is deliberately limited to JSON-equivalent values. It never
// evaluates expressions or accepts executable syntax.
type gemma4ValueParser struct {
	value string
	pos   int
}

func (p *gemma4ValueParser) skipSpace() {
	for p.pos < len(p.value) {
		switch p.value[p.pos] {
		case ' ', '\t', '\r', '\n':
			p.pos++
		default:
			return
		}
	}
}

func (p *gemma4ValueParser) parseValue() (string, error) {
	p.skipSpace()
	if p.pos >= len(p.value) {
		return "", errors.New("missing Gemma 4 tool argument value")
	}
	switch {
	case strings.HasPrefix(p.value[p.pos:], gemma4StringMarker):
		return p.parseString()
	case p.value[p.pos] == '{':
		return p.parseObject()
	case p.value[p.pos] == '[':
		return p.parseArray()
	default:
		return p.parseLiteral()
	}
}

func (p *gemma4ValueParser) parseString() (string, error) {
	p.pos += len(gemma4StringMarker)
	end := strings.Index(p.value[p.pos:], gemma4StringMarker)
	if end < 0 {
		return "", errors.New("unterminated Gemma 4 tool string")
	}
	raw := p.value[p.pos : p.pos+end]
	p.pos += end + len(gemma4StringMarker)
	encoded, err := json.Marshal(raw)
	return string(encoded), err
}

func (p *gemma4ValueParser) parseKey() (string, error) {
	p.skipSpace()
	if strings.HasPrefix(p.value[p.pos:], gemma4StringMarker) {
		encoded, err := p.parseString()
		if err != nil {
			return "", err
		}
		var key string
		if err := json.Unmarshal([]byte(encoded), &key); err != nil {
			return "", err
		}
		return key, nil
	}
	start := p.pos
	for p.pos < len(p.value) && p.value[p.pos] != ':' {
		switch p.value[p.pos] {
		case '{', '}', '[', ']', ',':
			return "", errors.New("invalid Gemma 4 tool argument key")
		}
		p.pos++
	}
	if p.pos >= len(p.value) {
		return "", errors.New("missing Gemma 4 tool argument separator")
	}
	key := strings.TrimSpace(p.value[start:p.pos])
	if key == "" {
		return "", errors.New("empty Gemma 4 tool argument key")
	}
	return key, nil
}

func (p *gemma4ValueParser) parseObject() (string, error) {
	p.pos++
	var result strings.Builder
	result.WriteByte('{')
	p.skipSpace()
	if p.pos < len(p.value) && p.value[p.pos] == '}' {
		p.pos++
		result.WriteByte('}')
		return result.String(), nil
	}
	for count := 0; ; count++ {
		if count > 0 {
			result.WriteByte(',')
		}
		key, err := p.parseKey()
		if err != nil {
			return "", err
		}
		if p.pos >= len(p.value) || p.value[p.pos] != ':' {
			return "", errors.New("missing Gemma 4 tool argument separator")
		}
		p.pos++
		encodedKey, err := json.Marshal(key)
		if err != nil {
			return "", err
		}
		encodedValue, err := p.parseValue()
		if err != nil {
			return "", err
		}
		result.Write(encodedKey)
		result.WriteByte(':')
		result.WriteString(encodedValue)
		p.skipSpace()
		if p.pos >= len(p.value) {
			return "", errors.New("unterminated Gemma 4 tool argument object")
		}
		switch p.value[p.pos] {
		case ',':
			p.pos++
		case '}':
			p.pos++
			result.WriteByte('}')
			return result.String(), nil
		default:
			return "", errors.New("invalid Gemma 4 tool argument object")
		}
	}
}

func (p *gemma4ValueParser) parseArray() (string, error) {
	p.pos++
	var result strings.Builder
	result.WriteByte('[')
	p.skipSpace()
	if p.pos < len(p.value) && p.value[p.pos] == ']' {
		p.pos++
		result.WriteByte(']')
		return result.String(), nil
	}
	for count := 0; ; count++ {
		if count > 0 {
			result.WriteByte(',')
		}
		encoded, err := p.parseValue()
		if err != nil {
			return "", err
		}
		result.WriteString(encoded)
		p.skipSpace()
		if p.pos >= len(p.value) {
			return "", errors.New("unterminated Gemma 4 tool argument array")
		}
		switch p.value[p.pos] {
		case ',':
			p.pos++
		case ']':
			p.pos++
			result.WriteByte(']')
			return result.String(), nil
		default:
			return "", errors.New("invalid Gemma 4 tool argument array")
		}
	}
}

func (p *gemma4ValueParser) parseLiteral() (string, error) {
	start := p.pos
	for p.pos < len(p.value) && !strings.ContainsRune(",}] \t\r\n", rune(p.value[p.pos])) {
		p.pos++
	}
	literal := p.value[start:p.pos]
	var decoded any
	if literal == "" || json.Unmarshal([]byte(literal), &decoded) != nil {
		return "", fmt.Errorf("invalid Gemma 4 tool argument literal %q", literal)
	}
	return literal, nil
}

func parseGemma4ToolCall(resp string) (openai.ChatCompletionMessageFunctionToolCallFunction, error) {
	toolCall := openai.ChatCompletionMessageFunctionToolCallFunction{}
	searchFrom := 0
	for searchFrom < len(resp) {
		startOffset := strings.Index(resp[searchFrom:], gemma4ToolCallStart)
		if startOffset < 0 {
			break
		}
		start := searchFrom + startOffset + len(gemma4ToolCallStart)
		endOffset := strings.Index(resp[start:], gemma4ToolCallEnd)
		if endOffset < 0 {
			break
		}
		payload := strings.TrimSpace(resp[start : start+endOffset])
		searchFrom = start + endOffset + len(gemma4ToolCallEnd)
		if !strings.HasPrefix(payload, "call:") {
			continue
		}
		payload = strings.TrimSpace(strings.TrimPrefix(payload, "call:"))
		objectStart := strings.IndexByte(payload, '{')
		if objectStart <= 0 {
			continue
		}
		name := strings.TrimSpace(payload[:objectStart])
		if name == "" || strings.ContainsAny(name, "{}[],: \t\r\n") {
			continue
		}
		parser := gemma4ValueParser{value: payload[objectStart:]}
		arguments, err := parser.parseObject()
		if err != nil {
			continue
		}
		parser.skipSpace()
		if parser.pos != len(parser.value) {
			continue
		}
		toolCall.Name = name
		toolCall.Arguments = arguments
		return toolCall, nil
	}
	return toolCall, errors.New("Gemma 4 tool call not match")
}

// balancedObjectEnd returns the index just past the '}' matching the '{' at
// start, skipping braces inside string literals. -1 if never closed.
func balancedObjectEnd(s string, start int) int {
	depth := 0
	inStr, escaped := false, false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return i + 1
			}
		}
	}
	return -1
}

// extractJSONObjects returns every balanced {...} substring in s, skipping
// stray or unterminated '{' so later objects are still found.
func extractJSONObjects(s string) []string {
	var objs []string
	for i := 0; i < len(s); i++ {
		if s[i] != '{' {
			continue
		}
		end := balancedObjectEnd(s, i)
		if end < 0 {
			continue
		}
		objs = append(objs, s[i:end])
		i = end - 1
	}
	return objs
}

// parseToolCallObject decodes one JSON object into a tool call, succeeding only
// when "name" is a string and "arguments" is an object or string.
func parseToolCallObject(obj string) (openai.ChatCompletionMessageFunctionToolCallFunction, error) {
	toolCall := openai.ChatCompletionMessageFunctionToolCallFunction{}

	name, err := sonic.GetFromString(obj, "name")
	if err != nil {
		return toolCall, err
	}
	if name.TypeSafe() != ast.V_STRING {
		return toolCall, errors.New("name is not a string")
	}
	toolCall.Name, err = name.String()
	if err != nil {
		return toolCall, err
	}

	arguments, err := sonic.GetFromString(obj, "arguments")
	if err != nil {
		return toolCall, err
	}
	switch arguments.TypeSafe() {
	case ast.V_OBJECT:
		toolCall.Arguments, _ = arguments.Raw()
	case ast.V_STRING:
		toolCall.Arguments, _ = arguments.String()
	default:
		return toolCall, errors.New("unknown arguments type")
	}

	return toolCall, nil
}

// ParseToolCalls returns the first balanced {...} object in resp that decodes
// into a valid tool call, skipping any malformed or non-tool-call objects.
func ParseToolCalls(resp string) (openai.ChatCompletionMessageFunctionToolCallFunction, error) {
	if toolCall, err := parseGemma4ToolCall(resp); err == nil {
		slog.Debug("Parsed Gemma 4 native tool call", "tool_call", toolCall)
		return toolCall, nil
	}
	for _, obj := range extractJSONObjects(resp) {
		if toolCall, err := parseToolCallObject(obj); err == nil {
			slog.Debug("Parsed tool call", "tool_call", toolCall)
			return toolCall, nil
		}
	}

	return openai.ChatCompletionMessageFunctionToolCallFunction{}, errors.New("tool call not match")
}
