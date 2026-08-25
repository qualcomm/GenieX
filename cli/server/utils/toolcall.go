// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"errors"
	"log/slog"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/bytedance/sonic/ast"
	"github.com/openai/openai-go/v3"
)

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

// ParseToolCalls returns the first tool call in resp. Gemma 4's non-JSON
// `<|tool_call>call:...` syntax is detected by its marker; every other model
// wraps a `{"name":..., "arguments":...}` JSON object we scan for directly.
func ParseToolCalls(resp string) (openai.ChatCompletionMessageFunctionToolCallFunction, error) {
	if strings.Contains(resp, gemma4ToolCallOpen) {
		return parseToolCallsGemma4(resp)
	}

	for _, obj := range extractJSONObjects(resp) {
		if toolCall, err := parseToolCallObject(obj); err == nil {
			slog.Debug("Parsed tool call", "tool_call", toolCall)
			return toolCall, nil
		}
	}

	return openai.ChatCompletionMessageFunctionToolCallFunction{}, errors.New("tool call not match")
}
