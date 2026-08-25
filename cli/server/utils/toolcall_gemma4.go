// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"errors"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/openai/openai-go/v3"
)

// Gemma 4 emits tool calls as `<|tool_call>call:NAME{key:value,...}<tool_call|>`
// instead of JSON. The body is a custom dict where strings are wrapped in
// `<|"|>...<|"|>`, keys are bare up to ':', and scalars (number/bool/null) are
// verbatim. Grammar mirrors llama.cpp's common_chat_params_init_gemma4.
const (
	gemma4ToolCallOpen = "<|tool_call>call:"
	gemma4StringMarker = `<|"|>`
)

var errGemma4 = errors.New("gemma4 tool call not match")

// parseToolCallsGemma4 extracts the first `<|tool_call>call:NAME{...}` in resp
// and returns it with the dict body transcoded to JSON arguments.
func parseToolCallsGemma4(resp string) (openai.ChatCompletionMessageFunctionToolCallFunction, error) {
	tc := openai.ChatCompletionMessageFunctionToolCallFunction{}

	start := strings.Index(resp, gemma4ToolCallOpen)
	if start < 0 {
		return tc, errGemma4
	}
	i := start + len(gemma4ToolCallOpen)

	brace := strings.IndexByte(resp[i:], '{')
	if brace < 0 {
		return tc, errGemma4
	}
	name := strings.TrimSpace(resp[i : i+brace])
	if name == "" {
		return tc, errGemma4
	}

	args, _, err := parseGemma4Dict(resp, i+brace)
	if err != nil {
		return tc, err
	}

	tc.Name = name
	tc.Arguments = args
	return tc, nil
}

func skipGemma4Space(s string, i int) int {
	for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r') {
		i++
	}
	return i
}

// parseGemma4Value parses one value at s[i] and returns its JSON encoding and
// the index just past it.
func parseGemma4Value(s string, i int) (string, int, error) {
	if i >= len(s) {
		return "", i, errGemma4
	}
	switch {
	case strings.HasPrefix(s[i:], gemma4StringMarker):
		return parseGemma4String(s, i)
	case s[i] == '{':
		return parseGemma4Dict(s, i)
	case s[i] == '[':
		return parseGemma4Array(s, i)
	default:
		return parseGemma4Scalar(s, i)
	}
}

func parseGemma4String(s string, i int) (string, int, error) {
	i += len(gemma4StringMarker)
	end := strings.Index(s[i:], gemma4StringMarker)
	if end < 0 {
		return "", i, errGemma4
	}
	quoted, err := sonic.MarshalString(s[i : i+end])
	if err != nil {
		return "", i, err
	}
	return quoted, i + end + len(gemma4StringMarker), nil
}

func parseGemma4Dict(s string, i int) (string, int, error) {
	return parseGemma4Seq(s, i, '{', '}', parseGemma4Member)
}

func parseGemma4Array(s string, i int) (string, int, error) {
	return parseGemma4Seq(s, i, '[', ']', parseGemma4Value)
}

// parseGemma4Member parses one `key:value` pair and returns it as `"key":value`.
// The key grammar is `[^:}]+`: it must be non-empty and cannot span the dict's
// closing '}', so a member missing its value fails instead of swallowing later
// text as the key.
func parseGemma4Member(s string, i int) (string, int, error) {
	colon := strings.IndexByte(s[i:], ':')
	if colon < 0 || strings.IndexByte(s[i:i+colon], '}') >= 0 {
		return "", i, errGemma4
	}
	key := strings.TrimSpace(s[i : i+colon])
	if key == "" {
		return "", i, errGemma4
	}
	val, next, err := parseGemma4Value(s, skipGemma4Space(s, i+colon+1))
	if err != nil {
		return "", i, err
	}
	quotedKey, err := sonic.MarshalString(key)
	if err != nil {
		return "", next, err
	}
	return quotedKey + ":" + val, next, nil
}

// parseGemma4Seq parses a comma-separated `open ... close` sequence, encoding
// each element with elem. s[i] must be open (guaranteed by the caller).
func parseGemma4Seq(s string, i int, open, close byte,
	elem func(s string, i int) (string, int, error)) (string, int, error) {
	i++ // past open

	var b strings.Builder
	b.WriteByte(open)

	i = skipGemma4Space(s, i)
	if i < len(s) && s[i] == close {
		b.WriteByte(close)
		return b.String(), i + 1, nil
	}

	for {
		enc, next, err := elem(s, skipGemma4Space(s, i))
		if err != nil {
			return "", next, err
		}
		b.WriteString(enc)
		i = skipGemma4Space(s, next)

		if i >= len(s) {
			return "", i, errGemma4
		}
		switch s[i] {
		case ',':
			i++
			b.WriteByte(',')
		case close:
			b.WriteByte(close)
			return b.String(), i + 1, nil
		default:
			return "", i, errGemma4
		}
	}
}

// parseGemma4Scalar reads a number/bool/null up to the next ',', '}' or ']'.
func parseGemma4Scalar(s string, i int) (string, int, error) {
	start := i
	for i < len(s) && s[i] != ',' && s[i] != '}' && s[i] != ']' {
		i++
	}
	tok := strings.TrimSpace(s[start:i])
	var v any
	if tok == "" || sonic.UnmarshalString(tok, &v) != nil {
		return "", start, errGemma4
	}
	return tok, i, nil
}
