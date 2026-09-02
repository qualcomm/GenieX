// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/server/utils"
)

// Flattens a message's content union to plain text. Returns "" for a nil
// content — an assistant turn that only issues tool calls carries none.
func contentText(msg openai.ChatCompletionMessageParamUnion) string {
	switch content := msg.GetContent().AsAny().(type) {
	case *string:
		return *content
	case *[]openai.ChatCompletionContentPartTextParam:
		var b strings.Builder
		for _, ct := range *content {
			b.WriteString(ct.Text)
		}
		return b.String()
	case *[]openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion:
		var b strings.Builder
		for _, ct := range *content {
			if *ct.GetType() == "text" {
				b.WriteString(*ct.GetText())
			}
		}
		return b.String()
	default:
		return ""
	}
}

// contentText in the VLM shape: one text part, or none for an assistant turn
// that carries only tool calls.
func textContents(msg openai.ChatCompletionMessageParamUnion) []geniex_sdk.VlmContent {
	text := contentText(msg)
	if text == "" {
		return nil
	}
	return []geniex_sdk.VlmContent{{Type: geniex_sdk.VlmContentTypeText, Text: text}}
}

// Tool calls must reach the SDK structured, not flattened into assistant text:
// chat templates render a tool response only from the tool_calls of the
// assistant turn before it, matched by call ID. On failure, writes a response
// and returns ok=false; the caller must return.
func buildToolCalls(c *gin.Context, msg openai.ChatCompletionMessageParamUnion) (calls []geniex_sdk.ToolCall, ok bool) {
	toolCalls := msg.GetToolCalls()
	calls = make([]geniex_sdk.ToolCall, 0, len(toolCalls))
	for _, tc := range toolCalls {
		fn := tc.GetFunction()
		if fn == nil {
			kind := ""
			if t := tc.GetType(); t != nil {
				kind = *t
			}
			slog.Error("Not support tool call type", "type", kind)
			c.JSON(http.StatusBadRequest, map[string]any{"error": "not support tool call type"})
			return nil, false
		}
		call := geniex_sdk.ToolCall{Name: fn.Name, Arguments: fn.Arguments}
		if id := tc.GetID(); id != nil {
			call.ID = *id
		}
		calls = append(calls, call)
	}
	return calls, true
}

// On failure, writes a response and returns ok=false; the caller must return.
func buildLLMMessages(c *gin.Context, param ChatCompletionRequest) (messages []geniex_sdk.LlmChatMessage, ok bool) {
	messages = make([]geniex_sdk.LlmChatMessage, 0, len(param.Messages))
	for _, msg := range param.Messages {
		if len(msg.GetToolCalls()) > 0 {
			calls, valid := buildToolCalls(c, msg)
			if !valid {
				return nil, false
			}
			messages = append(messages, geniex_sdk.LlmChatMessage{
				Role:      geniex_sdk.LlmRole(*msg.GetRole()),
				Content:   contentText(msg),
				ToolCalls: calls,
			})
			continue
		}

		if toolCallID := msg.GetToolCallID(); toolCallID != nil {
			messages = append(messages, geniex_sdk.LlmChatMessage{
				Role:       geniex_sdk.LlmRole(*msg.GetRole()),
				Content:    contentText(msg),
				ToolCallID: *toolCallID,
			})
			continue
		}

		switch content := msg.GetContent().AsAny().(type) {
		case *string:
			messages = append(messages, geniex_sdk.LlmChatMessage{
				Role:    geniex_sdk.LlmRole(*msg.GetRole()),
				Content: *content,
			})

		case *[]openai.ChatCompletionContentPartTextParam:
			for _, ct := range *content {
				messages = append(messages, geniex_sdk.LlmChatMessage{
					Role:    geniex_sdk.LlmRole(*msg.GetRole()),
					Content: ct.Text,
				})
			}
		case *[]openai.ChatCompletionContentPartUnionParam:
			for _, ct := range *content {
				switch *ct.GetType() {
				case "text":
					messages = append(messages, geniex_sdk.LlmChatMessage{
						Role:    geniex_sdk.LlmRole(*msg.GetRole()),
						Content: *ct.GetText(),
					})
				default:
					slog.Error("Not support content part type", "type", *ct.GetType())
					c.JSON(http.StatusBadRequest, map[string]any{"error": "not support content part type"})
					return nil, false
				}
			}
		case *[]openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion:
			for _, ct := range *content {
				switch *ct.GetType() {
				case "text":
					messages = append(messages, geniex_sdk.LlmChatMessage{
						Role:    geniex_sdk.LlmRole(*msg.GetRole()),
						Content: *ct.GetText(),
					})
				default:
					slog.Error("Not support content part type", "type", *ct.GetType())
					c.JSON(http.StatusBadRequest, map[string]any{"error": "not support content part type"})
					return nil, false
				}
			}

		default:
			slog.Error("Unknown content type in message", "content_type", fmt.Sprintf("%T", content))
			c.JSON(http.StatusBadRequest, map[string]any{"error": "unknown content type"})
			return nil, false
		}
	}
	return messages, true
}

// Image / audio data is spilled to tempFiles, which the caller must remove
// after generation (returned on the error path too, for partial writes).
func buildVLMMessages(c *gin.Context, param ChatCompletionRequest) (messages []geniex_sdk.VlmChatMessage, tempFiles []string, ok bool) {
	messages = make([]geniex_sdk.VlmChatMessage, 0, len(param.Messages))
	for _, msg := range param.Messages {
		if len(msg.GetToolCalls()) > 0 {
			calls, valid := buildToolCalls(c, msg)
			if !valid {
				return nil, tempFiles, false
			}
			messages = append(messages, geniex_sdk.VlmChatMessage{
				Role:      geniex_sdk.VlmRole(*msg.GetRole()),
				Contents:  textContents(msg),
				ToolCalls: calls,
			})
			continue
		}

		if toolCallID := msg.GetToolCallID(); toolCallID != nil {
			messages = append(messages, geniex_sdk.VlmChatMessage{
				Role:       geniex_sdk.VlmRole(*msg.GetRole()),
				Contents:   textContents(msg),
				ToolCallID: *toolCallID,
			})
			continue
		}

		switch content := msg.GetContent().AsAny().(type) {
		case *string:
			messages = append(messages, geniex_sdk.VlmChatMessage{
				Role: geniex_sdk.VlmRole(*msg.GetRole()),
				Contents: []geniex_sdk.VlmContent{
					{Type: geniex_sdk.VlmContentTypeText, Text: *msg.GetContent().AsAny().(*string)},
				},
			})

		case *[]openai.ChatCompletionContentPartTextParam:
			contents := make([]geniex_sdk.VlmContent, 0, len(*content))
			for _, ct := range *content {
				contents = append(contents, geniex_sdk.VlmContent{
					Type: geniex_sdk.VlmContentTypeText,
					Text: ct.Text,
				})
			}
			messages = append(messages, geniex_sdk.VlmChatMessage{
				Role:     geniex_sdk.VlmRole(*msg.GetRole()),
				Contents: contents,
			})

		case *[]openai.ChatCompletionContentPartUnionParam:
			contents := make([]geniex_sdk.VlmContent, 0, len(*content))
			for _, ct := range *content {
				switch *ct.GetType() {
				case "text":
					contents = append(contents, geniex_sdk.VlmContent{
						Type: geniex_sdk.VlmContentTypeText,
						Text: *ct.GetText(),
					})
				case "image_url":
					file, err := utils.SaveURIToTempFile(ct.GetImageURL().URL)
					slog.Debug("Saved image file", "file", file)
					if err != nil {
						c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error()})
						return nil, tempFiles, false
					}
					tempFiles = append(tempFiles, file)
					contents = append(contents, geniex_sdk.VlmContent{
						Type: geniex_sdk.VlmContentTypeImage,
						Text: file,
					})
				case "input_audio":
					file, err := utils.SaveURIToTempFile(ct.GetInputAudio().Data)
					slog.Debug("Saved audio file", "file", file)
					if err != nil {
						c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error()})
						return nil, tempFiles, false
					}
					tempFiles = append(tempFiles, file)
					contents = append(contents, geniex_sdk.VlmContent{
						Type: geniex_sdk.VlmContentTypeAudio,
						Text: file,
					})
				default:
					slog.Error("Not support content part type", "type", *ct.GetType())
					c.JSON(http.StatusBadRequest, map[string]any{"error": "not support content part type"})
					return nil, tempFiles, false
				}
			}
			messages = append(messages, geniex_sdk.VlmChatMessage{
				Role:     geniex_sdk.VlmRole(*msg.GetRole()),
				Contents: contents,
			})

		case *[]openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion:
			contents := make([]geniex_sdk.VlmContent, 0, len(*content))
			for _, ct := range *content {
				switch *ct.GetType() {
				case "text":
					contents = append(contents, geniex_sdk.VlmContent{
						Type: geniex_sdk.VlmContentTypeText,
						Text: *ct.GetText(),
					})
				default:
					slog.Error("Not support content part type", "type", *ct.GetType())
					c.JSON(http.StatusBadRequest, map[string]any{"error": "not support content part type"})
					return nil, tempFiles, false
				}
			}

			messages = append(messages, geniex_sdk.VlmChatMessage{
				Role:     geniex_sdk.VlmRole(*msg.GetRole()),
				Contents: contents,
			})

		default:
			slog.Error("Unknown content type in message")
			c.JSON(http.StatusBadRequest, map[string]any{"error": "unknown content type"})
			return nil, tempFiles, false
		}
	}
	return messages, tempFiles, true
}
