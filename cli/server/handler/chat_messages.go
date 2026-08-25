// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"fmt"
	"log/slog"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/server/utils"
)

// On failure, writes a response and returns ok=false; the caller must return.
func buildLLMMessages(c *gin.Context, param ChatCompletionRequest) (messages []geniex_sdk.LlmChatMessage, ok bool) {
	messages = make([]geniex_sdk.LlmChatMessage, 0, len(param.Messages))
	for _, msg := range param.Messages {
		if toolCalls := msg.GetToolCalls(); len(toolCalls) > 0 {
			for _, tc := range toolCalls {
				messages = append(messages, geniex_sdk.LlmChatMessage{
					Role: geniex_sdk.LlmRole(*msg.GetRole()),
					Content: fmt.Sprintf(`<tool_call>{"name":"%s","arguments":"%s"}</tool_call>`,
						tc.GetFunction().Name, tc.GetFunction().Arguments),
				})
			}
			continue
		}

		if toolResp := msg.GetToolCallID(); toolResp != nil {
			messages = append(messages, geniex_sdk.LlmChatMessage{
				Role:    geniex_sdk.LlmRole(*msg.GetRole()),
				Content: *msg.GetContent().AsAny().(*string),
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
		if toolCalls := msg.GetToolCalls(); len(toolCalls) > 0 {
			contents := make([]geniex_sdk.VlmContent, 0, len(toolCalls))
			for _, tc := range toolCalls {
				contents = append(contents, geniex_sdk.VlmContent{
					Type: geniex_sdk.VlmContentTypeText,
					Text: fmt.Sprintf(`<tool_call>{"name":"%s","arguments":"%s"}</tool_call>`,
						tc.GetFunction().Name, tc.GetFunction().Arguments),
				})
			}
			messages = append(messages, geniex_sdk.VlmChatMessage{
				Role:     geniex_sdk.VlmRole(*msg.GetRole()),
				Contents: contents,
			})
			continue
		}

		if toolResp := msg.GetToolCallID(); toolResp != nil {
			messages = append(messages, geniex_sdk.VlmChatMessage{
				Role: geniex_sdk.VlmRole(*msg.GetRole()),
				Contents: []geniex_sdk.VlmContent{{
					Type: geniex_sdk.VlmContentTypeText,
					Text: *msg.GetContent().AsAny().(*string),
				}},
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
