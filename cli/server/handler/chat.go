// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package handler

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"

	"github.com/bytedance/sonic"
	"github.com/bytedance/sonic/ast"
	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared/constant"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/store"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
	"github.com/qcom-it-nexa-ai/geniex/cli/server/service"
	"github.com/qcom-it-nexa-ai/geniex/cli/server/utils"
)

// =============== request types ===============

type ChatCompletionNewParams openai.ChatCompletionNewParams

type ChatCompletionRequest struct {
	ChatCompletionNewParams
	Stream bool `json:"stream"`

	EnableThink bool  `json:"enable_think"`
	NCtx        int32 `json:"nctx"`
	Ngl         int32 `json:"ngl"`

	ImageMaxLength int32 `json:"image_max_length"`

	TopK              int32   `json:"top_k"`
	MinP              float32 `json:"min_p"`
	RepetitionPenalty float32 `json:"repetition_penalty"`
	GrammarPath       string  `json:"grammar_path"`
	GrammarString     string  `json:"grammar_string"`
	EnableJson        bool    `json:"enable_json"`
}

func defaultChatCompletionRequest() ChatCompletionRequest {
	return ChatCompletionRequest{
		ChatCompletionNewParams: ChatCompletionNewParams{
			MaxCompletionTokens: param.NewOpt[int64](2048),
		},
		Stream: false,

		EnableThink:       true,
		NCtx:              0, // llama_cpp only; 0 = not set, resolved to 4096 for llama_cpp
		Ngl:               0, // llama_cpp only; 0 = not set, resolved to 999 for llama_cpp
		ImageMaxLength:    512,
		TopK:              0,
		MinP:              0.0,
		RepetitionPenalty: 1.0,
		GrammarPath:       "",
		GrammarString:     "",
		EnableJson:        false,
	}
}

func isWarmupRequest(param ChatCompletionRequest) bool {
	if len(param.Messages) == 0 {
		return true
	}
	if len(param.Messages) != 1 {
		return false
	}
	r := param.Messages[0].GetRole()
	return r != nil && *r == "system"
}

// =============== handlers ===============

func ChatCompletions(c *gin.Context) {
	param := defaultChatCompletionRequest()
	if err := c.ShouldBindJSON(&param); err != nil {
		slog.Error("Failed to bind JSON", "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	slog.Info("ChatCompletions", "param", param)
	s := store.Get()
	name, _ := model_hub.NormalizeModelName(param.Model)
	manifest, err := s.GetManifest(name)
	if err != nil {
		slog.Error("Failed to get model manifest", "model", param.Model, "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	// Automatically adjust NCtx if MaxCompletionTokens is larger (llama_cpp only — QAIRT
	// does not use NCtx and the 0-default must not be overwritten for non-llama_cpp plugins).
	if manifest.PluginId == geniex_sdk.PluginLlamaCpp && param.NCtx < int32(param.MaxCompletionTokens.Value) {
		slog.Debug("Adjust NCtx to MaxCompletionTokens", "from", param.NCtx, "to", param.MaxCompletionTokens.Value)
		param.NCtx = int32(param.MaxCompletionTokens.Value)
	}

	switch manifest.ModelType {
	case types.ModelTypeLLM:
		chatCompletionsLLM(c, param, manifest.PluginId)
	case types.ModelTypeVLM:
		chatCompletionsVLM(c, param, manifest.PluginId)
	default:
		slog.Error("Model type not support", "model_type", manifest.ModelType)
		c.JSON(http.StatusBadRequest, map[string]any{"error": "model type not support"})
		return
	}
}

func chatCompletionsLLM(c *gin.Context, param ChatCompletionRequest, pluginId string) {
	messages := make([]geniex_sdk.LlmChatMessage, 0, len(param.Messages))
	for _, msg := range param.Messages {
		if toolCalls := msg.GetToolCalls(); len(toolCalls) > 0 {
			for _, tc := range toolCalls {
				messages = append(messages, geniex_sdk.LlmChatMessage{
					Role: geniex_sdk.LLMRole(*msg.GetRole()),
					Content: fmt.Sprintf(`<tool_call>{"name":"%s","arguments":"%s"}</tool_call>`,
						tc.GetFunction().Name, tc.GetFunction().Arguments),
				})
			}
			continue
		}

		if toolResp := msg.GetToolCallID(); toolResp != nil {
			messages = append(messages, geniex_sdk.LlmChatMessage{
				Role:    geniex_sdk.LLMRole(*msg.GetRole()),
				Content: *msg.GetContent().AsAny().(*string),
			})
			continue
		}

		switch content := msg.GetContent().AsAny().(type) {
		case *string:
			messages = append(messages, geniex_sdk.LlmChatMessage{
				Role:    geniex_sdk.LLMRole(*msg.GetRole()),
				Content: *content,
			})

		case *[]openai.ChatCompletionContentPartTextParam:
			for _, ct := range *content {
				messages = append(messages, geniex_sdk.LlmChatMessage{
					Role:    geniex_sdk.LLMRole(*msg.GetRole()),
					Content: ct.Text,
				})
			}
		case *[]openai.ChatCompletionContentPartUnionParam:
			for _, ct := range *content {
				switch *ct.GetType() {
				case "text":
					messages = append(messages, geniex_sdk.LlmChatMessage{
						Role:    geniex_sdk.LLMRole(*msg.GetRole()),
						Content: *ct.GetText(),
					})
				default:
					slog.Error("Not support content part type", "type", *ct.GetType())
					c.JSON(http.StatusBadRequest, map[string]any{"error": "not support content part type"})
					return
				}
			}
		case *[]openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion:
			for _, ct := range *content {
				switch *ct.GetType() {
				case "text":
					messages = append(messages, geniex_sdk.LlmChatMessage{
						Role:    geniex_sdk.LLMRole(*msg.GetRole()),
						Content: *ct.GetText(),
					})
				default:
					slog.Error("Not support content part type", "type", *ct.GetType())
					c.JSON(http.StatusBadRequest, map[string]any{"error": "not support content part type"})
					return
				}
			}

		default:
			slog.Error("Unknown content type in message", "content_type", fmt.Sprintf("%T", content))
			c.JSON(http.StatusBadRequest, map[string]any{"error": "unknown content type"})
			return
		}
	}

	// Prepare tools if provided
	parseTool, tools, err := parseTools(param)
	if err != nil {
		slog.Error("Failed to parse tools", "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	samplerConfig := parseSamplerConfig(param)

	p, err := service.KeepAliveGet[geniex_sdk.LLM](
		string(param.Model),
		types.ModelParam{NCtx: param.NCtx, NGpuLayers: param.Ngl},
		c.GetHeader("GenieX-KeepCache") != "true",
	)
	if writeKeepAliveError(c, err, pluginId) {
		return
	}
	if isWarmupRequest(param) {
		c.JSON(http.StatusOK, nil)
		return
	}

	formatted, err := p.ApplyChatTemplate(geniex_sdk.LlmApplyChatTemplateInput{
		Messages:            messages,
		Tools:               tools,
		EnableThink:         param.EnableThink,
		AddGenerationPrompt: true,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
		return
	}

	if param.Stream {
		stopGen := false
		dataCh := make(chan string)

		var (
			res   geniex_sdk.LlmGenerateOutput
			err   error
			resWg sync.WaitGroup
		)

		resWg.Add(1)
		go func() {
			defer resWg.Done()
			res, err = p.Generate(geniex_sdk.LlmGenerateInput{
				PromptUTF8: formatted.FormattedText,
				OnToken: func(token string) bool {
					if stopGen {
						return false
					}
					dataCh <- token
					return true
				},
				Config: &geniex_sdk.GenerationConfig{
					MaxTokens:     int32(param.MaxCompletionTokens.Value),
					SamplerConfig: samplerConfig,
				},
			})
			close(dataCh)
		}()

		wait := func() error { resWg.Wait(); return err }
		usage := func() openai.CompletionUsage { return profile2Usage(res.ProfileData) }
		includeUsage := param.StreamOptions.IncludeUsage.Value
		if !parseTool {
			streamPlainText(c, dataCh, wait, includeUsage, usage)
		} else {
			streamToolCall(c, dataCh, wait, includeUsage, usage)
		}

		stopGen = true
		for range dataCh {
		}

	} else {
		genOut, err := p.Generate(geniex_sdk.LlmGenerateInput{
			PromptUTF8: formatted.FormattedText,
			Config: &geniex_sdk.GenerationConfig{
				MaxTokens:     int32(param.MaxCompletionTokens.Value),
				SamplerConfig: samplerConfig,
			},
		})
		if errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) {
			writeContextLengthExceeded(c, genOut.FullText, genOut.ProfileData)
			return
		}
		if err != nil {
			c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return
		}
		writeBlockingResponse(c, genOut.FullText, genOut.ProfileData, parseTool)
	}
}

func chatCompletionsVLM(c *gin.Context, param ChatCompletionRequest, pluginId string) {
	messages := make([]geniex_sdk.VlmChatMessage, 0, len(param.Messages))
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
						return
					}
					defer os.Remove(file)
					contents = append(contents, geniex_sdk.VlmContent{
						Type: geniex_sdk.VlmContentTypeImage,
						Text: file,
					})
				case "input_audio":
					file, err := utils.SaveURIToTempFile(ct.GetInputAudio().Data)
					slog.Debug("Saved audio file", "file", file)
					if err != nil {
						c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error()})
						return
					}
					defer os.Remove(file)
					contents = append(contents, geniex_sdk.VlmContent{
						Type: geniex_sdk.VlmContentTypeAudio,
						Text: file,
					})
				default:
					slog.Error("Not support content part type", "type", *ct.GetType())
					c.JSON(http.StatusBadRequest, map[string]any{"error": "not support content part type"})
					return
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
					return
				}
			}

			messages = append(messages, geniex_sdk.VlmChatMessage{
				Role:     geniex_sdk.VlmRole(*msg.GetRole()),
				Contents: contents,
			})

		default:
			slog.Error("Unknown content type in message")
			c.JSON(http.StatusBadRequest, map[string]any{"error": "unknown content type"})
			return
		}
	}

	// Prepare tools if provided
	parseTool, tools, err := parseTools(param)
	if err != nil {
		slog.Error("Failed to parse tools", "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	samplerConfig := parseSamplerConfig(param)

	p, err := service.KeepAliveGet[geniex_sdk.VLM](
		string(param.Model),
		types.ModelParam{NCtx: param.NCtx, NGpuLayers: param.Ngl},
		c.GetHeader("GenieX-KeepCache") != "true",
	)
	if writeKeepAliveError(c, err, pluginId) {
		return
	}
	if isWarmupRequest(param) {
		c.JSON(http.StatusOK, nil)
		return
	}

	// Format prompt using VLM chat template
	formatted, err := p.ApplyChatTemplate(geniex_sdk.VlmApplyChatTemplateInput{
		Messages:    messages,
		Tools:       tools,
		EnableThink: param.EnableThink,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
		return
	}
	images := make([]string, 0)
	audios := make([]string, 0)
	for _, content := range messages[len(messages)-1].Contents {
		switch content.Type {
		case geniex_sdk.VlmContentTypeImage:
			images = append(images, content.Text)
		case geniex_sdk.VlmContentTypeAudio:
			audios = append(audios, content.Text)
		}
	}

	if param.Stream {
		stopGen := false
		dataCh := make(chan string)

		var (
			res   *geniex_sdk.VlmGenerateOutput
			err   error
			resWg sync.WaitGroup
		)

		resWg.Add(1)
		go func() {
			defer resWg.Done()
			res, err = p.Generate(geniex_sdk.VlmGenerateInput{
				PromptUTF8: formatted.FormattedText,
				OnToken: func(token string) bool {
					if stopGen {
						return false
					}
					dataCh <- token
					return true
				},
				Config: &geniex_sdk.GenerationConfig{
					MaxTokens:      int32(param.MaxCompletionTokens.Value),
					SamplerConfig:  samplerConfig,
					ImagePaths:     images,
					AudioPaths:     audios,
					ImageMaxLength: param.ImageMaxLength,
				},
			})

			close(dataCh)
		}()

		wait := func() error { resWg.Wait(); return err }
		usage := func() openai.CompletionUsage { return profile2Usage(res.ProfileData) }
		includeUsage := param.StreamOptions.IncludeUsage.Value
		if !parseTool {
			streamPlainText(c, dataCh, wait, includeUsage, usage)
		} else {
			streamToolCall(c, dataCh, wait, includeUsage, usage)
		}

		stopGen = true
		for range dataCh {
		}

	} else {
		genOut, err := p.Generate(geniex_sdk.VlmGenerateInput{
			PromptUTF8: formatted.FormattedText,
			Config: &geniex_sdk.GenerationConfig{
				MaxTokens:      int32(param.MaxCompletionTokens.Value),
				SamplerConfig:  samplerConfig,
				ImagePaths:     images,
				AudioPaths:     audios,
				ImageMaxLength: param.ImageMaxLength,
			},
		})
		if errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) && genOut != nil {
			writeContextLengthExceeded(c, genOut.FullText, genOut.ProfileData)
			return
		}
		if err != nil {
			c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return
		}
		writeBlockingResponse(c, genOut.FullText, genOut.ProfileData, parseTool)
	}
}

// =============== request-side helpers ===============

func parseSamplerConfig(param ChatCompletionRequest) *geniex_sdk.SamplerConfig {
	return &geniex_sdk.SamplerConfig{
		Temperature:       float32(param.Temperature.Value),
		TopP:              float32(param.TopP.Value),
		TopK:              param.TopK,
		MinP:              param.MinP,
		RepetitionPenalty: param.RepetitionPenalty,
		PresencePenalty:   float32(param.PresencePenalty.Value),
		FrequencyPenalty:  float32(param.FrequencyPenalty.Value),
		Seed:              int32(param.Seed.Value),
		EnableJson:        param.EnableJson,
	}
}

func parseTools(param ChatCompletionRequest) (bool, string, error) {
	if len(param.Tools) == 0 {
		return false, "", nil
	}
	tools, err := sonic.MarshalString(param.Tools)
	return true, tools, err
}

// =============== response-side helpers ===============

// writeKeepAliveError maps a KeepAliveGet error to its HTTP response;
// returns true when handled (caller should return).
func writeKeepAliveError(c *gin.Context, err error, pluginId string) bool {
	if err == nil {
		return false
	}
	switch {
	case errors.Is(err, os.ErrNotExist):
		c.JSON(http.StatusNotFound, map[string]any{"error": "model not found"})
	case errors.Is(err, geniex_sdk.ErrCommonParamNotSupported):
		c.JSON(http.StatusBadRequest, map[string]any{
			"error": fmt.Sprintf("a parameter in the request is not supported by the %s plugin", pluginId),
			"code":  geniex_sdk.SDKErrorCode(err),
		})
	default:
		c.JSON(http.StatusInternalServerError, map[string]any{
			"error": err.Error(),
			"code":  geniex_sdk.SDKErrorCode(err),
		})
	}
	return true
}

// writeContextLengthExceeded mirrors OpenAI's HTTP 400 body but also surfaces
// the partial generation under `choices` so clients can still recover the
// truncated output. The SDK populates fullText / profile even on this error.
func writeContextLengthExceeded(c *gin.Context, fullText string, profile geniex_sdk.ProfileData) {
	choice := openai.ChatCompletionChoice{}
	choice.FinishReason = "length"
	choice.Message.Role = constant.Assistant(openai.MessageRoleAssistant)
	choice.Message.Content = fullText

	c.JSON(http.StatusBadRequest, map[string]any{
		"error": map[string]any{
			"message": "model context window exceeded; output truncated",
			"type":    "invalid_request_error",
			"code":    "context_length_exceeded",
		},
		"choices": []openai.ChatCompletionChoice{choice},
		"usage":   profile2Usage(profile),
	})
}

// writeBlockingResponse emits a non-streaming completion: tool-call response
// when parseTool matches, content response otherwise (or on parse failure).
func writeBlockingResponse(c *gin.Context, fullText string, profile geniex_sdk.ProfileData, parseTool bool) {
	if parseTool {
		toolCall, err := parseToolCalls(fullText)
		if err == nil {
			choice := openai.ChatCompletionChoice{}
			choice.FinishReason = "tool_calls"
			choice.Message.Role = constant.Assistant(openai.MessageRoleAssistant)
			choice.Message.ToolCalls = []openai.ChatCompletionMessageToolCallUnion{{Function: toolCall}}
			c.JSON(http.StatusOK, openai.ChatCompletion{
				ID:      fmt.Sprintf("call_%d", rand.Uint32()),
				Choices: []openai.ChatCompletionChoice{choice},
				Usage:   profile2Usage(profile),
			})
			return
		}
		slog.Warn("Tool call parse error, fallback to text", "error", err)
	}

	choice := openai.ChatCompletionChoice{}
	choice.FinishReason = mapFinishReason(profile.StopReason)
	choice.Message.Role = constant.Assistant(openai.MessageRoleAssistant)
	choice.Message.Content = fullText
	c.JSON(http.StatusOK, openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{choice},
		Usage:   profile2Usage(profile),
	})
}

// streamUsage is read once dataCh closes; it lets callers hide whether their
// generate result is a value or pointer.
type streamUsage func() openai.CompletionUsage

// streamPlainText drains dataCh as content chunks, then emits optional usage
// and [DONE].
func streamPlainText(c *gin.Context, dataCh <-chan string, wait func() error, includeUsage bool, usage streamUsage) {
	c.Stream(func(w io.Writer) bool {
		r, ok := <-dataCh
		if ok {
			chunk := openai.ChatCompletionChunk{}
			chunk.Choices = append(chunk.Choices, openai.ChatCompletionChunkChoice{
				Delta: openai.ChatCompletionChunkChoiceDelta{
					Content: r,
					Role:    string(openai.MessageRoleAssistant),
				},
			})
			c.SSEvent("", chunk)
			return true
		}
		if err := wait(); err != nil {
			c.SSEvent("", map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return false
		}
		if includeUsage {
			c.SSEvent("", openai.ChatCompletionChunk{
				Choices: []openai.ChatCompletionChunkChoice{},
				Usage:   usage(),
			})
		}
		c.SSEvent("", "[DONE]")
		return false
	})
}

// streamToolCall buffers the stream and emits a tool-call chunk once dataCh
// closes; falls back to a content chunk when parsing fails.
func streamToolCall(c *gin.Context, dataCh <-chan string, wait func() error, includeUsage bool, usage streamUsage) {
	buffer := strings.Builder{}
	c.Stream(func(w io.Writer) bool {
		r, ok := <-dataCh
		if ok {
			buffer.WriteString(r)
			return true
		}
		if err := wait(); err != nil {
			slog.Error("Generation error", "error", err)
			c.SSEvent("", map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return false
		}
		toolCall, err := parseToolCalls(buffer.String())
		if err != nil {
			slog.Warn("Tool call parse error, fallback to text", "error", err)
			chunk := openai.ChatCompletionChunk{}
			chunk.Choices = append(chunk.Choices, openai.ChatCompletionChunkChoice{
				Delta: openai.ChatCompletionChunkChoiceDelta{
					Content: buffer.String(),
					Role:    string(openai.MessageRoleAssistant),
				},
			})
			c.SSEvent("", chunk)
			return false
		}
		c.SSEvent("", openai.ChatCompletionChunk{
			Choices: []openai.ChatCompletionChunkChoice{{
				Delta: openai.ChatCompletionChunkChoiceDelta{
					ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{{
						ID: fmt.Sprintf("call_%d", rand.Uint32()),
						Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
							Name:      toolCall.Name,
							Arguments: toolCall.Arguments,
						},
					}},
				},
			}},
		})
		if includeUsage {
			c.SSEvent("", openai.ChatCompletionChunk{
				Choices: []openai.ChatCompletionChunkChoice{},
				Usage:   usage(),
			})
		}
		c.SSEvent("", "[DONE]")
		return false
	})
}

// =============== shared mappers ===============

func profile2Usage(p geniex_sdk.ProfileData) openai.CompletionUsage {
	return openai.CompletionUsage{
		CompletionTokens: p.GeneratedTokens,
		PromptTokens:     p.PromptTokens,
		TotalTokens:      p.TotalTokens(),
	}
}

// mapFinishReason translates the SDK's stop_reason values into the OpenAI
// finish_reason vocabulary.
func mapFinishReason(stopReason string) string {
	switch stopReason {
	case "length":
		return "length"
	case "user":
		return "stop"
	case "eos", "stop_sequence", "":
		return "stop"
	default:
		return "stop"
	}
}

// =============== tool-call parsing ===============

var toolCallRegex = regexp.MustCompile(`<tool_call>([\s\S]+)<\/tool_call>` + "|" + "```json([\\s\\S]+)```")

func parseToolCalls(resp string) (openai.ChatCompletionMessageFunctionToolCallFunction, error) {
	match := toolCallRegex.FindStringSubmatch(resp)
	if len(match) <= 1 {
		return openai.ChatCompletionMessageFunctionToolCallFunction{}, errors.New("tool call not match")
	}
	matched := match[1]
	if matched == "" && len(match) > 2 {
		matched = match[2]
	}

	slog.Debug("Tool call matched", "matched", matched)

	name, err := sonic.GetFromString(matched, "name")
	toolCall := openai.ChatCompletionMessageFunctionToolCallFunction{}
	if err != nil {
		return openai.ChatCompletionMessageFunctionToolCallFunction{}, err
	}
	toolCall.Name, err = name.String()
	if err != nil {
		return openai.ChatCompletionMessageFunctionToolCallFunction{}, err
	}

	arguments, err := sonic.GetFromString(matched, "arguments")
	if err != nil {
		return openai.ChatCompletionMessageFunctionToolCallFunction{}, err
	}
	switch arguments.TypeSafe() {
	case ast.V_OBJECT:
		toolCall.Arguments, _ = arguments.Raw()
	case ast.V_STRING:
		toolCall.Arguments, _ = arguments.String()
	default:
		return openai.ChatCompletionMessageFunctionToolCallFunction{}, errors.New("unknown arguments type")
	}

	slog.Debug("Parsed tool call", "tool_call", toolCall)

	return toolCall, nil
}
