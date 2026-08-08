// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/bytedance/sonic"
	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared/constant"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/config"
	"github.com/qualcomm/GenieX/cli/internal/store"
	"github.com/qualcomm/GenieX/cli/internal/types"
	"github.com/qualcomm/GenieX/cli/server/service"
	"github.com/qualcomm/GenieX/cli/server/utils"
)

// =============== request types ===============

type ChatCompletionNewParams openai.ChatCompletionNewParams

type ChatCompletionRequest struct {
	ChatCompletionNewParams
	Stream bool `json:"stream"`

	EnableThink bool `json:"enable_think"`
	// llama.cpp Web UI compatibility aliases. The UI uses llama-server's
	// repeat_penalty name and puts enable_thinking under chat_template_kwargs.
	RepeatPenalty      *float32 `json:"repeat_penalty"`
	ChatTemplateKwargs *struct {
		EnableThinking *bool `json:"enable_thinking"`
	} `json:"chat_template_kwargs"`
	NCtx    int32  `json:"nctx"`
	Ngl     int32  `json:"ngl"` // 0 = pure CPU, -1 = all layers, N = N layers; defaults to the server --ngl when omitted
	Compute string `json:"compute"`

	ImageMaxLength int32 `json:"image_max_length"`

	TopK              int32   `json:"top_k"`
	MinP              float32 `json:"min_p"`
	RepetitionPenalty float32 `json:"repetition_penalty"`
	GrammarPath       string  `json:"grammar_path"`
	GrammarString     string  `json:"grammar_string"`
	EnableJson        bool    `json:"enable_json"`

	SpecType       string  `json:"spec_type"`
	SpecDraftModel string  `json:"spec_draft_model"`
	SpecNMax       int32   `json:"spec_n_max"`
	SpecNMin       int32   `json:"spec_n_min"`
	SpecPMin       float32 `json:"spec_p_min"`
}

func defaultChatCompletionRequest() ChatCompletionRequest {
	// Prefill llama_cpp knobs with the server-wide defaults (--nctx / --ngl /
	// --compute). ShouldBindJSON only overwrites fields present in the body, so
	// an omitted knob keeps the server default while an explicit value (incl.
	// ngl 0 = pure CPU, -1 = all layers) passes through verbatim.
	cfg := config.Get()
	return ChatCompletionRequest{
		ChatCompletionNewParams: ChatCompletionNewParams{
			MaxCompletionTokens: param.NewOpt[int64](2048),
			Model:               openai.ChatModel(cfg.Model),
		},
		Stream: false,

		EnableThink:       true,
		NCtx:              cfg.NCtx,
		Ngl:               cfg.Ngl,
		Compute:           cfg.Compute,
		ImageMaxLength:    512,
		TopK:              0,
		MinP:              0.0,
		RepetitionPenalty: 1.0,
		GrammarPath:       "",
		GrammarString:     "",
		EnableJson:        false,
	}
}

func normalizeLlamaWebUIRequest(param *ChatCompletionRequest) {
	if param.MaxTokens.Valid() {
		param.MaxCompletionTokens = param.MaxTokens
	}
	if param.RepeatPenalty != nil {
		param.RepetitionPenalty = *param.RepeatPenalty
	}
	if param.ChatTemplateKwargs != nil && param.ChatTemplateKwargs.EnableThinking != nil {
		param.EnableThink = *param.ChatTemplateKwargs.EnableThinking
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
	normalizeLlamaWebUIRequest(&param)
	if param.Model == "" {
		c.JSON(http.StatusBadRequest, map[string]any{
			"error": "model is required; provide it in the request or configure geniex serve --model",
		})
		return
	}

	slog.Info("ChatCompletions", "param", param)
	name, _ := geniex_sdk.SplitNamePrecision(param.Model)
	modelType, err := geniex_sdk.ModelGetType(name)
	if err != nil {
		slog.Error("Failed to get model type", "model", param.Model, "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	paths, err := geniex_sdk.ModelGetPaths(name)
	if err != nil {
		slog.Error("Failed to resolve model paths", "model", param.Model, "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	// Fill unset request knobs from the server-wide defaults and resolve the
	// compute unit. Done before the MaxCompletionTokens floor so a body that
	// omits nctx picks up the server default, not the floor.
	modelParam, err := service.ResolveModelParam(paths.RuntimeID, paths.ModelName, param.NCtx, param.Ngl, param.Compute, store.Get().ResolveChipset(true), service.SpecParam{
		Type:       param.SpecType,
		DraftModel: param.SpecDraftModel,
		NMax:       param.SpecNMax,
		NMin:       param.SpecNMin,
		PMin:       param.SpecPMin,
	})
	if err != nil {
		slog.Error("Failed to resolve model params", "model", param.Model, "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	// Automatically adjust NCtx if MaxCompletionTokens is larger (llama_cpp only — QAIRT
	// does not use NCtx and the 0-default must not be overwritten for non-llama_cpp plugins).
	if paths.RuntimeID == geniex_sdk.RuntimeLlamaCpp && modelParam.NCtx < int32(param.MaxCompletionTokens.Value) {
		slog.Debug("Adjust NCtx to MaxCompletionTokens", "from", modelParam.NCtx, "to", param.MaxCompletionTokens.Value)
		modelParam.NCtx = int32(param.MaxCompletionTokens.Value)
	}

	effectiveType := modelType
	if effectiveType == geniex_sdk.ModelTypeVLM && param.SpecType != "" {
		slog.Warn("spec_type set on VLM-classified model; running LLM path, image/audio content will be ignored",
			"model", param.Model, "spec_type", param.SpecType)
		effectiveType = geniex_sdk.ModelTypeLLM
	}

	switch effectiveType {
	case geniex_sdk.ModelTypeLLM:
		chatCompletionsLLM(c, param, modelParam)
	case geniex_sdk.ModelTypeVLM:
		chatCompletionsVLM(c, param, modelParam)
	default:
		slog.Error("Model type not support", "model_type", modelType)
		c.JSON(http.StatusBadRequest, map[string]any{"error": "model type not support"})
		return
	}
}

func chatCompletionsLLM(c *gin.Context, param ChatCompletionRequest, modelParam types.ModelParam) {
	messages := make([]geniex_sdk.LlmChatMessage, 0, len(param.Messages))
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
					return
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

	p, err := service.RunOnInferenceThreadResult(func() (*geniex_sdk.LLM, error) {
		return service.KeepAliveGet[geniex_sdk.LLM](
			string(param.Model),
			modelParam,
			c.GetHeader("GenieX-KeepCache") != "true",
		)
	})
	if writeKeepAliveError(c, err) {
		return
	}
	if isWarmupRequest(param) {
		c.JSON(http.StatusOK, nil)
		return
	}

	formatted, err := service.RunOnInferenceThreadResult(func() (*geniex_sdk.LlmApplyChatTemplateOutput, error) {
		return p.ApplyChatTemplate(geniex_sdk.LlmApplyChatTemplateInput{
			Messages:            messages,
			Tools:               tools,
			EnableThink:         param.EnableThink,
			AddGenerationPrompt: true,
		})
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
		return
	}

	if param.Stream {
		stopGen := false
		dataCh := make(chan string)

		var (
			res   *geniex_sdk.LlmGenerateOutput
			err   error
			resWg sync.WaitGroup
		)

		resWg.Add(1)
		go func() {
			defer resWg.Done()
			res, err = service.RunOnInferenceThreadResult(func() (*geniex_sdk.LlmGenerateOutput, error) {
				return p.Generate(geniex_sdk.LlmGenerateInput{
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
			})
			close(dataCh)
		}()

		wait := func() error { resWg.Wait(); return err }
		usage := func() openai.CompletionUsage { return profile2Usage(res.ProfileData) }
		timings := func() llamaTimings { return profile2Timings(res.ProfileData) }
		finish := func() string { return mapFinishReason(res.ProfileData.StopReason) }
		includeUsage := param.StreamOptions.IncludeUsage.Value
		if !parseTool {
			streamPlainText(c, dataCh, wait, includeUsage, usage, timings, string(param.Model), finish)
		} else {
			streamToolCall(c, dataCh, wait, includeUsage, usage, timings, string(param.Model), finish)
		}

		stopGen = true
		for range dataCh {
		}

	} else {
		genOut, err := service.RunOnInferenceThreadResult(func() (*geniex_sdk.LlmGenerateOutput, error) {
			return p.Generate(geniex_sdk.LlmGenerateInput{
				PromptUTF8: formatted.FormattedText,
				Config: &geniex_sdk.GenerationConfig{
					MaxTokens:     int32(param.MaxCompletionTokens.Value),
					SamplerConfig: samplerConfig,
				},
			})
		})
		if errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) {
			writeContextLengthExceeded(c, genOut.FullText, genOut.ProfileData)
			return
		}
		if err != nil {
			c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return
		}
		writeBlockingResponse(c, genOut.FullText, genOut.ProfileData, parseTool, string(param.Model))
	}
}

func chatCompletionsVLM(c *gin.Context, param ChatCompletionRequest, modelParam types.ModelParam) {
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

	p, err := service.RunOnInferenceThreadResult(func() (*geniex_sdk.VLM, error) {
		return service.KeepAliveGet[geniex_sdk.VLM](
			string(param.Model),
			modelParam,
			c.GetHeader("GenieX-KeepCache") != "true",
		)
	})
	if writeKeepAliveError(c, err) {
		return
	}
	if isWarmupRequest(param) {
		c.JSON(http.StatusOK, nil)
		return
	}

	// Format prompt using VLM chat template
	formatted, err := service.RunOnInferenceThreadResult(func() (*geniex_sdk.VlmApplyChatTemplateOutput, error) {
		return p.ApplyChatTemplate(geniex_sdk.VlmApplyChatTemplateInput{
			Messages:    messages,
			Tools:       tools,
			EnableThink: param.EnableThink,
		})
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
			res, err = service.RunOnInferenceThreadResult(func() (*geniex_sdk.VlmGenerateOutput, error) {
				return p.Generate(geniex_sdk.VlmGenerateInput{
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
			})

			close(dataCh)
		}()

		wait := func() error { resWg.Wait(); return err }
		usage := func() openai.CompletionUsage { return profile2Usage(res.ProfileData) }
		timings := func() llamaTimings { return profile2Timings(res.ProfileData) }
		finish := func() string { return mapFinishReason(res.ProfileData.StopReason) }
		includeUsage := param.StreamOptions.IncludeUsage.Value
		if !parseTool {
			streamPlainText(c, dataCh, wait, includeUsage, usage, timings, string(param.Model), finish)
		} else {
			streamToolCall(c, dataCh, wait, includeUsage, usage, timings, string(param.Model), finish)
		}

		stopGen = true
		for range dataCh {
		}

	} else {
		genOut, err := service.RunOnInferenceThreadResult(func() (*geniex_sdk.VlmGenerateOutput, error) {
			return p.Generate(geniex_sdk.VlmGenerateInput{
				PromptUTF8: formatted.FormattedText,
				Config: &geniex_sdk.GenerationConfig{
					MaxTokens:      int32(param.MaxCompletionTokens.Value),
					SamplerConfig:  samplerConfig,
					ImagePaths:     images,
					AudioPaths:     audios,
					ImageMaxLength: param.ImageMaxLength,
				},
			})
		})
		if errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) && genOut != nil {
			writeContextLengthExceeded(c, genOut.FullText, genOut.ProfileData)
			return
		}
		if err != nil {
			c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return
		}
		writeBlockingResponse(c, genOut.FullText, genOut.ProfileData, parseTool, string(param.Model))
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
func writeKeepAliveError(c *gin.Context, err error) bool {
	if err == nil {
		return false
	}
	switch {
	case errors.Is(err, os.ErrNotExist):
		c.JSON(http.StatusNotFound, map[string]any{"error": "model not found"})
	case errors.Is(err, geniex_sdk.ErrCommonParamNotSupported):
		c.JSON(http.StatusBadRequest, map[string]any{
			"error": "a parameter in the request is not supported by the runtime",
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
		"timings": profile2Timings(profile),
	})
}

// writeBlockingResponse emits a non-streaming completion: tool-call response
// when parseTool matches, content response otherwise (or on parse failure).
func writeBlockingResponse(c *gin.Context, fullText string, profile geniex_sdk.ProfileData, parseTool bool, model string) {
	if parseTool {
		toolCall, err := utils.ParseToolCalls(fullText)
		if err == nil {
			choice := openai.ChatCompletionChoice{}
			choice.FinishReason = "tool_calls"
			choice.Message.Role = constant.Assistant(openai.MessageRoleAssistant)
			choice.Message.ToolCalls = []openai.ChatCompletionMessageToolCallUnion{{
				ID:       fmt.Sprintf("call_%d", rand.Uint32()),
				Type:     "function",
				Function: toolCall,
			}}
			c.JSON(http.StatusOK, map[string]any{
				"object":  "chat.completion",
				"model":   model,
				"choices": []openai.ChatCompletionChoice{choice},
				"usage":   profile2Usage(profile),
				"timings": profile2Timings(profile),
			})
			return
		}
		slog.Warn("Tool call parse error, fallback to text", "error", err)
	}

	reasoning, content := splitGemma4Response(fullText)
	message := map[string]any{
		"role":    openai.MessageRoleAssistant,
		"content": content,
	}
	if reasoning != "" {
		message["reasoning_content"] = reasoning
	}
	c.JSON(http.StatusOK, map[string]any{
		"object": "chat.completion",
		"model":  model,
		"choices": []map[string]any{{
			"index":         0,
			"finish_reason": mapFinishReason(profile.StopReason),
			"message":       message,
		}},
		"usage":   profile2Usage(profile),
		"timings": profile2Timings(profile),
	})
}

// streamUsage is read once dataCh closes; it lets callers hide whether their
// generate result is a value or pointer.
type streamUsage func() openai.CompletionUsage

// streamFinish is read once dataCh closes; it maps the generate result's
// stop_reason to the OpenAI finish_reason vocabulary for the final chunk.
type streamFinish func() string

// streamTimings exposes the completed SDK profile after generation finishes.
type streamTimings func() llamaTimings

// The openai-go response structs marshal FinishReason as a plain string, so
// every chunk carried `"finish_reason": ""` and no terminal chunk was sent.
// The OpenAI streaming spec requires null on intermediate chunks and
// "stop" / "length" / "tool_calls" on a final chunk with an empty delta —
// without it, clients never detect completion (#1243). These local types
// give the stream spec-compliant serialization.

type streamDelta struct {
	Role             string                                          `json:"role,omitempty"`
	Content          string                                          `json:"content,omitempty"`
	ReasoningContent string                                          `json:"reasoning_content,omitempty"`
	ToolCalls        []openai.ChatCompletionChunkChoiceDeltaToolCall `json:"tool_calls,omitempty"`
}

type streamChoice struct {
	Index        int64       `json:"index"`
	Delta        streamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"` // null until the final chunk
}

type streamChunk struct {
	Object  string                  `json:"object"`
	Model   string                  `json:"model,omitempty"`
	Choices []streamChoice          `json:"choices"`
	Usage   *openai.CompletionUsage `json:"usage,omitempty"`
	Timings *llamaTimings           `json:"timings,omitempty"`
}

const streamChunkObject = "chat.completion.chunk"

func contentChunk(content string) streamChunk {
	return streamChunk{
		Object: streamChunkObject,
		Choices: []streamChoice{{
			Delta: streamDelta{Role: string(openai.MessageRoleAssistant), Content: content},
		}},
	}
}

func reasoningChunk(content string) streamChunk {
	return streamChunk{
		Object: streamChunkObject,
		Choices: []streamChoice{{
			Delta: streamDelta{Role: string(openai.MessageRoleAssistant), ReasoningContent: content},
		}},
	}
}

const (
	gemma4ReasoningStart = "<|channel>thought"
	gemma4ReasoningEnd   = "<channel|>"
)

func splitGemma4Response(fullText string) (reasoning string, content string) {
	start := strings.Index(fullText, gemma4ReasoningStart)
	if start < 0 {
		return "", fullText
	}
	reasoningStart := start + len(gemma4ReasoningStart)
	endRelative := strings.Index(fullText[reasoningStart:], gemma4ReasoningEnd)
	if endRelative < 0 {
		return "", fullText
	}
	end := reasoningStart + endRelative
	reasoning = strings.TrimSpace(fullText[reasoningStart:end])
	content = fullText[end+len(gemma4ReasoningEnd):]
	return reasoning, content
}

// gemma4StreamParser mirrors llama.cpp's Gemma 4 reasoning boundary parser
// for the control-token pieces emitted by the embedded llama_cpp plugin.
// Non-Gemma output is passed through unchanged.
type gemma4StreamParser struct {
	inReasoning     bool
	trimReasoningLF bool
	pendingChannel  bool
}

func (p *gemma4StreamParser) parse(piece string) (streamChunk, bool) {
	if p.pendingChannel {
		p.pendingChannel = false
		if strings.HasPrefix(piece, "thought") {
			p.inReasoning = true
			p.trimReasoningLF = true
			piece = strings.TrimPrefix(piece, "thought")
		} else {
			return contentChunk("<|channel>" + piece), true
		}
	} else if piece == "<|channel>" {
		p.pendingChannel = true
		return streamChunk{}, false
	}

	if strings.HasPrefix(piece, gemma4ReasoningStart) {
		p.inReasoning = true
		p.trimReasoningLF = true
		piece = strings.TrimPrefix(piece, gemma4ReasoningStart)
	}

	if p.inReasoning {
		if idx := strings.Index(piece, gemma4ReasoningEnd); idx >= 0 {
			reasoning := piece[:idx]
			p.inReasoning = false
			p.trimReasoningLF = false
			remainder := piece[idx+len(gemma4ReasoningEnd):]
			if reasoning != "" && remainder != "" {
				return streamChunk{
					Object: streamChunkObject,
					Choices: []streamChoice{{Delta: streamDelta{
						Role:             string(openai.MessageRoleAssistant),
						ReasoningContent: reasoning,
						Content:          remainder,
					}}},
				}, true
			}
			if reasoning != "" {
				return reasoningChunk(reasoning), true
			}
			if remainder != "" {
				return contentChunk(remainder), true
			}
			return streamChunk{}, false
		}
		if p.trimReasoningLF {
			if piece == "" {
				return streamChunk{}, false
			}
			piece = strings.TrimPrefix(piece, "\n")
			p.trimReasoningLF = false
		}
		if piece == "" {
			return streamChunk{}, false
		}
		return reasoningChunk(piece), true
	}

	if piece == gemma4ReasoningEnd {
		return streamChunk{}, false
	}
	return contentChunk(piece), true
}

// finishChunk is the terminal chunk: empty delta, non-null finish_reason.
func finishChunk(reason string, timings llamaTimings, model string) streamChunk {
	return streamChunk{
		Object:  streamChunkObject,
		Model:   model,
		Choices: []streamChoice{{FinishReason: &reason}},
		Timings: &timings,
	}
}

func usageChunk(u openai.CompletionUsage) streamChunk {
	return streamChunk{Object: streamChunkObject, Choices: []streamChoice{}, Usage: &u}
}

// streamPlainText drains dataCh as content chunks, then emits the finishing
// chunk, optional usage and [DONE].
func streamPlainText(c *gin.Context, dataCh <-chan string, wait func() error, includeUsage bool, usage streamUsage, timings streamTimings, model string, finish streamFinish) {
	parser := gemma4StreamParser{}
	c.Stream(func(w io.Writer) bool {
		r, ok := <-dataCh
		if ok {
			if chunk, emit := parser.parse(r); emit {
				c.SSEvent("", chunk)
			}
			return true
		}
		if err := wait(); err != nil {
			c.SSEvent("", map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return false
		}
		c.SSEvent("", finishChunk(finish(), timings(), model))
		if includeUsage {
			c.SSEvent("", usageChunk(usage()))
		}
		c.SSEvent("", "[DONE]")
		return false
	})
}

// streamToolCall buffers the stream and emits a tool-call chunk once dataCh
// closes; falls back to a content chunk when parsing fails. Either way the
// stream ends with a finishing chunk, optional usage and [DONE].
func streamToolCall(c *gin.Context, dataCh <-chan string, wait func() error, includeUsage bool, usage streamUsage, timings streamTimings, model string, finish streamFinish) {
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
		finishReason := "tool_calls"
		toolCall, err := utils.ParseToolCalls(buffer.String())
		if err != nil {
			slog.Warn("Tool call parse error, fallback to text", "error", err)
			finishReason = finish()
			c.SSEvent("", contentChunk(buffer.String()))
		} else {
			c.SSEvent("", streamChunk{
				Object: streamChunkObject,
				Choices: []streamChoice{{
					Delta: streamDelta{
						ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{{
							ID:   fmt.Sprintf("call_%d", rand.Uint32()),
							Type: "function",
							Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
								Name:      toolCall.Name,
								Arguments: toolCall.Arguments,
							},
						}},
					},
				}},
			})
		}
		c.SSEvent("", finishChunk(finishReason, timings(), model))
		if includeUsage {
			c.SSEvent("", usageChunk(usage()))
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

// llamaTimings is llama-server's timing extension consumed by the upstream
// Web UI. It is derived from the completed GenieX SDK profile, not wall-clock
// proxy timing, so the UI and geniex-bench use the same token accounting.
type llamaTimings struct {
	PromptN            int64   `json:"prompt_n"`
	PromptMS           float64 `json:"prompt_ms"`
	PromptPerSecond    float64 `json:"prompt_per_second"`
	PredictedN         int64   `json:"predicted_n"`
	PredictedMS        float64 `json:"predicted_ms"`
	PredictedPerSecond float64 `json:"predicted_per_second"`
	TTFTMS             float64 `json:"ttft_ms"`
	CacheN             int64   `json:"cache_n"`
}

func profile2Timings(p geniex_sdk.ProfileData) llamaTimings {
	return llamaTimings{
		PromptN:            p.PromptTokens,
		PromptMS:           float64(p.PromptTime) / 1000.0,
		PromptPerSecond:    p.PrefillSpeed,
		PredictedN:         p.GeneratedTokens,
		PredictedMS:        float64(p.DecodeTime) / 1000.0,
		PredictedPerSecond: p.DecodingSpeed,
		TTFTMS:             float64(p.TTFT) / 1000.0,
		CacheN:             0,
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
