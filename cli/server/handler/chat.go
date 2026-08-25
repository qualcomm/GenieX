// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"errors"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/bytedance/sonic"
	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/config"
	"github.com/qualcomm/GenieX/cli/internal/store"
	"github.com/qualcomm/GenieX/cli/server/service"
	"github.com/qualcomm/GenieX/cli/server/types"
)

type ChatCompletionNewParams openai.ChatCompletionNewParams

type ChatCompletionRequest struct {
	ChatCompletionNewParams
	Stream bool `json:"stream"`

	EnableThink bool   `json:"enable_think"`
	NCtx        int32  `json:"nctx"`
	Ngl         int32  `json:"ngl"` // 0 = pure CPU, -1 = all layers, N = N layers; defaults to the server --ngl when omitted
	Compute     string `json:"compute"`

	// "" / "none" keeps thinking inline in content (default); "deepseek" /
	// "deepseek-legacy" / "auto" move it to reasoning_content.
	ReasoningFormat string `json:"reasoning_format"`

	TopK              int32   `json:"top_k"`
	MinP              float32 `json:"min_p"`
	RepetitionPenalty float32 `json:"repetition_penalty"`
	GrammarPath       string  `json:"grammar_path"`
	GrammarString     string  `json:"grammar_string"`

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
			// On the deprecated alias, so an explicit max_completion_tokens wins.
			MaxTokens: param.NewOpt[int64](2048),
		},
		Stream: false,

		EnableThink:       true,
		NCtx:              cfg.NCtx,
		Ngl:               cfg.Ngl,
		Compute:           cfg.Compute,
		TopK:              0,
		MinP:              0.0,
		RepetitionPenalty: 1.0,
		GrammarPath:       "",
		GrammarString:     "",
	}
}

func ChatCompletions(c *gin.Context) {
	param := defaultChatCompletionRequest()
	if err := c.ShouldBindJSON(&param); err != nil {
		slog.Error("Failed to bind JSON", "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	// max_tokens is the deprecated alias; below reads MaxCompletionTokens only.
	param.MaxCompletionTokens = openai.Int(param.MaxCompletionTokens.Or(param.MaxTokens.Value))

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
	modelParam, err := service.ResolveModelParam(paths.RuntimeID, paths.ModelName, param.NCtx, param.Ngl, param.Compute, store.Get().ResolveChipset(true), types.SpecParam{
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
		messages, ok := buildLLMMessages(c, param)
		if !ok {
			return
		}
		runChat(c, param, modelParam, messages, prepareLLM)
	case geniex_sdk.ModelTypeVLM:
		messages, tempFiles, ok := buildVLMMessages(c, param)
		for _, f := range tempFiles {
			defer os.Remove(f)
		}
		if !ok {
			return
		}
		runChat(c, param, modelParam, messages, prepareVLM)
	default:
		slog.Error("Model type not support", "model_type", modelType)
		c.JSON(http.StatusBadRequest, map[string]any{"error": "model type not support"})
		return
	}
}

// generateFn adapts LLM/VLM Generate to one shape; fullText is set even on error.
type generateFn func(prompt string, onToken func(string) bool) (geniex_sdk.ProfileData, string, error)

// prepareFn holds the only LLM/VLM-specific work: apply the chat template, then
// return the formatted prompt and a generateFn.
type prepareFn[T, M any] func(p *T, messages M, param ChatCompletionRequest, tools string, sampler *geniex_sdk.SamplerConfig) (prompt string, gen generateFn, err error)

func prepareLLM(p *geniex_sdk.LLM, messages []geniex_sdk.LlmChatMessage, param ChatCompletionRequest, tools string, sampler *geniex_sdk.SamplerConfig) (string, generateFn, error) {
	formatted, err := p.ApplyChatTemplate(geniex_sdk.LlmApplyChatTemplateInput{
		Messages:            messages,
		Tools:               tools,
		EnableThink:         param.EnableThink,
		AddGenerationPrompt: true,
	})
	if err != nil {
		return "", nil, err
	}
	gen := func(prompt string, onToken func(string) bool) (geniex_sdk.ProfileData, string, error) {
		out, err := p.Generate(geniex_sdk.LlmGenerateInput{
			PromptUTF8: prompt,
			OnToken:    onToken,
			Config: &geniex_sdk.GenerationConfig{
				MaxTokens:     int32(param.MaxCompletionTokens.Value),
				SamplerConfig: sampler,
			},
		})
		if out == nil {
			return geniex_sdk.ProfileData{}, "", err
		}
		return out.ProfileData, out.FullText, err
	}
	return formatted.FormattedText, gen, nil
}

func prepareVLM(p *geniex_sdk.VLM, messages []geniex_sdk.VlmChatMessage, param ChatCompletionRequest, tools string, sampler *geniex_sdk.SamplerConfig) (string, generateFn, error) {
	formatted, err := p.ApplyChatTemplate(geniex_sdk.VlmApplyChatTemplateInput{
		Messages:    messages,
		Tools:       tools,
		EnableThink: param.EnableThink,
	})
	if err != nil {
		return "", nil, err
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
	gen := func(prompt string, onToken func(string) bool) (geniex_sdk.ProfileData, string, error) {
		out, err := p.Generate(geniex_sdk.VlmGenerateInput{
			PromptUTF8: prompt,
			OnToken:    onToken,
			Config: &geniex_sdk.GenerationConfig{
				MaxTokens:     int32(param.MaxCompletionTokens.Value),
				SamplerConfig: sampler,
				ImagePaths:    images,
				AudioPaths:    audios,
			},
		})
		if out == nil {
			return geniex_sdk.ProfileData{}, "", err
		}
		return out.ProfileData, out.FullText, err
	}
	return formatted.FormattedText, gen, nil
}

// runChat is the shared flow wrapping the type-specific prepareFn.
func runChat[T, M any](c *gin.Context, param ChatCompletionRequest, modelParam types.ModelParam, messages M, prepare prepareFn[T, M]) {
	// ---- prepare: parse tools, load the model, apply the chat template ----
	parseTool, tools, err := parseTools(param)
	if err != nil {
		slog.Error("Failed to parse tools", "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	p, err := service.KeepAliveGet[T](
		string(param.Model),
		modelParam,
		c.GetHeader("GenieX-KeepCache") != "true",
	)
	if writeKeepAliveError(c, err) {
		return
	}
	// Warm-up: no messages, or a lone system message — model loaded, nothing to generate.
	var role *string
	if len(param.Messages) == 1 {
		role = param.Messages[0].GetRole()
	}
	if len(param.Messages) == 0 || role != nil && *role == "system" {
		c.JSON(http.StatusOK, nil)
		return
	}

	sampler := &geniex_sdk.SamplerConfig{
		Temperature:       float32(param.Temperature.Value),
		TopP:              float32(param.TopP.Value),
		TopK:              param.TopK,
		MinP:              param.MinP,
		RepetitionPenalty: param.RepetitionPenalty,
		PresencePenalty:   float32(param.PresencePenalty.Value),
		FrequencyPenalty:  float32(param.FrequencyPenalty.Value),
		Seed:              int32(param.Seed.Value),
	}
	prompt, gen, err := prepare(p, messages, param, tools, sampler)
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
		return
	}

	// ---- generate: stream SSE chunks or write one blocking response ----
	if param.Stream {
		// streaming
		var stopGen atomic.Bool
		dataCh := make(chan string)

		var (
			profile geniex_sdk.ProfileData
			genErr  error
			wg      sync.WaitGroup
		)

		wg.Add(1)
		go func() {
			defer wg.Done()
			profile, _, genErr = gen(prompt, func(token string) bool {
				if stopGen.Load() {
					return false
				}
				dataCh <- token
				return true
			})
			close(dataCh)
		}()

		wait := func() error { wg.Wait(); return genErr }
		includeUsage := param.StreamOptions.IncludeUsage.Value
		if parseTool {
			streamToolCall(c, dataCh, wait, includeUsage, &profile)
		} else {
			class := tokenClass(plainClass)
			if reasoningSeparated(param.ReasoningFormat) {
				class = reasoningClass()
			}
			streamPlainText(c, dataCh, wait, includeUsage, &profile, render(class))
		}

		stopGen.Store(true)
		for range dataCh {
		}
	} else {
		// blocking
		var content, reasoning strings.Builder
		class := tokenClass(plainClass)
		if !parseTool && reasoningSeparated(param.ReasoningFormat) {
			class = reasoningClass()
		}
		profile, _, err := gen(prompt, sink(class, &content, &reasoning))
		// A prompt that never fit is a client error (400). A window exhausted
		// mid-generation is a normal truncated completion (finish_reason=length),
		// so it falls through to the regular response below.
		if errors.Is(err, geniex_sdk.ErrLlmGenerationPromptTooLong) {
			writePromptTooLong(c, profile)
			return
		}
		if err != nil && !errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) {
			c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return
		}
		writeBlockingResponse(c, content.String(), reasoning.String(), profile, parseTool)
	}
}

func parseTools(param ChatCompletionRequest) (bool, string, error) {
	if len(param.Tools) == 0 {
		return false, "", nil
	}
	tools, err := sonic.MarshalString(param.Tools)
	return true, tools, err
}
