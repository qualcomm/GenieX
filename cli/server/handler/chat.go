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
	"github.com/qualcomm/GenieX/cli/server/service"
	"github.com/qualcomm/GenieX/cli/server/types"
	"github.com/qualcomm/GenieX/cli/server/utils"
)

type ChatCompletionNewParams openai.ChatCompletionNewParams

type ChatCompletionRequest struct {
	ChatCompletionNewParams
	Stream bool `json:"stream"`

	EnableThink bool   `json:"enable_think"`
	NCtx        int32  `json:"nctx"`
	Ngl         int32  `json:"ngl"` // 0 = pure CPU, -1 = all layers, N = N layers; defaults to the server --ngl when omitted
	Compute     string `json:"compute"`
	VitCompute  string `json:"vit_compute"`
	PowerMode   string `json:"power_mode"`

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
	// Prefill llama_cpp knobs with server defaults (--nctx/--ngl/--compute):
	// ShouldBindJSON only overwrites fields present in the body, so an omitted
	// knob keeps the default and an explicit one (incl. ngl 0) wins.
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
		VitCompute:        cfg.VitCompute,
		PowerMode:         cfg.PowerMode,
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
	// Keep the precision: KeepAliveGet resolves this same string.
	paths, err := geniex_sdk.ModelGetPaths(param.Model)
	if err != nil {
		slog.Error("Failed to resolve model paths", "model", param.Model, "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	// Fill unset knobs from the server defaults before the MaxCompletionTokens
	// floor, so a body that omits nctx picks up the default, not the floor.
	modelParam, err := service.ResolveModelParam(paths.RuntimeID, paths.ModelName, param.NCtx, param.Ngl, param.Compute, param.VitCompute, param.PowerMode, service.Chipset(), types.SpecParam{
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

	effectiveType := paths.ModelType
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
		runChat(c, param, modelParam, messages, utils.SessionKeyOf(messages), prepareLLM)
	case geniex_sdk.ModelTypeVLM:
		// Hash before buildVLMMessages swaps each image/audio part for a
		// per-request temp file path; see SessionKeyOfVLMRequest.
		session := utils.SessionKeyOfVLMRequest(param.Messages)
		messages, tempFiles, ok := buildVLMMessages(c, param)
		for _, f := range tempFiles {
			defer os.Remove(f)
		}
		if !ok {
			return
		}
		runChat(c, param, modelParam, messages, session, prepareVLM)
	default:
		slog.Error("Model type not support", "model_type", paths.ModelType)
		c.JSON(http.StatusBadRequest, map[string]any{"error": "model type not support"})
		return
	}
}

// generateFn adapts LLM/VLM Generate to one shape; fullText is set even on error.
type generateFn func(prompt string, onToken func(string) bool) (geniex_sdk.ProfileData, string, error)

type prepareInput[M any] struct {
	Messages M
	Param    ChatCompletionRequest
	Tools    string
	Sampler  *geniex_sdk.SamplerConfig
	Fresh    bool
}

// prepareFn holds the LLM/VLM-specific work: apply the chat template and
// return the formatted prompt plus a generateFn.
type prepareFn[T, M any] func(p *T, input prepareInput[M]) (prompt string, gen generateFn, err error)

func prepareLLM(p *geniex_sdk.LLM, input prepareInput[[]geniex_sdk.LlmChatMessage]) (string, generateFn, error) {
	formatted, err := p.ApplyChatTemplate(geniex_sdk.LlmApplyChatTemplateInput{
		Messages:            input.Messages,
		Tools:               input.Tools,
		EnableThink:         input.Param.EnableThink,
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
				MaxTokens:     int32(input.Param.MaxCompletionTokens.Value),
				SamplerConfig: input.Sampler,
			},
		})
		if out == nil {
			return geniex_sdk.ProfileData{}, "", err
		}
		return out.ProfileData, out.FullText, err
	}
	return formatted.FormattedText, gen, nil
}

func prepareVLM(p *geniex_sdk.VLM, input prepareInput[[]geniex_sdk.VlmChatMessage]) (string, generateFn, error) {
	formatted, err := p.ApplyChatTemplate(geniex_sdk.VlmApplyChatTemplateInput{
		Messages:    input.Messages,
		Tools:       input.Tools,
		EnableThink: input.Param.EnableThink,
	})
	if err != nil {
		return "", nil, err
	}
	start := len(input.Messages) - 1
	if input.Fresh {
		start = 0
	}
	var images, audios []string
	for _, message := range input.Messages[start:] {
		for _, content := range message.Contents {
			switch content.Type {
			case geniex_sdk.VlmContentTypeImage:
				images = append(images, content.Text)
			case geniex_sdk.VlmContentTypeAudio:
				audios = append(audios, content.Text)
			}
		}
	}
	gen := func(prompt string, onToken func(string) bool) (geniex_sdk.ProfileData, string, error) {
		out, err := p.Generate(geniex_sdk.VlmGenerateInput{
			PromptUTF8: prompt,
			OnToken:    onToken,
			Config: &geniex_sdk.GenerationConfig{
				MaxTokens:     int32(input.Param.MaxCompletionTokens.Value),
				SamplerConfig: input.Sampler,
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

// generateWithRetry retries a VLM prefix reuse failure once after resetting
// the model and rebuilding the full conversation.
func generateWithRetry[T, M any](model *T, prompt string, gen generateFn, input prepareInput[M], prepare prepareFn[T, M], onToken func(string) bool) (geniex_sdk.ProfileData, string, error) {
	profile, fullText, err := gen(prompt, onToken)
	if !errors.Is(err, geniex_sdk.ErrVlmPrefixReuseFailed) {
		return profile, fullText, err
	}

	vlm, ok := any(model).(*geniex_sdk.VLM)
	if !ok {
		slog.Error("Cannot retry VLM prefix reuse failure with non-VLM model")
		return profile, fullText, err
	}
	if resetErr := vlm.Reset(); resetErr != nil {
		slog.Error("Failed to reset model before VLM prefix reuse retry", "error", resetErr)
		return profile, fullText, err
	}
	slog.Warn("VLM prefix reuse failed; resetting and retrying", "error", err)
	input.Fresh = true
	prompt, gen, err = prepare(model, input)
	if err != nil {
		return profile, fullText, err
	}
	return gen(prompt, onToken)
}

// runChat is the shared flow wrapping the type-specific prepareFn. session
// identifies the conversation for the keepalive cache's reset decision.
func runChat[T, M any](c *gin.Context, param ChatCompletionRequest, modelParam types.ModelParam, messages M, session utils.SessionKey, prepare prepareFn[T, M]) {
	// ---- prepare: parse tools, load the model, apply the chat template ----
	parseTool, tools, err := parseTools(param)
	if err != nil {
		slog.Error("Failed to parse tools", "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	acquired, err := service.KeepAliveGet[T](
		string(param.Model),
		modelParam,
		session,
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
	prompt, gen, err := prepare(acquired.Model, prepareInput[M]{
		Messages: messages,
		Param:    param,
		Tools:    tools,
		Sampler:  sampler,
		Fresh:    acquired.Fresh,
	})
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
			profile, _, genErr = generateWithRetry(acquired.Model, prompt, gen, prepareInput[M]{
				Messages: messages,
				Param:    param,
				Tools:    tools,
				Sampler:  sampler,
				Fresh:    acquired.Fresh,
			}, prepare, func(token string) bool {
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
		class := tokenClass(plainClass)
		if reasoningSeparated(param.ReasoningFormat) {
			class = reasoningClass()
		}
		if parseTool {
			streamToolCall(c, dataCh, wait, includeUsage, &profile, class)
		} else {
			streamPlainText(c, dataCh, wait, includeUsage, &profile, render(class))
		}

		stopGen.Store(true)
		for range dataCh {
		}
	} else {
		// blocking
		var content, reasoning strings.Builder
		class := tokenClass(plainClass)
		if reasoningSeparated(param.ReasoningFormat) {
			class = reasoningClass()
		}
		profile, _, err := generateWithRetry(acquired.Model, prompt, gen, prepareInput[M]{
			Messages: messages,
			Param:    param,
			Tools:    tools,
			Sampler:  sampler,
			Fresh:    acquired.Fresh,
		}, prepare, sink(class, &content, &reasoning))
		// A prompt that never fit is a 400; a window exhausted mid-generation is a
		// normal truncated completion (finish_reason=length), handled below.
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
