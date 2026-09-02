// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"errors"
	"fmt"
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
	// Prefill the llama_cpp knobs with the server-wide defaults (--nctx / --ngl /
	// --compute): ShouldBindJSON only overwrites fields present in the body, so an
	// omitted knob keeps the default and an explicit one (incl. ngl 0) wins.
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

	// Request messages and grammar strings may contain private prompt data.
	// Keep operational logs useful without serializing the request body.
	slog.Info("ChatCompletions",
		"model", param.Model,
		"message_count", len(param.Messages),
		"stream", param.Stream,
		"managed_cache", managedCacheRequested(c))
	// Keep the precision: KeepAliveGet resolves this same string.
	paths, err := geniex_sdk.ModelGetPaths(param.Model)
	if err != nil {
		slog.Error("Failed to resolve model paths", "model", param.Model, "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	// Fill unset knobs from the server defaults before the MaxCompletionTokens
	// floor, so a body that omits nctx picks up the default, not the floor.
	modelParam, err := service.ResolveModelParam(paths.RuntimeID, paths.ModelName, param.NCtx, param.Ngl, param.Compute, service.Chipset(), types.SpecParam{
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
		runChat(c, param, modelParam, paths.RuntimeID, paths.ModelPath, paths.TokenizerPath, messages, utils.SessionKeyOf(messages), prepareLLM)
	case geniex_sdk.ModelTypeVLM:
		if managedCacheRequested(c) {
			c.JSON(http.StatusBadRequest, map[string]any{"error": "managed caching supports text LLM chat only in version 2"})
			return
		}
		// Hash before buildVLMMessages replaces media with per-request paths.
		autoSession := utils.SessionKeyOfVLMRequest(param.Messages)
		messages, tempFiles, ok := buildVLMMessages(c, param)
		for _, f := range tempFiles {
			defer os.Remove(f)
		}
		if !ok {
			return
		}
		runChat(c, param, modelParam, paths.RuntimeID, paths.ModelPath, paths.TokenizerPath, messages, autoSession, prepareVLM)
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

// prepareFn holds the only LLM/VLM-specific work: apply the chat template, then
// return the formatted prompt and a generateFn.
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

// runChat is the shared flow wrapping the type-specific prepareFn.
func runChat[T, M any](c *gin.Context, param ChatCompletionRequest, modelParam types.ModelParam, runtime, modelPath, tokenizerPath string, messages M, autoSession utils.SessionKey, prepare prepareFn[T, M]) {
	// ---- prepare: parse tools, load the model, apply the chat template ----
	parseTool, tools, err := parseTools(param)
	if err != nil {
		slog.Error("Failed to parse tools", "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	// Warm-up: no messages, or a lone system message — model loaded, nothing to generate.
	var role *string
	if len(param.Messages) == 1 {
		role = param.Messages[0].GetRole()
	}
	warmup := len(param.Messages) == 0 || role != nil && *role == "system"

	session := strings.TrimSpace(c.GetHeader(managedCacheSessionHeader))
	parent := strings.TrimSpace(c.GetHeader(managedCacheParentHeader))
	managed := session != "" || parent != ""
	if managed {
		c.Header(managedCacheProtocolHeader, managedCacheProtocolVersion)
	}
	legacyHeader := c.GetHeader("GenieX-KeepCache")
	if legacyHeader != "" && legacyHeader != "true" {
		c.JSON(http.StatusBadRequest, map[string]any{"error": "GenieX-KeepCache only accepts true"})
		return
	}
	if managed && legacyHeader != "" {
		c.JSON(http.StatusBadRequest, map[string]any{"error": "GenieX-KeepCache and managed-cache headers cannot be combined"})
		return
	}
	if managed && warmup {
		c.JSON(http.StatusBadRequest, map[string]any{"error": "managed caching requires a generative request"})
		return
	}

	reset := false
	var txnID uint64
	var plannedReuse bool
	if managed {
		artifact, artifactErr := resolveLineageArtifactIdentity(runtime, modelPath, tokenizerPath)
		if artifactErr != nil {
			c.JSON(http.StatusInternalServerError, map[string]any{"error": artifactErr.Error()})
			return
		}
		request, requestErr := lineageRequestFromChat(param, modelParam, artifact, session, parent)
		if requestErr != nil {
			c.JSON(http.StatusBadRequest, map[string]any{"error": requestErr.Error()})
			return
		}
		decision, beginErr := managedChatLineage.Begin(request)
		if beginErr != nil {
			c.JSON(http.StatusBadRequest, map[string]any{"error": beginErr.Error()})
			return
		}
		txnID = decision.TxnID
		plannedReuse = decision.Reuse
		reset = decision.ResetRequired
	} else {
		// An unmanaged request can mutate the same single model handle. It must
		// invalidate any managed lineage before the handle is touched.
		invalidateManagedLineageForUnmanagedRequest()
	}

	var p *T
	var fresh bool
	if managed {
		p, err = service.KeepAliveGetManaged[T](string(param.Model), modelParam, reset)
	} else if legacyHeader == "true" {
		p, err = service.KeepAliveGetLegacy[T](string(param.Model), modelParam)
	} else {
		var acquired service.AcquiredModel[T]
		acquired, err = service.KeepAliveGet[T](string(param.Model), modelParam, autoSession)
		p, fresh = acquired.Model, acquired.Fresh
	}
	if err != nil && managed {
		managedChatLineage.Abort(txnID)
	}
	if writeKeepAliveError(c, err) {
		return
	}
	if managed {
		bound, bindErr := managedChatLineage.BindGeneration(txnID, service.KeepAliveGeneration())
		if bindErr != nil {
			_ = abortManagedCache(txnID)
			c.JSON(http.StatusInternalServerError, map[string]any{"error": bindErr.Error()})
			return
		}
		// A generation change between Begin and binding invalidates a planned
		// hit. Reset the handle explicitly before accepting the full prompt;
		// merely changing the public status would leave unknown KV state live.
		if plannedReuse && !bound.Reuse {
			if resetErr := service.ResetKeepAlive(); resetErr != nil {
				managedChatLineage.Abort(txnID)
				c.JSON(http.StatusInternalServerError, map[string]any{"error": resetErr.Error()})
				return
			}
			if _, bindErr = managedChatLineage.BindGeneration(txnID, service.KeepAliveGeneration()); bindErr != nil {
				_ = abortManagedCache(txnID)
				c.JSON(http.StatusInternalServerError, map[string]any{"error": bindErr.Error()})
				return
			}
		}
	}

	if warmup {
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
	prompt, gen, err := prepare(p, prepareInput[M]{
		Messages: messages,
		Param:    param,
		Tools:    tools,
		Sampler:  sampler,
		Fresh:    fresh,
	})
	if err != nil {
		if managed {
			_ = abortManagedCache(txnID)
		}
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
		return
	}

	// ---- generate: stream SSE chunks or write one blocking response ----
	if param.Stream {
		// streaming
		var stopGen atomic.Bool
		dataCh := make(chan string)

		var (
			profile  geniex_sdk.ProfileData
			fullText string
			genErr   error
			wg       sync.WaitGroup
		)

		wg.Add(1)
		go func() {
			defer wg.Done()
			profile, fullText, genErr = gen(prompt, func(token string) bool {
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
		var finalize managedCacheFinalizer
		if managed {
			var once sync.Once
			var cache *managedCacheMetadata
			var finalizeErr error
			finalize = func(generationErr error) (*managedCacheMetadata, error) {
				once.Do(func() {
					if generationErr != nil && !errors.Is(generationErr, geniex_sdk.ErrLlmTokenizationContextLength) {
						finalizeErr = abortManagedCache(txnID)
						return
					}
					cache, finalizeErr = commitManagedCache(
						txnID, fullText, managedGenerationReusable(profile),
					)
				})
				return cache, finalizeErr
			}
		}
		class := tokenClass(plainClass)
		if reasoningSeparated(param.ReasoningFormat) {
			class = reasoningClass()
		}
		var disconnected bool
		if parseTool {
			disconnected = streamToolCall(c, dataCh, wait, includeUsage, &profile, class, finalize)
		} else {
			disconnected = streamPlainText(c, dataCh, wait, includeUsage, &profile, render(class), finalize)
		}

		stopGen.Store(true)
		for range dataCh {
		}
		wg.Wait()
		if disconnected && managed {
			_ = abortManagedCache(txnID)
		}
	} else {
		// blocking
		var content, reasoning strings.Builder
		class := tokenClass(plainClass)
		if reasoningSeparated(param.ReasoningFormat) {
			class = reasoningClass()
		}
		tokenSink := sink(class, &content, &reasoning)
		cancelled := false
		profile, fullText, err := gen(prompt, func(token string) bool {
			select {
			case <-c.Request.Context().Done():
				cancelled = true
				return false
			default:
				return tokenSink(token)
			}
		})
		if cancelled || c.Request.Context().Err() != nil {
			if managed {
				_ = abortManagedCache(txnID)
			}
			return
		}
		// A prompt that never fit is a 400; a window exhausted mid-generation is a
		// normal truncated completion (finish_reason=length), handled below.
		if errors.Is(err, geniex_sdk.ErrLlmGenerationPromptTooLong) {
			if managed {
				_ = abortManagedCache(txnID)
			}
			writePromptTooLong(c, profile)
			return
		}
		if err != nil && !errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) {
			if managed {
				_ = abortManagedCache(txnID)
			}
			c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return
		}
		var cache *managedCacheMetadata
		if managed {
			cache, err = commitManagedCache(
				txnID, fullText, managedGenerationReusable(profile),
			)
			if err != nil {
				c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error()})
				return
			}
		}
		writeBlockingResponse(c, content.String(), reasoning.String(), profile, parseTool, cache)
	}
}

func managedCacheRequested(c *gin.Context) bool {
	return strings.TrimSpace(c.GetHeader(managedCacheSessionHeader)) != "" ||
		strings.TrimSpace(c.GetHeader(managedCacheParentHeader)) != ""
}

func abortManagedCache(txnID uint64) error {
	managedChatLineage.Abort(txnID)
	if err := service.ResetKeepAlive(); err != nil {
		return fmt.Errorf("reset model after managed-cache abort: %w", err)
	}
	return nil
}

// Only an EOS-complete turn is safe for the next incremental QAIRT template.
// A length, stop-sequence, callback, or unknown stop can leave the physical KV
// state without the turn delimiter that a cold full-history template includes.
// Keep the logical transcript revision, but reset the model and mark the state
// non-reusable so the next exact extension is processed cold.
func managedGenerationReusable(profile geniex_sdk.ProfileData) bool {
	return profile.StopReason == "eos"
}

func commitManagedCache(txnID uint64, assistantContent string, reusable bool) (*managedCacheMetadata, error) {
	metadata, err := managedChatLineage.Commit(txnID, assistantContent, reusable)
	if err != nil {
		if resetErr := abortManagedCache(txnID); resetErr != nil {
			return nil, fmt.Errorf("commit managed cache: %v; %w", err, resetErr)
		}
		return nil, fmt.Errorf("commit managed cache: %w", err)
	}
	if !reusable {
		if resetErr := service.ResetKeepAlive(); resetErr != nil {
			managedChatLineage.Clear()
			return nil, fmt.Errorf("reset non-reusable managed cache: %w", resetErr)
		}
	}
	return &metadata, nil
}

func parseTools(param ChatCompletionRequest) (bool, string, error) {
	if len(param.Tools) == 0 {
		return false, "", nil
	}
	tools, err := sonic.MarshalString(param.Tools)
	return true, tools, err
}
