// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"errors"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"sync/atomic"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/config"
	"github.com/qualcomm/GenieX/cli/internal/store"
	"github.com/qualcomm/GenieX/cli/server/service"
	"github.com/qualcomm/GenieX/cli/server/types"
)

type CompletionNewParams openai.CompletionNewParams

type CompletionRequest struct {
	CompletionNewParams
	Stream bool `json:"stream"`

	NCtx    int32  `json:"nctx"`
	Ngl     int32  `json:"ngl"`
	Compute string `json:"compute"`

	TopK              int32   `json:"top_k"`
	MinP              float32 `json:"min_p"`
	RepetitionPenalty float32 `json:"repetition_penalty"`
}

func defaultCompletionRequest() CompletionRequest {
	cfg := config.Get()
	return CompletionRequest{
		CompletionNewParams: CompletionNewParams{
			// OpenAI's documented default for the legacy completions endpoint.
			MaxTokens: param.NewOpt[int64](16),
		},
		NCtx:              cfg.NCtx,
		Ngl:               cfg.Ngl,
		Compute:           cfg.Compute,
		RepetitionPenalty: 1.0,
	}
}

func completionPrompt(u openai.CompletionNewParamsPromptUnion) (string, error) {
	switch {
	case !param.IsOmitted(u.OfString):
		if u.OfString.Value == "" {
			return "", errors.New("prompt must be non-empty")
		}
		return u.OfString.Value, nil
	case u.OfArrayOfStrings != nil:
		if len(u.OfArrayOfStrings) != 1 || u.OfArrayOfStrings[0] == "" {
			return "", errors.New("only a single non-empty prompt is supported")
		}
		return u.OfArrayOfStrings[0], nil
	case u.OfArrayOfTokens != nil || u.OfArrayOfTokenArrays != nil:
		return "", errors.New("token prompts are not supported; send a string prompt")
	default:
		return "", errors.New("prompt is required")
	}
}

func completionStop(u openai.CompletionNewParamsStopUnion) []string {
	if !param.IsOmitted(u.OfString) {
		if u.OfString.Value == "" {
			return nil
		}
		return []string{u.OfString.Value}
	}
	return u.OfStringArray
}

func completionUnsupported(p CompletionNewParams) error {
	switch {
	case p.Suffix.Valid() && p.Suffix.Value != "":
		return errors.New("suffix is not supported; send a prompt containing the model's FIM template instead")
	case p.N.Valid() && p.N.Value > 1:
		return errors.New("n > 1 is not supported")
	case p.BestOf.Valid() && p.BestOf.Value > 1:
		return errors.New("best_of > 1 is not supported")
	case p.Logprobs.Valid():
		return errors.New("logprobs is not supported")
	case len(p.LogitBias) > 0:
		return errors.New("logit_bias is not supported")
	default:
		return nil
	}
}

// Completions serves POST /v1/completions. The prompt is passed to the model
// verbatim — no chat template — so clients that build their own prompt (e.g.
// editor FIM autocompletion) get a raw continuation back.
func Completions(c *gin.Context) {
	req := defaultCompletionRequest()
	if err := c.ShouldBindJSON(&req); err != nil {
		slog.Error("Failed to bind JSON", "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	slog.Info("Completions", "param", req)
	prompt, err := completionPrompt(req.Prompt)
	if err != nil {
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if err := completionUnsupported(req.CompletionNewParams); err != nil {
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	name, _ := geniex_sdk.SplitNamePrecision(string(req.Model))
	modelType, err := geniex_sdk.ModelGetType(name)
	if err != nil {
		slog.Error("Failed to get model type", "model", req.Model, "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if modelType != geniex_sdk.ModelTypeLLM {
		c.JSON(http.StatusBadRequest, map[string]any{"error": "completions is only supported for LLM models"})
		return
	}
	paths, err := geniex_sdk.ModelGetPaths(name)
	if err != nil {
		slog.Error("Failed to resolve model paths", "model", req.Model, "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	modelParam, err := service.ResolveModelParam(paths.RuntimeID, paths.ModelName, req.NCtx, req.Ngl, req.Compute, store.Get().ResolveChipset(true), types.SpecParam{})
	if err != nil {
		slog.Error("Failed to resolve model params", "model", req.Model, "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	// Automatically adjust NCtx if MaxTokens is larger (llama_cpp only — QAIRT
	// does not use NCtx and the 0-default must not be overwritten for non-llama_cpp plugins).
	if paths.RuntimeID == geniex_sdk.RuntimeLlamaCpp && modelParam.NCtx < int32(req.MaxTokens.Value) {
		slog.Debug("Adjust NCtx to MaxTokens", "from", modelParam.NCtx, "to", req.MaxTokens.Value)
		modelParam.NCtx = int32(req.MaxTokens.Value)
	}

	p, err := service.KeepAliveGet[geniex_sdk.LLM](
		string(req.Model),
		modelParam,
		c.GetHeader("GenieX-KeepCache") != "true",
	)
	if writeKeepAliveError(c, err) {
		return
	}

	genConfig := &geniex_sdk.GenerationConfig{
		MaxTokens: int32(req.MaxTokens.Value),
		Stop:      completionStop(req.Stop),
		SamplerConfig: &geniex_sdk.SamplerConfig{
			Temperature:       float32(req.Temperature.Value),
			TopP:              float32(req.TopP.Value),
			TopK:              req.TopK,
			MinP:              req.MinP,
			RepetitionPenalty: req.RepetitionPenalty,
			PresencePenalty:   float32(req.PresencePenalty.Value),
			FrequencyPenalty:  float32(req.FrequencyPenalty.Value),
			Seed:              int32(req.Seed.Value),
		},
	}
	echo := ""
	if req.Echo.Valid() && req.Echo.Value {
		echo = prompt
	}

	// ---- generate: stream SSE chunks or write one blocking response ----
	if req.Stream {
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
			out, err := p.Generate(geniex_sdk.LlmGenerateInput{
				PromptUTF8: prompt,
				OnToken: func(token string) bool {
					if stopGen.Load() {
						return false
					}
					dataCh <- token
					return true
				},
				Config: genConfig,
			})
			genErr = err
			if out != nil {
				profile = out.ProfileData
			}
			close(dataCh)
		}()

		wait := func() error { wg.Wait(); return genErr }
		streamCompletion(c, dataCh, wait, req.StreamOptions.IncludeUsage.Value, echo, &profile)

		stopGen.Store(true)
		for range dataCh {
		}
	} else {
		// blocking
		out, err := p.Generate(geniex_sdk.LlmGenerateInput{
			PromptUTF8: prompt,
			Config:     genConfig,
		})
		// A prompt that never fit is a client error (400). A window exhausted
		// mid-generation is a normal truncated completion (finish_reason=length),
		// so it falls through to the regular response below.
		if errors.Is(err, geniex_sdk.ErrLlmGenerationPromptTooLong) {
			writeCompletionPromptTooLong(c, out.ProfileData)
			return
		}
		if err != nil && !errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) {
			c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return
		}
		writeCompletionResponse(c, echo+out.FullText, out.ProfileData)
	}
}

// ---- streaming SSE chunks ----

type completionStreamChoice struct {
	Index        int64   `json:"index"`
	Text         string  `json:"text"`
	FinishReason *string `json:"finish_reason"` // null until the final chunk
}

type completionStreamChunk struct {
	Object  string                   `json:"object"`
	Choices []completionStreamChoice `json:"choices"`
	Usage   *openai.CompletionUsage  `json:"usage,omitempty"`
}

const completionObject = "text_completion"

func completionTextChunk(text string) completionStreamChunk {
	return completionStreamChunk{
		Object:  completionObject,
		Choices: []completionStreamChoice{{Text: text}},
	}
}

func completionFinishChunk(reason string) completionStreamChunk {
	return completionStreamChunk{
		Object:  completionObject,
		Choices: []completionStreamChoice{{FinishReason: &reason}},
	}
}

func completionUsageChunk(u openai.CompletionUsage) completionStreamChunk {
	return completionStreamChunk{Object: completionObject, Choices: []completionStreamChoice{}, Usage: &u}
}

// profile is read only after wait() returns, when generation has filled it.
func streamCompletion(c *gin.Context, dataCh <-chan string, wait func() error, includeUsage bool, echo string, profile *geniex_sdk.ProfileData) {
	first := echo != ""
	c.Stream(func(w io.Writer) bool {
		if first {
			first = false
			c.SSEvent("", completionTextChunk(echo))
			return true
		}
		r, ok := <-dataCh
		if ok {
			c.SSEvent("", completionTextChunk(r))
			return true
		}
		// A context window exhausted mid-stream is a normal truncated completion
		// (finish_reason=length): fall through to the finish chunk. Other errors
		// (including a too-long prompt) are surfaced as an error event.
		if err := wait(); err != nil && !errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) {
			c.SSEvent("", map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return false
		}
		c.SSEvent("", completionFinishChunk(mapFinishReason(profile.StopReason)))
		if includeUsage {
			c.SSEvent("", completionUsageChunk(profile2Usage(*profile)))
		}
		c.SSEvent("", "[DONE]")
		return false
	})
}

// ---- blocking response ----

type completionChoice struct {
	Index        int64  `json:"index"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"`
}

type completionResponse struct {
	Object  string                 `json:"object"`
	Choices []completionChoice     `json:"choices"`
	Usage   openai.CompletionUsage `json:"usage"`
}

func writeCompletionResponse(c *gin.Context, text string, profile geniex_sdk.ProfileData) {
	c.JSON(http.StatusOK, completionResponse{
		Object: completionObject,
		Choices: []completionChoice{{
			Text:         text,
			FinishReason: mapFinishReason(profile.StopReason),
		}},
		Usage: profile2Usage(profile),
	})
}

// OpenAI's 400 body for a prompt that is longer than the context window. No
// partial output exists (nothing was generated), so this returns only the error.
func writeCompletionPromptTooLong(c *gin.Context, profile geniex_sdk.ProfileData) {
	c.JSON(http.StatusBadRequest, map[string]any{
		"error": map[string]any{
			"message": "prompt is longer than the model's context window",
			"type":    "invalid_request_error",
			"code":    "context_length_exceeded",
		},
		"usage": profile2Usage(profile),
	})
}
