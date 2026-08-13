// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
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
	"github.com/qualcomm/GenieX/cli/internal/store"
	"github.com/qualcomm/GenieX/cli/server/service"
	"github.com/qualcomm/GenieX/cli/server/types"
)

type ChatCompletionNewParams openai.ChatCompletionNewParams

type ChatCompletionRequest struct {
	ChatCompletionNewParams
	Stream bool `json:"stream"`

	// openai-go can classify the standard named-function tool_choice object as
	// its allowed_tools union variant. Retain the raw bounded field so the
	// OpenAI-compatible named-function shape can be decoded strictly.
	rawToolChoice json.RawMessage

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

func (r *ChatCompletionRequest) UnmarshalJSON(data []byte) error {
	type requestAlias ChatCompletionRequest
	decoded := requestAlias(*r)
	if err := sonic.Unmarshal(data, &decoded); err != nil {
		return err
	}
	var envelope struct {
		ToolChoice json.RawMessage `json:"tool_choice"`
	}
	if err := sonic.Unmarshal(data, &envelope); err != nil {
		return err
	}
	*r = ChatCompletionRequest(decoded)
	r.rawToolChoice = append(r.rawToolChoice[:0], envelope.ToolChoice...)
	return nil
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

func ChatCompletions(c *gin.Context) {
	param := defaultChatCompletionRequest()
	if err := c.ShouldBindJSON(&param); err != nil {
		slog.Error("Failed to bind JSON", "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
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
type prepareFn[T, M any] func(p *T, messages M, param ChatCompletionRequest, tools string, policy toolSelectionPolicy, sampler *geniex_sdk.SamplerConfig) (prompt string, gen generateFn, err error)

func prepareLLM(p *geniex_sdk.LLM, messages []geniex_sdk.LlmChatMessage, param ChatCompletionRequest, tools string, policy toolSelectionPolicy, sampler *geniex_sdk.SamplerConfig) (string, generateFn, error) {
	messages = applyLlmToolSelectionDirective(messages, policy)
	formatted, err := p.ApplyChatTemplate(geniex_sdk.LlmApplyChatTemplateInput{
		Messages:            messages,
		Tools:               tools,
		ToolChoice:          policy.templateToolChoice(),
		EnableThink:         param.EnableThink,
		AddGenerationPrompt: true,
	})
	if err != nil {
		return "", nil, err
	}
	if policy.explicit() {
		if formatted.Grammar == "" {
			return "", nil, errors.New("active chat template did not provide a required-tool grammar")
		}
		requiredGrammar, err := strictRequiredToolGrammar(formatted.Grammar)
		if err != nil {
			return "", nil, err
		}
		sampler.GrammarString = requiredGrammar
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

func prepareVLM(p *geniex_sdk.VLM, messages []geniex_sdk.VlmChatMessage, param ChatCompletionRequest, tools string, policy toolSelectionPolicy, sampler *geniex_sdk.SamplerConfig) (string, generateFn, error) {
	messages = applyVlmToolSelectionDirective(messages, policy)
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
	toolPolicy, tools, err := parseTools(param)
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
	if isWarmupRequest(param) {
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
	prompt, gen, err := prepare(p, messages, param, tools, toolPolicy, sampler)
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
		if toolPolicy.parseResponse() {
			streamToolCall(c, dataCh, wait, includeUsage, &profile, toolPolicy)
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
		if !toolPolicy.parseResponse() && reasoningSeparated(param.ReasoningFormat) {
			class = reasoningClass()
		}
		profile, fullText, err := gen(prompt, sink(class, &content, &reasoning))
		if errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) {
			writeContextLengthExceeded(c, fullText, profile)
			return
		}
		if err != nil {
			c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_sdk.SDKErrorCode(err)})
			return
		}
		writeBlockingResponse(c, content.String(), reasoning.String(), profile, toolPolicy)
	}
}

type toolSelectionMode string

const (
	toolSelectionNone     toolSelectionMode = "none"
	toolSelectionAuto     toolSelectionMode = "auto"
	toolSelectionRequired toolSelectionMode = "required"
	toolSelectionNamed    toolSelectionMode = "named"
)

const (
	requiredToolCallParseFailedCode = "required_tool_call_parse_failed"
	requiredToolCallParseFailedText = "the explicitly selected tool was not emitted or could not be parsed"
)

type toolSelectionPolicy struct {
	mode         toolSelectionMode
	namedTool    string
	allowedTools []string
}

func (p toolSelectionPolicy) parseResponse() bool {
	return p.mode == toolSelectionAuto || p.mode == toolSelectionRequired || p.mode == toolSelectionNamed
}

func (p toolSelectionPolicy) explicit() bool {
	return p.mode == toolSelectionRequired || p.mode == toolSelectionNamed
}

func (p toolSelectionPolicy) templateToolChoice() string {
	if p.explicit() {
		// Named selection has already filtered the supplied list to exactly one
		// function, so llama.cpp required mode enforces that selected name.
		return string(toolSelectionRequired)
	}
	return string(p.mode)
}

func strictRequiredToolGrammar(grammar string) (string, error) {
	const permissiveRoot = "root ::= start message* scan-to-toolcall tool-call"
	const requiredRoot = "root ::= start tool-call"
	if !strings.Contains(grammar, permissiveRoot) {
		return "", errors.New("active chat template does not expose the expected required-tool grammar root")
	}
	return strings.Replace(grammar, permissiveRoot, requiredRoot, 1), nil
}

func (p toolSelectionPolicy) accepts(name string) bool {
	for _, allowed := range p.allowedTools {
		if name == allowed {
			return true
		}
	}
	return false
}

func (p toolSelectionPolicy) directive() string {
	switch p.mode {
	case toolSelectionRequired:
		return "API tool selection requirement: call exactly one of the provided functions before answering. " +
			"Emit only the tool call with its arguments; do not emit planning prose or a final answer."
	case toolSelectionNamed:
		return fmt.Sprintf(
			"API tool selection requirement: call the function %q before answering. "+
				"Emit only that tool call with its arguments; do not emit planning prose or a final answer.",
			p.namedTool,
		)
	default:
		return ""
	}
}

func applyLlmToolSelectionDirective(messages []geniex_sdk.LlmChatMessage, policy toolSelectionPolicy) []geniex_sdk.LlmChatMessage {
	directive := policy.directive()
	if directive == "" {
		return messages
	}
	result := make([]geniex_sdk.LlmChatMessage, 0, len(messages)+1)
	result = append(result, geniex_sdk.LlmChatMessage{Role: "system", Content: directive})
	return append(result, messages...)
}

func applyVlmToolSelectionDirective(messages []geniex_sdk.VlmChatMessage, policy toolSelectionPolicy) []geniex_sdk.VlmChatMessage {
	directive := policy.directive()
	if directive == "" {
		return messages
	}
	result := make([]geniex_sdk.VlmChatMessage, 0, len(messages)+1)
	result = append(result, geniex_sdk.VlmChatMessage{
		Role:     "system",
		Contents: []geniex_sdk.VlmContent{{Type: geniex_sdk.VlmContentTypeText, Text: directive}},
	})
	return append(result, messages...)
}

func parseTools(param ChatCompletionRequest) (toolSelectionPolicy, string, error) {
	policy := toolSelectionPolicy{mode: toolSelectionAuto}
	selectedTools := param.Tools

	switch {
	case len(param.rawToolChoice) > 0 && strings.HasPrefix(strings.TrimSpace(string(param.rawToolChoice)), "{"):
		var named struct {
			Type     string `json:"type"`
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		}
		if err := sonic.Unmarshal(param.rawToolChoice, &named); err != nil {
			return policy, "", fmt.Errorf("invalid named tool_choice: %w", err)
		}
		if named.Type != "function" || named.Function.Name == "" {
			return policy, "", errors.New("only named function tool_choice is supported")
		}
		policy.mode = toolSelectionNamed
		policy.namedTool = named.Function.Name
		selectedTools = nil
		for _, tool := range param.Tools {
			function := tool.GetFunction()
			if function != nil && function.Name == policy.namedTool {
				selectedTools = append(selectedTools, tool)
			}
		}
		if len(selectedTools) != 1 {
			return policy, "", fmt.Errorf("named tool_choice references unknown function %q", policy.namedTool)
		}
	case param.ToolChoice.OfAuto.Valid():
		switch param.ToolChoice.OfAuto.Value {
		case "", "auto":
			policy.mode = toolSelectionAuto
		case "none":
			policy.mode = toolSelectionNone
		case "required":
			policy.mode = toolSelectionRequired
		default:
			return policy, "", fmt.Errorf("unsupported tool_choice %q", param.ToolChoice.OfAuto.Value)
		}
	case param.ToolChoice.OfFunctionToolChoice != nil:
		policy.mode = toolSelectionNamed
		policy.namedTool = param.ToolChoice.OfFunctionToolChoice.Function.Name
		if policy.namedTool == "" {
			return policy, "", errors.New("named tool_choice requires a function name")
		}
		selectedTools = nil
		for _, tool := range param.Tools {
			function := tool.GetFunction()
			if function != nil && function.Name == policy.namedTool {
				selectedTools = append(selectedTools, tool)
			}
		}
		if len(selectedTools) != 1 {
			return policy, "", fmt.Errorf("named tool_choice references unknown function %q", policy.namedTool)
		}
	case param.ToolChoice.OfAllowedTools != nil || param.ToolChoice.OfCustomToolChoice != nil:
		return policy, "", errors.New("only function tool_choice is supported")
	}

	if len(param.Tools) == 0 {
		if policy.explicit() {
			return policy, "", errors.New("explicit tool_choice requires at least one function tool")
		}
		return toolSelectionPolicy{mode: toolSelectionNone}, "", nil
	}
	if policy.mode == toolSelectionNone {
		return policy, "", nil
	}
	for _, tool := range selectedTools {
		function := tool.GetFunction()
		if function == nil || function.Name == "" {
			return policy, "", errors.New("only named function tools are supported")
		}
		policy.allowedTools = append(policy.allowedTools, function.Name)
	}
	tools, err := sonic.MarshalString(selectedTools)
	return policy, tools, err
}
