// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import (
	"context"
	"fmt"
	"time"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// LLM is the subset of the Go LLM API needed for one structured text decision.
type LLM interface {
	ApplyChatTemplate(geniex_sdk.LlmApplyChatTemplateInput) (*geniex_sdk.LlmApplyChatTemplateOutput, error)
	Generate(geniex_sdk.LlmGenerateInput) (*geniex_sdk.LlmGenerateOutput, error)
}

// LLMRequest is the caller-owned text prompt for one untrusted candidate.
type LLMRequest struct {
	System    string
	User      string
	Grammar   string
	MaxTokens int32
}

// LLMDecider creates constrained text-only structured candidates. It does not
// accept image paths and cannot execute, approve, or authorize side effects.
type LLMDecider struct {
	LLM       LLM
	RuntimeID string
}

// Decide returns raw model output for strict host-side parsing. A non-empty
// partial output is retained so callers can apply their bounded retry policy.
func (d LLMDecider) Decide(ctx context.Context, request LLMRequest) (string, error) {
	response, _, err := d.DecideWithTiming(ctx, request)
	return response, err
}

// DecideWithTiming returns raw model output and separates native SDK timing
// from host-side template work. It does not parse or validate the response.
func (d LLMDecider) DecideWithTiming(_ context.Context, request LLMRequest) (string, DecisionTiming, error) {
	started := time.Now()
	promptHash, promptVersion := promptProvenance(request.System, request.User)
	if d.LLM == nil {
		return "", DecisionTiming{}, fmt.Errorf("LLM is required")
	}
	if request.System == "" || request.User == "" {
		return "", DecisionTiming{}, fmt.Errorf("structured decision system and user prompts are required")
	}
	templateStarted := time.Now()
	template, err := d.LLM.ApplyChatTemplate(geniex_sdk.LlmApplyChatTemplateInput{
		Messages: []geniex_sdk.LlmChatMessage{
			{Role: geniex_sdk.LlmRoleSystem, Content: request.System},
			{Role: geniex_sdk.LlmRoleUser, Content: request.User},
		},
		EnableThink:         false,
		AddGenerationPrompt: true,
	})
	templateTime := time.Since(templateStarted)
	if err != nil {
		return "", DecisionTiming{TemplateTime: templateTime, TotalTime: time.Since(started), PromptHash: promptHash, PromptVersion: promptVersion}, fmt.Errorf("apply LLM chat template: %w", err)
	}
	maxTokens := request.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 128
	}
	sampler := &geniex_sdk.SamplerConfig{Temperature: 0}
	if d.RuntimeID == geniex_sdk.RuntimeLlamaCpp && request.Grammar != "" {
		sampler.GrammarString = request.Grammar
	}
	generationStarted := time.Now()
	response, err := d.LLM.Generate(geniex_sdk.LlmGenerateInput{
		PromptUTF8: template.FormattedText,
		Config:     &geniex_sdk.GenerationConfig{MaxTokens: maxTokens, SamplerConfig: sampler},
	})
	timing := DecisionTiming{TemplateTime: templateTime, GenerationTime: time.Since(generationStarted), TotalTime: time.Since(started), PromptHash: promptHash, PromptVersion: promptVersion}
	if response != nil {
		timing.ProfileData = response.ProfileData
	}
	if response != nil && response.FullText != "" {
		return response.FullText, timing, nil
	}
	if err != nil {
		return "", timing, fmt.Errorf("generate structured decision: %w", err)
	}
	return "", timing, fmt.Errorf("generate structured decision: empty response")
}
