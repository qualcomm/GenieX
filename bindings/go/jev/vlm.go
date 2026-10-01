// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import (
	"context"
	"fmt"
	"time"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// VLM is the subset of the Go VLM API needed for one structured decision. It
// permits deterministic host-side tests without loading a model.
type VLM interface {
	ApplyChatTemplate(geniex_sdk.VlmApplyChatTemplateInput) (*geniex_sdk.VlmApplyChatTemplateOutput, error)
	Generate(geniex_sdk.VlmGenerateInput) (*geniex_sdk.VlmGenerateOutput, error)
}

// VLMRequest is the caller-owned prompt and media for one untrusted candidate.
type VLMRequest struct {
	System          string
	User            string
	ImagePaths []string
	Grammar    string
	MaxTokens  int32
}

// VLMDecider creates constrained structured candidates. It has no execution,
// approval, browser, shell, or filesystem-management capability.
type VLMDecider struct {
	VLM       VLM
	RuntimeID string
}

// Decide returns raw model output for strict host-side parsing. A non-empty
// partial output is returned even when generation reports an error so callers
// can validate or retry it under their own bounded policy.
func (d VLMDecider) Decide(ctx context.Context, request VLMRequest) (string, error) {
	response, _, err := d.DecideWithTiming(ctx, request)
	return response, err
}

// DecideWithTiming returns raw model output and separates native SDK timing
// from host-side template work. It does not parse or validate the response.
func (d VLMDecider) DecideWithTiming(_ context.Context, request VLMRequest) (string, DecisionTiming, error) {
	started := time.Now()
	promptHash, promptVersion := promptProvenance(request.System, request.User)
	if d.VLM == nil {
		return "", DecisionTiming{}, fmt.Errorf("VLM is required")
	}
	if request.System == "" || request.User == "" {
		return "", DecisionTiming{}, fmt.Errorf("structured decision system and user prompts are required")
	}
	contents := []geniex_sdk.VlmContent{{Type: geniex_sdk.VlmContentTypeText, Text: request.User}}
	for _, imagePath := range request.ImagePaths {
		if imagePath == "" {
			return "", DecisionTiming{}, fmt.Errorf("structured decision image path is empty")
		}
		contents = append(contents, geniex_sdk.VlmContent{Type: geniex_sdk.VlmContentTypeImage, Text: imagePath})
	}
	templateStarted := time.Now()
	template, err := d.VLM.ApplyChatTemplate(geniex_sdk.VlmApplyChatTemplateInput{
		Messages: []geniex_sdk.VlmChatMessage{
			{Role: geniex_sdk.VlmRoleSystem, Contents: []geniex_sdk.VlmContent{{Type: geniex_sdk.VlmContentTypeText, Text: request.System}}},
			{Role: geniex_sdk.VlmRoleUser, Contents: contents},
		},
		EnableThink: false,
	})
	templateTime := time.Since(templateStarted)
	if err != nil {
		return "", DecisionTiming{TemplateTime: templateTime, TotalTime: time.Since(started), PromptHash: promptHash, PromptVersion: promptVersion}, fmt.Errorf("apply VLM chat template: %w", err)
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
	response, err := d.VLM.Generate(geniex_sdk.VlmGenerateInput{
		PromptUTF8: template.FormattedText,
		Config: &geniex_sdk.GenerationConfig{
			MaxTokens:     maxTokens,
			ImagePaths:    request.ImagePaths,
			SamplerConfig: sampler,
		},
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
