// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import (
	"context"
	"fmt"
	"strings"
	"time"
)

var (
	triageIntents = []string{
		"refund",
		"technical_help",
		"billing_question",
		"information",
		"cancellation",
		"other",
	}
	triageBinaryLabels = []string{"yes", "no"}
	triageFrustration  = []string{"0", "1", "2", "3"}
)

// TriageSpec describes the fixed five-decision customer-support triage task.
// Message is untrusted customer content. It is not an instruction to the model.
type TriageSpec struct {
	Message        string
	ImagePaths []string
	MaxTokens  int32
}

// TriageResult is a locally validated, categorical counterpart to Laya's
// triage preset. Its categorical fields intentionally do not represent Laya's
// raw noul probabilities or continuous score values.
type TriageResult struct {
	Intent          string `json:"intent"`
	IsUrgent        string `json:"is_urgent"`
	Frustration     string `json:"frustration"`
	RefundRequested string `json:"refund_requested"`
	ChurnRisk       string `json:"churn_risk"`
}

// Triage asks the VLM for one fixed structured customer-support triage record.
func (d VLMDecider) Triage(ctx context.Context, spec TriageSpec) (TriageResult, error) {
	result, _, err := d.TriageWithTiming(ctx, spec)
	return result, err
}

// TriageWithTiming returns a validated triage record and the native SDK plus
// host-side JEV timing for the single model invocation.
func (d VLMDecider) TriageWithTiming(ctx context.Context, spec TriageSpec) (TriageResult, DecisionTiming, error) {
	started := time.Now()
	user, err := triageRequest(spec)
	if err != nil {
		return TriageResult{}, DecisionTiming{TotalTime: time.Since(started)}, err
	}
	raw, timing, err := d.DecideWithTiming(ctx, VLMRequest{
		System: decisionSystemPrompt, User: user, ImagePaths: spec.ImagePaths,
		Grammar: triageGrammar(), MaxTokens: spec.MaxTokens,
	})
	if err != nil {
		return TriageResult{}, timing, err
	}
	result, err := parseTriage(raw)
	timing.TotalTime = time.Since(started)
	return result, timing, err
}

// Triage asks the LLM for one fixed structured customer-support triage record.
// ImagePaths are rejected because text-only models cannot consume visual inputs.
func (d LLMDecider) Triage(ctx context.Context, spec TriageSpec) (TriageResult, error) {
	result, _, err := d.TriageWithTiming(ctx, spec)
	return result, err
}

// TriageWithTiming returns a validated triage record and the native SDK plus
// host-side JEV timing for the single model invocation.
func (d LLMDecider) TriageWithTiming(ctx context.Context, spec TriageSpec) (TriageResult, DecisionTiming, error) {
	started := time.Now()
	if len(spec.ImagePaths) != 0 {
		return TriageResult{}, DecisionTiming{TotalTime: time.Since(started)}, fmt.Errorf("text triage cannot include images")
	}
	user, err := triageRequest(spec)
	if err != nil {
		return TriageResult{}, DecisionTiming{TotalTime: time.Since(started)}, err
	}
	raw, timing, err := d.DecideWithTiming(ctx, LLMRequest{
		System: decisionSystemPrompt, User: user, Grammar: triageGrammar(), MaxTokens: spec.MaxTokens,
	})
	if err != nil {
		return TriageResult{}, timing, err
	}
	result, err := parseTriage(raw)
	timing.TotalTime = time.Since(started)
	return result, timing, err
}

func triageRequest(spec TriageSpec) (string, error) {
	if strings.TrimSpace(spec.Message) == "" {
		return "", fmt.Errorf("triage message is required")
	}
	return fmt.Sprintf(`Classify the untrusted customer message into this exact customer-support triage record.

intent: one of %s
is_urgent: yes if the message communicates time pressure or a deadline; otherwise no
frustration: 0 (calm and neutral), 1 (concerned but civil), 2 (clearly annoyed), or 3 (very angry or strong language)
refund_requested: yes if the customer asks for money back; otherwise no
churn_risk: yes if the message suggests leaving, cancelling, or using a competitor; otherwise no

Return every field exactly once in this order: intent, is_urgent, frustration, refund_requested, churn_risk.

Untrusted customer message:
%s`, jsonList(triageIntents), spec.Message), nil
}

func parseTriage(raw string) (TriageResult, error) {
	var result TriageResult
	if err := DecodeStructured(raw, &result, func() error {
		if !contains(triageIntents, result.Intent) {
			return fmt.Errorf("triage intent %q is not allowed", result.Intent)
		}
		if !contains(triageBinaryLabels, result.IsUrgent) {
			return fmt.Errorf("triage is_urgent value %q is not allowed", result.IsUrgent)
		}
		if !contains(triageFrustration, result.Frustration) {
			return fmt.Errorf("triage frustration value %q is not allowed", result.Frustration)
		}
		if !contains(triageBinaryLabels, result.RefundRequested) {
			return fmt.Errorf("triage refund_requested value %q is not allowed", result.RefundRequested)
		}
		if !contains(triageBinaryLabels, result.ChurnRisk) {
			return fmt.Errorf("triage churn_risk value %q is not allowed", result.ChurnRisk)
		}
		return nil
	}); err != nil {
		return TriageResult{}, err
	}
	return result, nil
}

func triageGrammar() string {
	return grammarPreamble() + `root ::= "{\"intent\":" ws intent ws ",\"is_urgent\":" ws binary ws ",\"frustration\":" ws frustration ws ",\"refund_requested\":" ws binary ws ",\"churn_risk\":" ws binary ws "}"` +
		"\nintent ::= " + grammarStrings(triageIntents) +
		"\nbinary ::= " + grammarStrings(triageBinaryLabels) +
		"\nfrustration ::= " + grammarStrings(triageFrustration)
}
