// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import (
	"context"
	"fmt"
)

// DecideAction formats a fresh visual observation and asks the VLM for one
// candidate browser action. It does not parse, validate, approve, or execute
// the response. Browser hosts retain all side-effect authority.
func (d VLMDecider) DecideAction(ctx context.Context, task string, observation Observation, ledger []string, correction string) (string, error) {
	if observation.ScreenshotPath == "" {
		return "", fmt.Errorf("observation screenshot path is required")
	}
	promptText, err := RoundPrompt(task, observation, ledger, correction)
	if err != nil {
		return "", err
	}
	return d.Decide(ctx, VLMRequest{
		System:         SystemPrompt(),
		User:           promptText,
		ImagePaths:     []string{observation.ScreenshotPath},
		Grammar:   Grammar,
		MaxTokens: 256,
	})
}
