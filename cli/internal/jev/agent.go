// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

var (
	// ErrMaxSteps prevents unbounded browser activity.
	ErrMaxSteps = errors.New("browser agent reached its step limit")
	// ErrNoProgress prevents the same proposed action being repeated forever.
	ErrNoProgress = errors.New("browser agent made no progress")
	// ErrApprovalAborted identifies a human-selected abort separately from a
	// browser or inference failure.
	ErrApprovalAborted = errors.New("browser action was aborted by the user")
	// ErrStaleObservation means the trusted page state changed after an action
	// candidate was validated. The agent must observe and decide again rather
	// than dispatching against stale state.
	ErrStaleObservation = errors.New("browser observation is stale")
)

// Browser is deliberately minimal. The concrete CDP implementation must obtain
// a new observation for each step and revalidate a target before acting.
type Browser interface {
	Observe(context.Context) (Observation, error)
	Execute(context.Context, Action, Observation) (string, error)
	Close() error
}

// Decider receives a screenshot path in the observation and returns only a
// candidate JSON action. It never has direct browser access.
type Decider interface {
	DecideAction(context.Context, string, Observation, []string, string) (string, error)
}

// AgentConfig places hard bounds around a browser task.
type AgentConfig struct {
	Task          string
	MaxSteps      int
	MaxRetries    int
	ApprovalMode  ApprovalMode
	DryRun        bool
	NoProgressMax int
}

// Result contains the trusted agent ledger, never unvalidated model prose.
type Result struct {
	FinalAnswer string
	Steps       []string
}

// Agent coordinates untrusted model output, safety policy, and browser I/O.
type Agent struct {
	Browser  Browser
	Decider  Decider
	Approver ApprovalGate
	Config   AgentConfig
}

func (a *Agent) Run(ctx context.Context) (Result, error) {
	if a.Browser == nil || a.Decider == nil {
		return Result{}, fmt.Errorf("browser and decider are required")
	}
	if strings.TrimSpace(a.Config.Task) == "" {
		return Result{}, fmt.Errorf("browser task is required")
	}
	if a.Config.MaxSteps <= 0 {
		a.Config.MaxSteps = 25
	}
	if a.Config.MaxRetries < 0 {
		a.Config.MaxRetries = 0
	}
	if a.Config.NoProgressMax <= 0 {
		a.Config.NoProgressMax = 2
	}

	var result Result
	previous := ""
	noProgress := 0
	for step := 0; step < a.Config.MaxSteps; step++ {
		observation, err := a.Browser.Observe(ctx)
		if err != nil {
			return result, fmt.Errorf("observe browser: %w", err)
		}

		var action Action
		var correction string
		valid := false
		for retry := 0; retry <= a.Config.MaxRetries; retry++ {
			response, err := a.Decider.DecideAction(ctx, a.Config.Task, observation, result.Steps, correction)
			if err != nil {
				return result, fmt.Errorf("decide browser action: %w", err)
			}
			action, err = ParseAction(response, observation)
			if err == nil {
				valid = true
				break
			}
			correction = err.Error()
		}
		if !valid {
			return result, fmt.Errorf("agent did not produce a valid action after %d retries: %s", a.Config.MaxRetries+1, correction)
		}

		fingerprint := actionFingerprint(action, observation.URL)
		if fingerprint == previous {
			noProgress++
			if noProgress >= a.Config.NoProgressMax {
				return result, ErrNoProgress
			}
		} else {
			previous = fingerprint
			noProgress = 0
		}

		if action.Action == ActionFinish {
			result.FinalAnswer = action.FinalAnswer
			return result, nil
		}

		var target *Element
		if action.Index != nil {
			element, ok := observation.Element(*action.Index)
			if !ok {
				return result, fmt.Errorf("validated element %d disappeared before approval", *action.Index)
			}
			target = &element
		}
		if required, reason := ApprovalRequired(a.Config.ApprovalMode, action, observation); required {
			if a.Approver == nil {
				return result, fmt.Errorf("approval required for %s but no approval gate is configured", action.Action)
			}
			approval, err := a.Approver.Approve(ApprovalRequest{Action: action, Target: target, Reason: reason})
			if err != nil {
				return result, fmt.Errorf("approve browser action: %w", err)
			}
			switch approval {
			case ApprovalAbort:
				return result, ErrApprovalAborted
			case ApprovalSkip:
				result.Steps = append(result.Steps, "skipped "+string(action.Action)+" (approval denied)")
				continue
			case ApprovalApprove:
			default:
				return result, fmt.Errorf("invalid approval decision %q", approval)
			}
		}

		if a.Config.DryRun {
			result.Steps = append(result.Steps, "dry-run "+describeAction(action, target))
			continue
		}
		outcome, err := a.Browser.Execute(ctx, action, observation)
		if err != nil {
			if errors.Is(err, ErrStaleObservation) {
				// A new page state requires a new decision; it is not evidence the
				// model repeated an action without progress.
				previous = ""
				noProgress = 0
				result.Steps = append(result.Steps, "re-observe stale "+string(action.Action))
				continue
			}
			return result, fmt.Errorf("execute %s: %w", action.Action, err)
		}
		result.Steps = append(result.Steps, describeAction(action, target, outcome))
	}
	return result, ErrMaxSteps
}

func actionFingerprint(action Action, pageURL string) string {
	return fmt.Sprintf("%s|%d|%s|%s|%t|%s", action.Action, dereference(action.Index), action.URL, action.Text, action.Submit, pageURL)
}

func dereference(value *int) int {
	if value == nil {
		return -1
	}
	return *value
}

func describeAction(action Action, target *Element, outcomes ...string) string {
	description := string(action.Action)
	if target != nil {
		description += fmt.Sprintf(" %s[%d]", target.Tag, target.Index)
	}
	if action.Action == ActionType {
		description += " " + RedactedText(action, target)
	}
	if len(outcomes) > 0 && outcomes[0] != "" {
		description += ": " + RedactSensitiveText(outcomes[0], target)
	}
	return description
}
