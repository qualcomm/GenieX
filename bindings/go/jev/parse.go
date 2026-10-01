// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

// ParseAction decodes exactly one constrained action. A response may contain one
// fenced JSON object, but prose outside it and unknown fields are rejected.
func ParseAction(response string, observation Observation) (Action, error) {
	var action Action
	if err := DecodeStructured(response, &action, func() error {
		return action.Validate(observation)
	}); err != nil {
		return Action{}, err
	}
	return action, nil
}
