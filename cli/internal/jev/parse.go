// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import bindingjev "github.com/qualcomm/GenieX/bindings/go/jev"

// ParseAction is retained for CLI callers while the public decision SDK owns
// strict constrained-action parsing and generic shape validation.
func ParseAction(response string, observation Observation) (Action, error) {
	return bindingjev.ParseAction(response, observation)
}
