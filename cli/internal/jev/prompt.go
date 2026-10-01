// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import bindingjev "github.com/qualcomm/GenieX/bindings/go/jev"

// SystemPrompt is the public fixed instruction for model decisions.
func SystemPrompt() string { return bindingjev.SystemPrompt() }

// Grammar is the public constrained-action grammar used with llama.cpp.
const Grammar = bindingjev.Grammar

// RoundPrompt is retained for CLI callers while the public decision SDK owns
// trusted observation rendering.
func RoundPrompt(task string, observation Observation, ledger []string, correction string) (string, error) {
	return bindingjev.RoundPrompt(task, observation, ledger, correction)
}
