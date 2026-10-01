// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import (
	"crypto/sha256"
	"encoding/hex"
	"time"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// DecisionTiming separates SDK-reported generation timing from host-side JEV
// work. ProfileData measures native inference only; the wall-clock fields also
// include template formatting and local structured-output validation.
type DecisionTiming struct {
	ProfileData    geniex_sdk.ProfileData
	TemplateTime   time.Duration
	GenerationTime time.Duration
	TotalTime      time.Duration
	// PromptHash identifies the exact system/user prompt pair without retaining
	// its potentially untrusted or sensitive content. PromptVersion changes when
	// the JEV prompt contract changes.
	PromptHash    string
	PromptVersion string
}

const promptVersion = "jev-structured-v1"

func promptProvenance(system, user string) (string, string) {
	sum := sha256.Sum256([]byte(system + "\x00" + user))
	return hex.EncodeToString(sum[:]), promptVersion
}
