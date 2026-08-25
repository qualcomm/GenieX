// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package common

import (
	"errors"
	"testing"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// hintFor returns the hint errorHints would render for err, or "" if none match.
// Mirrors PrintError's first-match-wins loop.
func hintFor(err error) string {
	for _, h := range errorHints {
		if errors.Is(err, h.sentinel) {
			return h.hint
		}
	}
	return ""
}

// The SDK returns the same NOT_SUPPORTED code for an unsupported model type and
// for a CPU-feature failure at init. InitSDK maps the init case to
// ErrCPUUnsupported so it resolves to the CPU hint, not the model-type hint, even
// though ErrCommonNotSupport is listed before it in errorHints.
func TestCPUUnsupportedHintWins(t *testing.T) {
	if got := hintFor(ErrCPUUnsupported); got != hintCPUUnsupported {
		t.Fatalf("ErrCPUUnsupported resolved to wrong hint:\n%s", got)
	}
	// A bare NOT_SUPPORTED (e.g. unsupported model type) must still get its own hint.
	if got := hintFor(geniex_sdk.ErrCommonNotSupport); got != hintNotSupport {
		t.Fatalf("bare NOT_SUPPORTED resolved to wrong hint:\n%s", got)
	}
}
