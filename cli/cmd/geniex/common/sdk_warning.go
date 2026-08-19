// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package common

import (
	"fmt"
	"os"
	"sync"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"

	"github.com/qualcomm/GenieX/cli/internal/render"
)

// lastPrinted dedups identical warnings within one CLI invocation. Both
// `pull` (query then pull) and `pickChipset` (list-chipsets then detect)
// hit the same aihub resolve path twice, so without this the same
// "release pointer unreachable, using v0.60.0..." line would print twice.
var (
	lastPrintedMu  sync.Mutex
	lastPrintedMsg string
)

// FlushSDKWarning prints any user-facing warning left by the most recent
// geniex_sdk call on this goroutine's OS thread to stderr, decorated with
// the theme's Warning colour. Bypasses slog so it surfaces at the default
// log level ("none"), where SDK log warnings are suppressed.
//
// Silent when the SDK didn't flag anything or the same message was
// already printed earlier in this process, so it's safe to call after
// every SDK entry point unconditionally. Call it AFTER any spinner /
// progress bar has been stopped so the warning line isn't overwritten.
func FlushSDKWarning() {
	w := geniex_sdk.ModelLastWarningMessage()
	if w == "" {
		return
	}
	lastPrintedMu.Lock()
	defer lastPrintedMu.Unlock()
	if w == lastPrintedMsg {
		return
	}
	lastPrintedMsg = w
	fmt.Fprintln(os.Stderr, render.GetTheme().Warning.Sprintf("⚠  %s", w))
}
