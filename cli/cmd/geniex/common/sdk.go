// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package common

import (
	"errors"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// InitSDK calls the SDK init and maps its generic NOT_SUPPORTED result to the
// CPU-specific sentinel so callers surface hintCPUUnsupported. Other init
// failures are returned unchanged.
func InitSDK() error {
	err := geniex_sdk.Init()
	if errors.Is(err, geniex_sdk.ErrCommonNotSupport) {
		// Return ErrCPUUnsupported alone, not wrapping the SDK error: the SDK
		// uses the same NOT_SUPPORTED code for the unsupported-model-type case,
		// so keeping it in the Is-chain would match hintNotSupport first.
		return ErrCPUUnsupported
	}
	return err
}
