// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package common

import (
	"errors"
	"fmt"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// InitSDK calls the SDK init and maps its generic NOT_SUPPORTED result to the
// CPU-specific sentinel so callers surface hintCPUUnsupported. Other init
// failures are returned unchanged.
func InitSDK() error {
	err := geniex_sdk.Init()
	if errors.Is(err, geniex_sdk.ErrCommonNotSupport) {
		// Wrap with %v, not %w: the SDK returns the generic NOT_SUPPORTED code
		// for both this and the unsupported-model-type case, so keeping it in
		// the Is-chain would match hintNotSupport first. ErrCPUUnsupported is
		// the only sentinel we want PrintError to see here.
		return fmt.Errorf("%w: %v", ErrCPUUnsupported, err)
	}
	return err
}
