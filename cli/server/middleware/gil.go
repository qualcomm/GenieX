// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package middleware

import (
	"sync"

	"github.com/gin-gonic/gin"
)

// GILock serializes all API requests. The keep-alive sweeper shares it so a
// model is only freed while no request is in flight, never mid-generation
// (#1322).
var GILock sync.Mutex

// pendingRelease runs when the request releases the GIL; guarded by GILock,
// reset per request. See RunOnRelease.
var pendingRelease func()

func GIL(c *gin.Context) {
	// Block rather than fail so briefly queued requests don't 429.
	GILock.Lock()
	pendingRelease = nil
	defer func() {
		if pendingRelease != nil {
			pendingRelease()
		}
		GILock.Unlock()
	}()

	c.Next()
}

// RunOnRelease schedules f to run when the current request releases the GIL.
// Caller holds GILock (i.e. runs inside a request handler).
func RunOnRelease(f func()) {
	pendingRelease = f
}
