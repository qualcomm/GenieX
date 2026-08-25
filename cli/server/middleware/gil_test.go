// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func serveWith(handler gin.HandlerFunc) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(GIL)
	r.GET("/", handler)
	r.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/", nil))
}

// A handler that registers a release hook has it run when the request releases
// the GIL — the keep-alive cache uses this to stamp its idle timer at request
// end, measuring idle from when generation finished (#1322).
func TestGILRunsReleaseHook(t *testing.T) {
	ran := false
	serveWith(func(*gin.Context) { RunOnRelease(func() { ran = true }) })

	if !ran {
		t.Fatal("release hook did not run")
	}
}

// A request that registers no hook clears the previous request's hook, so
// management endpoints like /api/model don't run a stale release action.
func TestGILClearsHookBetweenRequests(t *testing.T) {
	serveWith(func(*gin.Context) { RunOnRelease(func() {}) })
	serveWith(func(*gin.Context) {})

	if pendingRelease != nil {
		t.Fatal("pendingRelease not cleared after a request without a hook")
	}
}
