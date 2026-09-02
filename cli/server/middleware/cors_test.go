// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package middleware

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestCORSAllowsManagedCacheHeaders(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	context, _ := gin.CreateTestContext(recorder)
	context.Request = httptest.NewRequest(http.MethodOptions, "/v1/chat/completions", nil)

	CORS(context)

	if recorder.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNoContent)
	}
	allowed := recorder.Header().Get("Access-Control-Allow-Headers")
	for _, header := range []string{"GenieX-KeepCache", "GenieX-Cache-Session", "GenieX-Cache-Parent"} {
		if !strings.Contains(allowed, header) {
			t.Errorf("Access-Control-Allow-Headers = %q, missing %q", allowed, header)
		}
	}
	if exposed := recorder.Header().Get("Access-Control-Expose-Headers"); !strings.Contains(exposed, "GenieX-Cache-Protocol") {
		t.Errorf("Access-Control-Expose-Headers = %q, missing protocol header", exposed)
	}
}
