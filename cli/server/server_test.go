// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestHostBindingHint(t *testing.T) {
	tests := []struct {
		name     string
		host     string
		wantHint bool
		wantPort string // if wantHint, the port that should appear in the message
	}{
		{"ipv4 loopback", "127.0.0.1:18181", true, "18181"},
		{"ipv4 loopback other subnet", "127.0.1.5:8080", true, "8080"},
		{"localhost", "localhost:18181", true, "18181"},
		{"localhost uppercase", "LOCALHOST:18181", true, "18181"},
		{"ipv6 loopback", "[::1]:18181", true, "18181"},
		{"any ipv4", "0.0.0.0:18181", false, ""},
		{"any ipv6", "[::]:18181", false, ""},
		{"lan ipv4", "192.168.1.10:18181", false, ""},
		{"malformed no port", "127.0.0.1", false, ""},
		{"empty", "", false, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hostBindingHint(tt.host)
			if tt.wantHint && got == "" {
				t.Errorf("hostBindingHint(%q) = %q, want non-empty hint", tt.host, got)
			}
			if !tt.wantHint && got != "" {
				t.Errorf("hostBindingHint(%q) = %q, want empty", tt.host, got)
			}
			if tt.wantHint && !strings.Contains(got, "--host 0.0.0.0:"+tt.wantPort) {
				t.Errorf("hostBindingHint(%q) = %q, want it to suggest --host 0.0.0.0:%s", tt.host, got, tt.wantPort)
			}
		})
	}
}

func TestRegisterRootServesUnmodifiedWebUIAndCompatibilityProps(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "index.html"), []byte("upstream-ui"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "asset.js"), []byte("upstream-asset"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GENIEX_WEBUIDIR", dir)
	t.Setenv("GENIEX_NCTX", "4096")
	t.Setenv("GENIEX_MODEL", "local/test-model:Q4_0")

	gin.SetMode(gin.TestMode)
	engine := gin.New()
	if err := RegisterRoot(engine); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		path string
		want string
	}{
		{path: "/", want: "upstream-ui"},
		{path: "/asset.js", want: "upstream-asset"},
		{path: "/chat/example", want: "upstream-ui"},
	} {
		recorder := httptest.NewRecorder()
		engine.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, tc.path, nil))
		if recorder.Code != http.StatusOK || recorder.Body.String() != tc.want {
			t.Fatalf("GET %s = (%d, %q), want (200, %q)", tc.path, recorder.Code, recorder.Body.String(), tc.want)
		}
	}

	propsRecorder := httptest.NewRecorder()
	engine.ServeHTTP(propsRecorder, httptest.NewRequest(http.MethodGet, "/props", nil))
	var props map[string]any
	if err := json.Unmarshal(propsRecorder.Body.Bytes(), &props); err != nil {
		t.Fatal(err)
	}
	if propsRecorder.Code != http.StatusOK || props["role"] != "model" || props["model_alias"] != "local/test-model:Q4_0" {
		t.Fatalf("unexpected /props response: %d %s", propsRecorder.Code, propsRecorder.Body.String())
	}

	missingAPI := httptest.NewRecorder()
	engine.ServeHTTP(missingAPI, httptest.NewRequest(http.MethodGet, "/v1/missing", nil))
	if missingAPI.Code != http.StatusNotFound {
		t.Fatalf("unknown API route returned %d, want 404", missingAPI.Code)
	}
}

func TestRegisterRootRejectsMissingWebUIBuild(t *testing.T) {
	t.Setenv("GENIEX_WEBUIDIR", t.TempDir())
	if err := RegisterRoot(gin.New()); err == nil {
		t.Fatal("expected missing index.html to fail")
	}
}
