// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package browser

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/qualcomm/GenieX/cli/internal/jev"
)

func TestStartBrowserRejectsMissingExecutable(t *testing.T) {
	if _, _, err := startBrowser(t.Context(), filepath.Join(t.TempDir(), "missing-browser")); err == nil {
		t.Fatal("startBrowser() accepted a missing executable")
	}
}

func TestLaunchArgsRestrictsCDPOrigin(t *testing.T) {
	args := launchArgs(9222, "profile", true, "https://example.test")
	joined := strings.Join(args, "\n")
	if !strings.Contains(joined, "--remote-allow-origins="+cdpClientOrigin) {
		t.Fatalf("launch args do not allow the client origin: %#v", args)
	}
	if strings.Contains(joined, "--remote-allow-origins=*") {
		t.Fatalf("launch args allow every origin: %#v", args)
	}
}

func TestValidateAttachURL(t *testing.T) {
	for _, test := range []struct {
		name string
		raw  string
		ok   bool
	}{
		{"IPv4 HTTP", "http://127.0.0.1:9222", true},
		{"IPv4 WebSocket", "ws://127.0.0.1:9222/devtools/page/1", true},
		{"IPv6 HTTPS", "https://[::1]:9222", true},
		{"IPv6 secure WebSocket", "wss://[::1]:9222/devtools/page/1", true},
		{"localhost", "http://localhost:9222", true},
		{"loopback range", "http://127.1.2.3:9222", true},
		{"remote host", "http://example.com:9222", false},
		{"remote IP", "ws://192.0.2.1:9222", false},
		{"unspecified IPv4", "http://0.0.0.0:9222", false},
		{"unspecified IPv6", "http://[::]:9222", false},
		{"unsupported scheme", "file:///tmp/debug", false},
		{"relative URL", "/json/list", false},
		{"credentials", "http://user:password@127.0.0.1:9222", false},
		{"malformed", "http://[::1", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := validateAttachURL(test.raw)
			if (err == nil) != test.ok {
				t.Fatalf("validateAttachURL(%q) error = %v, want success=%v", test.raw, err, test.ok)
			}
		})
	}
}

func TestObservationOutputDirLifecycle(t *testing.T) {
	t.Run("temporary directory is removed", func(t *testing.T) {
		browser := &Browser{}
		dir, err := browser.observationOutputDir()
		if err != nil {
			t.Fatalf("observationOutputDir() error = %v", err)
		}
		if browser.removeObservationDir != true {
			t.Fatal("temporary observation directory is not owned")
		}
		if again, err := browser.observationOutputDir(); err != nil || again != dir {
			t.Fatalf("observationOutputDir() = (%q, %v), want (%q, nil)", again, err, dir)
		}
		if err := os.WriteFile(filepath.Join(dir, "screenshot.png"), []byte("image"), 0o600); err != nil {
			t.Fatalf("write temporary screenshot: %v", err)
		}
		if err := browser.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if _, err := os.Stat(dir); !os.IsNotExist(err) {
			t.Fatalf("temporary observation directory still exists: %v", err)
		}
	})

	t.Run("explicit trace directory is retained", func(t *testing.T) {
		dir := t.TempDir()
		browser := &Browser{traceDir: dir}
		got, err := browser.observationOutputDir()
		if err != nil || got != dir {
			t.Fatalf("observationOutputDir() = (%q, %v), want (%q, nil)", got, err, dir)
		}
		file := filepath.Join(dir, "screenshot.png")
		if err := os.WriteFile(file, []byte("image"), 0o600); err != nil {
			t.Fatalf("write trace screenshot: %v", err)
		}
		if err := browser.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if _, err := os.Stat(file); err != nil {
			t.Fatalf("explicit trace screenshot was removed: %v", err)
		}
	})
}

func TestExecuteReadWithoutIndexFails(t *testing.T) {
	browser := &Browser{}
	if _, err := browser.Execute(t.Context(), jev.Action{Action: jev.ActionRead}, jev.Observation{}); err == nil {
		t.Fatal("Execute() accepted an unindexed read")
	}
}

func TestDecodeEvaluateResult(t *testing.T) {
	for _, test := range []struct {
		name  string
		body  string
		value string
		stale bool
		ok    bool
	}{
		{"value", `{"result":{"value":"clicked"}}`, `"clicked"`, false, true},
		{"stale", `{"result":{"type":"undefined"},"exceptionDetails":{"text":"Uncaught","exception":{"description":"Error: target is stale"}}}`, "", true, false},
		{"occluded", `{"result":{"type":"undefined"},"exceptionDetails":{"text":"Uncaught","exception":{"description":"Error: target is occluded"}}}`, "", true, false},
		{"other exception", `{"result":{"type":"undefined"},"exceptionDetails":{"text":"Uncaught","exception":{"description":"TypeError: failed"}}}`, "", false, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			value, err := decodeEvaluateResult([]byte(test.body))
			if (err == nil) != test.ok {
				t.Fatalf("decodeEvaluateResult() error = %v, want success=%v", err, test.ok)
			}
			if errors.Is(err, jev.ErrStaleObservation) != test.stale {
				t.Fatalf("errors.Is(err, ErrStaleObservation) = %v, want %v", errors.Is(err, jev.ErrStaleObservation), test.stale)
			}
			if test.ok && string(value) != test.value {
				t.Fatalf("value = %s, want %s", value, test.value)
			}
		})
	}
}

func TestSelectPageTarget(t *testing.T) {
	targets := []targetInfo{
		{Type: "page", URL: "about:blank", WebSocketDebuggerURL: "ws://blank"},
		{Type: "page", URL: "https://example.test/start", WebSocketDebuggerURL: "ws://requested"},
	}
	if got := selectPageTarget(targets, false, "https://example.test/start"); got != "ws://requested" {
		t.Fatalf("launched target = %q", got)
	}
	if got := selectPageTarget(targets, true, "https://example.test/start"); got != "ws://blank" {
		t.Fatalf("attached target = %q", got)
	}
	if got := selectPageTarget(targets, false, ""); got != "ws://blank" {
		t.Fatalf("default target = %q", got)
	}
}

func TestSelectPageTargetWaitsForRequestedLaunch(t *testing.T) {
	targets := []targetInfo{{Type: "page", URL: "about:blank", WebSocketDebuggerURL: "ws://blank"}}
	if got := selectPageTarget(targets, false, "https://example.test/start"); got != "" {
		t.Fatalf("target = %q, want no target while initial URL is pending", got)
	}
}
