// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model_hub

import (
	"io"
	"os"
	"strings"
	"sync"
	"testing"
)

func TestHFMaxConcurrency_WarnsWithoutToken(t *testing.T) {
	home := t.TempDir()
	t.Setenv("GENIEX_HFTOKEN", "")
	t.Setenv("HF_TOKEN", "")
	t.Setenv("USERPROFILE", home)
	t.Setenv("HOME", home)

	hfTokenWarnOnce = sync.Once{}
	h := NewHuggingFace()
	out := captureStdout(t, func() {
		if got := h.MaxConcurrency(); got != 1 {
			t.Fatalf("expected concurrency 1 without token, got %d", got)
		}
	})

	if !strings.Contains(out, "Cannot find a HuggingFace token") {
		t.Fatalf("expected missing-token warning, got %q", out)
	}

	out2 := captureStdout(t, func() {
		h.MaxConcurrency()
	})
	if out2 != "" {
		t.Fatalf("expected warning to be emitted once, got %q", out2)
	}
}

func TestHFMaxConcurrency_NoWarningWithToken(t *testing.T) {
	t.Setenv("GENIEX_HFTOKEN", "geniex_token")
	t.Setenv("HF_TOKEN", "")

	hfTokenWarnOnce = sync.Once{}
	h := NewHuggingFace()
	out := captureStdout(t, func() {
		if got := h.MaxConcurrency(); got != 8 {
			t.Fatalf("expected concurrency 8 with token, got %d", got)
		}
	})

	if out != "" {
		t.Fatalf("expected no warning with token, got %q", out)
	}
}

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()

	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe error: %v", err)
	}

	os.Stdout = w
	defer func() { os.Stdout = orig }()
	fn()
	_ = w.Close()

	buf, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read stdout error: %v", err)
	}
	_ = r.Close()
	return string(buf)
}
