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

package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveHFToken_GenieXPreferredOverHFToken(t *testing.T) {
	t.Setenv("HF_TOKEN", "hf_token")

	got := resolveHFToken("geniex_token")
	if got != "geniex_token" {
		t.Fatalf("expected GENIEX token, got %q", got)
	}
}

func TestResolveHFToken_UsesHFTokenWhenGenieXMissing(t *testing.T) {
	t.Setenv("HF_TOKEN", "hf_token")

	got := resolveHFToken("")
	if got != "hf_token" {
		t.Fatalf("expected HF_TOKEN value, got %q", got)
	}
}

func TestResolveHFToken_UsesDefaultTokenFileWhenEnvMissing(t *testing.T) {
	home := t.TempDir()
	tokenPath := filepath.Join(home, ".cache", "huggingface", "token")
	if err := os.MkdirAll(filepath.Dir(tokenPath), 0o755); err != nil {
		t.Fatalf("failed to create token dir: %v", err)
	}
	if err := os.WriteFile(tokenPath, []byte("  file_token\n"), 0o644); err != nil {
		t.Fatalf("failed to write token file: %v", err)
	}

	t.Setenv("HF_TOKEN", "")
	t.Setenv("USERPROFILE", home)
	t.Setenv("HOME", home)

	got := resolveHFToken("")
	if got != "file_token" {
		t.Fatalf("expected token from file fallback, got %q", got)
	}
}

func TestResolveHFToken_EmptyWhenNoSourceAvailable(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HF_TOKEN", "")
	t.Setenv("USERPROFILE", home)
	t.Setenv("HOME", home)

	got := resolveHFToken("")
	if got != "" {
		t.Fatalf("expected empty token when no source is available, got %q", got)
	}
}

// TestGet_GenieXHFTokenBeatsHFToken verifies the full precedence chain through
// Get(): GENIEX_HFTOKEN (via viper) must beat HF_TOKEN. This guards against
// viper config drift silently breaking the documented order.
func TestGet_GenieXHFTokenBeatsHFToken(t *testing.T) {
	t.Setenv("GENIEX_HFTOKEN", "geniex_value")
	t.Setenv("HF_TOKEN", "hf_value")

	if got := Get().HFToken; got != "geniex_value" {
		t.Fatalf("expected geniex_value (GENIEX_HFTOKEN should win), got %q", got)
	}
}

// TestGet_HFTokenUsedWhenGenieXUnset verifies that Get() falls back to
// HF_TOKEN when GENIEX_HFTOKEN is unset, exercising the env path through viper.
func TestGet_HFTokenUsedWhenGenieXUnset(t *testing.T) {
	t.Setenv("GENIEX_HFTOKEN", "")
	t.Setenv("HF_TOKEN", "hf_value")

	if got := Get().HFToken; got != "hf_value" {
		t.Fatalf("expected hf_value via HF_TOKEN fallback, got %q", got)
	}
}
