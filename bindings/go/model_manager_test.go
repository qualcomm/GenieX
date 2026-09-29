// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package geniex_sdk

import "testing"

// No ModelInit: the alias table is static, so no store is involved.
func TestResolveAlias(t *testing.T) {
	got, ok := ResolveAlias("qwen3")
	if !ok {
		t.Fatal("qwen3 is the SDK's alias fixture")
	}
	if want := "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	for _, alias := range []string{"", "no-such-alias-xyz-9999", "ggml-org/Qwen3-1.7B-GGUF"} {
		if got, ok := ResolveAlias(alias); ok {
			t.Errorf("ResolveAlias(%q): expected miss, got %q", alias, got)
		}
	}
}

func TestSplitNamePrecision(t *testing.T) {
	cases := []struct{ arg, name, precision string }{
		{"org/repo:Q4_0", "org/repo", "Q4_0"},
		{"org/repo", "org/repo", ""},
		{"org/repo:N/A", "org/repo", "N/A"},
		{"docker.io/ai/gemma3:latest", "docker.io/ai/gemma3", "latest"},
		{
			"https://modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF:Q4_K_M",
			"https://modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF",
			"Q4_K_M",
		},
		{
			"https://huggingface.co/Qwen/Qwen3-0.6B-GGUF",
			"https://huggingface.co/Qwen/Qwen3-0.6B-GGUF",
			"",
		},
	}
	for _, c := range cases {
		name, precision := SplitNamePrecision(c.arg)
		if name != c.name || precision != c.precision {
			t.Errorf("SplitNamePrecision(%q) = (%q, %q), want (%q, %q)",
				c.arg, name, precision, c.name, c.precision)
		}
	}
}

func TestJoinNamePrecision(t *testing.T) {
	cases := []struct{ name, precision, want string }{
		{"org/repo", "Q4_0", "org/repo:Q4_0"},
		{"org/repo", "", "org/repo"},
		// N/A is a manifest key like any other; hiding it is the caller's job.
		{"org/repo", PrecisionNA, "org/repo:N/A"},
	}
	for _, c := range cases {
		if got := JoinNamePrecision(c.name, c.precision); got != c.want {
			t.Errorf("JoinNamePrecision(%q, %q) = %q, want %q", c.name, c.precision, got, c.want)
		}
	}
	// Round-trips with the splitter it mirrors.
	name, precision := SplitNamePrecision("org/repo:Q4_0")
	if got := JoinNamePrecision(name, precision); got != "org/repo:Q4_0" {
		t.Errorf("round-trip: got %q", got)
	}
}
