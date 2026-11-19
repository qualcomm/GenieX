// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package geniex_sdk

import "testing"

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
}
