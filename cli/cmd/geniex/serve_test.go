// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/spf13/viper"

	"github.com/qualcomm/GenieX/cli/internal/config"
)

func resetServeTestViper(t *testing.T) {
	t.Helper()
	viper.Reset()
	viper.SetDefault("hftoken", "")
	viper.SetDefault("log", "none")
	viper.SetEnvPrefix("geniex")
	viper.AutomaticEnv()
	t.Cleanup(func() {
		viper.Reset()
		viper.SetDefault("hftoken", "")
		viper.SetDefault("log", "none")
		viper.SetEnvPrefix("geniex")
		viper.AutomaticEnv()
	})
}

func TestServeFlagGroups(t *testing.T) {
	resetServeTestViper(t)
	cmd := serve()
	var out bytes.Buffer
	cmd.SetOut(&out)
	if err := cmd.Usage(); err != nil {
		t.Fatal(err)
	}
	usage := out.String()
	server := strings.Index(usage, "Server Flags:")
	defaults := strings.Index(usage, "Model Default Flags:")
	if server < 0 || defaults <= server {
		t.Fatalf("serve help missing ordered flag groups: %s", usage)
	}
	serverFlags, defaultFlags := usage[server:defaults], usage[defaults+len("Model Default Flags:"):]
	for _, flag := range []string{"host", "origins", "keepalive", "metrics", "qairt-lib", "https", "certfile", "keyfile", "help"} {
		if !strings.Contains(serverFlags, "--"+flag) || strings.Contains(defaultFlags, "--"+flag) {
			t.Errorf("--%s must appear only in Server Flags", flag)
		}
	}
	for _, flag := range []string{"model", "nctx", "ngl", "compute", "vit-compute", "power-mode", "spec-type", "draft-model", "draft-tokens", "draft-min", "draft-p-min", "think"} {
		if !strings.Contains(defaultFlags, "--"+flag) || strings.Contains(serverFlags, "--"+flag) {
			t.Errorf("--%s must appear only in Model Default Flags", flag)
		}
	}
	if strings.Contains(serverFlags, "Default server address") || strings.Contains(serverFlags, "Default CORS origins") {
		t.Errorf("host and origins are server settings, not model defaults: %s", serverFlags)
	}
	if strings.Contains(defaultFlags, "Default ") {
		t.Errorf("Model Default Flags descriptions must not repeat Default: %s", defaultFlags)
	}
}

func TestServeModelDefault(t *testing.T) {
	resetServeTestViper(t)
	t.Setenv("GENIEX_MODEL", "")
	cmd := serve()
	if flag := cmd.Flags().Lookup("model"); flag == nil || flag.DefValue != "" {
		t.Fatalf("serve --model should default to empty: %v", flag)
	}
	if got := config.Get().Model; got != "" {
		t.Errorf("unconfigured model = %q, want empty", got)
	}
	t.Setenv("GENIEX_MODEL", "local/env:Q4_0")
	if got := config.Get().Model; got != "local/env:Q4_0" {
		t.Errorf("model from environment = %q", got)
	}
	if err := cmd.Flags().Set("model", "local/flag:Q4_0"); err != nil {
		t.Fatal(err)
	}
	if got := config.Get().Model; got != "local/flag:Q4_0" {
		t.Errorf("--model must override GENIEX_MODEL, got %q", got)
	}
}

func TestServeThinkDefault(t *testing.T) {
	resetServeTestViper(t)
	t.Setenv("GENIEX_THINK", "")
	cmd := serve()
	if flag := cmd.Flags().Lookup("think"); flag == nil || flag.DefValue != "true" {
		t.Fatalf("serve --think should default to true: %v", flag)
	}
	if !config.Get().Think {
		t.Error("unconfigured think = false, want true")
	}
	t.Setenv("GENIEX_THINK", "false")
	if config.Get().Think {
		t.Error("GENIEX_THINK=false should disable thinking")
	}
	if err := cmd.Flags().Set("think", "true"); err != nil {
		t.Fatal(err)
	}
	if !config.Get().Think {
		t.Error("--think must override GENIEX_THINK")
	}
}

func TestServeSpeculativeDecodingFlags(t *testing.T) {
	resetServeTestViper(t)
	for _, name := range []string{"GENIEX_SPEC_TYPE", "GENIEX_DRAFT_MODEL", "GENIEX_DRAFT_TOKENS", "GENIEX_DRAFT_MIN", "GENIEX_DRAFT_P_MIN"} {
		t.Setenv(name, "")
	}
	cmd := serve()

	defaults := []struct {
		name string
		want string
	}{
		{"spec-type", ""},
		{"draft-model", ""},
		// Unlike infer, the server preserves the SDK/plugin's zero-value default.
		{"draft-tokens", "0"},
		{"draft-min", "0"},
		{"draft-p-min", "0"},
	}
	for _, tc := range defaults {
		flag := cmd.Flags().Lookup(tc.name)
		if flag == nil {
			t.Fatalf("serve is missing --%s", tc.name)
		}
		if flag.DefValue != tc.want {
			t.Errorf("--%s default = %q, want %q", tc.name, flag.DefValue, tc.want)
		}
	}
	cfg := config.Get()
	if cfg.SpecType != "" || cfg.DraftModel != "" || cfg.DraftTokens != 0 || cfg.DraftMin != 0 || cfg.DraftPMin != 0 {
		t.Errorf("unconfigured serve MTP defaults = %+v, want zero values", cfg)
	}

	values := []struct {
		flag string
		key  string
		set  string
		want string
	}{
		{"spec-type", "spectype", "draft-mtp", "draft-mtp"},
		{"draft-model", "draftmodel", "draft.gguf", "draft.gguf"},
		{"draft-tokens", "drafttokens", "7", "7"},
		{"draft-min", "draftmin", "2", "2"},
		{"draft-p-min", "draftpmin", "0.65", "0.65"},
	}
	for _, tc := range values {
		if err := cmd.Flags().Set(tc.flag, tc.set); err != nil {
			t.Fatalf("set --%s: %v", tc.flag, err)
		}
		if got := viper.GetString(tc.key); got != tc.want {
			t.Errorf("--%s binds %s = %q, want %q", tc.flag, tc.key, got, tc.want)
		}
	}
	cfg = config.Get()
	if cfg.SpecType != "draft-mtp" || cfg.DraftModel != "draft.gguf" || cfg.DraftTokens != 7 || cfg.DraftMin != 2 || cfg.DraftPMin != 0.65 {
		t.Errorf("serve MTP flag defaults = %+v, want type=draft-mtp, model=draft.gguf, tokens=7, min=2, p_min=0.65", cfg)
	}
	t.Setenv("GENIEX_SPEC_TYPE", "ngram-simple")
	if got := config.Get().SpecType; got != "draft-mtp" {
		t.Errorf("--spec-type should override GENIEX_SPEC_TYPE: got %q", got)
	}
}

func TestServeSpeculativeDecodingEnvironmentBindings(t *testing.T) {
	resetServeTestViper(t)
	for name, value := range map[string]string{
		"GENIEX_SPEC_TYPE":    "draft-mtp",
		"GENIEX_DRAFT_MODEL":  "draft.gguf",
		"GENIEX_DRAFT_TOKENS": "7",
		"GENIEX_DRAFT_MIN":    "2",
		"GENIEX_DRAFT_P_MIN":  "0.65",
	} {
		t.Setenv(name, value)
	}

	serve()
	values := []struct {
		key  string
		want string
	}{
		{"spectype", "draft-mtp"},
		{"draftmodel", "draft.gguf"},
		{"drafttokens", "7"},
		{"draftmin", "2"},
		{"draftpmin", "0.65"},
	}
	for _, tc := range values {
		if got := viper.GetString(tc.key); got != tc.want {
			t.Errorf("environment binding for %s = %q, want %q", tc.key, got, tc.want)
		}
	}
	cfg := config.Get()
	if cfg.SpecType != "draft-mtp" || cfg.DraftModel != "draft.gguf" || cfg.DraftTokens != 7 || cfg.DraftMin != 2 || cfg.DraftPMin != 0.65 {
		t.Errorf("serve MTP environment defaults = %+v, want type=draft-mtp, model=draft.gguf, tokens=7, min=2, p_min=0.65", cfg)
	}
}
