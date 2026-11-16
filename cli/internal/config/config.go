// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package config

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/viper"
)

type Config struct {
	// Global settings
	DataDir string

	// Server settings
	Host      string // Server host and port (default: "127.0.0.1:18181")
	Origins   string // Allowed CORS origins (default: "*")
	KeepAlive int64  // Connection keep-alive timeout in seconds (default: 300)
	Metrics   bool   // Whether to expose GET /metrics (default: false)
	// Server defaults applied when a request omits them; per-request fields
	// still override. Compute is resolved by the SDK (sdk/src/device.cpp).
	Model       string  // Model used when a request omits model (default: "")
	NCtx        int32   // Default context window size (default: 4096)
	Ngl         int32   // Default GPU/NPU layers to offload, -1 = all (default: -1)
	Compute     string  // Default compute unit: cpu, gpu, npu, hybrid (default: "")
	VitCompute  string  // Default VLM vision encoder compute unit (default: "")
	PowerMode   string  // Default HTP power/clock-management mode (default: "" = burst)
	SpecType    string  // Default speculative decoding type (default: disabled)
	DraftModel  string  // Default draft model name or path (default: "")
	DraftTokens int32   // Default max draft tokens (0 = plugin default of 3)
	DraftMin    int32   // Default min draft tokens (0 = llama.cpp default)
	DraftPMin   float32 // Default min greedy draft probability (0 = llama.cpp default)
	Think       bool    // Default chat thinking mode (default: true)
	// HTTPS / TLS settings
	HTTPS    bool   // Whether to serve over HTTPS (default: false)
	CertFile string // TLS certificate file path
	KeyFile  string // TLS private key file path

	// Env only params
	HFToken string
	Log     string
}

// init sets up viper defaults and env binding. Runs once at package load.
func init() {
	// ENV only param need to set default here
	viper.SetDefault("hftoken", "") // Default empty token
	viper.SetDefault("log", "none") // Default log level

	viper.SetDefault("think", true) // Default chat thinking mode, overridden by serve --think

	viper.SetEnvPrefix("geniex")
	viper.AutomaticEnv()
}

// Get returns a fresh snapshot of the current viper state. Unmarshalling on
// every call is intentional: cobra populates subcommand flags in stages, so
// callers at different points in startup observe different values. Sharing a
// cached *Config via sync.Once would freeze whichever snapshot was observed
// first — usually before the subcommand's own flags are visible.
func Get() *Config {
	c := &Config{}
	viper.Unmarshal(c)
	c.HFToken = resolveHFToken(c.HFToken)
	return c
}

func resolveHFToken(geniexToken string) string {
	if geniexToken != "" {
		return geniexToken
	}

	if token := os.Getenv("HF_TOKEN"); token != "" {
		return token
	}

	homeDir, err := os.UserHomeDir()
	if err != nil || homeDir == "" {
		return ""
	}

	tokenPath := filepath.Join(homeDir, ".cache", "huggingface", "token")
	data, err := os.ReadFile(tokenPath)
	if err != nil {
		return ""
	}

	return strings.TrimSpace(string(data))
}
