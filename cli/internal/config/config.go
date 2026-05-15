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
	"strings"

	"github.com/spf13/viper"
)

// DefaultAIHubBaseURL is the public root for Qualcomm AI Hub release assets.
// Override via the GENIEX_AIHUBBASEURL env var (tests / mirrors).
const DefaultAIHubBaseURL = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models"

// DefaultAIHubVersion is the pinned aihm release the CLI consumes. The public
// bucket has no `latest` alias; manifests are only at
// <base>/releases/<version>/manifest.json. Override via GENIEX_AIHUBVERSION.
const DefaultAIHubVersion = "v0.53.1"

type Config struct {
	// Global settings
	DataDir string

	// Server settings
	Host      string // Server host and port (default: "127.0.0.1:18181")
	Origins   string // Allowed CORS origins (default: "*")
	KeepAlive int64  // Connection keep-alive timeout in seconds (default: 300)
	// HTTPS / TLS settings
	HTTPS    bool   // Whether to serve over HTTPS (default: false)
	CertFile string // TLS certificate file path
	KeyFile  string // TLS private key file path

	// Env only params
	HFToken      string
	Log          string
	AIHubBaseURL string // Override the AI Hub public assets base URL (rarely used)
	AIHubVersion string // Override the pinned aihm release version
}

// init sets up viper defaults and env binding. Runs once at package load.
func init() {
	// ENV only param need to set default here
	viper.SetDefault("hftoken", "")                       // Default empty token
	viper.SetDefault("log", "none")                       // Default log level
	viper.SetDefault("aihubbaseurl", DefaultAIHubBaseURL) // AI Hub public assets base URL
	viper.SetDefault("aihubversion", DefaultAIHubVersion) // Pinned aihm release version

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
