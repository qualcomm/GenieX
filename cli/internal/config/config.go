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
	"sync"

	"github.com/spf13/viper"
)

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
	HFToken string
	Log     string
}

var config *Config
var once sync.Once

// Get returns the singleton configuration instance.
// Uses sync.Once to ensure configuration is loaded only once.
func Get() *Config {
	once.Do(get)
	return config
}

// NOTE: Avoid calling Get before subcommand initialization to prevent premature config initialization
func GetLog() string {
	get()
	return config.Log
}

// init sets up default configuration values using Viper.
// These defaults are used if no environment variables are provided.
func init() {
	// ENV only param need to set default here
	viper.SetDefault("hftoken", "") // Default empty token
	viper.SetDefault("log", "none") // Default log level
}

// get initializes the configuration by reading from environment variables.
// Environment variables should be prefixed with "GENIEX_" (e.g., GENIEX_HOST).
// This function is called only once via sync.Once for thread safety.
func get() {
	config = &Config{}

	// Set environment variable prefix to "GENIEX_"
	viper.SetEnvPrefix("geniex")
	// Automatically read environment variables
	viper.AutomaticEnv()
	// Unmarshal configuration into the Config struct
	viper.Unmarshal(config)
}
