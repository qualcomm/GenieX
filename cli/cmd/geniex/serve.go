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

package main

import (
	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/cli/server"
)

// serve creates a command to start the GenieX server.
// This command initializes the GenieX-CLI, starts the HTTP server for AI services,
// and properly cleans up resources when the server shuts down.
// The server provides REST API endpoints for AI model inference and management.
// Usage: GenieX serve
func serve() *cobra.Command {
	serveCmd := &cobra.Command{
		GroupID: "inference",
		Use:     "serve",
		Short:   "Run the GenieX Server",
	}

	serveCmd.Flags().SortFlags = false
	serveCmd.Flags().String("host", "127.0.0.1:18181", "Default server address (env: GENIEX_HOST)")
	serveCmd.Flags().String("origins", "*", "Default CORS origins (env: GENIEX_ORIGINS)")
	serveCmd.Flags().Int("keepalive", 300, "Keepalive seconds (env: GENIEX_KEEPALIVE)")
	// HTTPS / TLS flags
	serveCmd.Flags().Bool("https", false, "Enable HTTPS/TLS (env: GENIEX_HTTPS)")
	serveCmd.Flags().String("certfile", "cert.pem", "TLS certificate file path (env: GENIEX_CERTFILE)")
	serveCmd.Flags().String("keyfile", "key.pem", "TLS private key file path (env: GENIEX_KEYFILE)")

	viper.BindPFlag("host", serveCmd.Flags().Lookup("host"))
	viper.BindPFlag("origins", serveCmd.Flags().Lookup("origins"))
	viper.BindPFlag("keepalive", serveCmd.Flags().Lookup("keepalive"))
	viper.BindPFlag("enablehttps", serveCmd.Flags().Lookup("https"))
	viper.BindPFlag("certfile", serveCmd.Flags().Lookup("certfile"))
	viper.BindPFlag("keyfile", serveCmd.Flags().Lookup("keyfile"))

	serveCmd.Run = func(cmd *cobra.Command, args []string) {
		checkAudioDependency()
		geniex_sdk.Init()

		server.Serve()

		geniex_sdk.DeInit()
	}

	return serveCmd
}
