// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/cmd/geniex/common"
	"github.com/qualcomm/GenieX/cli/server"
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
	// Model-load defaults applied when a request omits them (llama_cpp only;
	// per-request body fields still override).
	serveCmd.Flags().Int32("nctx", 4096, "Default context window size, llama_cpp only (env: GENIEX_NCTX)")
	serveCmd.Flags().Int32P("ngl", "n", -1, "Default layers to offload to gpu/npu, -1 = all, llama_cpp only (env: GENIEX_NGL)")
	serveCmd.Flags().StringP("compute", "c", "", "Default compute unit: cpu, gpu, npu, or hybrid (env: GENIEX_COMPUTE)")
	// HTTPS / TLS flags
	serveCmd.Flags().Bool("https", false, "Enable HTTPS/TLS (env: GENIEX_HTTPS)")
	serveCmd.Flags().String("certfile", "cert.pem", "TLS certificate file path (env: GENIEX_CERTFILE)")
	serveCmd.Flags().String("keyfile", "key.pem", "TLS private key file path (env: GENIEX_KEYFILE)")

	viper.BindPFlag("host", serveCmd.Flags().Lookup("host"))
	viper.BindPFlag("origins", serveCmd.Flags().Lookup("origins"))
	viper.BindPFlag("keepalive", serveCmd.Flags().Lookup("keepalive"))
	viper.BindPFlag("nctx", serveCmd.Flags().Lookup("nctx"))
	viper.BindPFlag("ngl", serveCmd.Flags().Lookup("ngl"))
	viper.BindPFlag("compute", serveCmd.Flags().Lookup("compute"))
	viper.BindPFlag("enablehttps", serveCmd.Flags().Lookup("https"))
	viper.BindPFlag("certfile", serveCmd.Flags().Lookup("certfile"))
	viper.BindPFlag("keyfile", serveCmd.Flags().Lookup("keyfile"))

	serveCmd.Run = func(cmd *cobra.Command, args []string) {
		checkAudioDependency()
		if err := common.InitSDK(); err != nil {
			common.PrintError(err)
			os.Exit(1)
		}

		server.Serve()

		geniex_sdk.DeInit()
	}

	return serveCmd
}
