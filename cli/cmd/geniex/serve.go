// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/cmd/geniex/common"
	"github.com/qualcomm/GenieX/cli/server"
	"github.com/qualcomm/GenieX/cli/server/handler"
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

	serverFlags := pflag.NewFlagSet("Server", pflag.ExitOnError)
	serverFlags.SortFlags = false
	serverFlags.String("host", "127.0.0.1:18181", "Server listen address (env: GENIEX_HOST)")
	serverFlags.String("origins", "*", "Allowed CORS origins (env: GENIEX_ORIGINS)")
	serverFlags.Int("keepalive", 300, "Model idle timeout in seconds (env: GENIEX_KEEPALIVE)")
	serverFlags.Bool("metrics", false, "Expose Prometheus metrics at /metrics (env: GENIEX_METRICS)")
	serverFlags.String("qairt-lib", "", "Run against a different QAIRT runtime: path to a QAIRT SDK root or a folder of QNN libraries, qairt only (env: GENIEX_QAIRT_LIB)")
	serverFlags.Bool("https", false, "Enable HTTPS/TLS (env: GENIEX_HTTPS)")
	serverFlags.String("certfile", "cert.pem", "TLS certificate file path (env: GENIEX_CERTFILE)")
	serverFlags.String("keyfile", "key.pem", "TLS private key file path (env: GENIEX_KEYFILE)")
	serverFlags.BoolP("help", "h", false, "help for serve")

	// Model-load defaults applied when a request omits them; request fields override them.
	defaultFlags := pflag.NewFlagSet("Model Default", pflag.ExitOnError)
	defaultFlags.SortFlags = false
	defaultFlags.String("model", "", "Model to use when request omits model: cached name[:precision] (env: GENIEX_MODEL)")
	defaultFlags.Int32("nctx", 4096, "Context window size, llama_cpp only (env: GENIEX_NCTX)")
	defaultFlags.Int32P("ngl", "n", -1, "Layers to offload to gpu/npu, -1 = all, llama_cpp only (env: GENIEX_NGL)")
	defaultFlags.StringP("compute", "c", "", "Compute unit: cpu, gpu, npu, or hybrid (env: GENIEX_COMPUTE)")
	defaultFlags.String("vit-compute", "", "VLM vision encoder compute unit, e.g. CPU or HTP2 (env: GENIEX_VIT_COMPUTE)")
	defaultFlags.String("power-mode", "", "HTP power/clock-management mode: low_power_saver, power_saver, high_power_saver, low_balanced, balanced, high_performance, sustained_high_performance, burst (burst when omitted) (env: GENIEX_POWER_MODE)")
	defaultFlags.String("spec-type", "", "Speculative decoding type(s), llama_cpp only (env: GENIEX_SPEC_TYPE)")
	defaultFlags.String("draft-model", "", "Draft/MTP model for draft-* types: cached name or local GGUF path (env: GENIEX_DRAFT_MODEL)")
	defaultFlags.Int32("draft-tokens", 0, "Max draft tokens per step, 0 = plugin default of 3 (env: GENIEX_DRAFT_TOKENS)")
	defaultFlags.Int32("draft-min", 0, "Min draft tokens per step, 0 = llama.cpp default (env: GENIEX_DRAFT_MIN)")
	defaultFlags.Float32("draft-p-min", 0, "Min greedy draft probability, 0 = llama.cpp default (env: GENIEX_DRAFT_P_MIN)")
	defaultFlags.Bool("think", true, "Enable thinking mode when request omits enable_think, use --think=false to disable (env: GENIEX_THINK)")

	groups := []*pflag.FlagSet{serverFlags, defaultFlags}
	for _, flags := range groups {
		serveCmd.Flags().AddFlagSet(flags)
	}
	serveCmd.SetUsageFunc(func(c *cobra.Command) error { return groupedFlagUsage(c, groups) })

	viper.BindPFlag("host", serveCmd.Flags().Lookup("host"))
	viper.BindPFlag("origins", serveCmd.Flags().Lookup("origins"))
	viper.BindPFlag("keepalive", serveCmd.Flags().Lookup("keepalive"))
	viper.BindPFlag("metrics", serveCmd.Flags().Lookup("metrics"))
	viper.BindPFlag("model", serveCmd.Flags().Lookup("model"))
	viper.BindPFlag("nctx", serveCmd.Flags().Lookup("nctx"))
	viper.BindPFlag("ngl", serveCmd.Flags().Lookup("ngl"))
	viper.BindPFlag("compute", serveCmd.Flags().Lookup("compute"))
	viper.BindPFlag("vitcompute", serveCmd.Flags().Lookup("vit-compute"))
	viper.BindEnv("vitcompute", "GENIEX_VIT_COMPUTE")
	viper.BindPFlag("powermode", serveCmd.Flags().Lookup("power-mode"))
	viper.BindEnv("powermode", "GENIEX_POWER_MODE")
	viper.BindPFlag("qairtlib", serveCmd.Flags().Lookup("qairt-lib"))
	// Bound explicitly so the plugin's own spelling is the only one that works;
	// AutomaticEnv would otherwise make GENIEX_QAIRTLIB a silent second alias.
	viper.BindEnv("qairtlib", "GENIEX_QAIRT_LIB")
	viper.BindPFlag("spectype", serveCmd.Flags().Lookup("spec-type"))
	viper.BindEnv("spectype", "GENIEX_SPEC_TYPE")
	viper.BindPFlag("draftmodel", serveCmd.Flags().Lookup("draft-model"))
	viper.BindEnv("draftmodel", "GENIEX_DRAFT_MODEL")
	viper.BindPFlag("drafttokens", serveCmd.Flags().Lookup("draft-tokens"))
	viper.BindEnv("drafttokens", "GENIEX_DRAFT_TOKENS")
	viper.BindPFlag("draftmin", serveCmd.Flags().Lookup("draft-min"))
	viper.BindEnv("draftmin", "GENIEX_DRAFT_MIN")
	viper.BindPFlag("draftpmin", serveCmd.Flags().Lookup("draft-p-min"))
	viper.BindEnv("draftpmin", "GENIEX_DRAFT_P_MIN")
	viper.BindPFlag("think", serveCmd.Flags().Lookup("think"))
	viper.BindPFlag("https", serveCmd.Flags().Lookup("https"))
	viper.BindPFlag("certfile", serveCmd.Flags().Lookup("certfile"))
	viper.BindPFlag("keyfile", serveCmd.Flags().Lookup("keyfile"))

	serveCmd.Run = func(cmd *cobra.Command, args []string) {
		checkAudioDependency()

		// Handed to the SDK rather than exported, matching `infer`; every model the server
		// loads, LLM or VLM, then goes through the same SDK-side resolution.
		if qairtLib := viper.GetString("qairtlib"); qairtLib != "" {
			if err := geniex_sdk.SetQairtRuntimePath(qairtLib); err != nil {
				common.PrintError(err)
				os.Exit(1)
			}
		}

		if err := common.InitSDK(); err != nil {
			common.PrintError(err)
			os.Exit(1)
		}

		if _, err := geniex_sdk.ResolvePowerMode(viper.GetString("powermode")); err != nil {
			common.PrintError(err)
			os.Exit(1)
		}

		handler.Version = Version
		err := server.Serve()

		geniex_sdk.DeInit()
		if err != nil {
			common.PrintError(err)
			os.Exit(1)
		}
	}

	return serveCmd
}
