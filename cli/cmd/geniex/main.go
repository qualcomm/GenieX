// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"log/slog"
	"os"
	"slices"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/cmd/geniex/common"
	"github.com/qualcomm/GenieX/cli/internal/render"
	"github.com/qualcomm/GenieX/cli/internal/store"
)

var (
	dataDir    string
	verbose    bool
	skipUpdate bool
	testMode   bool
)

// RootCmd creates the main GenieX CLI command with all subcommands.
// It sets up the command tree structure for model management,
// inference, and server operations.
func RootCmd() *cobra.Command {
	cobra.EnableCommandSorting = false

	rootCmd := &cobra.Command{
		Use:           "geniex",
		SilenceUsage:  true,
		SilenceErrors: true,
		PersistentPreRun: func(cmd *cobra.Command, args []string) {
			cmd.SilenceErrors = true

			subCmd := cmd.CalledAs()

			// Skip ModelInit for commands that don't touch the model manager
			if !slices.Contains([]string{
				"geniex",
				"version", "update",
				"help", "completion",
			}, subCmd) {
				s := store.Get()
				if err := geniex_sdk.ModelInit(s.DataPath()); err != nil {
					slog.Error("failed to initialize model manager", "err", err)
				}
			}

			if !skipUpdate {
				// `update` fetches and prints the latest version itself; the
				// cached notify banner would be redundant and possibly stale.
				if subCmd != "update" {
					notifyUpdate()
				}
				// skip network probe for quick commands
				if !slices.Contains([]string{
					"geniex",
					"remove", "rm", "clean", "list", "ls", "model",
					"config",
					"version", "update",
					"help", "completion",
				}, subCmd) {
					go checkUpdate()
				}
			}
		},
		Run: func(cmd *cobra.Command, args []string) {
			if showVer, _ := cmd.Flags().GetBool("version"); showVer {
				runVersion()
				return
			}
			cmd.Help()
		},
	}
	rootCmd.PersistentFlags().StringVarP(&dataDir, "data-dir", "", "", "Custom data directory (env: GENIEX_DATADIR)")
	viper.BindPFlag("datadir", rootCmd.PersistentFlags().Lookup("data-dir"))
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "", false, "Enable verbose output")
	rootCmd.PersistentFlags().BoolVarP(&skipUpdate, "skip-update", "", false, "Skip checking for updates")
	rootCmd.PersistentFlags().BoolVarP(&testMode, "test-mode", "", false, "Enable test mode")
	rootCmd.PersistentFlags().MarkHidden("test-mode")

	rootCmd.Flags().BoolP("version", "v", false, "Print version information")

	rootCmd.AddGroup(
		&cobra.Group{ID: "model", Title: "Model Commands"},
		&cobra.Group{ID: "inference", Title: "Inference Commands"},
		&cobra.Group{ID: "management", Title: "Management Commands"},
	)

	rootCmd.AddCommand(
		pull(), remove(), clean(), list(),
		modelCmd(),
		infer(),
		serve(), run(),
		configCmd(),
		version(), update(),
	)

	return rootCmd
}

// main is the entry point that executes the root command.
func main() {
	// log
	common.ApplyLogLevel()
	common.EnableUTF8Console()

	cmd := RootCmd()
	applyHelpStyle(cmd)
	cmd.SetErr(render.NewStyledWriter(os.Stderr, render.GetTheme().Error))
	if err := cmd.Execute(); err != nil {
		common.PrintError(err)
		os.Exit(1)
	}
}
