// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
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

// RootCmd builds the CLI command tree.
func RootCmd() *cobra.Command {
	cobra.EnableCommandSorting = false

	rootCmd := &cobra.Command{
		Use:           "geniex",
		SilenceUsage:  true,
		SilenceErrors: true,
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			// Re-apply now that --log is parsed; the main() call only saw GENIEX_LOG.
			common.ApplyLogLevel()

			// Cobra passes the leaf command being executed, so CalledAs() would
			// yield `get` for `geniex config get`. "" is the bare `geniex`.
			subCmd := ""
			for c := cmd; c.HasParent(); c = c.Parent() {
				if !c.Parent().HasParent() {
					subCmd = c.Name()
					break
				}
			}

			// Skip ModelInit for commands that don't touch the model manager
			if !slices.Contains([]string{
				"",
				"run", // pure HTTP client, no local store
				"version", "update",
				"help", "completion", cobra.ShellCompRequestCmd,
			}, subCmd) {
				s := store.Get()
				if err := geniex_sdk.ModelInit(s.DataPath()); err != nil {
					// Carrying on fails every later call as NOT_INITIALIZED.
					return fmt.Errorf("initialize model manager at %s: %w", s.DataPath(), err)
				}
			}

			if !skipUpdate {
				// `update` prints the latest version itself; completion's
				// stdout is parsed by the shell.
				if !slices.Contains([]string{
					"update",
					"completion", cobra.ShellCompRequestCmd,
				}, subCmd) {
					notifyUpdate()
				}
				// skip network probe for quick commands
				if !slices.Contains([]string{
					"",
					"remove", "clean", "list", "model",
					"config",
					"version", "update",
					"help", "completion", cobra.ShellCompRequestCmd,
				}, subCmd) {
					go checkUpdate()
				}
			}
			return nil
		},
		// Mirrors the ModelInit above. Unchecked: the SDK's deinit is a no-op that
		// never fails, the same reason Cobra skipping it on error is free.
		PersistentPostRun: func(cmd *cobra.Command, args []string) {
			geniex_sdk.ModelDeinit()
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
	rootCmd.PersistentFlags().String("log", "none", "Log level: none, error, warn, info, debug, trace (env: GENIEX_LOG)")
	viper.BindPFlag("log", rootCmd.PersistentFlags().Lookup("log"))
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
		infer(), evalCmd(),
		serve(), run(),
		configCmd(),
		version(), update(),
	)

	return rootCmd
}

func checkAudioDependency() {
	if _, err := exec.LookPath("sox"); err != nil {
		fmt.Println(render.GetTheme().Warning.Sprintf("SoX is not installed, some features may not work. Try:"))
		switch runtime.GOOS {
		case "linux":
			fmt.Println(render.GetTheme().Warning.Sprintf("  sudo apt install sox       # Debian/Ubuntu"))
			fmt.Println(render.GetTheme().Warning.Sprintf("  sudo yum install sox       # RHEL/CentOS/Fedora"))
			fmt.Println(render.GetTheme().Warning.Sprintf("  sudo pacman -S sox         # Arch Linux"))
		case "windows":
			fmt.Println(render.GetTheme().Warning.Sprintf("  winget install --id=ChrisBagwell.SoX -e"))
			fmt.Println(render.GetTheme().Warning.Sprintf("Then restart your terminal to make sure sox is in PATH"))
		default:
			fmt.Println(render.GetTheme().Warning.Sprintf("Please install it manually for your OS: %s\n", runtime.GOOS))
		}
	}
}

// main is the entry point that executes the root command.
func main() {
	// Honor GENIEX_LOG for early logs; the SDK callback is set later once --log
	// is parsed. Setting it here (default "none") would null its built-in handler.
	common.ApplySlog()
	common.EnableUTF8Console()

	cmd := RootCmd()
	applyHelpStyle(cmd)
	cmd.SetErr(render.NewStyledWriter(os.Stderr, render.GetTheme().Error))
	if err := cmd.Execute(); err != nil {
		common.PrintError(err)
		os.Exit(1)
	}
}
