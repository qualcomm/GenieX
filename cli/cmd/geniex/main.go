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
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"runtime"
	"slices"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/qcom-it-nexa-ai/geniex/cli/cmd/geniex/common"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/store"
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
		Use: "geniex",
		PersistentPreRun: func(cmd *cobra.Command, args []string) {
			// Register ModelHub
			s := store.Get()
			model_hub.RegisterHub(model_hub.NewHuggingFace())
			model_hub.RegisterHub(model_hub.NewAIHub(chipsetGet(s)))

			subCmd := cmd.CalledAs()
			if !skipUpdate {
				notifyUpdate()
				// skip network probe for quick commands
				if !slices.Contains([]string{
					"remove", "rm", "clean", "list", "ls", "model",
					"config",
					"version", "update",
					"help", "completion",
				}, subCmd) {
					go checkUpdate()
				}
			}
		},
	}
	rootCmd.PersistentFlags().StringVarP(&dataDir, "data-dir", "", "", "Custom data directory (env: GENIEX_DATADIR)")
	viper.BindPFlag("datadir", rootCmd.PersistentFlags().Lookup("data-dir"))
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "Enable verbose output")
	rootCmd.PersistentFlags().BoolVarP(&skipUpdate, "skip-update", "", false, "Skip checking for updates")
	rootCmd.PersistentFlags().BoolVarP(&testMode, "test-mode", "", false, "Enable test mode")
	rootCmd.PersistentFlags().MarkHidden("test-mode")

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

func checkDependency() {
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
	// log
	common.ApplyLogLevel()

	if err := RootCmd().Execute(); err != nil {
		slog.Error("geniex failed", "err", err)
		os.Exit(1)
	}
}
