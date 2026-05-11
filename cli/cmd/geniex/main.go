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
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/qcom-it-nexa-ai/geniex/cli/cmd/geniex/common"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/config"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/store"
)

func registerAIHub() {
	s := store.Get()
	model_hub.RegisterHub(model_hub.NewHuggingFace())
	model_hub.RegisterHub(model_hub.NewAIHub(
		func() string {
			v, _, _ := s.ConfigGet(store.ConfigKeyDevice)
			return v
		},
		filepath.Join(s.DataPath(), "aihub"),
	))
}

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
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			// log
			common.ApplyLogLevel()

			registerAIHub()

			// subCmd := cmd.CalledAs()
			//
			// // skip update check
			// if !skipUpdate {
			// 	notifyUpdate()
			// 	// skip some quick commands
			// 	if !slices.Contains([]string{
			// 		"remove", "rm", "clean", "list", "ls",
			// 		"config",
			// 		"version", "update",
			// 	}, subCmd) {
			// 		go checkUpdate()
			// 	}
			// }

			return nil
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
		infer(),
		serve(), run(),
		configCmd(),
		version(), update(),
	)

	return rootCmd
}

func normalizeModelName(name string) (string, string) {
	// split quant
	parts := strings.SplitN(name, ":", 2)
	name = parts[0]
	quant := ""
	if len(parts) == 2 {
		quant = strings.ToUpper(parts[1])
	}

	// support shortcuts
	if actualName, exists := config.GetModelMapping(name); exists {
		return actualName, quant
	}

	// support qwen3 -> qualcomm/qwen3
	if !strings.Contains(name, "/") {
		return "qualcomm/" + name, quant
	}

	// support https://huggingface.co/Qwen/Qwen3-0.6B-GGUF -> Qwen/Qwen3-0.6B-GGUF
	if strings.HasPrefix(name, model_hub.HF_ENDPOINT) {
		return strings.TrimPrefix(name, model_hub.HF_ENDPOINT+"/"), quant
	}

	return name, quant
}

// main is the entry point that executes the root command.
func main() {
	if err := RootCmd().Execute(); err != nil {
		slog.Error("geniex failed", "err", err)
		os.Exit(1)
	}
}
