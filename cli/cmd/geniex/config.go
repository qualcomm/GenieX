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
	"slices"
	"sort"
	"strings"

	"github.com/spf13/cobra"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/store"
)

// configCmd builds the `geniex config` command tree:
//
//	geniex config get <key>
//	geniex config set <key> <value>
//	geniex config list
//
// The command persists values under <data-dir>/config.json via the store.
func configCmd() *cobra.Command {
	cmd := &cobra.Command{
		GroupID: "management",
		Use:     "config",
		Short:   "Manage GenieX CLI configuration",
		Long: "Commands to manage GenieX CLI configuration, including setting and getting " +
			"configuration values.\n\n" +
			"Available keys: " + strings.Join(store.ConfigKeys, ", "),
	}

	cmd.AddCommand(
		configGetCmd(),
		configSetCmd(),
		configListCmd(),
	)

	return cmd
}

// validConfigKeyArg validates that args[0] is a known configuration key.
func validConfigKeyArg(cmd *cobra.Command, args []string) error {
	if !slices.Contains(store.ConfigKeys, args[0]) {
		return fmt.Errorf("unknown config key %q (valid keys: %s)", args[0], strings.Join(store.ConfigKeys, ", "))
	}
	return nil
}

func configGetCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "get <key>",
		Short: "Get a configuration value",
		Long: "Retrieve the value of a specific configuration key.\n\n" +
			"Available keys: " + strings.Join(store.ConfigKeys, ", "),
		Args:      cobra.MatchAll(cobra.ExactArgs(1), validConfigKeyArg),
		ValidArgs: store.ConfigKeys,
		RunE: func(cmd *cobra.Command, args []string) error {
			key := args[0]
			value, ok, err := store.Get().ConfigGet(key)
			if err != nil {
				return fmt.Errorf("failed to get configuration: %w", err)
			}
			if !ok {
				// Unset keys print nothing so the output is easy to use in
				// scripts (e.g. `$(geniex config get device)`).
				return nil
			}
			fmt.Println(value)
			return nil
		},
	}
}

func configSetCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "set <key> [value]",
		Short: "Set a configuration value",
		Long: "Set a specific configuration key to a new value. Pass an empty string to clear the value.\n\n" +
			"Available keys: " + strings.Join(store.ConfigKeys, ", "),
		Args:      cobra.MatchAll(cobra.RangeArgs(1, 2), validConfigKeyArg),
		ValidArgs: store.ConfigKeys,
		RunE: func(cmd *cobra.Command, args []string) error {
			key := args[0]
			if len(args) < 2 {
				if key == store.ConfigKeyDevice {
					return fmt.Errorf("automatic device detection is not supported; pass an explicit chipset (run: geniex config set device <chipset>)")
				}
				return fmt.Errorf("key %q requires a value argument", key)
			}

			value := args[1]
			if err := store.Get().ConfigSet(key, value); err != nil {
				return fmt.Errorf("failed to set configuration: %w", err)
			}
			fmt.Println(render.GetTheme().Info.Sprintf("%s = %s", key, value))
			return nil
		},
	}
}

func configListCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List all configuration values",
		Long:  "Display all known configuration keys and their corresponding values. Unset keys are shown as blank.",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := store.Get().ConfigList()
			if err != nil {
				return fmt.Errorf("failed to read configuration: %w", err)
			}

			keys := make([]string, 0, len(store.ConfigKeys))
			keys = append(keys, store.ConfigKeys...)
			sort.Strings(keys)

			for _, key := range keys {
				value := cfg[key]
				fmt.Println(render.GetTheme().Info.Sprintf("%s: %s", key, value))
			}
			return nil
		},
	}
}
