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

	"github.com/charmbracelet/huh"
	"github.com/spf13/cobra"

	geniex_sdk "github.com/qualcomm/nexa-sdk/bindings/go"
	"github.com/qualcomm/nexa-sdk/cli/internal/render"
	"github.com/qualcomm/nexa-sdk/cli/internal/store"
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
			value, _, err := store.Get().ConfigGet(key)
			if err != nil {
				return fmt.Errorf("failed to get configuration: %w", err)
			}
			// The chipset key falls back to host autodetection when unset,
			// reporting "unknown" if the host cannot be probed.
			if key == store.ConfigKeyChipset && value == "" {
				value = resolveChipset()
			}
			// Other unset keys print nothing so the output is easy to use in
			// scripts (e.g. `$(geniex config get chipset)`).
			if value != "" {
				fmt.Println(value)
			}
			return nil
		},
	}
}

func configSetCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "set <key> [value]",
		Short: "Set a configuration value",
		Long: "Set a specific configuration key to a new value. Pass an empty string to clear the value.\n\n" +
			"For the \"chipset\" key, omit the value to launch an interactive chipset picker.\n\n" +
			"Available keys: " + strings.Join(store.ConfigKeys, ", "),
		Args:      cobra.MatchAll(cobra.RangeArgs(1, 2), validConfigKeyArg),
		ValidArgs: store.ConfigKeys,
		RunE: func(cmd *cobra.Command, args []string) error {
			key := args[0]

			// Chipset key with no explicit value: launch the interactive picker.
			switch key {
			case store.ConfigKeyChipset:
				if len(args) < 2 {
					_, err := pickChipset()
					return err
				}
			default:
				if len(args) < 2 {
					return fmt.Errorf("key %q requires a value argument", key)
				}
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

// resolveChipset returns the host's autodetected chipset, or "unknown" when the
// host cannot be probed. Used as the fallback for the unset "chipset" key.
func resolveChipset() string {
	detected, err := geniex_sdk.ModelDetectChipset()
	if err != nil || detected == "" {
		return "unknown"
	}
	return detected
}

// pickChipset lists the chipsets Qualcomm AI Hub supports and lets the user
// select one interactively, defaulting to the host's detected chipset when it
// can be probed. The chosen chipset name is persisted under the "chipset" key
// and returned to the caller.
func pickChipset() (string, error) {
	chipsets, err := geniex_sdk.ModelListChipsets()
	if err != nil {
		return "", fmt.Errorf("failed to fetch chipset list: %w", err)
	}
	if len(chipsets) == 0 {
		return "", fmt.Errorf("no chipsets available from the hub")
	}

	// Probe the host so we can preselect its chipset; failure is non-fatal.
	detected, _ := geniex_sdk.ModelDetectChipset()

	// matches reports whether the detected chipset is c's name or an alias.
	matches := func(c geniex_sdk.ChipsetInfo) bool {
		if strings.EqualFold(c.Name, detected) {
			return true
		}
		return slices.ContainsFunc(c.Aliases, func(a string) bool {
			return strings.EqualFold(a, detected)
		})
	}

	var selected string
	options := make([]huh.Option[string], 0, len(chipsets))
	for _, c := range chipsets {
		label := c.Name
		if len(c.Aliases) > 0 {
			label = fmt.Sprintf("%s (%s)", c.Name, strings.Join(c.Aliases, ", "))
		}
		options = append(options, huh.NewOption(label, c.Name))
		if detected != "" && matches(c) {
			selected = c.Name
		}
	}

	if err := huh.NewSelect[string]().
		Title("Select your chipset").
		Options(options...).
		Value(&selected).
		Run(); err != nil {
		return "", err
	}

	if err := store.Get().ConfigSet(store.ConfigKeyChipset, selected); err != nil {
		return "", fmt.Errorf("failed to save chipset: %w", err)
	}
	fmt.Println(render.GetTheme().Info.Sprintf("chipset = %s", selected))
	return selected, nil
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
				if key == store.ConfigKeyChipset && value == "" {
					value = resolveChipset()
				}
				fmt.Println(render.GetTheme().Info.Sprintf("%s: %s", key, value))
			}
			return nil
		},
	}
}
