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
	"context"
	"fmt"
	"runtime"
	"slices"
	"sort"
	"strings"

	"github.com/charmbracelet/huh"
	"github.com/spf13/cobra"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub/aihub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/sochost"
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
			"For the \"device\" key, omit the value to launch an interactive device picker.\n\n" +
			"Available keys: " + strings.Join(store.ConfigKeys, ", "),
		Args:      cobra.MatchAll(cobra.RangeArgs(1, 2), validConfigKeyArg),
		ValidArgs: store.ConfigKeys,
		RunE: func(cmd *cobra.Command, args []string) error {
			key := args[0]

			// Device key with no explicit value: launch interactive picker.
			switch key {
			case store.ConfigKeyDevice:
				if len(args) == 1 {
					if err := pickDevice(cmd.Context()); err != nil {
						return fmt.Errorf("failed to pick device: %w", err)
					}
					return nil
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

// chipsetGet returns a closure that reads the cached chipset config key.
// Returns "" when unset — callers must run chipsetEnsure first if they
// need a real value. Kept passive so AIHub can hold it across a spinner
// without triggering UI from inside the spinner.
func chipsetGet(s *store.Store) func() string {
	return func() string {
		v, _, _ := s.ConfigGet(store.ConfigKeyDevice)
		return v
	}
}

// chipsetEnsure makes sure a chipset is configured: cached value wins,
// then host probe, then interactive picker. Returns an error if the user
// bails on the picker. Must be called before any flow that surrounds the
// AIHub fetch with a spinner.
func chipsetEnsure(ctx context.Context, s *store.Store) error {
	if v, _, _ := s.ConfigGet(store.ConfigKeyDevice); v != "" {
		return nil
	}

	plat, err := loadPlatform(ctx)
	if err != nil {
		return err
	}

	if alias, ok := sochost.DetectChipsetAlias(); ok {
		if canonical, err := aihub.ResolveChipset(plat, alias); err == nil {
			if err := s.ConfigSet(store.ConfigKeyDevice, canonical); err == nil {
				fmt.Println(render.GetTheme().Info.Sprintf("device = %s (auto-detected)", canonical))
				return nil
			}
		}
	}

	fmt.Println(render.GetTheme().Info.Sprint("No device configured. Please select your device first."))
	if err := pickDeviceFrom(plat); err != nil {
		return fmt.Errorf("device chipset not configured (run: geniex config set device <chipset>): %w", err)
	}
	return nil
}

func loadPlatform(ctx context.Context) (*aihub.PlatformInfo, error) {
	client := aihub.NewClient()
	defer client.Close()

	spin := render.NewSpinner("fetching device list...")
	spin.Start()
	plat, err := client.LoadPlatformDirect(ctx)
	spin.Stop()
	if err != nil {
		return nil, fmt.Errorf("failed to fetch device list: %w", model_hub.TranslateAIHubError(err))
	}
	return plat, nil
}

func pickDevice(ctx context.Context) error {
	plat, err := loadPlatform(ctx)
	if err != nil {
		return err
	}
	return pickDeviceFrom(plat)
}

// pickDeviceFrom presents an interactive selector over plat's devices for the
// host OS, persisting the chosen chipset under the "device" config key.
func pickDeviceFrom(plat *aihub.PlatformInfo) error {
	var osType aihub.OperatingSystemType
	switch runtime.GOOS {
	case "windows":
		osType = aihub.OSTypeWindows
	case "linux":
		osType = aihub.OSTypeQCLinux
	default:
		return fmt.Errorf("unsupported operating system %q for AI Hub device selection", runtime.GOOS)
	}

	displayByChipset := map[string]string{}
	var options []huh.Option[string]
	for _, d := range plat.Devices {
		if d.OS.OSType != osType || d.Chipset == "" {
			continue
		}
		displayByChipset[d.Chipset] = d.Name
		options = append(options, huh.NewOption(d.Name, d.Chipset))
	}
	if len(options) == 0 {
		return fmt.Errorf("no devices found for this operating system")
	}

	var selected string
	if err := huh.NewSelect[string]().
		Title("Select your device").
		Options(options...).
		Value(&selected).
		Run(); err != nil {
		return err
	}

	if err := store.Get().ConfigSet(store.ConfigKeyDevice, selected); err != nil {
		return fmt.Errorf("failed to save device: %w", err)
	}
	fmt.Println(render.GetTheme().Info.Sprintf("device = %s (%s)", displayByChipset[selected], selected))
	return nil
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
