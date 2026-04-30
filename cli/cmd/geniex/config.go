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
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/charmbracelet/huh"
	"github.com/spf13/cobra"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub/aihub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/qaihm"
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
	if len(args) < 1 {
		return fmt.Errorf("requires a key argument (one of: %s)", strings.Join(store.ConfigKeys, ", "))
	}
	if !store.IsValidConfigKey(args[0]) {
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
		Run: func(cmd *cobra.Command, args []string) {
			key := args[0]
			value, ok, err := store.Get().ConfigGet(key)
			if err != nil {
				fmt.Fprintln(os.Stderr, render.GetTheme().Error.Sprintf("Failed to get configuration: %s", err))
				os.Exit(1)
			}
			if !ok {
				// Unset keys print nothing so the output is easy to use in
				// scripts (e.g. `$(geniex config get device)`).
				return
			}
			fmt.Println(value)
		},
	}
}

func configSetCmd() *cobra.Command {
	var noConfigCache bool

	cmd := &cobra.Command{
		Use:   "set <key> [value]",
		Short: "Set a configuration value",
		Long: "Set a specific configuration key to a new value. Pass an empty string to clear the value.\n\n" +
			"For the \"device\" key, omit the value to launch an interactive device picker.\n\n" +
			"Available keys: " + strings.Join(store.ConfigKeys, ", "),
		Args: cobra.MatchAll(
			cobra.RangeArgs(1, 2),
			func(cmd *cobra.Command, args []string) error {
				if err := validConfigKeyArg(cmd, args[:1]); err != nil {
					return err
				}
				// Non-device keys always require an explicit value.
				if args[0] != store.ConfigKeyDevice && len(args) < 2 {
					return fmt.Errorf("key %q requires a value argument", args[0])
				}
				return nil
			},
		),
		ValidArgs: store.ConfigKeys,
		RunE: func(cmd *cobra.Command, args []string) error {
			key := args[0]

			// Device key with no explicit value: launch interactive picker.
			if key == store.ConfigKeyDevice && len(args) == 1 {
				return pickDevice(cmd.Context(), noConfigCache)
			}

			value := args[1]
			if err := store.Get().ConfigSet(key, value); err != nil {
				fmt.Fprintln(os.Stderr, render.GetTheme().Error.Sprintf("Failed to set configuration: %s", err))
				os.Exit(1)
			}
			fmt.Println(render.GetTheme().Info.Sprintf("%s = %s", key, value))
			return nil
		},
	}

	cmd.Flags().BoolVar(&noConfigCache, "no-config-cache", false, "bypass local metadata cache and fetch the latest device list from remote")
	return cmd
}

// hostOSType maps runtime.GOOS to the protobuf OperatingSystemType used in
// DeviceInfo so we can filter the device list to only relevant entries.
// Returns an error for operating systems the CLI does not support.
func hostOSType() (qaihm.OperatingSystemType, error) {
	switch runtime.GOOS {
	case "windows":
		return qaihm.OperatingSystemType_OPERATING_SYSTEM_TYPE_WINDOWS, nil
	case "linux":
		return qaihm.OperatingSystemType_OPERATING_SYSTEM_TYPE_QC_LINUX, nil
	default:
		return qaihm.OperatingSystemType_OPERATING_SYSTEM_TYPE_UNSPECIFIED,
			fmt.Errorf("unsupported operating system %q for AI Hub device selection", runtime.GOOS)
	}
}

// pickDevice fetches platform.json, filters devices by the host OS, and
// presents an interactive selector. The selected device's chipset string is
// stored under the "device" config key.
func pickDevice(ctx context.Context, noConfigCache bool) error {
	cacheDir := filepath.Join(store.Get().DataPath(), "aihub")
	client := aihub.NewClient(cacheDir)
	defer client.Close()

	var fetchOpts []aihub.FetchOption
	if noConfigCache {
		fetchOpts = append(fetchOpts, aihub.WithSkipCache())
	}

	spin := render.NewSpinner("fetching device list...")
	spin.Start()
	plat, err := client.LoadPlatformDirect(ctx, fetchOpts...)
	spin.Stop()
	if err != nil {
		fmt.Fprintln(os.Stderr, render.GetTheme().Error.Sprintf("Failed to fetch device list: %s", err))
		return err
	}

	osType, err := hostOSType()
	if err != nil {
		fmt.Fprintln(os.Stderr, render.GetTheme().Error.Sprint(err.Error()))
		return err
	}

	var options []huh.Option[string]
	for _, d := range plat.GetDevices() {
		if d.GetOs().GetOstype() != osType {
			continue
		}
		if d.GetChipset() == "" {
			continue
		}
		options = append(options, huh.NewOption(d.GetName(), d.GetChipset()))
	}

	if len(options) == 0 {
		fmt.Fprintln(os.Stderr, render.GetTheme().Error.Sprint("No devices found for this operating system."))
		return fmt.Errorf("no devices available")
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
		fmt.Fprintln(os.Stderr, render.GetTheme().Error.Sprintf("Failed to save device: %s", err))
		return err
	}

	// Find the display name for the confirmation message.
	displayName := selected
	for _, d := range plat.GetDevices() {
		if d.GetChipset() == selected {
			displayName = d.GetName()
			break
		}
	}
	fmt.Println(render.GetTheme().Info.Sprintf("device = %s (%s)", displayName, selected))
	return nil
}

func configListCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List all configuration values",
		Long:  "Display all known configuration keys and their corresponding values. Unset keys are shown as blank.",
		Args:  cobra.NoArgs,
		Run: func(cmd *cobra.Command, args []string) {
			cfg, err := store.Get().ConfigList()
			if err != nil {
				fmt.Fprintln(os.Stderr, render.GetTheme().Error.Sprintf("Failed to read configuration: %s", err))
				os.Exit(1)
			}

			keys := make([]string, 0, len(store.ConfigKeys))
			keys = append(keys, store.ConfigKeys...)
			sort.Strings(keys)

			for _, key := range keys {
				value := cfg[key]
				fmt.Println(render.GetTheme().Info.Sprintf("%s: %s", key, value))
			}
		},
	}
}
