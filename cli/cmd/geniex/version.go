// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/bytedance/sonic"
	"github.com/spf13/cobra"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/store"
)

var Version string

// pluginIDs are the plugins whose versions `geniex version` reports, in display
// order.
var pluginIDs = []string{"qairt", "llama_cpp"}

// pluginVersionCache stores plugin versions keyed by the SDK bridge version.
//
// Reading a plugin version otherwise forces geniex_sdk.Init(), which scans and
// dlopen's every plugin's runtime libraries — llama.cpp's OpenCL backend
// compiles kernels on load, which makes `geniex version` take many seconds. The
// bridge version (geniex_sdk.Version()) is a compile-time constant that needs no
// Init and bumps on every SDK/plugin update, so it doubles as the cache key:
// when it changes the cache is stale and we re-init to repopulate.
type pluginVersionCache struct {
	BridgeVersion string            `json:"bridge_version"`
	Plugins       map[string]string `json:"plugins"`
}

const pluginVersionCacheFile = "plugin_versions.json"

func pluginVersionCachePath() string {
	return filepath.Join(store.Get().DataPath(), pluginVersionCacheFile)
}

// loadPluginVersionCache returns the cached versions if the file exists and was
// written for the given bridge version. ok is false on any miss (absent file,
// decode error, stale bridge version, or a missing plugin entry).
func loadPluginVersionCache(path, bridgeVersion string) (versions map[string]string, ok bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}
	var c pluginVersionCache
	if err := sonic.Unmarshal(data, &c); err != nil {
		slog.Debug("plugin version cache decode failed", "error", err)
		return nil, false
	}
	if c.BridgeVersion != bridgeVersion || c.Plugins == nil {
		return nil, false
	}
	for _, id := range pluginIDs {
		if _, present := c.Plugins[id]; !present {
			return nil, false
		}
	}
	return c.Plugins, true
}

// savePluginVersionCache persists versions for bridgeVersion. Cache write
// failures are non-fatal: the next `version` run just re-inits.
func savePluginVersionCache(path, bridgeVersion string, versions map[string]string) {
	data, err := sonic.Marshal(pluginVersionCache{BridgeVersion: bridgeVersion, Plugins: versions})
	if err != nil {
		slog.Debug("plugin version cache marshal failed", "error", err)
		return
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		slog.Debug("plugin version cache save failed", "error", err)
	}
}

// resolvePluginVersions returns each plugin's reported version, preferring the
// on-disk cache. On a miss it inits the SDK (the slow path that loads every
// plugin), reads the versions, and repopulates the cache.
func resolvePluginVersions() map[string]string {
	bridge := geniex_sdk.Version()
	path := pluginVersionCachePath()
	if cached, ok := loadPluginVersionCache(path, bridge); ok {
		return cached
	}

	if err := geniex_sdk.Init(); err != nil {
		// Best-effort path: without init we just report no plugin versions.
		return nil
	}
	defer geniex_sdk.DeInit()

	versions := make(map[string]string, len(pluginIDs))
	for _, id := range pluginIDs {
		versions[id] = geniex_sdk.GetPluginVersion(id)
	}
	savePluginVersionCache(path, bridge, versions)
	return versions
}

func runVersion() {
	versions := resolvePluginVersions()

	fmt.Printf("%-23s %s\n", "GenieX CLI Version:", Version)
	fmt.Printf("%-23s %s\n", "QAIRT Runtime Version:", versions["qairt"])
	fmt.Printf("%-23s %s\n", "LlamaCPP Runtime Hash:", versions["llama_cpp"])
}

func version() *cobra.Command {
	return &cobra.Command{
		GroupID: "management",
		Use:     "version",
		Short:   "show geniex version",
		Run:     func(cmd *cobra.Command, args []string) { runVersion() },
	}
}
