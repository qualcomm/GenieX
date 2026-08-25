// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package store

import (
	"errors"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/bytedance/sonic"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// Known configuration keys. The set is intentionally small and explicit so
// that invalid keys can be rejected at the CLI boundary.
const (
	ConfigKeyChipset = "chipset"
)

// ConfigKeys is the list of all known configuration keys.
var ConfigKeys = []string{
	ConfigKeyChipset,
}

const configFileName = "config.json"

// configPath returns the on-disk path of the config file.
func (s *Store) configPath() string {
	return filepath.Join(s.home, configFileName)
}

// loadConfig reads and decodes the config file. If the file does not exist
// an empty map is returned.
func (s *Store) loadConfig() (map[string]string, error) {
	data, err := os.ReadFile(s.configPath())
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return map[string]string{}, nil
		}
		return nil, err
	}
	if len(data) == 0 {
		return map[string]string{}, nil
	}
	cfg := map[string]string{}
	if err := sonic.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return cfg, nil
}

// saveConfig atomically writes cfg to the config file.
func (s *Store) saveConfig(cfg map[string]string) error {
	data, err := sonic.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}

	path := s.configPath()
	tmp, err := os.CreateTemp(filepath.Dir(path), ".config-*.json.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op after a successful rename

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// ConfigGet returns the stored value for key. If key is unset the second
// return value is false.
func (s *Store) ConfigGet(key string) (string, bool, error) {
	slog.Debug("ConfigGet", "key", key)

	cfg, err := s.loadConfig()
	if err != nil {
		return "", false, err
	}
	value, ok := cfg[key]
	return value, ok, nil
}

// ConfigSet stores value under key, persisting it to disk. An empty value
// clears the key.
func (s *Store) ConfigSet(key, value string) error {
	slog.Debug("ConfigSet", "key", key, "value", value)

	cfg, err := s.loadConfig()
	if err != nil {
		return err
	}
	if value == "" {
		delete(cfg, key)
	} else {
		cfg[key] = value
	}
	return s.saveConfig(cfg)
}

// ConfigList returns a snapshot of all persisted configuration entries.
func (s *Store) ConfigList() (map[string]string, error) {
	return s.loadConfig()
}

// ResolveChipset resolves the host chipset without prompting: the configured
// "chipset" key wins, else a host probe, else "". Pass offline=true on the
// infer / serve path so the probe never touches the network.
func (s *Store) ResolveChipset(offline bool) string {
	if c, _, err := s.ConfigGet(ConfigKeyChipset); err == nil && c != "" {
		return c
	}
	detected, _ := geniex_sdk.ModelDetectChipset(offline)
	return detected
}
