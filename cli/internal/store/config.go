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

package store

import (
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"sync"

	"github.com/bytedance/sonic"
)

// Known configuration keys. The set is intentionally small and explicit so
// that invalid keys can be rejected at the CLI boundary.
const (
	ConfigKeyDevice = "device"
)

// ConfigKeys is the list of all known configuration keys.
var ConfigKeys = []string{
	ConfigKeyDevice,
}

// IsValidConfigKey returns true if key is a known configuration key.
func IsValidConfigKey(key string) bool {
	for _, k := range ConfigKeys {
		if k == key {
			return true
		}
	}
	return false
}

const configFileName = "config.json"

var configMu sync.Mutex

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
	defer func() {
		// If rename succeeded the temp file is already gone; ignore error.
		_ = os.Remove(tmpName)
	}()

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Chmod(0o600); err != nil {
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

	configMu.Lock()
	defer configMu.Unlock()

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

	configMu.Lock()
	defer configMu.Unlock()

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
	configMu.Lock()
	defer configMu.Unlock()

	return s.loadConfig()
}
