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
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/config"
)

// Store resolves the geniex data directory for the CLI. Model storage itself
// is owned by the SDK model-manager; this only exposes the data-dir root (for
// the config file, update cache, and REPL history) via DataPath().
type Store struct {
	home string
}

var (
	instance *Store
	once     sync.Once
)

// Get returns the singleton instance of Store.
func Get() *Store {
	once.Do(func() {
		instance = &Store{}
		instance.init()
	})
	return instance
}

// init resolves the data directory. The SDK model-manager creates the models
// subdirectory itself, so the store only needs the root to exist.
func (s *Store) init() {
	if config.Get().DataDir != "" {
		s.home = config.Get().DataDir
	} else {
		homeDir, e := os.UserHomeDir()
		if e != nil {
			fmt.Fprintf(os.Stderr, "geniex: failed to resolve user home directory: %s\n", e)
			os.Exit(1)
		}
		s.home = filepath.Join(homeDir, ".cache", "geniex")
	}
	slog.Info("Using data directory", "path", s.home)

	if e := os.MkdirAll(s.home, 0o770); e != nil {
		fmt.Fprintf(os.Stderr, "geniex: failed to create data directory %s: %s\n", s.home, e)
		os.Exit(1)
	}
}

// DataPath returns the geniex data directory root.
func (s *Store) DataPath() string {
	return s.home
}
