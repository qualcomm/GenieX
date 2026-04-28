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
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/gofrs/flock"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/config"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub"
)

type Store struct {
	home       string
	modelLocks sync.Map // Model locks mapping map[string]*flock.Flock
}

var (
	instance *Store
	once     sync.Once
)

// Get returns the singleton instance of Store
func Get() *Store {
	once.Do(func() {
		instance = &Store{}
		instance.init()
	})
	return instance
}

// init sets up the store's directory structure
func (s *Store) init() {
	if config.Get().DataDir != "" {
		s.home = config.Get().DataDir
	} else {
		// Get user's cache directory (OS-specific)
		homeDir, e := os.UserHomeDir()
		if e != nil {
			panic(e)
		}

		// Set geniex cache directory
		s.home = filepath.Join(homeDir, ".cache", "geniex")
	}
	slog.Info("Using data directory", "path", s.home)

	// Create models directory structure
	for _, d := range []string{"models"} {
		e := os.MkdirAll(filepath.Join(s.home, d), 0o770)
		if e != nil {
			panic(e)
		}
	}

	s.cleanCorruptedDirectories()
}

func (s *Store) Close() error {
	s.modelLocks.Range(func(key, value any) bool {
		fl := value.(*flock.Flock)
		if fl != nil {
			fl.Unlock()
		}
		s.modelLocks.Delete(key)
		return true
	})

	return nil
}

func (s *Store) cleanCorruptedDirectories() {
	// Sweep stale *.partial directories left by a killed AI Hub pull.
	// On success the pull atomically renames the .partial suffix away;
	// any remaining dir is by definition incomplete.
	s.cleanPartialDirs()

	models, err := s.scanModelDir()
	if err != nil {
		slog.Error("Failed to scan model directory", "err", err)
		return
	}

	for _, models := range models {
		slog.Info("Checking model directory", "name", models)
		if s.isCorruptedModelDirectory(models) {
			if err := s.LockModel(models); err != nil {
				slog.Warn("Skipping cleanup of directory", "name", models, "err", err)
				continue
			}

			slog.Info("Cleaning corrupted model directory", "name", models)
			if err := os.RemoveAll(s.ModelfilePath(models, "")); err != nil {
				slog.Error("Failed to remove corrupted directory", "name", models, "err", err)
			}

			s.UnlockModel(models)
		}
	}
}

func (s *Store) isCorruptedModelDirectory(name string) bool {
	manifestPath := s.ModelfilePath(name, "geniex.json")
	if _, err := os.Stat(manifestPath); err == nil {
		return false
	}
	dir := s.ModelfilePath(name, "")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return true
	}
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), model_hub.ProgressSuffix) {
			return false
		}
	}
	slog.Info("Cleaning corrupted model directory", "name", name)
	return true
}

// cleanPartialDirs removes stale <name>.partial directories at any depth
// under the models root. Used on Store init to recover from a crash during
// an AI Hub zip download/extract.
func (s *Store) cleanPartialDirs() {
	root := s.ModelDirPath()
	orgs, err := os.ReadDir(root)
	if err != nil {
		return
	}
	for _, org := range orgs {
		if !org.IsDir() {
			continue
		}
		orgPath := filepath.Join(root, org.Name())
		children, err := os.ReadDir(orgPath)
		if err != nil {
			continue
		}
		for _, c := range children {
			if !c.IsDir() {
				continue
			}
			if strings.HasSuffix(c.Name(), PartialSuffix) {
				victim := filepath.Join(orgPath, c.Name())
				slog.Info("Removing stale partial model directory", "path", victim)
				if err := os.RemoveAll(victim); err != nil {
					slog.Warn("Failed to remove stale partial directory", "path", victim, "err", err)
				}
			}
		}
	}
}
