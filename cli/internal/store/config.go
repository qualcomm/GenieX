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
)

func (s *Store) ConfigGet(key string) (string, error) {
	slog.Debug("ConfigGet", "key", key)
	data, err := os.ReadFile(filepath.Join(s.home, "config"))
	return string(data), err
}

func (s *Store) ConfigSet(key string, value string) error {
	slog.Debug("ConfigSet", "key", key, "value", value)
	return os.WriteFile(filepath.Join(s.home, "config"), []byte(value), 0o600)
}
