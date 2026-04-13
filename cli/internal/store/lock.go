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

	"github.com/gofrs/flock"
)

func (s *Store) LockModel(modelName string) error {
	slog.Debug("LockModel", "modelName", modelName)

	if modelName == "" {
		return ErrModelNameEmpty
	}

	modelsDir := s.ModelDirPath()
	os.MkdirAll(filepath.Join(modelsDir, modelName), 0o770)
	lockPath := filepath.Join(modelsDir, modelName+".lock")

	fl := flock.New(lockPath)

	locked, err := fl.TryLock()
	if err != nil {
		return err
	}
	if !locked {
		return ErrModelLocked
	}

	s.modelLocks.Store(modelName, fl)
	return nil
}

func (s *Store) UnlockModel(modelName string) error {
	slog.Debug("UnLockModel", "modelName", modelName)

	if modelName == "" {
		return nil
	}

	if info, ok := s.modelLocks.Load(modelName); ok {
		fl := info.(*flock.Flock)
		fl.Unlock()
		s.modelLocks.Delete(modelName)
	}

	return nil
}
