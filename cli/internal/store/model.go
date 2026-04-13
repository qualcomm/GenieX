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
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/shirou/gopsutil/disk"

	"github.com/qcom-it-nexa-ai/geniex/internal/model_hub"
	"github.com/qcom-it-nexa-ai/geniex/internal/types"
)

// isBundlePath checks if a path is a macOS bundle (.mlmodelc or .mlpackage)
// These are directory references, not downloadable files
func isBundlePath(path string) bool {
	return strings.HasSuffix(strings.ToLower(path), ".mlmodelc") ||
		strings.HasSuffix(strings.ToLower(path), ".mlpackage")
}

func (s *Store) ensureEnoughDiskSpace(requiredBytes int64) error {
	usage, err := disk.Usage(s.ModelDirPath())
	if err != nil {
		return err
	}

	free := int64(usage.Free)
	slog.Debug("Disk space check", "required_bytes", requiredBytes, "free_bytes", free)
	if free < requiredBytes {
		return fmt.Errorf("not enough disk space: required %d bytes, available %d bytes", requiredBytes, free)
	}

	return nil
}

// List returns all locally stored models by reading their manifest files
func (s *Store) List() ([]types.ModelManifest, error) {
	res := make([]types.ModelManifest, 0)
	models, err := s.scanModelDir()
	if err != nil {
		return nil, err
	}
	for _, model := range models {
		// Parse each model directory's manifest
		model, err := s.GetManifest(model)
		if err != nil {
			slog.Warn("GetManifest Error", "err", err)
			continue
		}

		res = append(res, *model)
	}

	return res, nil
}

// Remove deletes a specific model and all its files
func (s *Store) Remove(name string) error {
	slog.Debug("Remove model", "model", name)

	err := s.LockModel(name)
	if err != nil {
		return err
	}
	defer s.UnlockModel(name)
	return os.RemoveAll(filepath.Join(s.home, "models", name))
}

// Clean removes all stored models and the models directory
func (s *Store) Clean() int {
	slog.Debug("Start clean model")

	models, err := s.scanModelDir()
	if err != nil {
		return 0
	}

	// Get list of all model names to remove
	count := 0
	for _, model := range models {
		if err := s.Remove(model); err != nil {
			slog.Warn("Failed to remove model", "model", model, "err", err)
			continue
		}
		count += 1
	}

	return count
}

func (s *Store) GetManifest(name string) (*types.ModelManifest, error) {
	err := s.LockModel(name)
	if err != nil {
		return nil, err
	}
	defer s.UnlockModel(name)

	dir := filepath.Join(s.home, "models")
	// Read manifest file
	data, e := os.ReadFile(filepath.Join(dir, name, "geniex.manifest"))
	if e != nil {
		return nil, e
	}

	// Parse manifest JSON
	model := types.ModelManifest{}
	e = sonic.Unmarshal(data, &model)
	if e != nil {
		return nil, e
	}
	return &model, nil
}

// Pull downloads a model from HuggingFace and stores it locally
// It fetches the model tree, finds .gguf files, downloads them, and saves metadata
// if model not specify, all is set true, and autodetect true
func (s *Store) Pull(ctx context.Context, mf types.ModelManifest) (infoCh <-chan types.DownloadInfo, errCh <-chan error) {
	infoC := make(chan types.DownloadInfo, 10)
	infoCh = infoC
	errC := make(chan error, 1)
	errCh = errC

	go func() {
		defer close(errC)
		defer close(infoC)

		// check free disk space
		if err := s.ensureEnoughDiskSpace(mf.GetSize()); err != nil {
			errC <- err
			return
		}

		modelDir := filepath.Join(s.home, "models", mf.Name)
		hasProgress := false
		if entries, _ := os.ReadDir(modelDir); entries != nil {
			for _, e := range entries {
				if !e.IsDir() && strings.HasSuffix(e.Name(), model_hub.ProgressSuffix) {
					hasProgress = true
					break
				}
			}
		}
		if !hasProgress {
			if err := s.Remove(mf.Name); err != nil {
				errC <- err
				return
			}
		}

		if err := s.LockModel(mf.Name); err != nil {
			errC <- err
			return
		}
		defer s.UnlockModel(mf.Name)

		// filter download file
		var needs []model_hub.ModelFileInfo
		for _, f := range mf.ModelFile {
			if f.Downloaded {
				// Skip bundle paths (.mlmodelc and .mlpackage) - they are directory references, not files
				// Only the individual files within the bundles will be downloaded via ExtraFiles
				if isBundlePath(f.Name) {
					continue
				}
				needs = append(needs, model_hub.ModelFileInfo{Name: f.Name, Size: f.Size})
			}
		}
		if mf.MMProjFile.Name != "" {
			if mf.MMProjFile.Downloaded {
				needs = append(needs, model_hub.ModelFileInfo{Name: mf.MMProjFile.Name, Size: mf.MMProjFile.Size})
			}
		}
		if mf.TokenizerFile.Name != "" {
			if mf.TokenizerFile.Downloaded {
				needs = append(needs, model_hub.ModelFileInfo{Name: mf.TokenizerFile.Name, Size: mf.TokenizerFile.Size})
			}
		}
		for _, f := range mf.ExtraFiles {
			if f.Downloaded {
				needs = append(needs, model_hub.ModelFileInfo{Name: f.Name, Size: f.Size})
			}
		}

		// Create model directory structure
		err := os.MkdirAll(filepath.Join(s.home, "models", mf.Name), 0o770)
		if err != nil {
			errC <- err
			return
		}

		// Create modelfile for storing downloaded content
		resCh, errCh := model_hub.StartDownload(ctx, mf.Name, filepath.Join(s.home, "models", mf.Name), needs)
		for d := range resCh {
			infoC <- d
		}
		for e := range errCh {
			errC <- e
			return
		}

		model := types.ModelManifest{
			Name:          mf.Name,
			ModelName:     mf.ModelName,
			ModelType:     mf.ModelType,
			PluginId:      mf.PluginId,
			DeviceId:      mf.DeviceId,
			MinSDKVersion: mf.MinSDKVersion,
			ModelFile:     mf.ModelFile,
			MMProjFile:    mf.MMProjFile,
			TokenizerFile: mf.TokenizerFile,
			ExtraFiles:    mf.ExtraFiles,
		}
		manifestPath := filepath.Join(s.home, "models", mf.Name, "geniex.manifest")
		manifestData, _ := sonic.Marshal(model) // JSON marshal won't fail, ignore error
		err = os.WriteFile(manifestPath, manifestData, 0o664)
		if err != nil {
			errC <- err
			return
		}
	}()

	return
}

func (s *Store) PullExtraQuant(ctx context.Context, omf, nmf types.ModelManifest) (infoCh <-chan types.DownloadInfo, errCh <-chan error) {
	infoC := make(chan types.DownloadInfo, 10)
	infoCh = infoC
	errC := make(chan error, 1)
	errCh = errC

	go func() {
		defer close(errC)
		defer close(infoC)

		if err := s.LockModel(nmf.Name); err != nil {
			errC <- err
			return
		}
		defer s.UnlockModel(nmf.Name)

		// filter download file
		var needs []model_hub.ModelFileInfo
		for q, f := range nmf.ModelFile {
			if f.Downloaded && !omf.ModelFile[q].Downloaded {
				// Skip bundle paths (.mlmodelc and .mlpackage) - they are directory references, not files
				if isBundlePath(f.Name) {
					continue
				}
				needs = append(needs, model_hub.ModelFileInfo{Name: f.Name, Size: f.Size})
			}
		}
		if nmf.TokenizerFile.Downloaded && !omf.TokenizerFile.Downloaded {
			needs = append(needs, model_hub.ModelFileInfo{Name: nmf.TokenizerFile.Name, Size: nmf.TokenizerFile.Size})
		}
		for q, f := range nmf.ExtraFiles {
			if f.Downloaded && !omf.ExtraFiles[q].Downloaded {
				needs = append(needs, model_hub.ModelFileInfo{Name: f.Name, Size: f.Size})
			}
		}

		// check free disk space
		totalNeeded := int64(0)
		for _, n := range needs {
			totalNeeded += n.Size
		}
		if err := s.ensureEnoughDiskSpace(totalNeeded); err != nil {
			errC <- err
			return
		}

		// Create model directory structure
		err := os.MkdirAll(filepath.Join(s.home, "models", nmf.Name), 0o770)
		if err != nil {
			errC <- err
			return
		}

		resCh, errCh := model_hub.StartDownload(ctx, nmf.Name, filepath.Join(s.home, "models", nmf.Name), needs)
		for d := range resCh {
			infoC <- d
		}
		for e := range errCh {
			errC <- e
			return
		}

		model := types.ModelManifest{
			Name:          nmf.Name,
			ModelName:     nmf.ModelName,
			ModelType:     nmf.ModelType,
			PluginId:      nmf.PluginId,
			DeviceId:      nmf.DeviceId,
			ModelFile:     nmf.ModelFile,
			MMProjFile:    nmf.MMProjFile,
			TokenizerFile: nmf.TokenizerFile,
			ExtraFiles:    nmf.ExtraFiles,
		}
		manifestPath := filepath.Join(s.home, "models", nmf.Name, "geniex.manifest")
		manifestData, _ := sonic.Marshal(model) // JSON marshal won't fail, ignore error
		err = os.WriteFile(manifestPath, manifestData, 0o664)
		if err != nil {
			errC <- err
			return
		}
	}()

	return
}

func (s *Store) DataPath() string {
	return s.home
}

func (s *Store) ModelDirPath() string {
	return filepath.Join(s.home, "models")
}

// ModelfilePath returns the full path to a model's data file
func (s *Store) ModelfilePath(name string, file string) string {
	return filepath.Join(s.home, "models", name, file)
}

func (s *Store) scanModelDir() ([]string, error) {
	orgs, e := os.ReadDir(s.ModelDirPath())
	if e != nil {
		slog.Warn("Failed to read model directory", "err", e)
		return nil, e
	}

	// Parse each model directory's manifest
	res := make([]string, 0)
	for _, org := range orgs {
		if !org.IsDir() {
			continue
		}

		ignoreDirs := []string{".cache"}
		if slices.Contains(ignoreDirs, org.Name()) {
			continue
		}

		repos, e := os.ReadDir(filepath.Join(s.ModelDirPath(), org.Name()))
		if e != nil {
			slog.Warn("Failed to read model subdirectory", "org", org.Name(), "err", e)
			continue
		}

		for _, repo := range repos {
			if !repo.IsDir() {
				continue
			}

			res = append(res, org.Name()+"/"+repo.Name())
		}
	}

	return res, nil
}
