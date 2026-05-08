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

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
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
	data, e := os.ReadFile(filepath.Join(dir, name, "geniex.json"))
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
		err := os.MkdirAll(modelDir, 0o770)
		if err != nil {
			errC <- err
			return
		}

		// Create modelfile for storing downloaded content
		resCh, errCh := model_hub.StartDownload(ctx, mf.Name, modelDir, needs)
		for d := range resCh {
			infoC <- d
		}
		for e := range errCh {
			errC <- e
			return
		}

		if err := model_hub.PostDownload(ctx, mf.Name, modelDir, &mf); err != nil {
			errC <- err
			return
		}

		model := types.ModelManifest{
			Name:          mf.Name,
			ModelName:     mf.ModelName,
			ModelType:     mf.ModelType,
			PluginId:      mf.PluginId,
			DeviceId:      mf.DeviceId,
			ModelFile:     mf.ModelFile,
			MMProjFile:    mf.MMProjFile,
			TokenizerFile: mf.TokenizerFile,
			ExtraFiles:    mf.ExtraFiles,
		}
		manifestPath := filepath.Join(modelDir, "geniex.json")
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
		manifestPath := filepath.Join(s.home, "models", nmf.Name, "geniex.json")
		manifestData, _ := sonic.Marshal(model) // JSON marshal won't fail, ignore error
		err = os.WriteFile(manifestPath, manifestData, 0o664)
		if err != nil {
			errC <- err
			return
		}
	}()

	return
}

// PartialSuffix is appended to a model directory while download + extract
// are in flight.
const PartialSuffix = ".partial"

// PullZipAsset downloads a single zip asset and
// extracts it flat into a staging directory.
func (s *Store) PullZipAsset(
	ctx context.Context,
	mf types.ModelManifest,
	downloadURL string,
	zipSize int64,
) (<-chan types.DownloadInfo, <-chan error) {
	infoC := make(chan types.DownloadInfo, 10)
	errC := make(chan error, 1)

	go func() {
		defer close(errC)
		defer close(infoC)

		// Conservative: zip + extracted payload + slack ~= 3x zipSize.
		if err := s.ensureEnoughDiskSpace(zipSize * 3); err != nil {
			errC <- err
			return
		}

		// Remove any healthy same-named model before re-pulling.
		finalDir := filepath.Join(s.home, "models", mf.Name)
		if _, err := os.Stat(filepath.Join(finalDir, "geniex.json")); err == nil {
			if rerr := s.Remove(mf.Name); rerr != nil {
				errC <- fmt.Errorf("remove existing model: %w", rerr)
				return
			}
		}

		if err := s.LockModel(mf.Name); err != nil {
			errC <- err
			return
		}
		defer s.UnlockModel(mf.Name)

		// LockModel created finalDir as a side-effect; remove it before staging.
		_ = os.RemoveAll(finalDir)
		partialDir := finalDir + PartialSuffix
		if err := os.RemoveAll(partialDir); err != nil {
			errC <- fmt.Errorf("clean stale partial: %w", err)
			return
		}
		if err := os.MkdirAll(partialDir, 0o770); err != nil {
			errC <- fmt.Errorf("mkdir partial: %w", err)
			return
		}

		zipName := filepath.Base(mf.Name) + ".zip"
		dlCh, dlErrCh := model_hub.StartDownloadURL(ctx, downloadURL, partialDir, zipName, zipSize)
		for d := range dlCh {
			infoC <- d
		}
		for e := range dlErrCh {
			errC <- e
			return
		}

		zipPath := filepath.Join(partialDir, zipName)
		fmt.Println(render.GetTheme().Info.Sprintf("Extracting %s...", zipName))
		res, err := aihub.ExtractFlat(zipPath, partialDir)
		if err != nil {
			errC <- fmt.Errorf("unzip %s: %w", zipName, err)
			return
		}
		if err := os.Remove(zipPath); err != nil {
			slog.Warn("PullZipAsset: failed to remove zip after extract", "path", zipPath, "err", err)
		}

		if err := finalizeAIHubManifest(mf, res, partialDir, finalDir); err != nil {
			errC <- err
			return
		}

		slog.Info("PullZipAsset complete",
			"model", mf.Name, "entrypoint", res.EntrypointBasename,
			"files", len(res.Files), "size", res.TotalSize)
	}()

	return infoC, errC
}

// finalizeAIHubManifest builds the geniex.json for an AI Hub (qairt) model
// from the flat-extracted contents of a zip asset and atomically promotes the
// staging directory into its final location.
func finalizeAIHubManifest(mf types.ModelManifest, res *aihub.ExtractResult, partialDir, finalDir string) error {
	// --- Step 1: auto-detect model type ---
	if mf.ModelType == "" {
		var hasMetadata bool
		for _, f := range res.Files {
			if f.Name == "metadata.json" {
				hasMetadata = true
				break
			}
		}
		if hasMetadata {
			metaBytes, readErr := os.ReadFile(filepath.Join(partialDir, "metadata.json"))
			if readErr != nil {
				slog.Warn("finalizeAIHubManifest: failed to read metadata.json, defaulting to LLM", "err", readErr)
				fmt.Println(render.GetTheme().Warning.Sprintf("Warning: could not read metadata.json (%s); defaulting model type to LLM", readErr))
				mf.ModelType = types.ModelTypeLLM
			} else {
				var meta qaihm.ModelMetadata
				if unmarshalErr := protojson.Unmarshal(metaBytes, &meta); unmarshalErr != nil {
					slog.Warn("finalizeAIHubManifest: failed to parse metadata.json, defaulting to LLM", "err", unmarshalErr)
					fmt.Println(render.GetTheme().Warning.Sprintf("Warning: could not parse metadata.json (%s); defaulting model type to LLM", unmarshalErr))
					mf.ModelType = types.ModelTypeLLM
				} else if meta.GetGenie().GetSupportsVision() {
					slog.Info("finalizeAIHubManifest: detected VLM via metadata.json supports_vision")
					mf.ModelType = types.ModelTypeVLM
				} else {
					slog.Info("finalizeAIHubManifest: supports_vision=false in metadata.json, treating as LLM")
					mf.ModelType = types.ModelTypeLLM
				}
			}
		} else {
			slog.Info("finalizeAIHubManifest: no metadata.json in zip, defaulting to LLM")
			mf.ModelType = types.ModelTypeLLM
		}
	}

	// --- Step 2: populate manifest fields from extract result ---
	mf.ModelFile = map[string]types.ModelFileInfo{
		"N/A": {
			Name:       res.EntrypointBasename,
			Downloaded: true,
			Size:       res.TotalSize,
		},
	}
	mf.MMProjFile = types.ModelFileInfo{}
	mf.TokenizerFile = types.ModelFileInfo{}
	mf.ExtraFiles = mf.ExtraFiles[:0]
	for _, f := range res.Files {
		if f.Name == res.EntrypointBasename {
			continue
		}
		mf.ExtraFiles = append(mf.ExtraFiles, f)
	}

	// --- Step 3: write geniex.json ---
	manifestPath := filepath.Join(partialDir, "geniex.json")
	manifestData, _ := sonic.Marshal(mf)
	if err := os.WriteFile(manifestPath, manifestData, 0o664); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}

	// --- Step 4: atomic promotion ---
	if err := os.Rename(partialDir, finalDir); err != nil {
		return fmt.Errorf("promote partial dir: %w", err)
	}

	return nil
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

// SetModelType updates the ModelType field in an already-downloaded model's
// geniex.json manifest. It is safe to call concurrently — the model lock is
// held for the duration of the read-modify-write.
func (s *Store) SetModelType(name string, modelType types.ModelType) error {
	if err := s.LockModel(name); err != nil {
		return err
	}
	defer s.UnlockModel(name)

	manifestPath := filepath.Join(s.home, "models", name, "geniex.json")
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return fmt.Errorf("read manifest: %w", err)
	}
	var mf types.ModelManifest
	if err := sonic.Unmarshal(data, &mf); err != nil {
		return fmt.Errorf("parse manifest: %w", err)
	}
	mf.ModelType = modelType
	out, _ := sonic.Marshal(mf)
	if err := os.WriteFile(manifestPath, out, 0o664); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	return nil
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
