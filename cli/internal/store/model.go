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
	"regexp"
	"slices"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/shirou/gopsutil/disk"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
)

// matches gguf multi-part suffix like "-00001-of-00003.gguf".
var ggufPartRegex = regexp.MustCompile(`-\d+-of-\d+\.gguf$`)

// readManifestAt reads and unmarshals the manifest file at the given directory
// path. It does NOT acquire the model lock — callers are responsible for
// locking when needed.
func readManifestAt(dir string) (*types.ModelManifest, error) {
	data, err := os.ReadFile(filepath.Join(dir, types.ManifestFileName))
	if err != nil {
		return nil, err
	}
	var mf types.ModelManifest
	if err := sonic.Unmarshal(data, &mf); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}
	return &mf, nil
}

// writeManifestAt marshals mf and writes it to types.ManifestFileName inside dir.
// It does NOT acquire the model lock — callers are responsible for locking
// when needed.
func writeManifestAt(dir string, mf types.ModelManifest) error {
	data, _ := sonic.Marshal(mf) // Marshal of a plain struct never fails
	if err := os.WriteFile(filepath.Join(dir, types.ManifestFileName), data, 0o664); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	return nil
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

// Remove deletes a cached model. Empty quant removes the whole model;
// otherwise only that quant's files are removed (and the model directory
// itself when no downloaded quants remain).
func (s *Store) Remove(name, quant string) error {
	slog.Debug("Remove model", "model", name, "quant", quant)

	if err := s.LockModel(name); err != nil {
		return err
	}
	defer s.UnlockModel(name)

	dir := s.ModelfilePath(name, "")
	if quant == "" {
		return os.RemoveAll(dir)
	}

	mf, err := readManifestAt(dir)
	if err != nil {
		return fmt.Errorf("read manifest: %w", err)
	}

	target, ok := mf.ModelFile[quant]
	if !ok || !target.Downloaded {
		return fmt.Errorf("quant %s not downloaded", quant)
	}

	remaining := 0
	for q, f := range mf.ModelFile {
		if q != quant && f.Downloaded {
			remaining++
		}
	}
	if remaining == 0 {
		return os.RemoveAll(dir)
	}

	// Keep manifest entries (Downloaded=false) so a later pull re-fetches all fragments.
	base := ggufPartRegex.ReplaceAllString(target.Name, "")
	toDelete := []string{target.Name}
	for i, f := range mf.ExtraFiles {
		if f.Downloaded && ggufPartRegex.ReplaceAllString(f.Name, "") == base {
			toDelete = append(toDelete, f.Name)
			mf.ExtraFiles[i].Downloaded = false
		}
	}

	for _, fname := range toDelete {
		if err := os.Remove(filepath.Join(dir, fname)); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("remove %s: %w", fname, err)
		}
	}

	mf.ModelFile[quant] = types.ModelFileInfo{
		Name:       target.Name,
		Downloaded: false,
		Size:       target.Size,
	}

	return writeManifestAt(dir, *mf)
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
		if err := s.Remove(model, ""); err != nil {
			slog.Warn("Failed to remove model", "model", model, "err", err)
			continue
		}
		count += 1
	}

	return count
}

func (s *Store) GetManifest(name string) (*types.ModelManifest, error) {
	if err := s.LockModel(name); err != nil {
		return nil, err
	}
	defer s.UnlockModel(name)

	return readManifestAt(s.ModelfilePath(name, ""))
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

		modelDir := s.ModelfilePath(mf.Name, "")
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
			if err := s.Remove(mf.Name, ""); err != nil {
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
				needs = append(needs, model_hub.ModelFileInfo{Name: f.Name, Size: f.Size})
			}
		}
		if mf.MMProjFile.Name != "" {
			if mf.MMProjFile.Downloaded {
				needs = append(needs, model_hub.ModelFileInfo{Name: mf.MMProjFile.Name, Size: mf.MMProjFile.Size})
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

		if err := writeManifestAt(s.ModelfilePath(mf.Name, ""), mf); err != nil {
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
				needs = append(needs, model_hub.ModelFileInfo{Name: f.Name, Size: f.Size})
			}
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
		err := os.MkdirAll(s.ModelfilePath(nmf.Name, ""), 0o770)
		if err != nil {
			errC <- err
			return
		}

		resCh, errCh := model_hub.StartDownload(ctx, nmf.Name, s.ModelfilePath(nmf.Name, ""), needs)
		for d := range resCh {
			infoC <- d
		}
		for e := range errCh {
			errC <- e
			return
		}

		if err := writeManifestAt(s.ModelfilePath(nmf.Name, ""), nmf); err != nil {
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

// SetModelType updates the ModelType field in an already-downloaded model's
// manifest. It is safe to call concurrently — the model lock is held for the
// duration of the read-modify-write.
func (s *Store) SetModelType(name string, modelType types.ModelType) error {
	if err := s.LockModel(name); err != nil {
		return err
	}
	defer s.UnlockModel(name)

	dir := s.ModelfilePath(name, "")
	mf, err := readManifestAt(dir)
	if err != nil {
		return fmt.Errorf("read manifest: %w", err)
	}
	mf.ModelType = modelType
	return writeManifestAt(dir, *mf)
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
