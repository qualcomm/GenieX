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

package model_hub

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
)

// LocalFS side-loads a model from a local directory or a single .zip file.
type LocalFS struct {
	basePath string
}

func NewLocalFS(base string) *LocalFS {
	return &LocalFS{base}
}

func (d *LocalFS) MaxConcurrency() int {
	return 4
}

// isZipPath: basePath is a single .zip (extracted in PostDownload) rather
// than a directory walked recursively.
func (d *LocalFS) isZipPath() bool {
	info, err := os.Stat(d.basePath)
	if err != nil {
		return false
	}
	return !info.IsDir() && strings.HasSuffix(strings.ToLower(d.basePath), ".zip")
}

func (d *LocalFS) CheckAvailable(ctx context.Context, name string) error {
	info, err := os.Stat(d.basePath)
	if err != nil {
		return err
	}
	if info.IsDir() || strings.HasSuffix(strings.ToLower(d.basePath), ".zip") {
		return nil
	}
	return os.ErrNotExist
}

// PostDownload finalises the manifest by PluginId: GGUF → mmproj-based
// ModelType, qairt → unzip (if needed) + metadata.json.
func (d *LocalFS) PostDownload(ctx context.Context, modelName, outputDir string, mf *types.ModelManifest) error {
	// gguf model
	if mf.PluginId == PluginLlamaCpp {
		applyGGUFModelType(mf)
		return nil
	}
	// qairt zip
	if d.isZipPath() {
		if err := extractQairtZip(filepath.Join(outputDir, filepath.Base(d.basePath)), outputDir, mf); err != nil {
			return err
		}
	}
	// qairt folder
	applyQairtMetadata(outputDir, mf)
	return nil
}

func (d *LocalFS) ModelInfo(ctx context.Context, name string) ([]ModelFileInfo, error) {
	if d.isZipPath() {
		info, err := os.Stat(d.basePath)
		if err != nil {
			return nil, err
		}
		return []ModelFileInfo{{Name: filepath.Base(d.basePath), Size: info.Size()}}, nil
	}

	res := make([]ModelFileInfo, 0)
	err := filepath.Walk(d.basePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			relPath, err := filepath.Rel(d.basePath, path)
			if err != nil {
				return err
			}
			res = append(res, ModelFileInfo{
				Name: relPath,
				Size: info.Size(),
			})
		}
		return nil
	})
	return res, err
}

func (d *LocalFS) GetFileContent(ctx context.Context, modelName, fileName string, offset, limit int64, writer io.Writer) error {
	path := filepath.Join(d.basePath, fileName)
	if d.isZipPath() {
		// basePath is the zip itself; fileName is its basename.
		path = d.basePath
	}
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// seek to offset
	if offset > 0 {
		_, err = file.Seek(offset, io.SeekStart)
		if err != nil {
			return err
		}
	}

	var reader io.Reader = file
	if limit > 0 {
		reader = io.LimitReader(file, limit)
	}

	_, err = io.Copy(writer, reader)
	if err != nil {
		return err
	}

	return nil
}
