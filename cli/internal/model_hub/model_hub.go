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
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"

	"github.com/bytedance/sonic"
	"golang.org/x/sync/errgroup"
	"resty.dev/v3"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/config"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
)

const ProgressSuffix = ".progress"

// Plugin IDs used in ModelManifest.PluginId.
const (
	PluginLlamaCpp = "llama_cpp"
	PluginQairt    = "qairt"
)

type ModelFileInfo struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
}

// ChoosePluginId returns llama_cpp for GGUF, qairt for a single genie .zip
// or a layout containing genie_config.json, and errors otherwise.
func ChoosePluginId(files []ModelFileInfo) (string, error) {
	slog.Debug("choosing plugin", "files", files)
	// gguf
	for _, f := range files {
		if strings.HasSuffix(strings.ToLower(f.Name), ".gguf") {
			return PluginLlamaCpp, nil
		}
	}
	// qairt zip
	if len(files) == 1 {
		name := strings.ToLower(files[0].Name)
		if strings.HasSuffix(name, ".zip") && strings.Contains(name, "genie") {
			return PluginQairt, nil
		}
	}
	// qairt folder
	for _, f := range files {
		if filepath.Base(f.Name) == "genie_config.json" {
			return PluginQairt, nil
		}
	}
	return "", fmt.Errorf("neither gguf nor aihub format detected, cannot determine plugin")
}

type ModelHub interface {
	MaxConcurrency() int
	CheckAvailable(ctx context.Context, modelName string) error
	ModelInfo(ctx context.Context, modelName string) ([]ModelFileInfo, error)
	GetFileContent(ctx context.Context, modelName, fileName string, offset, limit int64, writer io.Writer) error
	PostDownload(ctx context.Context, modelName, outputDir string, mf *types.ModelManifest) error
}

var hubs = []ModelHub{}

var errUnavailable = fmt.Errorf("no model hub contains the model")

// SetHub replaces the hub list with a single hub.
func SetHub(h ModelHub) {
	hubs = []ModelHub{h}
}

// RegisterHub prepends h to the hub list. Used by downstream packages
// (e.g. store) to plug in hubs needing runtime dependencies the model_hub
// package cannot take directly.
func RegisterHub(h ModelHub) {
	hubs = append([]ModelHub{h}, hubs...)
}

func ModelInfo(ctx context.Context, modelName string) ([]ModelFileInfo, *types.ModelManifest, error) {
	slog.Debug("fetching model info", "model", modelName)

	hub, err := getHub(ctx, modelName)
	if err != nil {
		return nil, nil, err
	}

	files, err := hub.ModelInfo(ctx, modelName)
	if err != nil {
		return nil, nil, err
	}

	// Strip the manifest entry (if any) from files; parse it separately.
	var hasManifest bool
	for i := 0; i < len(files); i++ {
		if files[i].Name == types.ManifestFileName {
			files = append(files[:i], files[i+1:]...)
			hasManifest = true
			break
		}
	}
	if !hasManifest {
		return files, nil, nil
	}

	data, err := GetFileContent(ctx, modelName, types.ManifestFileName)
	if err != nil {
		slog.Warn("failed to get manifest file, ignore", "error", err)
		return nil, nil, err
	}

	var manifest types.ModelManifest
	if err := sonic.Unmarshal(data, &manifest); err != nil {
		slog.Warn("failed to parse manifest file, ignore", "error", err)
		return nil, nil, err
	}

	return files, &manifest, nil

}

func PostDownload(ctx context.Context, modelName, outputDir string, mf *types.ModelManifest) error {
	hub, err := getHub(ctx, modelName)
	if err != nil {
		return err
	}
	return hub.PostDownload(ctx, modelName, outputDir, mf)
}

func GetFileContent(ctx context.Context, modelName, fileName string) ([]byte, error) {
	slog.Debug("fetching file content", "model", modelName, "file", fileName)

	hub, err := getHub(ctx, modelName)
	if err != nil {
		return nil, err
	}

	buf := bytes.NewBuffer(nil)
	if err := hub.GetFileContent(ctx, modelName, fileName, 0, 0, buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

const minChunkSize = 16 * 1024 * 1024 // 16MiB

// chunkFetcher writes the bytes [offset, offset+limit) into w. Must be
// safe for concurrent use.
type chunkFetcher func(ctx context.Context, offset, limit int64, w io.Writer) error

// downloadChunked pre-allocates outPath, dispatches missing chunks via g,
// and reports per-chunk progress on resCh. Resume is supported through
// the marker sidecar file at markerPath.
//
// downloaded is the caller's running total (shared across files);
// totalSize is the grand total for progress reporting. The caller waits
// on g and removes markerPath on success.
func downloadChunked(
	ctx context.Context,
	outPath, markerPath string,
	size, totalSize int64,
	downloaded *int64,
	g *errgroup.Group,
	resCh chan<- types.DownloadInfo,
	fetch chunkFetcher,
) error {
	chunkSize := max(minChunkSize, size/128)
	nChunks := int((size + chunkSize - 1) / chunkSize)

	// Initialise or reuse marker file.
	markers, err := os.ReadFile(markerPath)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err != nil || len(markers) != nChunks {
		markers = make([]byte, nChunks)
		if werr := os.WriteFile(markerPath, markers, 0o644); werr != nil {
			return werr
		}
	}

	// Pre-allocate output file.
	f, err := os.OpenFile(outPath, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return err
	}
	if fi, _ := f.Stat(); fi == nil || fi.Size() < size {
		if err := f.Truncate(size); err != nil {
			f.Close()
			return err
		}
	}
	f.Close()

	for i, marker := range markers {
		if marker == 0x01 {
			atomic.AddInt64(downloaded, min(chunkSize, size-int64(i)*chunkSize))
			continue
		}
		offset := int64(i) * chunkSize
		limit := min(chunkSize, size-offset)
		idx := i
		g.Go(func() error {
			f, err := os.OpenFile(outPath, os.O_WRONLY, 0o644)
			if err != nil {
				return err
			}
			defer f.Close()
			if _, err := f.Seek(offset, io.SeekStart); err != nil {
				return err
			}
			if err := fetch(ctx, offset, limit, f); err != nil {
				return err
			}
			m, err := os.OpenFile(markerPath, os.O_WRONLY, 0o644)
			if err != nil {
				return err
			}
			defer m.Close()
			if _, err := m.WriteAt([]byte{0x01}, int64(idx)); err != nil {
				return err
			}
			resCh <- types.DownloadInfo{
				TotalDownloaded: atomic.AddInt64(downloaded, limit),
				TotalSize:       totalSize,
			}
			return nil
		})
	}
	return nil
}

func StartDownload(ctx context.Context, modelName, outputPath string, files []ModelFileInfo) (resChan chan types.DownloadInfo, errChan chan error) {
	slog.Info("Starting download", "model", modelName, "outputPath", outputPath, "files", files)

	hub, err := getHub(ctx, modelName)
	if err != nil {
		resCh := make(chan types.DownloadInfo)
		errCh := make(chan error, 1)
		close(resCh)
		errCh <- err
		close(errCh)
		return resCh, errCh
	}

	maxConcurrency := hub.MaxConcurrency()
	resCh := make(chan types.DownloadInfo)
	errCh := make(chan error, maxConcurrency)
	slog.Info("GetHub", "hub", reflect.TypeOf(hub), "maxConcurrency", maxConcurrency)

	go func() {
		defer close(errCh)
		defer close(resCh)

		var totalSize int64
		for _, f := range files {
			totalSize += f.Size
		}
		var downloaded int64
		var markerPaths []string
		g, gctx := errgroup.WithContext(ctx)
		g.SetLimit(maxConcurrency)

		for _, f := range files {
			if err := os.MkdirAll(filepath.Dir(filepath.Join(outputPath, f.Name)), 0o755); err != nil {
				errCh <- fmt.Errorf("failed to create directory: %v, %s", err, f.Name)
				return
			}
			outPath := filepath.Join(outputPath, f.Name)
			markerPath := filepath.Join(outputPath, f.Name+ProgressSuffix)
			markerPaths = append(markerPaths, markerPath)

			slog.Info("Download file", "name", f.Name, "size", f.Size)

			fetch := func(ctx context.Context, offset, limit int64, w io.Writer) error {
				return hub.GetFileContent(ctx, modelName, f.Name, offset, limit, w)
			}
			if err := downloadChunked(gctx, outPath, markerPath, f.Size, totalSize, &downloaded, g, resCh, fetch); err != nil {
				errCh <- err
				return
			}
		}

		if err := g.Wait(); err != nil {
			errCh <- err
			return
		}
		for _, p := range markerPaths {
			_ = os.Remove(p)
		}
		slog.Info("download complete", "model", modelName, "outputPath", outputPath)
	}()

	return resCh, errCh
}

// QuantPriority mirrors `QUANT_PRIORITY` in
// sdk/model-manager/crates/core/src/manifest_builder.rs. Update both sides
// together — pull (chooseFiles) and load (KeepAliveGet) read from the same
// list so a no-quant pull and a no-quant load agree on which file wins.
var QuantPriority = []string{"Q8_0", "Q4_K_M", "Q4_0"}

// PickDefaultQuant returns the highest-priority quant from `available`
// according to QuantPriority. Quants outside the priority list fall back to
// lexicographic min, matching the historical `slices.Min` behaviour.
// Caller must ensure the slice is non-empty.
func PickDefaultQuant(available []string) string {
	for _, p := range QuantPriority {
		if slices.Contains(available, p) {
			return p
		}
	}
	return slices.Min(available)
}

// NormalizeModelName splits "<name>[:<quant>]" and applies shortcuts:
// bare name → "qualcomm/<name>", HF URL → repo path.
func NormalizeModelName(name string) (string, string) {
	parts := strings.SplitN(name, ":", 2)
	name = parts[0]
	quant := ""
	if len(parts) == 2 {
		quant = strings.ToUpper(parts[1])
	}

	if actualName, exists := config.GetModelMapping(name); exists {
		return actualName, quant
	}
	if !strings.Contains(name, "/") {
		return "qualcomm/" + name, quant
	}
	if strings.HasPrefix(name, HF_ENDPOINT) {
		return strings.TrimPrefix(name, HF_ENDPOINT+"/"), quant
	}
	return name, quant
}

func getHub(ctx context.Context, modelName string) (ModelHub, error) {
	if len(hubs) == 1 {
		h := hubs[0]
		slog.Info("specified single hub", "hub", reflect.TypeOf(h))
		return h, h.CheckAvailable(ctx, modelName)
	}
	for _, h := range hubs {
		if err := h.CheckAvailable(ctx, modelName); err != nil {
			slog.Warn("hub not available, try next", "hub", reflect.TypeOf(h), "err", err)
		} else {
			slog.Info("hub available", "hub", reflect.TypeOf(h))
			return h, nil
		}
	}
	return nil, errUnavailable
}

func code2error(client *resty.Client, response *resty.Response) error {
	switch response.StatusCode() {
	case http.StatusOK:
		return nil
	case http.StatusNotFound:
		return fmt.Errorf("%w: %s", ErrModelNotFound, response.Request.URL)
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("%w: %s", ErrAuthRequired, response.Request.URL)
	default:
		return fmt.Errorf("%w: %s", ErrUnreachable, response.Status())
	}
}
