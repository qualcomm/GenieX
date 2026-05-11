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
	"sync/atomic"

	"github.com/bytedance/sonic"
	"golang.org/x/sync/errgroup"
	"resty.dev/v3"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/downloader"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
)

const ProgressSuffix = ".progress"

type ModelFileInfo struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
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

// Specify hub to use
func SetHub(h ModelHub) {
	hubs = []ModelHub{h}
}

// RegisterHub prepends h to the hub list. Used by downstream packages
// (e.g. store) to plug in hubs that need runtime dependencies the
// model_hub package can't take directly, like the AI Hub hub which
// needs access to the chipset configuration and on-disk cache dir.
func RegisterHub(h ModelHub) {
	hubs = append([]ModelHub{h}, hubs...)
}

// list model files

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

	// check manifest available
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

	// parse manifest
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

// PostDownload resolves the active hub for modelName and delegates to it.
// See ModelHub.PostDownload.
func PostDownload(ctx context.Context, modelName, outputDir string, mf *types.ModelManifest) error {
	hub, err := getHub(ctx, modelName)
	if err != nil {
		return err
	}
	return hub.PostDownload(ctx, modelName, outputDir, mf)
}

// Get single file content

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

// Batch download

const (
	minChunkSize = 16 * 1024 * 1024 // 16MiB
)

// chunkFetcher writes the bytes [offset, offset+limit) of some remote resource
// into w. Implementations must be safe for concurrent use.
type chunkFetcher func(ctx context.Context, offset, limit int64, w io.Writer) error

// downloadChunked is the shared engine behind StartDownload and
// StartDownloadURL. It pre-allocates outPath to size bytes, initialises (or
// reuses) a marker sidecar file for resume support, and dispatches missing
// chunks to the errgroup g. Completed-chunk progress is reported on resCh.
//
// downloaded must be a pointer to the caller's running total (shared across
// multiple files in StartDownload). totalSize is the grand total used only for
// progress reporting.
//
// The caller is responsible for calling g.Wait() and removing markerPath on
// clean completion.
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

	// Dispatch chunks.
	for i, marker := range markers {
		if marker == 0x01 {
			// Already downloaded; count it toward progress immediately.
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

// StartDownloadURL downloads a single remote URL into a specific file on
// disk using the same chunked, resumable, parallel machinery as StartDownload.
// Used by the AI Hub (qairt) flow, where the asset is a single .zip whose
// direct URL is resolved before the download begins.
//
// outputDir must exist. dstName is the basename under outputDir. A sidecar
// <dstName>.progress marker file is created for resume support and removed
// on clean completion. size must be the exact Content-Length of the URL.
func StartDownloadURL(ctx context.Context, urlStr, outputDir, dstName string, size int64) (chan types.DownloadInfo, chan error) {
	const maxConcurrency = 8
	resCh := make(chan types.DownloadInfo)
	errCh := make(chan error, maxConcurrency)

	slog.Info("Starting URL download", "url", urlStr, "outputDir", outputDir, "file", dstName, "size", size)

	go func() {
		defer close(errCh)
		defer close(resCh)

		if size <= 0 {
			errCh <- fmt.Errorf("StartDownloadURL: invalid size %d", size)
			return
		}
		if err := os.MkdirAll(outputDir, 0o755); err != nil {
			errCh <- fmt.Errorf("create outputDir: %w", err)
			return
		}

		outPath := filepath.Join(outputDir, dstName)
		markerPath := outPath + ProgressSuffix

		hd := downloader.NewDownloader("")
		fetch := func(ctx context.Context, offset, limit int64, w io.Writer) error {
			return hd.DownloadChunk(ctx, urlStr, offset, limit, w)
		}

		var downloaded int64
		g, gctx := errgroup.WithContext(ctx)
		g.SetLimit(maxConcurrency)

		if err := downloadChunked(gctx, outPath, markerPath, size, size, &downloaded, g, resCh, fetch); err != nil {
			errCh <- err
			return
		}

		if err := g.Wait(); err != nil {
			errCh <- err
			return
		}
		_ = os.Remove(markerPath)
		slog.Info("url download complete", "url", urlStr, "outPath", outPath)
	}()

	return resCh, errCh
}

func getHub(ctx context.Context, modelName string) (ModelHub, error) {
	// if only one hub specified, check availability first
	if len(hubs) == 1 {
		h := hubs[0]
		slog.Info("specified single hub", "hub", reflect.TypeOf(h))
		return h, h.CheckAvailable(ctx, modelName)
	}

	// try each hub
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
	case http.StatusNotFound, http.StatusUnauthorized:
		return fmt.Errorf("model not found, please check the model name or auth token")
	default:
		return fmt.Errorf("HTTPError: %s", response.Status())
	}
}
