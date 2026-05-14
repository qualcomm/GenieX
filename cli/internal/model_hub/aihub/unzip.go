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

package aihub

import (
	"archive/zip"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
)

// ErrNoBinShard signals that the zip contains no *.bin file; the qairt
// plugin has nothing to load without one.
var ErrNoBinShard = errors.New("aihub: no .bin shard in archive")

// ErrFileCollision signals that two zip entries share the same basename.
// Flattening them would silently overwrite one, so we fail instead.
var ErrFileCollision = errors.New("aihub: duplicate basename in archive")

// ExtractResult summarises the outcome of ExtractFlat.
type ExtractResult struct {
	// EntrypointBasename is the lex-first *.bin shard. The CLI stores this
	// in ModelManifest.ModelFile["N/A"].Name so the qairt plugin can take
	// <dir>/<entrypoint> and parent_path() back to the model directory.
	EntrypointBasename string

	// Files is every extracted file (basename, uncompressed size, Downloaded=true).
	Files []types.ModelFileInfo

	// TotalSize is the sum of uncompressed sizes across all extracted files.
	TotalSize int64
}

// ExtractFlat unzips zipPath into destDir, collapsing sub-directories so
// every file lands at destDir/<basename>. Directory entries and "."/".."
// names are skipped; basename collisions raise ErrFileCollision.
func ExtractFlat(zipPath, destDir string) (*ExtractResult, error) {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return nil, fmt.Errorf("open zip: %w", err)
	}
	defer r.Close()

	// Detect basename collisions before writing anything.
	seen := make(map[string]string) // basename -> original zip path
	for _, e := range r.File {
		if e.FileInfo().IsDir() {
			continue
		}
		base := filepath.Base(filepath.ToSlash(e.Name))
		if base == "" || base == "." || base == ".." {
			continue
		}
		if prev, ok := seen[base]; ok {
			return nil, fmt.Errorf("%w: %s (from %q and %q)",
				ErrFileCollision, base, prev, e.Name)
		}
		seen[base] = e.Name
	}

	// DEFLATE is CPU-bound; decompress shards in parallel. Each *zip.File
	// opens its own SectionReader over the underlying file, so concurrent
	// Open/Copy from different goroutines is safe.
	type job struct {
		entry *zip.File
		base  string
	}
	var jobs []job
	for _, e := range r.File {
		if e.FileInfo().IsDir() {
			continue
		}
		base := filepath.Base(filepath.ToSlash(e.Name))
		if base == "" || base == "." || base == ".." {
			continue
		}
		jobs = append(jobs, job{entry: e, base: base})
	}

	result := &ExtractResult{Files: make([]types.ModelFileInfo, len(jobs))}
	limit := runtime.NumCPU()
	if limit > 8 {
		limit = 8
	}
	if limit > len(jobs) {
		limit = len(jobs)
	}
	if limit < 1 {
		limit = 1
	}

	var mu sync.Mutex
	g := new(errgroup.Group)
	g.SetLimit(limit)
	for i, j := range jobs {
		i, j := i, j
		g.Go(func() error {
			size, werr := extractOne(j.entry, filepath.Join(destDir, j.base))
			if werr != nil {
				return fmt.Errorf("extract %s: %w", j.entry.Name, werr)
			}
			result.Files[i] = types.ModelFileInfo{
				Name:       j.base,
				Size:       size,
				Downloaded: true,
			}
			mu.Lock()
			result.TotalSize += size
			mu.Unlock()
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}

	sort.Slice(result.Files, func(i, j int) bool {
		return result.Files[i].Name < result.Files[j].Name
	})

	for _, f := range result.Files {
		if strings.HasSuffix(strings.ToLower(f.Name), ".bin") {
			result.EntrypointBasename = f.Name
			break
		}
	}
	if result.EntrypointBasename == "" {
		return nil, ErrNoBinShard
	}

	slog.Debug("aihub: extracted",
		"zip", zipPath, "dest", destDir,
		"files", len(result.Files), "size", result.TotalSize,
		"entrypoint", result.EntrypointBasename)
	return result, nil
}

// extractOne streams one zip entry to outPath and returns the uncompressed
// size written. O_EXCL guards against overwriting any leftover from a prior
// aborted extract.
func extractOne(e *zip.File, outPath string) (int64, error) {
	rc, err := e.Open()
	if err != nil {
		return 0, err
	}
	defer rc.Close()

	f, err := os.OpenFile(outPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	return io.Copy(f, rc)
}
