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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/bytedance/sonic"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/config"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/downloader"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub/aihub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
)

var ErrChipsetNotConfigured = errors.New("aihub: device chipset not configured (run: geniex config set device <chipset>)")

// ErrModelTypeDetection is returned by PostDownload when metadata.json exists
// but could not be parsed. The model type has been defaulted to LLM; the caller
// should warn the user to correct it with `geniex model set-type`.
type ErrModelTypeDetection struct{ Cause error }

func (e *ErrModelTypeDetection) Error() string {
	return "model type detection failed: " + e.Cause.Error()
}
func (e *ErrModelTypeDetection) Unwrap() error { return e.Cause }

type AIHub struct {
	client        *aihub.Client
	chipsetGetter func() string
	http          *downloader.HTTPDownloader

	mu       sync.Mutex
	resolved map[string]resolvedAsset // keyed by modelName ("<org>/<repo>")
}

type resolvedAsset struct {
	zipURL       string
	zipSize      int64
	zipBasename  string // "<repo>.zip"
	precision    string // quant-like label, e.g. "W4A16"
	manifestJSON []byte // serialised pseudo geniex.json
}

func NewAIHub(chipsetGetter func() string, cacheDir string) *AIHub {
	return &AIHub{
		client:        aihub.NewClient(cacheDir),
		chipsetGetter: chipsetGetter,
		http:          downloader.NewDownloader(""),
		resolved:      make(map[string]resolvedAsset),
	}
}

func (*AIHub) MaxConcurrency() int { return 8 }

func (*AIHub) CheckAvailable(ctx context.Context, name string) error {
	if _, ok := aihub.IsAIHubName(name); !ok {
		return fmt.Errorf("aihub: %q is not an AI Hub model", name)
	}
	return nil
}

// ModelInfo resolves manifest/platform/release-assets, picks the first
// matching asset for the configured chipset, and returns a two-file listing:
//   - "<repo>.zip" — what store.Pull will actually download via GetFileContent
//   - "geniex.json" — a pseudo manifest (PluginId/DeviceId/Precision/…)
//     that the upper ModelInfo parses into *types.ModelManifest so
//     pullModel can populate the real manifest fields automatically.
func (h *AIHub) ModelInfo(ctx context.Context, name string) ([]ModelFileInfo, error) {
	repo, ok := aihub.IsAIHubName(name)
	if !ok {
		return nil, fmt.Errorf("aihub: %q is not an AI Hub model", name)
	}

	chipset := h.chipsetGetter()
	if chipset == "" {
		return nil, ErrChipsetNotConfigured
	}

	var fetchOpts []aihub.FetchOption
	if config.Get().AIHubNoCache {
		fetchOpts = append(fetchOpts, aihub.WithSkipCache())
	}

	manifest, err := h.client.LoadManifest(ctx, fetchOpts...)
	if err != nil {
		return nil, err
	}
	model, err := h.client.LookupModelByDisplayName(repo)
	if err != nil {
		return nil, err
	}
	if _, rerr := aihub.RuntimeForDomain(model.GetDomain()); rerr != nil {
		return nil, rerr
	}

	plat, err := h.client.LoadPlatform(ctx, manifest, fetchOpts...)
	if err != nil {
		return nil, err
	}
	ra, err := h.client.LoadReleaseAssets(ctx, manifest, model.GetId(), fetchOpts...)
	if err != nil {
		return nil, err
	}
	candidates, err := aihub.MatchAll(ra, plat, model.GetDomain(), chipset)
	if err != nil {
		return nil, h.formatMatchError(err, name, chipset)
	}
	asset := candidates[0] // TODO: ask user when len(candidates) > 1

	zipSize, err := aihub.HeadContentLength(ctx, asset.GetDownloadUrl())
	if err != nil {
		return nil, fmt.Errorf("HEAD %s: %w", asset.GetDownloadUrl(), err)
	}

	precisionLabel := strings.TrimPrefix(asset.GetPrecision().String(), "PRECISION_")
	pseudo := types.ModelManifest{
		Name:      name,
		ModelName: model.GetId(),
		PluginId:  "qairt",
		DeviceId:  asset.GetChipset(),
		// ModelType intentionally left blank so pullModel prompts the user
		// via chooseModelType(). ModelFile is filled in by PostDownload
		// once the zip has been extracted.
	}
	manifestJSON, err := sonic.Marshal(pseudo)
	if err != nil {
		return nil, fmt.Errorf("marshal pseudo manifest: %w", err)
	}

	zipBasename := repo + ".zip"
	h.mu.Lock()
	h.resolved[name] = resolvedAsset{
		zipURL:       asset.GetDownloadUrl(),
		zipSize:      zipSize,
		zipBasename:  zipBasename,
		precision:    precisionLabel,
		manifestJSON: manifestJSON,
	}
	h.mu.Unlock()

	slog.Info("aihub: resolved asset",
		"name", name, "chipset", asset.GetChipset(), "precision", precisionLabel,
		"url", asset.GetDownloadUrl(), "size", zipSize)

	return []ModelFileInfo{
		{Name: zipBasename, Size: zipSize},
		{Name: "geniex.json", Size: int64(len(manifestJSON))},
	}, nil
}

// GetFileContent serves the pseudo manifest from memory and proxies the
// zip download into the shared chunk downloader.
func (h *AIHub) GetFileContent(ctx context.Context, name, fileName string, offset, limit int64, w io.Writer) error {
	h.mu.Lock()
	r, ok := h.resolved[name]
	h.mu.Unlock()
	if !ok {
		return fmt.Errorf("aihub: ModelInfo not called for %q", name)
	}

	switch fileName {
	case "geniex.json":
		// Upper GetFileContent calls with offset=0,limit=0. Ignore them.
		_, err := io.Copy(w, bytes.NewReader(r.manifestJSON))
		return err
	case r.zipBasename:
		return h.http.DownloadChunk(ctx, r.zipURL, offset, limit, w)
	default:
		return fmt.Errorf("aihub: unknown file %q for model %q", fileName, name)
	}
}

// PostDownload unpacks the downloaded zip into outputDir (flat), rewrites
// mf.ModelFile / mf.ExtraFiles to reflect the extracted contents, and
// removes the zip.
func (h *AIHub) PostDownload(ctx context.Context, name, outputDir string, mf *types.ModelManifest) error {
	h.mu.Lock()
	r, ok := h.resolved[name]
	h.mu.Unlock()
	if !ok {
		return fmt.Errorf("aihub: PostDownload called without prior ModelInfo for %q", name)
	}

	zipPath := filepath.Join(outputDir, r.zipBasename)
	spin := render.NewSpinner("Extracting " + r.zipBasename)
	spin.Start()
	res, err := aihub.ExtractFlat(zipPath, outputDir)
	spin.Stop()
	if err != nil {
		return fmt.Errorf("aihub: unzip %s: %w", r.zipBasename, err)
	}
	if err := os.Remove(zipPath); err != nil {
		slog.Warn("aihub: failed to remove zip after extract", "path", zipPath, "err", err)
	}

	quant := r.precision
	if quant == "" {
		quant = "N/A"
	}
	var entrypointSize int64
	for _, f := range res.Files {
		if f.Name == res.EntrypointBasename {
			entrypointSize = f.Size
			break
		}
	}
	mf.ModelFile = map[string]types.ModelFileInfo{
		quant: {
			Name:       res.EntrypointBasename,
			Downloaded: true,
			Size:       entrypointSize,
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

	// Detect model type from metadata.json if present.
	// supports_vision absent or false → LLM; true → VLM.
	if mf.ModelType == "" {
		mt, detErr := detectModelTypeFromDir(outputDir)
		mf.ModelType = mt
		if detErr != nil {
			// Return as ErrModelTypeDetection so the caller can distinguish it
			// from a fatal error and show a user-facing warning instead of aborting.
			return &ErrModelTypeDetection{Cause: detErr}
		}
	}

	h.mu.Lock()
	delete(h.resolved, name)
	h.mu.Unlock()
	return nil
}

// aiHubMetaJSON is the minimal subset of metadata.json needed for type detection.
type aiHubMetaJSON struct {
	Genie *struct {
		SupportsVision bool `json:"supports_vision"`
	} `json:"genie"`
}

// detectModelTypeFromDir reads metadata.json from dir and returns VLM when
// supports_vision is true, LLM otherwise. Returns a non-nil error only when
// metadata.json exists but could not be parsed; an absent file is not an error.
func detectModelTypeFromDir(dir string) (types.ModelType, error) {
	data, err := os.ReadFile(filepath.Join(dir, "metadata.json"))
	if err != nil {
		// File absent is not an error — default to LLM.
		slog.Debug("aihub: no metadata.json, treating as LLM", "dir", dir)
		return types.ModelTypeLLM, nil
	}
	var meta aiHubMetaJSON
	if err := json.Unmarshal(data, &meta); err != nil {
		slog.Warn("aihub: failed to parse metadata.json, defaulting to LLM", "err", err)
		return types.ModelTypeLLM, fmt.Errorf("parse metadata.json: %w", err)
	}
	if meta.Genie != nil && meta.Genie.SupportsVision {
		slog.Info("aihub: detected VLM via metadata.json supports_vision")
		return types.ModelTypeVLM, nil
	}
	slog.Debug("aihub: supports_vision=false, treating as LLM")
	return types.ModelTypeLLM, nil
}

// formatMatchError turns aihub.ChipsetNotAvailableError into a friendlier
// message listing available chipsets; other errors pass through unchanged.
func (h *AIHub) formatMatchError(err error, name, chipset string) error {
	var cnae *aihub.ChipsetNotAvailableError
	if !errors.As(err, &cnae) {
		return err
	}
	seen := make(map[string]struct{})
	var chipsets []string
	for _, a := range cnae.Available {
		if _, ok := seen[a.Chipset]; ok {
			continue
		}
		seen[a.Chipset] = struct{}{}
		chipsets = append(chipsets, a.Chipset)
	}
	sort.Strings(chipsets)
	return fmt.Errorf("no AI Hub asset for model=%q chipset=%q; available: %s",
		name, chipset, strings.Join(chipsets, ", "))
}
