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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/downloader"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/model_hub/aihub"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/render"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/types"
)

// AIHub pulls QAIRT models from Qualcomm AI Hub.
type AIHub struct {
	client        *aihub.Client
	chipsetGetter func() string
	http          *downloader.HTTPDownloader

	mu       sync.Mutex
	resolved map[string]resolvedAsset // keyed by modelName ("<org>/<repo>")
}

type resolvedAsset struct {
	zipURL      string
	zipSize     int64
	zipBasename string // "<repo>.zip"
}

func NewAIHub(chipsetGetter func() string) *AIHub {
	return &AIHub{
		client:        aihub.NewClient(),
		chipsetGetter: chipsetGetter,
		http:          downloader.NewDownloader(),
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

// ModelInfo picks the first asset matching the configured chipset and
// returns a single-entry listing for its zip. Per-model details come
// from metadata.json in PostDownload.
func (h *AIHub) ModelInfo(ctx context.Context, name string) ([]ModelFileInfo, error) {
	repo, ok := aihub.IsAIHubName(name)
	if !ok {
		return nil, fmt.Errorf("aihub: %q is not an AI Hub model", name)
	}

	chipset := h.chipsetGetter()
	manifest, err := h.client.LoadManifest(ctx)
	if err != nil {
		return nil, TranslateAIHubError(err)
	}
	model, err := h.client.LookupModelByDisplayName(repo)
	if err != nil {
		return nil, TranslateAIHubError(err)
	}
	if _, rerr := aihub.RuntimeForDomain(model.Domain); rerr != nil {
		return nil, rerr
	}

	plat, err := h.client.LoadPlatform(ctx, manifest)
	if err != nil {
		return nil, TranslateAIHubError(err)
	}
	ra, err := h.client.LoadReleaseAssets(ctx, manifest, model.ID)
	if err != nil {
		return nil, TranslateAIHubError(err)
	}
	candidates, err := aihub.MatchAll(ra, plat, model.Domain, chipset)
	if err != nil {
		return nil, h.formatMatchError(err, name, chipset)
	}
	asset := candidates[0] // TODO: ask user when len(candidates) > 1

	zipSize, err := aihub.HeadContentLength(ctx, asset.DownloadURL)
	if err != nil {
		return nil, TranslateAIHubError(err)
	}

	zipURL := asset.DownloadURL
	zipBasename := path.Base(zipURL)
	if path.Ext(zipBasename) != ".zip" {
		return nil, fmt.Errorf("expected zip file from AI Hub, got %q", zipBasename)
	}
	h.mu.Lock()
	h.resolved[name] = resolvedAsset{
		zipURL:      zipURL,
		zipSize:     zipSize,
		zipBasename: zipBasename,
	}
	h.mu.Unlock()

	slog.Info("aihub: resolved asset",
		"name", name, "chipset", asset.Chipset,
		"url", zipURL, "size", zipSize)

	return []ModelFileInfo{{Name: zipBasename, Size: zipSize}}, nil
}

func (h *AIHub) GetFileContent(ctx context.Context, name, fileName string, offset, limit int64, w io.Writer) error {
	h.mu.Lock()
	r, ok := h.resolved[name]
	h.mu.Unlock()
	if !ok {
		return fmt.Errorf("aihub: ModelInfo not called for %q", name)
	}

	if fileName != r.zipBasename {
		return fmt.Errorf("aihub: unknown file %q for model %q", fileName, name)
	}
	return h.http.DownloadChunk(ctx, r.zipURL, offset, limit, w)
}

// PostDownload unpacks the zip and finalises the manifest from metadata.json.
func (h *AIHub) PostDownload(ctx context.Context, name, outputDir string, mf *types.ModelManifest) error {
	h.mu.Lock()
	r, ok := h.resolved[name]
	h.mu.Unlock()
	if !ok {
		return fmt.Errorf("aihub: PostDownload called without prior ModelInfo for %q", name)
	}

	if err := extractQairtZip(filepath.Join(outputDir, r.zipBasename), outputDir, mf); err != nil {
		return err
	}
	applyQairtMetadata(outputDir, mf)
	h.mu.Lock()
	delete(h.resolved, name)
	h.mu.Unlock()
	return nil
}

// extractQairtZip flat-extracts zipPath into outputDir, removes the zip,
// and resets mf.{ModelFile, MMProjFile, ExtraFiles}. ModelFile is left nil
// for applyQairtMetadata to populate.
func extractQairtZip(zipPath, outputDir string, mf *types.ModelManifest) error {
	base := filepath.Base(zipPath)
	spin := render.NewSpinner("Extracting " + base)
	spin.Start()
	res, err := aihub.ExtractFlat(zipPath, outputDir)
	spin.Stop()
	if err != nil {
		return fmt.Errorf("unzip %s: %w", base, err)
	}
	if err := os.Remove(zipPath); err != nil {
		slog.Warn("failed to remove zip after extract", "path", zipPath, "err", err)
	}

	mf.MMProjFile = types.ModelFileInfo{}
	mf.ExtraFiles = append(mf.ExtraFiles[:0], res.Files...)
	mf.ModelFile = nil
	return nil
}

// qairtMetaJSON: subset of the AI Hub model metadata schema we care about.
type qairtMetaJSON struct {
	ModelID   string `json:"model_id"`
	Precision string `json:"precision"`
	Genie     *struct {
		SupportsVision bool `json:"supports_vision"`
	} `json:"genie"`
}

// applyQairtMetadata fills mf.{ModelFile, ModelName, ModelType} from
// metadata.json. Pre-set fields win. On read/parse failure ModelType is
// left empty so the caller can prompt the user to set it.
func applyQairtMetadata(outputDir string, mf *types.ModelManifest) {
	var meta qairtMetaJSON
	var parsed bool
	data, rerr := os.ReadFile(filepath.Join(outputDir, "metadata.json"))
	switch {
	case errors.Is(rerr, os.ErrNotExist):
		slog.Debug("no metadata.json", "dir", outputDir)
		parsed = true
	case rerr != nil:
		slog.Warn("read metadata.json", "err", rerr)
	default:
		if err := json.Unmarshal(data, &meta); err != nil {
			slog.Warn("parse metadata.json", "err", err)
		} else {
			parsed = true
		}
	}

	quant := types.QuantNA
	if p := strings.ToUpper(meta.Precision); p != "" {
		quant = p
	}
	// qairt loads the directory whole; ModelFile name is a placeholder.
	if len(mf.ModelFile) == 0 {
		mf.ModelFile = map[string]types.ModelFileInfo{
			quant: {Name: "qairt-placeholder", Downloaded: true},
		}
	}
	if mf.ModelName == "" {
		mf.ModelName = meta.ModelID
	}
	if mf.ModelType == "" && parsed {
		if meta.Genie != nil && meta.Genie.SupportsVision {
			mf.ModelType = types.ModelTypeVLM
		} else {
			mf.ModelType = types.ModelTypeLLM
		}
	}
}

// formatMatchError tags ChipsetNotAvailableError with the model name.
func (h *AIHub) formatMatchError(err error, name, chipset string) error {
	var cnae *aihub.ChipsetNotAvailableError
	if !errors.As(err, &cnae) {
		return err
	}
	seen := make(map[string]struct{})
	chipsets := make([]string, 0, len(cnae.Available))
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
