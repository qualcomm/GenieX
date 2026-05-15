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
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"google.golang.org/protobuf/encoding/protojson"
	"resty.dev/v3"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/config"
	"github.com/qcom-it-nexa-ai/geniex/cli/internal/qaihm"
)

// ErrModelNotFound signals that an id was not present in the AI Hub manifest.
// The CLI uses this sentinel to fall back to the HuggingFace-style pull path.
var ErrModelNotFound = errors.New("aihub: model not found in manifest")

// ErrNoReleaseAssets signals that the manifest entry exists but has no
// release_assets URL (some legacy/LLM entries lack one).
var ErrNoReleaseAssets = errors.New("aihub: model has no release_assets URL")

// Client fetches AI Hub index JSONs over the network. Use NewClient; the zero
// value is not usable.
type Client struct {
	baseURL string
	version string

	http *resty.Client

	modelIndex map[string]*qaihm.ManifestModelEntry
}

// NewClient builds a Client. Base URL and pinned aihm version come from CLI
// config.
func NewClient() *Client {
	cfg := config.Get()

	base := strings.TrimRight(cfg.AIHubBaseURL, "/")
	if base == "" {
		base = strings.TrimRight(config.DefaultAIHubBaseURL, "/")
	}
	version := strings.Trim(cfg.AIHubVersion, "/")
	if version == "" {
		version = config.DefaultAIHubVersion
	}

	c := resty.New()
	c.SetTimeout(30 * time.Second)

	return &Client{
		baseURL: base,
		version: version,
		http:    c,
	}
}

// Close releases the underlying HTTP client.
func (c *Client) Close() error {
	if c.http != nil {
		return c.http.Close()
	}
	return nil
}

// LoadManifest fetches the manifest.json for the pinned aihm release (the
// public bucket has no `latest` alias) and builds an O(1) model lookup index
// on first success.
func (c *Client) LoadManifest(ctx context.Context) (*qaihm.ReleaseManifest, error) {
	url := fmt.Sprintf("%s/releases/%s/manifest.json", c.baseURL, c.version)

	data, err := c.fetchJSON(ctx, url)
	if err != nil {
		return nil, fmt.Errorf("load manifest: %w", err)
	}

	var m qaihm.ReleaseManifest
	if err := protojson.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}

	c.modelIndex = make(map[string]*qaihm.ManifestModelEntry, len(m.GetModels()))
	for _, model := range m.GetModels() {
		c.modelIndex[model.GetDisplayName()] = model
	}

	return &m, nil
}

// LookupModelByDisplayName returns the manifest entry whose display_name
// matches the given string, or ErrModelNotFound.
// Must be called after LoadManifest has succeeded.
func (c *Client) LookupModelByDisplayName(displayName string) (*qaihm.ManifestModelEntry, error) {
	if c.modelIndex == nil {
		return nil, errors.New("aihub: LookupModelByDisplayName called before LoadManifest")
	}
	m, ok := c.modelIndex[displayName]
	if !ok {
		return nil, ErrModelNotFound
	}
	return m, nil
}

// LoadPlatformDirect fetches platform.json by constructing the URL directly
// from the client's base URL and version — no manifest required.
func (c *Client) LoadPlatformDirect(ctx context.Context) (*qaihm.PlatformInfo, error) {
	url := fmt.Sprintf("%s/releases/%s/platform.json", c.baseURL, c.version)

	data, err := c.fetchJSON(ctx, url)
	if err != nil {
		return nil, fmt.Errorf("load platform: %w", err)
	}

	var p qaihm.PlatformInfo
	if err := protojson.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("parse platform: %w", err)
	}
	return &p, nil
}

// LoadPlatform fetches platform.json (URL referenced by manifest).
func (c *Client) LoadPlatform(ctx context.Context, m *qaihm.ReleaseManifest) (*qaihm.PlatformInfo, error) {
	if m == nil || m.GetPlatformUrl() == "" {
		return nil, errors.New("aihub: manifest has no platform_url")
	}

	data, err := c.fetchJSON(ctx, m.GetPlatformUrl())
	if err != nil {
		return nil, fmt.Errorf("load platform: %w", err)
	}

	var p qaihm.PlatformInfo
	if err := protojson.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("parse platform: %w", err)
	}
	return &p, nil
}

// LoadReleaseAssets fetches release-assets.json for a given model id.
// Returns ErrNoReleaseAssets if the manifest entry lacks the URL.
func (c *Client) LoadReleaseAssets(ctx context.Context, m *qaihm.ReleaseManifest, id string) (*qaihm.ModelReleaseAssets, error) {
	var model *qaihm.ManifestModelEntry
	for _, entry := range m.GetModels() {
		if entry.GetId() == id {
			model = entry
			break
		}
	}
	if model == nil {
		return nil, ErrModelNotFound
	}
	if model.GetManifestUrls().GetReleaseAssets() == "" {
		return nil, ErrNoReleaseAssets
	}

	data, err := c.fetchJSON(ctx, model.GetManifestUrls().GetReleaseAssets())
	if err != nil {
		return nil, fmt.Errorf("load release assets for %s: %w", id, err)
	}

	var ra qaihm.ModelReleaseAssets
	if err := protojson.Unmarshal(data, &ra); err != nil {
		return nil, fmt.Errorf("parse release assets: %w", err)
	}
	return &ra, nil
}

// fetchJSON GETs url and returns the body bytes.
func (c *Client) fetchJSON(ctx context.Context, url string) ([]byte, error) {
	slog.Debug("aihub: fetching", "url", url)
	resp, err := c.http.R().SetContext(ctx).Get(url)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode() != http.StatusOK {
		return nil, fmt.Errorf("http %d from %s", resp.StatusCode(), url)
	}
	return resp.Bytes(), nil
}

// aiHubOrgs is the allowlist of HuggingFace-style org names that route through
// the AI Hub S3/QAIRT pull path instead of HuggingFace.
var aiHubOrgs = []string{"qualcomm", "qai-hub-models"}

// IsAIHubName reports whether name belongs to an AI Hub org
// (e.g. "qualcomm/Qwen3-4B"). It returns the repo portion (after the slash)
// as a convenience.
func IsAIHubName(name string) (repo string, ok bool) {
	parts := strings.SplitN(name, "/", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", false
	}
	for _, o := range aiHubOrgs {
		if strings.EqualFold(parts[0], o) {
			return parts[1], true
		}
	}
	return "", false
}

// HeadContentLength issues a HEAD against url and returns its Content-Length.
func HeadContentLength(ctx context.Context, url string) (int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, url, nil)
	if err != nil {
		return 0, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("HEAD %s: %s", url, resp.Status)
	}
	if resp.ContentLength <= 0 {
		return 0, fmt.Errorf("HEAD %s: missing Content-Length", url)
	}
	return resp.ContentLength, nil
}
