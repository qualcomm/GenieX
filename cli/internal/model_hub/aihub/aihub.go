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

	"github.com/bytedance/sonic"
	"resty.dev/v3"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/config"
)

// ErrModelNotFound: id is not in the AI Hub manifest. The CLI uses this
// sentinel to fall back to the HuggingFace-style pull path.
var ErrModelNotFound = errors.New("aihub: model not found in manifest")

// ErrNoReleaseAssets: manifest entry exists but has no release_assets URL.
var ErrNoReleaseAssets = errors.New("aihub: model has no release_assets URL")

// Client fetches AI Hub index JSONs. Build with NewClient.
type Client struct {
	baseURL string
	version string

	http *resty.Client

	modelIndex map[string]*ManifestModelEntry
}

// NewClient: base URL and pinned aihm version come from CLI config.
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

func (c *Client) Close() error {
	if c.http != nil {
		return c.http.Close()
	}
	return nil
}

// LoadManifest fetches manifest.json for the pinned aihm release and
// builds an O(1) display_name → entry index. The public bucket has no
// `latest` alias, so the version must be pinned.
func (c *Client) LoadManifest(ctx context.Context) (*ReleaseManifest, error) {
	url := fmt.Sprintf("%s/releases/%s/manifest.json", c.baseURL, c.version)

	data, err := c.fetchJSON(ctx, url)
	if err != nil {
		return nil, fmt.Errorf("load manifest: %w", err)
	}

	var m ReleaseManifest
	if err := sonic.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}

	c.modelIndex = make(map[string]*ManifestModelEntry, len(m.Models))
	for i := range m.Models {
		c.modelIndex[m.Models[i].DisplayName] = &m.Models[i]
	}

	return &m, nil
}

// LookupModelByDisplayName returns the entry for displayName or
// ErrModelNotFound. Requires LoadManifest first.
func (c *Client) LookupModelByDisplayName(displayName string) (*ManifestModelEntry, error) {
	if c.modelIndex == nil {
		return nil, errors.New("aihub: LookupModelByDisplayName called before LoadManifest")
	}
	m, ok := c.modelIndex[displayName]
	if !ok {
		return nil, ErrModelNotFound
	}
	return m, nil
}

// LoadPlatformDirect fetches platform.json without needing a manifest.
func (c *Client) LoadPlatformDirect(ctx context.Context) (*PlatformInfo, error) {
	return c.loadPlatform(ctx, fmt.Sprintf("%s/releases/%s/platform.json", c.baseURL, c.version))
}

// LoadPlatform fetches platform.json via the URL in m.
func (c *Client) LoadPlatform(ctx context.Context, m *ReleaseManifest) (*PlatformInfo, error) {
	if m == nil || m.PlatformURL == "" {
		return nil, errors.New("aihub: manifest has no platform_url")
	}
	return c.loadPlatform(ctx, m.PlatformURL)
}

func (c *Client) loadPlatform(ctx context.Context, url string) (*PlatformInfo, error) {
	data, err := c.fetchJSON(ctx, url)
	if err != nil {
		return nil, fmt.Errorf("load platform: %w", err)
	}
	var p PlatformInfo
	if err := sonic.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("parse platform: %w", err)
	}
	return &p, nil
}

// LoadReleaseAssets fetches release-assets.json for id. Returns
// ErrModelNotFound or ErrNoReleaseAssets.
func (c *Client) LoadReleaseAssets(ctx context.Context, m *ReleaseManifest, id string) (*ModelReleaseAssets, error) {
	var entry *ManifestModelEntry
	for i := range m.Models {
		if m.Models[i].ID == id {
			entry = &m.Models[i]
			break
		}
	}
	if entry == nil {
		return nil, ErrModelNotFound
	}
	url := entry.ManifestUrls.ReleaseAssets
	if url == "" {
		return nil, ErrNoReleaseAssets
	}

	data, err := c.fetchJSON(ctx, url)
	if err != nil {
		return nil, fmt.Errorf("load release assets for %s: %w", id, err)
	}
	var ra ModelReleaseAssets
	if err := sonic.Unmarshal(data, &ra); err != nil {
		return nil, fmt.Errorf("parse release assets: %w", err)
	}
	return &ra, nil
}

func (c *Client) fetchJSON(ctx context.Context, url string) ([]byte, error) {
	slog.Debug("aihub: fetching", "url", url)
	resp, err := c.http.R().SetContext(ctx).Get(url)
	if err != nil {
		return nil, fmt.Errorf("GET %s: %w", url, err)
	}
	if resp.StatusCode() != http.StatusOK {
		return nil, &HTTPError{URL: url, Status: resp.StatusCode()}
	}
	return resp.Bytes(), nil
}

// HTTPError carries a non-2xx response from the AI Hub endpoint. The
// model_hub package translates it into one of its hub sentinels based on
// Status (see model_hub.TranslateAIHubError).
type HTTPError struct {
	URL    string
	Status int
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("aihub: http %d from %s", e.Status, e.URL)
}

// aiHubOrgs are HF-style orgs that route to the AI Hub pull path.
var aiHubOrgs = []string{"qualcomm", "ai-hub-models"}

// IsAIHubName reports whether name belongs to an AI Hub org and returns
// the repo portion (e.g. "qualcomm/Qwen3-4B" → "Qwen3-4B", true).
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

// HeadContentLength returns the Content-Length of url via HEAD.
func HeadContentLength(ctx context.Context, url string) (int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, url, nil)
	if err != nil {
		return 0, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, fmt.Errorf("HEAD %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, &HTTPError{URL: url, Status: resp.StatusCode}
	}
	if resp.ContentLength <= 0 {
		return 0, fmt.Errorf("HEAD %s: missing Content-Length", url)
	}
	return resp.ContentLength, nil
}
