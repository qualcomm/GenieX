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
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/bytedance/sonic"
	"resty.dev/v3"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/config"
)

// DefaultCacheTTL is how long cached index JSONs are considered fresh.
const DefaultCacheTTL = 24 * time.Hour

// ErrModelNotFound signals that an id was not present in the AI Hub manifest.
// The CLI uses this sentinel to fall back to the HuggingFace-style pull path.
var ErrModelNotFound = errors.New("aihub: model not found in manifest")

// ErrNoReleaseAssets signals that the manifest entry exists but has no
// release_assets URL (some legacy/LLM entries lack one).
var ErrNoReleaseAssets = errors.New("aihub: model has no release_assets URL")

// Client fetches AI Hub index JSONs with a small on-disk TTL cache. Use
// NewClient; the zero value is not usable.
type Client struct {
	baseURL  string
	version  string
	cacheDir string
	ttl      time.Duration
	noCache  bool

	http *resty.Client

	modelIndex map[string]*Model
}

// NewClient builds a Client rooted at cacheDir (typically <data-dir>/aihub).
// Base URL and pinned aihm version come from CLI config.
func NewClient(cacheDir string) *Client {
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
		baseURL:  base,
		version:  version,
		cacheDir: cacheDir,
		ttl:      DefaultCacheTTL,
		noCache:  cfg.AIHubNoCache,
		http:     c,
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
func (c *Client) LoadManifest(ctx context.Context) (*Manifest, error) {
	url := fmt.Sprintf("%s/releases/%s/manifest.json", c.baseURL, c.version)
	cachePath := filepath.Join(c.cacheDir, fmt.Sprintf("manifest-%s.json", sanitizeForFilename(c.version)))

	data, err := c.fetchJSON(ctx, url, cachePath)
	if err != nil {
		return nil, fmt.Errorf("load manifest: %w", err)
	}

	var m Manifest
	if err := sonic.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}

	c.modelIndex = make(map[string]*Model, len(m.Models))
	for i := range m.Models {
		c.modelIndex[m.Models[i].ID] = &m.Models[i]
	}

	return &m, nil
}

// LookupModel returns the Model entry for id, or ErrModelNotFound.
// Must be called after LoadManifest has succeeded.
func (c *Client) LookupModel(id string) (*Model, error) {
	if c.modelIndex == nil {
		return nil, errors.New("aihub: LookupModel called before LoadManifest")
	}
	m, ok := c.modelIndex[id]
	if !ok {
		return nil, ErrModelNotFound
	}
	return m, nil
}

// LoadPlatform fetches and caches platform.json (referenced by manifest).
func (c *Client) LoadPlatform(ctx context.Context, m *Manifest) (*Platform, error) {
	if m == nil || m.PlatformURL == "" {
		return nil, errors.New("aihub: manifest has no platform_url")
	}

	cacheName := fmt.Sprintf("platform-%s.json", sanitizeForFilename(m.Version))
	cachePath := filepath.Join(c.cacheDir, cacheName)

	data, err := c.fetchJSON(ctx, m.PlatformURL, cachePath)
	if err != nil {
		return nil, fmt.Errorf("load platform: %w", err)
	}

	var p Platform
	if err := sonic.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("parse platform: %w", err)
	}
	return &p, nil
}

// LoadReleaseAssets fetches release-assets.json for a given model id.
// Returns ErrNoReleaseAssets if the manifest entry lacks the URL.
func (c *Client) LoadReleaseAssets(ctx context.Context, m *Manifest, id string) (*ReleaseAssets, error) {
	model, err := c.LookupModel(id)
	if err != nil {
		return nil, err
	}
	if model.ManifestURLs.ReleaseAssets == "" {
		return nil, ErrNoReleaseAssets
	}

	cacheName := fmt.Sprintf("release-assets-%s-%s.json",
		sanitizeForFilename(id), sanitizeForFilename(m.Version))
	cachePath := filepath.Join(c.cacheDir, cacheName)

	data, err := c.fetchJSON(ctx, model.ManifestURLs.ReleaseAssets, cachePath)
	if err != nil {
		return nil, fmt.Errorf("load release assets for %s: %w", id, err)
	}

	var ra ReleaseAssets
	if err := sonic.Unmarshal(data, &ra); err != nil {
		return nil, fmt.Errorf("parse release assets: %w", err)
	}
	return &ra, nil
}

// fetchJSON returns the bytes of url, serving from cachePath if it's younger
// than c.ttl. Cache write failures are logged and swallowed.
func (c *Client) fetchJSON(ctx context.Context, url, cachePath string) ([]byte, error) {
	if !c.noCache && cachePath != "" {
		if info, err := os.Stat(cachePath); err == nil && time.Since(info.ModTime()) < c.ttl {
			if data, err := os.ReadFile(cachePath); err == nil {
				slog.Debug("aihub: cache hit", "url", url, "path", cachePath)
				return data, nil
			}
		}
	}

	slog.Debug("aihub: fetching", "url", url)
	resp, err := c.http.R().SetContext(ctx).Get(url)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode() != http.StatusOK {
		return nil, fmt.Errorf("http %d from %s", resp.StatusCode(), url)
	}
	body := resp.Bytes()

	if !c.noCache && cachePath != "" {
		if err := os.MkdirAll(filepath.Dir(cachePath), 0o770); err == nil {
			if werr := os.WriteFile(cachePath, body, 0o664); werr != nil {
				slog.Warn("aihub: cache write failed", "path", cachePath, "err", werr)
			}
		}
	}

	return body, nil
}

// sanitizeForFilename strips characters unsafe on Windows paths.
func sanitizeForFilename(s string) string {
	return strings.Map(func(r rune) rune {
		switch r {
		case '/', '\\', ':', '*', '?', '"', '<', '>', '|':
			return '_'
		}
		return r
	}, path.Base(s))
}
