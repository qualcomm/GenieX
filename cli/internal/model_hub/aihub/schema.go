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

// Package aihub wraps the Qualcomm AI Hub public release assets bucket.
//
// Three JSON indices drive the download flow, fetched via Client:
//
//	manifest.json          - top-level catalog of models + URLs
//	platform.json          - chipset alias table
//	release-assets.json    - per-model downloadable artefacts
//
// Given a model id and the user's --chipset flag, aihub resolves exactly
// one Asset to download.
package aihub

// Model domain enum values published in manifest.json.
const (
	DomainGenerativeAI = "MODEL_DOMAIN_GENERATIVE_AI"
	DomainMultimodal   = "MODEL_DOMAIN_MULTIMODAL"
	DomainComputer     = "MODEL_DOMAIN_COMPUTER_VISION"
)

// Runtime enum values used by release-assets.json. Only GENIE is consumed.
const RuntimeGenie = "RUNTIME_GENIE"

// Manifest mirrors the top-level manifest.json.
type Manifest struct {
	Version     string  `json:"version"`
	PlatformURL string  `json:"platform_url"`
	Models      []Model `json:"models"`
}

type Model struct {
	ID           string       `json:"id"`
	Domain       string       `json:"domain"`
	ManifestURLs ManifestURLs `json:"manifest_urls"`
}

type ManifestURLs struct {
	ReleaseAssets string `json:"release_assets,omitempty"`
}

// Platform mirrors the chipset table from platform.json. Only Name + Aliases
// are consumed by the selector; HTPVersion is kept for schema round-trip tests
// and potential future compatibility checks.
type Platform struct {
	AIHMVersion string    `json:"aihm_version"`
	Chipsets    []Chipset `json:"chipsets"`
}

type Chipset struct {
	Name       string   `json:"name"`
	Aliases    []string `json:"aliases"`
	HTPVersion int      `json:"htp_version"`
}

// ReleaseAssets mirrors release-assets.json for one model.
type ReleaseAssets struct {
	AIHMVersion string  `json:"aihm_version"`
	ModelID     string  `json:"model_id"`
	Assets      []Asset `json:"assets"`
}

type Asset struct {
	Precision   string `json:"precision"`
	Runtime     string `json:"runtime"`
	Chipset     string `json:"chipset"`
	DownloadURL string `json:"download_url"`
}
