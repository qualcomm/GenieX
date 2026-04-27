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

import (
	"github.com/qcom-it-nexa-ai/geniex/cli/gen/qaihm"
)

// Domain enum values used for supported-domain checks.
var (
	DomainGenerativeAI = qaihm.ModelDomain_MODEL_DOMAIN_GENERATIVE_AI
	DomainMultimodal   = qaihm.ModelDomain_MODEL_DOMAIN_MULTIMODAL
)

// RuntimeGenie is the runtime enum value the CLI downloads and executes.
var RuntimeGenie = qaihm.Runtime_RUNTIME_GENIE

// Type aliases so call-sites outside this package don't need to import qaihm
// directly for the types they reference.
type (
	Manifest      = qaihm.ReleaseManifest
	Model         = qaihm.ManifestModelEntry
	Platform      = qaihm.PlatformInfo
	ReleaseAssets = qaihm.ModelReleaseAssets
	Asset         = qaihm.ModelReleaseAssets_AssetDetails
)
