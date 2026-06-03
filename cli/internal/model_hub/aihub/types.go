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

// Hand-rolled subset of the AI Hub manifest / platform / release-assets
// schemas. Field names match protojson-style snake_case JSON. Enums are
// kept as typed strings — protojson emits the enum names verbatim.

type ModelDomain string

const (
	ModelDomainComputerVision ModelDomain = "MODEL_DOMAIN_COMPUTER_VISION"
	ModelDomainMultimodal     ModelDomain = "MODEL_DOMAIN_MULTIMODAL"
	ModelDomainGenerativeAI   ModelDomain = "MODEL_DOMAIN_GENERATIVE_AI"
)

type Runtime string

const (
	RuntimeGenie Runtime = "RUNTIME_GENIE"
)

type OperatingSystemType string

const (
	OSTypeWindows OperatingSystemType = "OPERATING_SYSTEM_TYPE_WINDOWS"
	OSTypeQCLinux OperatingSystemType = "OPERATING_SYSTEM_TYPE_QC_LINUX"
)

type Precision string

// precisionRank mirrors the order in the original precision.proto so
// MatchAll keeps the same "lowest precision first" sort.
var precisionRank = map[Precision]int{
	"PRECISION_UNSPECIFIED":       0,
	"PRECISION_FLOAT":             1,
	"PRECISION_W8A8":              2,
	"PRECISION_W8A16":             3,
	"PRECISION_W16A16":            4,
	"PRECISION_W4A16":             5,
	"PRECISION_W4":                6,
	"PRECISION_W8A8_MIXED_INT16":  7,
	"PRECISION_W8A16_MIXED_INT16": 8,
	"PRECISION_W8A8_MIXED_FP16":   9,
	"PRECISION_W8A16_MIXED_FP16":  10,
	"PRECISION_MXFP4":             11,
	"PRECISION_Q8_0":              12,
	"PRECISION_Q4_0":              13,
	"PRECISION_MIXED":             14,
	"PRECISION_MIXED_WITH_FLOAT":  15,
}

type ModelManifestUrls struct {
	ReleaseAssets string `json:"release_assets"`
}

type ManifestModelEntry struct {
	ID           string            `json:"id"`
	DisplayName  string            `json:"display_name"`
	Domain       ModelDomain       `json:"domain"`
	ManifestUrls ModelManifestUrls `json:"manifest_urls"`
}

type ReleaseManifest struct {
	PlatformURL string               `json:"platform_url"`
	Models      []ManifestModelEntry `json:"models"`
}

type OperatingSystem struct {
	OSType OperatingSystemType `json:"ostype"`
}

type DeviceInfo struct {
	Name    string          `json:"name"`
	Chipset string          `json:"chipset"`
	OS      OperatingSystem `json:"os"`
}

type ChipsetInfo struct {
	Name    string   `json:"name"`
	Aliases []string `json:"aliases"`
}

type PlatformInfo struct {
	Devices  []DeviceInfo  `json:"devices"`
	Chipsets []ChipsetInfo `json:"chipsets"`
}

type AssetDetails struct {
	Precision   Precision `json:"precision"`
	Runtime     Runtime   `json:"runtime"`
	Chipset     string    `json:"chipset"`
	DownloadURL string    `json:"download_url"`
}

type ModelReleaseAssets struct {
	Assets []AssetDetails `json:"assets"`
}
