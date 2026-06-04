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
	"errors"
	"testing"

	"github.com/bytedance/sonic"
)

// Verbatim slices of the real v0.53.1 AI Hub JSONs (manifest trimmed to
// two model entries; platform trimmed to two chipsets). Captured from
// https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/releases/v0.53.1/.
// We embed literals instead of reading files to keep the test hermetic
// under Bazel sandboxing.

const sampleManifestJSON = `{
  "version": "0.53.1",
  "platform_url": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/releases/v0.53.1/platform.json",
  "models": [
    {
      "id": "qwen3_4b_instruct_2507",
      "display_name": "Qwen3-4B-Instruct-2507",
      "domain": "MODEL_DOMAIN_GENERATIVE_AI",
      "manifest_urls": {
        "info": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/qwen3_4b_instruct_2507/releases/v0.53.1/info.json",
        "perf": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/qwen3_4b_instruct_2507/releases/v0.53.1/perf.json",
        "release_assets": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/qwen3_4b_instruct_2507/releases/v0.53.1/release-assets.json"
      }
    },
    {
      "id": "baichuan2_7b",
      "display_name": "Baichuan2-7B",
      "domain": "MODEL_DOMAIN_GENERATIVE_AI",
      "manifest_urls": {
        "info": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/baichuan2_7b/releases/v0.53.1/info.json",
        "perf": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/baichuan2_7b/releases/v0.53.1/perf.json"
      }
    }
  ]
}`

const samplePlatformJSON = `{
  "aihm_version": "0.53.1",
  "chipsets": [
    {
      "name": "qualcomm-snapdragon-8gen1",
      "aliases": [
        "qualcomm-snapdragon-8gen1",
        "sm8450"
      ],
      "marketing_name": "Snapdragon® 8 Gen 1 Mobile",
      "world": "WEBSITE_WORLD_MOBILE",
      "supports_fp16": true,
      "htp_version": 69,
      "soc_model": 36,
      "reference_device": "Samsung Galaxy S22 Ultra 5G"
    },
    {
      "name": "qualcomm-snapdragon-x-elite",
      "aliases": [
        "qualcomm-snapdragon-x-elite",
        "sc8380xp"
      ],
      "marketing_name": "Snapdragon® X Elite",
      "world": "WEBSITE_WORLD_COMPUTE",
      "supports_fp16": true,
      "htp_version": 73,
      "soc_model": 60,
      "reference_device": "Snapdragon X Elite CRD"
    }
  ]
}`

const sampleReleaseAssetsJSON = `{
  "aihm_version": "0.53.1",
  "model_id": "qwen3_4b_instruct_2507",
  "assets": [
    {
      "precision": "PRECISION_W4A16",
      "runtime": "RUNTIME_GENIE",
      "chipset": "qualcomm-qcs8275",
      "download_url": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/qwen3_4b_instruct_2507/releases/v0.53.1/qwen3_4b_instruct_2507-genie-w4a16-qualcomm_qcs8275.zip",
      "tool_versions": {
        "qairt": "2.45.0.260326154327"
      }
    },
    {
      "precision": "PRECISION_W4A16",
      "runtime": "RUNTIME_GENIE",
      "chipset": "qualcomm-qcs9075",
      "download_url": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/qwen3_4b_instruct_2507/releases/v0.53.1/qwen3_4b_instruct_2507-genie-w4a16-qualcomm_qcs9075.zip",
      "tool_versions": {
        "qairt": "2.45.0.260326154327"
      }
    },
    {
      "precision": "PRECISION_W4A16",
      "runtime": "RUNTIME_GENIE",
      "chipset": "qualcomm-snapdragon-8-elite",
      "download_url": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/qwen3_4b_instruct_2507/releases/v0.53.1/qwen3_4b_instruct_2507-genie-w4a16-qualcomm_snapdragon_8_elite.zip",
      "tool_versions": {
        "qairt": "2.45.0.260326154327"
      }
    },
    {
      "precision": "PRECISION_W4A16",
      "runtime": "RUNTIME_GENIE",
      "chipset": "qualcomm-snapdragon-8-elite-gen5",
      "download_url": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/qwen3_4b_instruct_2507/releases/v0.53.1/qwen3_4b_instruct_2507-genie-w4a16-qualcomm_snapdragon_8_elite_gen5.zip",
      "tool_versions": {
        "qairt": "2.45.0.260326154327"
      }
    },
    {
      "precision": "PRECISION_W4A16",
      "runtime": "RUNTIME_GENIE",
      "chipset": "qualcomm-snapdragon-x-elite",
      "download_url": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/qwen3_4b_instruct_2507/releases/v0.53.1/qwen3_4b_instruct_2507-genie-w4a16-qualcomm_snapdragon_x_elite.zip",
      "tool_versions": {
        "qairt": "2.45.0.260326154327"
      }
    },
    {
      "precision": "PRECISION_W4A16",
      "runtime": "RUNTIME_GENIE",
      "chipset": "qualcomm-snapdragon-x2-elite",
      "download_url": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/qwen3_4b_instruct_2507/releases/v0.53.1/qwen3_4b_instruct_2507-genie-w4a16-qualcomm_snapdragon_x2_elite.zip",
      "tool_versions": {
        "qairt": "2.45.0.260326154327"
      }
    }
  ]
}`

func loadFixtures(t *testing.T) (*PlatformInfo, *ModelReleaseAssets) {
	t.Helper()
	var p PlatformInfo
	if err := sonic.Unmarshal([]byte(samplePlatformJSON), &p); err != nil {
		t.Fatalf("unmarshal platform: %v", err)
	}
	var ra ModelReleaseAssets
	if err := sonic.Unmarshal([]byte(sampleReleaseAssetsJSON), &ra); err != nil {
		t.Fatalf("unmarshal release assets: %v", err)
	}
	return &p, &ra
}

func TestUnmarshalManifest(t *testing.T) {
	var m ReleaseManifest
	if err := sonic.Unmarshal([]byte(sampleManifestJSON), &m); err != nil {
		t.Fatalf("unmarshal manifest: %v", err)
	}
	if len(m.Models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(m.Models))
	}
	if m.Models[0].ID != "qwen3_4b_instruct_2507" {
		t.Errorf("unexpected first model: %+v", m.Models[0])
	}
	if m.Models[0].ManifestUrls.ReleaseAssets == "" {
		t.Errorf("qwen3 should have release_assets url")
	}
	if m.Models[1].ManifestUrls.ReleaseAssets != "" {
		t.Errorf("baichuan2 should NOT have release_assets url")
	}
}

func TestUnmarshalPlatformAndReleaseAssets(t *testing.T) {
	var p PlatformInfo
	if err := sonic.Unmarshal([]byte(samplePlatformJSON), &p); err != nil {
		t.Fatalf("unmarshal platform: %v", err)
	}
	if len(p.Chipsets) != 2 {
		t.Fatalf("expected 2 chipsets, got %d", len(p.Chipsets))
	}

	var ra ModelReleaseAssets
	if err := sonic.Unmarshal([]byte(sampleReleaseAssetsJSON), &ra); err != nil {
		t.Fatalf("unmarshal release assets: %v", err)
	}
	if len(ra.Assets) != 6 {
		t.Fatalf("expected 6 assets, got %d", len(ra.Assets))
	}
	if ra.Assets[0].Runtime != RuntimeGenie {
		t.Errorf("bad runtime: %s", ra.Assets[0].Runtime)
	}
}

func TestResolveChipset(t *testing.T) {
	p, _ := loadFixtures(t)

	// Canonical name.
	got, err := ResolveChipset(p, "qualcomm-snapdragon-x-elite")
	if err != nil || got != "qualcomm-snapdragon-x-elite" {
		t.Errorf("canonical: got (%q, %v)", got, err)
	}
	// Alias resolves to canonical.
	got, err = ResolveChipset(p, "sc8380xp")
	if err != nil || got != "qualcomm-snapdragon-x-elite" {
		t.Errorf("alias: got (%q, %v)", got, err)
	}
	// Case-insensitive.
	got, err = ResolveChipset(p, "QUALCOMM-SNAPDRAGON-X-ELITE")
	if err != nil || got != "qualcomm-snapdragon-x-elite" {
		t.Errorf("case-insensitive: got (%q, %v)", got, err)
	}
	// Unknown.
	if _, err = ResolveChipset(p, "nvidia-a100"); !errors.Is(err, ErrUnknownChipset) {
		t.Errorf("unknown chipset: expected ErrUnknownChipset, got %v", err)
	}
}

func TestMatchAll_HappyPath(t *testing.T) {
	p, ra := loadFixtures(t)

	cands, err := MatchAll(ra, p, ModelDomainGenerativeAI, "qualcomm-snapdragon-x-elite")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(cands) != 1 {
		t.Fatalf("expected 1 candidate, got %d", len(cands))
	}
	asset := cands[0]
	if asset.Chipset != "qualcomm-snapdragon-x-elite" {
		t.Errorf("wrong chipset: %s", asset.Chipset)
	}
	if asset.Runtime != RuntimeGenie {
		t.Errorf("wrong runtime: %s", asset.Runtime)
	}
	if asset.DownloadURL == "" {
		t.Errorf("missing download_url")
	}
}

func TestMatchAll_ChipsetNotAvailable(t *testing.T) {
	p, ra := loadFixtures(t)

	_, err := MatchAll(ra, p, ModelDomainGenerativeAI, "qualcomm-snapdragon-8gen1")
	if !errors.Is(err, ErrChipsetNotAvailable) {
		t.Fatalf("expected ErrChipsetNotAvailable, got %v", err)
	}

	var cnae *ChipsetNotAvailableError
	if !errors.As(err, &cnae) {
		t.Fatalf("expected ChipsetNotAvailableError, got %T", err)
	}
	if len(cnae.Available) == 0 {
		t.Errorf("ChipsetNotAvailableError should carry at least one available entry")
	}
}

func TestMatchAll_UnsupportedDomain(t *testing.T) {
	p, ra := loadFixtures(t)

	_, err := MatchAll(ra, p, ModelDomainComputerVision, "qualcomm-snapdragon-x-elite")
	if !errors.Is(err, ErrUnsupportedDomain) {
		t.Errorf("expected ErrUnsupportedDomain, got %v", err)
	}
}

func TestMatchAll_UnknownChipset(t *testing.T) {
	p, ra := loadFixtures(t)

	_, err := MatchAll(ra, p, ModelDomainGenerativeAI, "nvidia-a100")
	if !errors.Is(err, ErrUnknownChipset) {
		t.Errorf("expected ErrUnknownChipset, got %v", err)
	}
}
