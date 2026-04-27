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
	"testing"

	"google.golang.org/protobuf/encoding/protojson"

	"github.com/qcom-it-nexa-ai/geniex/cli/gen/qaihm"
)

// Minimal samples pulled from the real AI Hub JSONs attached to the
// feature discussion. We embed trimmed literals instead of reading files
// to keep the test hermetic under Bazel sandboxing.

const sampleManifestJSON = `{
  "version": "0.51.1.dev1",
  "platform_url": "https://example.com/platform.json",
  "models": [
    {
      "id": "qwen3_4b_instruct_2507",
      "display_name": "Qwen3-4B-Instruct-2507",
      "domain": "MODEL_DOMAIN_GENERATIVE_AI",
      "manifest_urls": {
        "info": "https://example.com/qwen3/info.json",
        "perf": "https://example.com/qwen3/perf.json",
        "release_assets": "https://example.com/qwen3/release-assets.json"
      }
    },
    {
      "id": "baichuan2_7b",
      "display_name": "Baichuan2-7B",
      "domain": "MODEL_DOMAIN_GENERATIVE_AI",
      "manifest_urls": {
        "info": "https://example.com/baichuan2/info.json",
        "perf": "https://example.com/baichuan2/perf.json"
      }
    }
  ]
}`

const samplePlatformJSON = `{
  "aihm_version": "0.51.1.dev1",
  "chipsets": [
    {
      "name": "qualcomm-snapdragon-x-elite",
      "aliases": ["qualcomm-snapdragon-x-elite", "sc8380xp"],
      "marketing_name": "Snapdragon X Elite",
      "supports_fp16": true,
      "htp_version": 73,
      "soc_model": 60,
      "reference_device": "Snapdragon X Elite CRD"
    },
    {
      "name": "qualcomm-snapdragon-8gen1",
      "aliases": ["qualcomm-snapdragon-8gen1", "sm8450"],
      "marketing_name": "Snapdragon 8 Gen 1 Mobile",
      "supports_fp16": true,
      "htp_version": 69,
      "soc_model": 36,
      "reference_device": "Samsung Galaxy S22 Ultra 5G"
    }
  ]
}`

const sampleReleaseAssetsJSON = `{
  "aihm_version": "0.51.1.dev1",
  "model_id": "qwen3_4b_instruct_2507",
  "assets": [
    {
      "precision": "PRECISION_W4A16",
      "runtime": "RUNTIME_GENIE",
      "chipset": "qualcomm-snapdragon-x-elite",
      "download_url": "https://example.com/qwen3-x-elite.zip",
      "tool_versions": {"qairt": "2.45.0"}
    },
    {
      "precision": "PRECISION_W4A16",
      "runtime": "RUNTIME_GENIE",
      "chipset": "qualcomm-snapdragon-8-elite",
      "download_url": "https://example.com/qwen3-8-elite.zip"
    }
  ]
}`

func TestUnmarshalManifest(t *testing.T) {
	var m qaihm.ReleaseManifest
	if err := protojson.Unmarshal([]byte(sampleManifestJSON), &m); err != nil {
		t.Fatalf("unmarshal manifest: %v", err)
	}
	if m.GetVersion() != "0.51.1.dev1" {
		t.Errorf("bad version: %s", m.GetVersion())
	}
	if len(m.GetModels()) != 2 {
		t.Fatalf("expected 2 models, got %d", len(m.GetModels()))
	}
	if m.GetModels()[0].GetId() != "qwen3_4b_instruct_2507" {
		t.Errorf("unexpected first model: %+v", m.GetModels()[0])
	}
	if m.GetModels()[0].GetManifestUrls().GetReleaseAssets() == "" {
		t.Errorf("qwen3 should have release_assets url")
	}
	if m.GetModels()[1].GetManifestUrls().GetReleaseAssets() != "" {
		t.Errorf("baichuan2 should NOT have release_assets url")
	}
}

func TestUnmarshalPlatformAndReleaseAssets(t *testing.T) {
	var p qaihm.PlatformInfo
	if err := protojson.Unmarshal([]byte(samplePlatformJSON), &p); err != nil {
		t.Fatalf("unmarshal platform: %v", err)
	}
	if len(p.GetChipsets()) != 2 {
		t.Fatalf("expected 2 chipsets, got %d", len(p.GetChipsets()))
	}
	if p.GetChipsets()[0].GetHtpVersion() != 73 {
		t.Errorf("bad htp_version: %d", p.GetChipsets()[0].GetHtpVersion())
	}

	var ra qaihm.ModelReleaseAssets
	if err := protojson.Unmarshal([]byte(sampleReleaseAssetsJSON), &ra); err != nil {
		t.Fatalf("unmarshal release assets: %v", err)
	}
	if ra.GetModelId() != "qwen3_4b_instruct_2507" {
		t.Errorf("bad model_id: %s", ra.GetModelId())
	}
	if len(ra.GetAssets()) != 2 {
		t.Fatalf("expected 2 assets, got %d", len(ra.GetAssets()))
	}
	if ra.GetAssets()[0].GetRuntime() != RuntimeGenie {
		t.Errorf("bad runtime: %s", ra.GetAssets()[0].GetRuntime())
	}
}
