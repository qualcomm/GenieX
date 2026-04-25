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

func loadFixtures(t *testing.T) (*Platform, *ReleaseAssets) {
	t.Helper()
	var p Platform
	if err := sonic.UnmarshalString(samplePlatformJSON, &p); err != nil {
		t.Fatalf("unmarshal platform: %v", err)
	}
	var ra ReleaseAssets
	if err := sonic.UnmarshalString(sampleReleaseAssetsJSON, &ra); err != nil {
		t.Fatalf("unmarshal release assets: %v", err)
	}
	return &p, &ra
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

func TestMatch_HappyPath(t *testing.T) {
	p, ra := loadFixtures(t)

	asset, err := Match(ra, p, DomainGenerativeAI, "qualcomm-snapdragon-x-elite")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
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

func TestMatch_ChipsetNotAvailable(t *testing.T) {
	p, ra := loadFixtures(t)

	_, err := Match(ra, p, DomainGenerativeAI, "qualcomm-snapdragon-8gen1")
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

func TestMatch_UnsupportedDomain(t *testing.T) {
	p, ra := loadFixtures(t)

	_, err := Match(ra, p, DomainComputer, "qualcomm-snapdragon-x-elite")
	if !errors.Is(err, ErrUnsupportedDomain) {
		t.Errorf("expected ErrUnsupportedDomain, got %v", err)
	}
}

func TestMatch_UnknownChipset(t *testing.T) {
	p, ra := loadFixtures(t)

	_, err := Match(ra, p, DomainGenerativeAI, "nvidia-a100")
	if !errors.Is(err, ErrUnknownChipset) {
		t.Errorf("expected ErrUnknownChipset, got %v", err)
	}
}
