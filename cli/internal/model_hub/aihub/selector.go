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
	"fmt"
	"sort"
	"strings"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/qaihm"
)

// Sentinel errors returned by MatchAll.
var (
	ErrUnknownChipset      = errors.New("aihub: chipset not found in platform.json")
	ErrChipsetNotAvailable = errors.New("aihub: chipset is not a target for this model")
	ErrUnsupportedDomain   = errors.New("aihub: model domain not supported by CLI (only LLM/VLM)")
)

// Availability: one (chipset, runtime, precision) triple from ReleaseAssets.
type Availability struct {
	Chipset   string
	Runtime   string
	Precision string
}

// ChipsetNotAvailableError carries the chipsets actually supported by
// the model so callers can render a hint. Wraps ErrChipsetNotAvailable.
type ChipsetNotAvailableError struct {
	Requested string
	Available []Availability
}

func (e *ChipsetNotAvailableError) Error() string {
	names := make([]string, 0, len(e.Available))
	seen := make(map[string]struct{})
	for _, a := range e.Available {
		if _, ok := seen[a.Chipset]; ok {
			continue
		}
		seen[a.Chipset] = struct{}{}
		names = append(names, a.Chipset)
	}
	sort.Strings(names)
	return fmt.Sprintf("aihub: chipset %q not available; model supports: %s",
		e.Requested, strings.Join(names, ", "))
}

func (e *ChipsetNotAvailableError) Is(target error) bool {
	return target == ErrChipsetNotAvailable
}

// RuntimeForDomain maps a ModelDomain to the runtime the CLI can run.
// Returns ErrUnsupportedDomain for non-LLM/VLM domains.
func RuntimeForDomain(domain qaihm.ModelDomain) (qaihm.Runtime, error) {
	switch domain {
	case qaihm.ModelDomain_MODEL_DOMAIN_GENERATIVE_AI, qaihm.ModelDomain_MODEL_DOMAIN_MULTIMODAL:
		return qaihm.Runtime_RUNTIME_GENIE, nil
	default:
		return 0, fmt.Errorf("%w: %s", ErrUnsupportedDomain, domain)
	}
}

// ResolveChipset returns the canonical chipset name matching chipset
// (case-insensitive, against name + aliases) or ErrUnknownChipset.
func ResolveChipset(plat *qaihm.PlatformInfo, chipset string) (string, error) {
	if plat == nil {
		return "", errors.New("aihub: nil platform")
	}
	if chipset == "" {
		return "", errors.New("aihub: empty chipset")
	}

	target := strings.ToLower(strings.TrimSpace(chipset))
	for _, cs := range plat.GetChipsets() {
		if strings.ToLower(cs.GetName()) == target {
			return cs.GetName(), nil
		}
		for _, a := range cs.GetAliases() {
			if strings.ToLower(a) == target {
				return cs.GetName(), nil
			}
		}
	}

	return "", fmt.Errorf("%w: %s", ErrUnknownChipset, chipset)
}

// MatchAll returns assets matching chipset+domain, sorted by precision.
// Returns ChipsetNotAvailableError when no asset matches the chipset.
func MatchAll(ra *qaihm.ModelReleaseAssets, plat *qaihm.PlatformInfo, domain qaihm.ModelDomain, chipset string) ([]*qaihm.ModelReleaseAssets_AssetDetails, error) {
	if ra == nil || len(ra.GetAssets()) == 0 {
		return nil, errors.New("aihub: empty release assets")
	}

	runtime, err := RuntimeForDomain(domain)
	if err != nil {
		return nil, err
	}

	canonical, err := ResolveChipset(plat, chipset)
	if err != nil {
		return nil, err
	}

	avail := make([]Availability, 0, len(ra.GetAssets()))
	var candidates []*qaihm.ModelReleaseAssets_AssetDetails
	for _, a := range ra.GetAssets() {
		avail = append(avail, Availability{
			Chipset: a.GetChipset(), Runtime: a.GetRuntime().String(), Precision: a.GetPrecision().String(),
		})
		if a.GetChipset() != canonical {
			continue
		}
		if a.GetRuntime() != runtime {
			continue
		}
		candidates = append(candidates, a)
	}

	if len(candidates) == 0 {
		return nil, &ChipsetNotAvailableError{
			Requested: chipset,
			Available: avail,
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].GetPrecision() < candidates[j].GetPrecision()
	})

	return candidates, nil
}

