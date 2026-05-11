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
	"log/slog"
	"sort"
	"strings"

	"github.com/qcom-it-nexa-ai/geniex/cli/internal/qaihm"
)

// Errors returned by Match. Callers pattern-match these so they can print
// actionable hints (e.g. the list of supported chipsets).
var (
	ErrUnknownChipset      = errors.New("aihub: chipset not found in platform.json")
	ErrChipsetNotAvailable = errors.New("aihub: chipset is not a target for this model")
	ErrUnsupportedDomain   = errors.New("aihub: model domain not supported by CLI (only LLM/VLM)")
)

// Availability lists the (chipset, runtime, precision) triples a
// ReleaseAssets file publishes, for error messages.
type Availability struct {
	Chipset   string
	Runtime   string
	Precision string
}

// ChipsetNotAvailableError is the typed form of ErrChipsetNotAvailable and
// carries the list of chipsets actually supported by the model, so the
// caller can render a user-friendly hint.
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

// RuntimeForDomain maps a ModelDomain enum value to the runtime enum the CLI
// knows how to download + execute. Returns ErrUnsupportedDomain otherwise.
func RuntimeForDomain(domain qaihm.ModelDomain) (qaihm.Runtime, error) {
	switch domain {
	case qaihm.ModelDomain_MODEL_DOMAIN_GENERATIVE_AI, qaihm.ModelDomain_MODEL_DOMAIN_MULTIMODAL:
		return qaihm.Runtime_RUNTIME_GENIE, nil
	default:
		return 0, fmt.Errorf("%w: %s", ErrUnsupportedDomain, domain)
	}
}

// ResolveChipset looks up the user-supplied chipset string against
// platform.json, matching either ChipsetInfo.Name or any of its aliases. It
// returns the canonical name.
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

// SupportedChipsetsFor returns the sorted list of canonical chipset names
// that appear in the given release assets. Used only for error messages.
func SupportedChipsetsFor(ra *qaihm.ModelReleaseAssets) []string {
	seen := make(map[string]struct{})
	for _, a := range ra.GetAssets() {
		seen[a.GetChipset()] = struct{}{}
	}
	names := make([]string, 0, len(seen))
	for n := range seen {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// MatchAll returns all assets that match the requested chipset + model domain,
// sorted by Precision enum value for deterministic ordering.
//
// Returns ChipsetNotAvailableError (wrapping ErrChipsetNotAvailable) when no
// asset matches the requested chipset.
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

	// Sort by Precision enum int value for deterministic display order.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].GetPrecision() < candidates[j].GetPrecision()
	})

	return candidates, nil
}

// Match picks exactly one asset for the requested chipset + model domain.
// When multiple precision variants are available, the one with the lowest
// Precision enum value is returned. Prefer MatchAll when the caller wants to
// let the user choose a precision interactively.
func Match(ra *qaihm.ModelReleaseAssets, plat *qaihm.PlatformInfo, domain qaihm.ModelDomain, chipset string) (*qaihm.ModelReleaseAssets_AssetDetails, error) {
	candidates, err := MatchAll(ra, plat, domain, chipset)
	if err != nil {
		return nil, err
	}
	if len(candidates) > 1 {
		extras := make([]string, 0, len(candidates))
		for _, c := range candidates {
			extras = append(extras, c.GetPrecision().String())
		}
		slog.Debug("aihub: multiple assets match, picking first",
			"model", ra.GetModelId(), "chipset", chipset, "runtime", candidates[0].GetRuntime(),
			"picked", candidates[0].GetPrecision(), "available", extras)
	}
	return candidates[0], nil
}
