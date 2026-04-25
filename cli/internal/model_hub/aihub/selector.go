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

// RuntimeForDomain maps a MODEL_DOMAIN_* value to the runtime enum the CLI
// knows how to download + execute. Returns ErrUnsupportedDomain otherwise.
func RuntimeForDomain(domain string) (string, error) {
	switch domain {
	case DomainGenerativeAI, DomainMultimodal:
		return RuntimeGenie, nil
	default:
		return "", fmt.Errorf("%w: %s", ErrUnsupportedDomain, domain)
	}
}

// ResolveChipset looks up the user-supplied chipset string against
// platform.json, matching either Chipset.Name or any of its aliases. It
// returns the canonical name.
func ResolveChipset(plat *Platform, chipset string) (string, error) {
	if plat == nil {
		return "", errors.New("aihub: nil platform")
	}
	if chipset == "" {
		return "", errors.New("aihub: empty chipset")
	}

	target := strings.ToLower(strings.TrimSpace(chipset))
	for i := range plat.Chipsets {
		cs := &plat.Chipsets[i]
		if strings.ToLower(cs.Name) == target {
			return cs.Name, nil
		}
		for _, a := range cs.Aliases {
			if strings.ToLower(a) == target {
				return cs.Name, nil
			}
		}
	}

	return "", fmt.Errorf("%w: %s", ErrUnknownChipset, chipset)
}

// SupportedChipsetsFor returns the sorted list of canonical chipset names
// that appear in the given release assets. Used only for error messages.
func SupportedChipsetsFor(ra *ReleaseAssets) []string {
	seen := make(map[string]struct{})
	for _, a := range ra.Assets {
		seen[a.Chipset] = struct{}{}
	}
	names := make([]string, 0, len(seen))
	for n := range seen {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// Match picks exactly one Asset for the requested chipset + model domain:
// domain is mapped to a runtime, chipset is resolved to canonical form via
// Platform aliases, and assets are filtered by both. If multiple candidates
// remain (e.g. several precisions) the first in publication order wins and
// the choice is logged at debug level.
func Match(ra *ReleaseAssets, plat *Platform, domain, chipset string) (*Asset, error) {
	if ra == nil || len(ra.Assets) == 0 {
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

	avail := make([]Availability, 0, len(ra.Assets))
	var candidates []*Asset
	for i := range ra.Assets {
		a := &ra.Assets[i]
		avail = append(avail, Availability{
			Chipset: a.Chipset, Runtime: a.Runtime, Precision: a.Precision,
		})
		if a.Chipset != canonical {
			continue
		}
		if a.Runtime != runtime {
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
	if len(candidates) > 1 {
		extras := make([]string, 0, len(candidates))
		for _, c := range candidates {
			extras = append(extras, c.Precision)
		}
		slog.Debug("aihub: multiple assets match, picking first",
			"model", ra.ModelID, "chipset", canonical, "runtime", runtime,
			"picked", candidates[0].Precision, "available", extras)
	}

	return candidates[0], nil
}
