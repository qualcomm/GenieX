// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package config

import "strings"

// cpuDefaultChipsets default to CPU because their llama.cpp NPU path is slow.
// Keys are lower-cased and cover both forms ResolveChipset can yield: the
// canonical id (offline probe) and the reference-device name (stored by the
// picker).
var cpuDefaultChipsets = map[string]bool{
	"qualcomm-qcs6490":                true,
	"dragonwing rb3 gen 2 vision kit": true,
}

// ComputeDefault keeps an explicit compute; an empty one becomes "cpu" on
// cpuDefaultChipsets (overridden=true), else stays empty for the SDK's default.
func ComputeDefault(compute, chipset string) (result string, overridden bool) {
	if strings.TrimSpace(compute) != "" {
		return compute, false
	}
	if cpuDefaultChipsets[strings.ToLower(strings.TrimSpace(chipset))] {
		return "cpu", true
	}
	return compute, false
}
