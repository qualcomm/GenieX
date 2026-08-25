// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package config

import "testing"

func TestComputeDefault(t *testing.T) {
	tests := []struct {
		name           string
		compute        string
		chipset        string
		want           string
		wantOverridden bool
	}{
		{"explicit npu is kept on rb3", "npu", "qualcomm-qcs6490", "npu", false},
		{"explicit cpu is kept", "cpu", "qualcomm-snapdragon-x-elite", "cpu", false},
		{"whitespace-only is treated as unset", "  ", "qualcomm-qcs6490", "cpu", true},
		{"unset on rb3 canonical becomes cpu", "", "qualcomm-qcs6490", "cpu", true},
		{"unset on rb3 display name becomes cpu", "", "Dragonwing RB3 Gen 2 Vision Kit", "cpu", true},
		{"unset on rb3 is case-insensitive", "", "Qualcomm-QCS6490", "cpu", true},
		{"unset on other chipset stays unset", "", "qualcomm-snapdragon-x-elite", "", false},
		{"unset with no detected chipset stays unset", "", "", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, overridden := ComputeDefault(tt.compute, tt.chipset)
			if got != tt.want || overridden != tt.wantOverridden {
				t.Fatalf("ComputeDefault(%q, %q) = (%q, %v), want (%q, %v)",
					tt.compute, tt.chipset, got, overridden, tt.want, tt.wantOverridden)
			}
		})
	}
}
