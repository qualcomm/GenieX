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

// Package sochost is a best-effort host SoC probe — Windows reads the
// registry CPU brand, Linux reads the boot Device Tree `compatible` node.
// Mirrors sdk/model-manager/crates/core/src/source/ai_hub/detect.rs.
package sochost

import "strings"

// First marker that appears in the probe wins. `qcm6490` is the
// commercial SKU spelling many upstream QCS6490 board dts files use for
// the SoC row instead of `qcs6490`.
var chipsetMarkers = []struct{ marker, alias string }{
	{"x1e", "qualcomm-snapdragon-x-elite"},
	{"x1p", "qualcomm-snapdragon-x-plus-8-core"},
	{"x2e", "qualcomm-snapdragon-x2-elite"},
	{"qcs6490", "qualcomm-qcs6490"},
	{"qcm6490", "qualcomm-qcs6490"},
	{"qcs9075", "qualcomm-qcs9075"},
}

// DetectChipsetAlias returns ("", false) on any failure so callers can
// fall back to interactive selection.
func DetectChipsetAlias() (string, bool) {
	return chipsetFromProbe(readHostProbe())
}

func chipsetFromProbe(probe string) (string, bool) {
	probe = strings.ToLower(probe)
	for _, m := range chipsetMarkers {
		if strings.Contains(probe, m.marker) {
			return m.alias, true
		}
	}
	return "", false
}
