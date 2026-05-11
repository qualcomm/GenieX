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

// Package sochost provides best-effort host SoC detection so the CLI can
// pick an AI Hub chipset without asking the user. The Rust model-manager
// has an equivalent probe in
// sdk/model-manager/crates/core/src/source/ai_hub/detect.rs; this file
// mirrors its SKU table.
package sochost

// CPUNameToChipsetAlias maps a Qualcomm Oryon CPU brand string to the
// matching AI Hub chipset alias. Returns ("", false) for unknown CPUs.
func CPUNameToChipsetAlias(brand string) (string, bool) {
	sku, ok := extractOryonSKU(brand)
	if !ok {
		return "", false
	}
	switch sku[:3] {
	case "X1E":
		return "qualcomm-snapdragon-x-elite", true
	case "X1P":
		return "qualcomm-snapdragon-x-plus-8-core", true
	case "X2E":
		return "qualcomm-snapdragon-x2-elite", true
	default:
		return "", false
	}
}

func extractOryonSKU(brand string) (string, bool) {
	start := -1
	for i := 0; i <= len(brand); i++ {
		if i == len(brand) || !isAlphanumeric(brand[i]) {
			if start >= 0 {
				tok := brand[start:i]
				if isOryonSKU(tok) {
					return toUpper(tok), true
				}
				start = -1
			}
		} else if start < 0 {
			start = i
		}
	}
	return "", false
}

func isOryonSKU(tok string) bool {
	if len(tok) < 6 {
		return false
	}
	if toUpperByte(tok[0]) != 'X' {
		return false
	}
	if !isDigit(tok[1]) {
		return false
	}
	c := toUpperByte(tok[2])
	if c != 'E' && c != 'P' {
		return false
	}
	for i := 3; i < len(tok); i++ {
		if !isDigit(tok[i]) {
			return false
		}
	}
	return true
}

func isAlphanumeric(b byte) bool {
	return isDigit(b) || (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

func isDigit(b byte) bool {
	return b >= '0' && b <= '9'
}

func toUpperByte(b byte) byte {
	if b >= 'a' && b <= 'z' {
		return b - 32
	}
	return b
}

func toUpper(s string) string {
	out := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		out[i] = toUpperByte(s[i])
	}
	return string(out)
}
