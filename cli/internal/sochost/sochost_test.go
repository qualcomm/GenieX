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

package sochost

import "testing"

func TestCPUNameToChipsetAlias(t *testing.T) {
	cases := []struct {
		name  string
		brand string
		want  string
		ok    bool
	}{
		{"x elite brand", "Snapdragon(R) X Elite - X1E80100 - Qualcomm(R) Oryon(TM) CPU", "qualcomm-snapdragon-x-elite", true},
		{"x plus brand", "Snapdragon X Plus - X1P64100 - Qualcomm Oryon CPU", "qualcomm-snapdragon-x-plus-8-core", true},
		{"x2 elite brand", "Snapdragon X2 Elite - X2E80100 - Qualcomm Oryon CPU", "qualcomm-snapdragon-x2-elite", true},
		{"windows reg format", "Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz", "qualcomm-snapdragon-x-elite", true},
		{"x1e lower bin", "X1E78100", "qualcomm-snapdragon-x-elite", true},
		{"x1p lower bin", "X1P42100", "qualcomm-snapdragon-x-plus-8-core", true},
		{"intel", "Intel(R) Core(TM) i7-12700H @ 2.30GHz", "", false},
		{"amd", "AMD Ryzen 7 7840U", "", false},
		{"empty", "", "", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, ok := CPUNameToChipsetAlias(c.brand)
			if got != c.want || ok != c.ok {
				t.Fatalf("CPUNameToChipsetAlias(%q) = (%q, %v); want (%q, %v)", c.brand, got, ok, c.want, c.ok)
			}
		})
	}
}

func TestIsOryonSKURejectsNearShapedTokens(t *testing.T) {
	cases := []string{
		"X1E80100A", // trailing letter
		"XEE80100",  // second char not digit
		"X1E10",     // too short
		"",
	}
	for _, tok := range cases {
		if isOryonSKU(tok) {
			t.Errorf("isOryonSKU(%q) = true; want false", tok)
		}
	}
}
