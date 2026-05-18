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

func TestChipsetFromProbe(t *testing.T) {
	cases := []struct {
		name  string
		probe string
		want  string
		ok    bool
	}{
		// Windows CPU brand string.
		{name: "x elite brand", probe: "Snapdragon(R) X Elite - X1E80100 - Qualcomm(R) Oryon(TM) CPU", want: "qualcomm-snapdragon-x-elite", ok: true},
		{name: "x plus brand", probe: "Snapdragon X Plus - X1P64100 - Qualcomm Oryon CPU", want: "qualcomm-snapdragon-x-plus-8-core", ok: true},
		{name: "x2 elite brand", probe: "Snapdragon X2 Elite - X2E80100 - Qualcomm Oryon CPU", want: "qualcomm-snapdragon-x2-elite", ok: true},
		{name: "windows reg format", probe: "Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz", want: "qualcomm-snapdragon-x-elite", ok: true},
		{name: "intel", probe: "Intel(R) Core(TM) i7-12700H @ 2.30GHz"},
		{name: "amd", probe: "AMD Ryzen 7 7840U"},

		// Linux DT compatible. The 9075 fixture was captured from a real
		// IQ-9075 EVK; QCS6490 boards split between the `qcs6490` and
		// `qcm6490` SoC spellings depending on dts vendor.
		{name: "qcs9075 evk", probe: "qcom,qcs9075-addons-iq-9075-evk\x00qcom,qcs9075\x00qcom,sa8775p\x00", want: "qualcomm-qcs9075", ok: true},
		{name: "qcs6490 rb3 gen2", probe: "qcom,qcs6490-rb3gen2-vision-kit\x00qcom,qcs6490\x00", want: "qualcomm-qcs6490", ok: true},
		{name: "qcm6490 radxa dragon-q6a", probe: "radxa,dragon-q6a\x00qcom,qcm6490\x00", want: "qualcomm-qcs6490", ok: true},
		{name: "family only", probe: "qcom,sa8775p\x00"},
		{name: "non qualcomm vendor", probe: "brcm,bcm2837\x00"},

		{name: "empty probe"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, ok := chipsetFromProbe(c.probe)
			if got != c.want || ok != c.ok {
				t.Fatalf("chipsetFromProbe(%q) = (%q, %v); want (%q, %v)", c.probe, got, ok, c.want, c.ok)
			}
		})
	}
}
