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

package model_hub

import "testing"

func TestNormalizeModelName(t *testing.T) {
	cases := []struct {
		in        string
		wantName  string
		wantQuant string
	}{
		{"Qwen3-4B", "qualcomm/Qwen3-4B", ""},
		{"qualcomm/Qwen3-4B", "qualcomm/Qwen3-4B", ""},
		{"ai-hub-models/Qwen3-4B", "qualcomm/Qwen3-4B", ""},
		{"ai-hub-models/Qwen3-4B:q4_k_m", "qualcomm/Qwen3-4B", "Q4_K_M"},
		{"ggml-org/Qwen3-1.7B-GGUF", "ggml-org/Qwen3-1.7B-GGUF", ""},
	}
	for _, c := range cases {
		gotName, gotQuant := NormalizeModelName(c.in)
		if gotName != c.wantName || gotQuant != c.wantQuant {
			t.Errorf("NormalizeModelName(%q) = (%q, %q), want (%q, %q)",
				c.in, gotName, gotQuant, c.wantName, c.wantQuant)
		}
	}
}
