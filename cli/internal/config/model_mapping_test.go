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

package config

import (
	"testing"
)

func TestGetModelMapping(t *testing.T) {
	tests := []struct {
		shortcut    string
		expected    string
		shouldExist bool
	}{
		{"qwen3", "Qwen/Qwen3-4B-GGUF", true},
		{"qwen2vl", "ggml-org/Qwen2-VL-2B-Instruct-GGUF", true},
		{"qwen2.5vl", "Qwen/Qwen2.5-VL-3B-Instruct", true},
		{"gemma3", "ggml-org/gemma-3-4b-it-GGUF", true},
		{"smolvlm", "ggml-org/SmolVLM-500M-Instruct-GGUF", true},
		{"unknown", "", false},
		{"", "", false},
	}

	for _, test := range tests {
		actual, exists := GetModelMapping(test.shortcut)
		if exists != test.shouldExist {
			t.Errorf("GetModelMapping(%q) exists = %v, want %v", test.shortcut, exists, test.shouldExist)
		}
		if exists && actual != test.expected {
			t.Errorf("GetModelMapping(%q) = %q, want %q", test.shortcut, actual, test.expected)
		}
	}
}
