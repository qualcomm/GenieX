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

package types

import "testing"

func TestModelManifest_GetSize(t *testing.T) {
	tests := []struct {
		name string
		m    ModelManifest
		want int64
	}{
		{
			name: "empty manifest is zero",
			m:    ModelManifest{},
			want: 0,
		},
		{
			name: "skips files that are not downloaded",
			m: ModelManifest{
				ModelFile: map[string]ModelFileInfo{
					"Q4_K_M": {Size: 100, Downloaded: true},
					"Q8_0":   {Size: 200, Downloaded: false},
				},
				MMProjFile: ModelFileInfo{Size: 1000, Downloaded: false},
				ExtraFiles: []ModelFileInfo{
					{Size: 10, Downloaded: true},
					{Size: 20, Downloaded: false},
				},
			},
			want: 110,
		},
		{
			name: "sums every downloaded source",
			m: ModelManifest{
				ModelFile: map[string]ModelFileInfo{
					"Q4_K_M": {Size: 100, Downloaded: true},
					"Q8_0":   {Size: 200, Downloaded: true},
				},
				MMProjFile: ModelFileInfo{Size: 1000, Downloaded: true},
				ExtraFiles: []ModelFileInfo{
					{Size: 10, Downloaded: true},
					{Size: 20, Downloaded: true},
				},
			},
			want: 1330,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.GetSize(); got != tt.want {
				t.Fatalf("GetSize() = %d, want %d", got, tt.want)
			}
		})
	}
}
