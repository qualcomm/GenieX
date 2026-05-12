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

package main

import (
	"testing"
)

func TestCompareVersion(t *testing.T) {
	tests := []struct {
		name     string
		v1       string
		v2       string
		expected int
		wantErr  bool
	}{
		// Equal versions
		{
			name:     "equal versions with v prefix",
			v1:       "v1.0.0",
			v2:       "v1.0.0",
			expected: 0,
			wantErr:  false,
		},
		{
			name:     "equal versions without v prefix",
			v1:       "1.0.0",
			v2:       "1.0.0",
			expected: 0,
			wantErr:  false,
		},
		{
			name:     "equal versions mixed prefix",
			v1:       "v2.5.10",
			v2:       "2.5.10",
			expected: 0,
			wantErr:  false,
		},
		{
			name:     "equal versions with large numbers",
			v1:       "v10.20.30",
			v2:       "v10.20.30",
			expected: 0,
			wantErr:  false,
		},

		// v1 < v2 cases
		{
			name:     "major version less",
			v1:       "v1.0.0",
			v2:       "v2.0.0",
			expected: -1,
			wantErr:  false,
		},
		{
			name:     "minor version less",
			v1:       "v1.0.0",
			v2:       "v1.1.0",
			expected: -1,
			wantErr:  false,
		},
		{
			name:     "patch version less",
			v1:       "v1.0.0",
			v2:       "v1.0.1",
			expected: -1,
			wantErr:  false,
		},
		{
			name:     "large numbers less",
			v1:       "v10.20.30",
			v2:       "v10.20.31",
			expected: -1,
			wantErr:  false,
		},
		{
			name:     "major version large numbers less",
			v1:       "v99.99.99",
			v2:       "v100.0.0",
			expected: -1,
			wantErr:  false,
		},

		// v1 > v2 cases
		{
			name:     "major version greater",
			v1:       "v2.0.0",
			v2:       "v1.0.0",
			expected: 1,
			wantErr:  false,
		},
		{
			name:     "minor version greater",
			v1:       "v1.1.0",
			v2:       "v1.0.0",
			expected: 1,
			wantErr:  false,
		},
		{
			name:     "patch version greater",
			v1:       "v1.0.1",
			v2:       "v1.0.0",
			expected: 1,
			wantErr:  false,
		},
		{
			name:     "large numbers greater",
			v1:       "v10.20.31",
			v2:       "v10.20.30",
			expected: 1,
			wantErr:  false,
		},
		{
			name:     "complex comparison",
			v1:       "v1.2.3",
			v2:       "v1.2.2",
			expected: 1,
			wantErr:  false,
		},
		{
			name:     "complex comparison reverse",
			v1:       "v1.2.2",
			v2:       "v1.2.3",
			expected: -1,
			wantErr:  false,
		},

		// Edge cases
		{
			name:     "zero versions",
			v1:       "v0.0.0",
			v2:       "v0.0.0",
			expected: 0,
			wantErr:  false,
		},
		{
			name:     "zero to one",
			v1:       "v0.0.0",
			v2:       "v0.0.1",
			expected: -1,
			wantErr:  false,
		},
		{
			name:     "very large numbers",
			v1:       "v999.999.999",
			v2:       "v999.999.998",
			expected: 1,
			wantErr:  false,
		},

		// Pre-release versions (should compare base version only)
		{
			name:     "rc version equals stable",
			v1:       "v0.2.68-rc2",
			v2:       "v0.2.68",
			expected: 0,
			wantErr:  false,
		},
		{
			name:     "rc version greater than older stable",
			v1:       "v0.2.68-rc2",
			v2:       "v0.2.67",
			expected: 1,
			wantErr:  false,
		},
		{
			name:     "rc version less than newer stable",
			v1:       "v0.2.68-rc2",
			v2:       "v0.2.69",
			expected: -1,
			wantErr:  false,
		},
		{
			name:     "beta version comparison",
			v1:       "v1.0.0-beta",
			v2:       "v1.0.0",
			expected: 0,
			wantErr:  false,
		},
		{
			name:     "alpha version comparison",
			v1:       "v2.5.0-alpha.1",
			v2:       "v2.4.9",
			expected: 1,
			wantErr:  false,
		},
		{
			name:     "build metadata with plus",
			v1:       "v1.0.0+build123",
			v2:       "v1.0.0",
			expected: 0,
			wantErr:  false,
		},
		{
			name:     "complex prerelease version",
			v1:       "v3.2.1-rc1+build456",
			v2:       "v3.2.1",
			expected: 0,
			wantErr:  false,
		},
		{
			name:     "two rc versions",
			v1:       "v0.2.68-rc1",
			v2:       "v0.2.68-rc2",
			expected: 0,
			wantErr:  false,
		},

		// Invalid format cases
		{
			name:     "invalid format - missing parts",
			v1:       "v1.0",
			v2:       "v1.0.0",
			expected: 0,
			wantErr:  true,
		},
		{
			name:     "invalid format - too many parts",
			v1:       "v1.0.0.0",
			v2:       "v1.0.0",
			expected: 0,
			wantErr:  true,
		},
		{
			name:     "invalid format - non-numeric",
			v1:       "v1.0.a",
			v2:       "v1.0.0",
			expected: 0,
			wantErr:  true,
		},
		{
			name:     "invalid format - empty string",
			v1:       "",
			v2:       "v1.0.0",
			expected: 0,
			wantErr:  true,
		},
		{
			name:     "invalid format - both invalid",
			v1:       "invalid",
			v2:       "also-invalid",
			expected: 0,
			wantErr:  true,
		},
		{
			name:     "invalid format - negative numbers",
			v1:       "v-1.0.0",
			v2:       "v1.0.0",
			expected: 0,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := compareVersion(tt.v1, tt.v2)
			if (err != nil) != tt.wantErr {
				t.Errorf("compareVersion() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && result != tt.expected {
				t.Errorf("compareVersion() = %v, want %v", result, tt.expected)
			}
		})
	}
}
