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
	"runtime"
	"sync"
)

// key: shortcut
// value:
//   - list of (os, arch, actual model_name)
//   - os and arch can be empty string to match any
//   - the later value will override the previous one if both match
var modelMappingSrc = map[string][][3]string{}

var modelMappingInit sync.Once

var modelMapping = make(map[string]string)

func GetModelMapping(shortcut string) (string, bool) {
	modelMappingInit.Do(func() {
		for model, aliases := range modelMappingSrc {
			for _, entry := range aliases {
				if (entry[0] == "" || entry[0] == runtime.GOOS) && (entry[1] == "" || entry[1] == runtime.GOARCH) {
					modelMapping[model] = entry[2]
				}
			}
		}
	})

	actualPath, exists := modelMapping[shortcut]
	return actualPath, exists
}
