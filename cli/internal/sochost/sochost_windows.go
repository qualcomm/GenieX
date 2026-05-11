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

//go:build windows

package sochost

import (
	"golang.org/x/sys/windows/registry"
)

// DetectChipsetAlias reads the CPU brand string from the registry and
// maps it to an AI Hub chipset alias. Returns ("", false) on any failure
// so callers can fall back to interactive selection.
func DetectChipsetAlias() (string, bool) {
	brand, ok := readCPUBrand()
	if !ok {
		return "", false
	}
	return CPUNameToChipsetAlias(brand)
}

func readCPUBrand() (string, bool) {
	k, err := registry.OpenKey(
		registry.LOCAL_MACHINE,
		`HARDWARE\DESCRIPTION\System\CentralProcessor\0`,
		registry.QUERY_VALUE,
	)
	if err != nil {
		return "", false
	}
	defer k.Close()

	v, _, err := k.GetStringValue("ProcessorNameString")
	if err != nil {
		return "", false
	}
	return v, v != ""
}
