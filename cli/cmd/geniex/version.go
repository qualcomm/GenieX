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
	"fmt"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
)

var Version string

func version() *cobra.Command {
	versionCmd := &cobra.Command{
		GroupID: "management",
		Use:     "version",
		Short:   "show geniex version",
	}

	versionCmd.Run = func(cmd *cobra.Command, args []string) {
		fmt.Println("QAIRT Version:      " + geniex_sdk.QairtVersion())
		fmt.Println("GenieX CLI Version: " + Version)
	}

	return versionCmd
}

// compareVersion compares two version strings in format v0.0.0
// Returns: -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
func compareVersion(v1, v2 string) (int, error) {
	parseVersion := func(v string) ([3]int, error) {
		origV := v
		v = strings.TrimPrefix(v, "v")
		// Strip pre-release suffixes like -rc2, -beta, -alpha, etc.
		if idx := strings.IndexAny(v, "-+"); idx != -1 {
			v = v[:idx]
		}
		parts := strings.Split(v, ".")
		if len(parts) != 3 {
			return [3]int{}, fmt.Errorf("invalid format: %s", origV)
		}
		var nums [3]int
		for i, p := range parts {
			n, err := strconv.Atoi(p)
			if err != nil {
				return [3]int{}, fmt.Errorf("invalid format: %s", origV)
			}
			if n < 0 {
				return [3]int{}, fmt.Errorf("invalid format: %s", origV)
			}
			nums[i] = n
		}
		return nums, nil
	}

	n1, err := parseVersion(v1)
	if err != nil {
		return 0, err
	}

	n2, err := parseVersion(v2)
	if err != nil {
		return 0, err
	}

	for i := range 3 {
		if n1[i] < n2[i] {
			return -1, nil
		}
		if n1[i] > n2[i] {
			return 1, nil
		}
	}
	return 0, nil
}
