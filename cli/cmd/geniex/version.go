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

	"github.com/spf13/cobra"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
)

var Version string

func runVersion() {
	geniex_sdk.Init()
	defer geniex_sdk.DeInit()

	fmt.Println("GenieX CLI Version:      " + Version)
	fmt.Println("QAIRT Plugin Version:    " + geniex_sdk.GetPluginVersion("qairt"))
	fmt.Println("LlamaCPP Plugin Hash:    " + geniex_sdk.GetPluginVersion("llama_cpp"))
}

func version() *cobra.Command {
	return &cobra.Command{
		GroupID: "management",
		Use:     "version",
		Short:   "show geniex version",
		Run:     func(cmd *cobra.Command, args []string) { runVersion() },
	}
}
