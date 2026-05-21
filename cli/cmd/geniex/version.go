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
	"io"
	"strings"

	"github.com/spf13/cobra"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
)

var Version string

func printVersion(w io.Writer) {
	fmt.Fprintln(w, "GenieX CLI Version:      "+Version)
	fmt.Fprintln(w, "QAIRT Plugin Version:    "+geniex_sdk.QairtVersion())
	fmt.Fprintln(w, "LlamaCPP Plugin Hash:    "+geniex_sdk.LlamaCppVersion())
}

func versionTemplate() string {
	var b strings.Builder
	printVersion(&b)
	return b.String()
}

func version() *cobra.Command {
	versionCmd := &cobra.Command{
		GroupID: "management",
		Use:     "version",
		Short:   "show geniex version",
	}

	versionCmd.Run = func(cmd *cobra.Command, args []string) {
		printVersion(cmd.OutOrStdout())
	}

	return versionCmd
}
