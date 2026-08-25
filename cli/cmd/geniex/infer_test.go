// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"testing"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

func TestModelLoadedLine(t *testing.T) {
	tests := []struct {
		name                   string
		runtimeID, computeUnit string
		ngl, nctx              int32
		want                   string
	}{
		{
			name:      "llama_cpp npu shows alias, ngl/nctx",
			runtimeID: geniex_sdk.RuntimeLlamaCpp, computeUnit: "npu", ngl: 32, nctx: 4096,
			want: "Model loaded: runtime=llama_cpp compute=npu ngl=32 nctx=4096",
		},
		{
			name:      "llama_cpp cpu echoes alias (device_id is empty)",
			runtimeID: geniex_sdk.RuntimeLlamaCpp, computeUnit: "cpu", ngl: 0, nctx: 2048,
			want: "Model loaded: runtime=llama_cpp compute=cpu ngl=0 nctx=2048",
		},
		{
			name:      "llama_cpp hybrid echoes alias (device_id is empty)",
			runtimeID: geniex_sdk.RuntimeLlamaCpp, computeUnit: "hybrid", ngl: -1, nctx: 4096,
			want: "Model loaded: runtime=llama_cpp compute=hybrid ngl=-1 nctx=4096",
		},
		{
			name:      "llama_cpp empty compute defaults to npu",
			runtimeID: geniex_sdk.RuntimeLlamaCpp, computeUnit: "", ngl: -1, nctx: 4096,
			want: "Model loaded: runtime=llama_cpp compute=npu ngl=-1 nctx=4096",
		},
		{
			name:      "qairt fixed to npu, hides ngl/nctx",
			runtimeID: geniex_sdk.RuntimeQairt, computeUnit: "cpu", ngl: 0, nctx: 0,
			want: "Model loaded: runtime=qairt compute=npu",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := modelLoadedLine(tt.runtimeID, tt.computeUnit, tt.ngl, tt.nctx)
			if got != tt.want {
				t.Errorf("modelLoadedLine\n got: %q\nwant: %q", got, tt.want)
			}
		})
	}
}
