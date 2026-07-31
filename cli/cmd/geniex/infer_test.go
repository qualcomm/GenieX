// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package main

import (
	"testing"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

func TestModelLoadedLine(t *testing.T) {
	tests := []struct {
		name              string
		runtimeID, device string
		ngl, nctx         int32
		want              string
	}{
		{
			name:      "llama_cpp shows ngl/nctx",
			runtimeID: geniex_sdk.RuntimeLlamaCpp, device: "HTP0", ngl: 32, nctx: 4096,
			want: "Model loaded: backend=llama_cpp device=HTP0 ngl=32 nctx=4096",
		},
		{
			name:      "llama_cpp with ngl=-1 (all layers) still prints",
			runtimeID: geniex_sdk.RuntimeLlamaCpp, device: "CPU", ngl: -1, nctx: 2048,
			want: "Model loaded: backend=llama_cpp device=CPU ngl=-1 nctx=2048",
		},
		{
			name:      "qairt hides ngl/nctx (not applicable)",
			runtimeID: geniex_sdk.RuntimeQairt, device: "HTP0", ngl: 0, nctx: 0,
			want: "Model loaded: backend=qairt device=HTP0",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := modelLoadedLine(tt.runtimeID, tt.device, tt.ngl, tt.nctx)
			if got != tt.want {
				t.Errorf("modelLoadedLine\n got: %q\nwant: %q", got, tt.want)
			}
		})
	}
}
