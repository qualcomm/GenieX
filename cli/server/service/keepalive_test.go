// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package service

import (
	"testing"

	"github.com/spf13/viper"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// TestResolveModelParam_LlamaCppFallsBackToServerDefault verifies that a request
// omitting nctx / ngl (0) picks up the server-wide defaults for llama_cpp. The
// compute alias is "gpu" so ngl is passed through verbatim — hybrid/npu would
// force ngl=999 in the SDK regardless of the default.
func TestResolveModelParam_LlamaCppFallsBackToServerDefault(t *testing.T) {
	viper.Reset()
	viper.Set("nctx", int32(8192))
	viper.Set("ngl", int32(42))
	t.Cleanup(viper.Reset)

	got, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 0, 0, "gpu")
	if err != nil {
		t.Fatalf("ResolveModelParam: %v", err)
	}
	if got.NCtx != 8192 {
		t.Errorf("NCtx = %d, want 8192 (server default)", got.NCtx)
	}
	if got.NGpuLayers != 42 {
		t.Errorf("NGpuLayers = %d, want 42 (server default via gpu)", got.NGpuLayers)
	}
}

// TestResolveModelParam_RequestOverridesServerDefault verifies that a non-zero
// request nctx / ngl wins over the server default.
func TestResolveModelParam_RequestOverridesServerDefault(t *testing.T) {
	viper.Reset()
	viper.Set("nctx", int32(8192))
	viper.Set("ngl", int32(42))
	t.Cleanup(viper.Reset)

	got, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 2048, 10, "gpu")
	if err != nil {
		t.Fatalf("ResolveModelParam: %v", err)
	}
	if got.NCtx != 2048 {
		t.Errorf("NCtx = %d, want 2048 (request override)", got.NCtx)
	}
	if got.NGpuLayers != 10 {
		t.Errorf("NGpuLayers = %d, want 10 (request override)", got.NGpuLayers)
	}
}

// TestResolveModelParam_ComputeDefaultResolvesDevice verifies that an omitted
// compute falls back to the server default and gets resolved to a concrete
// device id (llama_cpp "npu" pins HTP0 and forces ngl=999).
func TestResolveModelParam_ComputeDefaultResolvesDevice(t *testing.T) {
	viper.Reset()
	viper.Set("compute", "npu")
	t.Cleanup(viper.Reset)

	got, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 0, 0, "")
	if err != nil {
		t.Fatalf("ResolveModelParam: %v", err)
	}
	if got.DeviceID != "HTP0" {
		t.Errorf("DeviceID = %q, want HTP0 (server compute default)", got.DeviceID)
	}
	if got.NGpuLayers != 999 {
		t.Errorf("NGpuLayers = %d, want 999 (npu forces full offload)", got.NGpuLayers)
	}
}

// TestResolveModelParam_NonLlamaCppKeepsZeroNCtx verifies that for non-llama_cpp
// runtimes NCtx stays 0 (server nctx/ngl defaults must not trip the plugin's
// param-guard).
func TestResolveModelParam_NonLlamaCppKeepsZeroNCtx(t *testing.T) {
	viper.Reset()
	viper.Set("nctx", int32(8192))
	viper.Set("ngl", int32(42))
	t.Cleanup(viper.Reset)

	got, err := ResolveModelParam(geniex_sdk.RuntimeQairt, "some-model", 0, 0, "")
	if err != nil {
		t.Fatalf("ResolveModelParam: %v", err)
	}
	if got.NCtx != 0 {
		t.Errorf("NCtx = %d, want 0 for non-llama_cpp", got.NCtx)
	}
}
