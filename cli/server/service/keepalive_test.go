// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package service

import (
	"testing"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/server/middleware"
	"github.com/qualcomm/GenieX/cli/server/types"
	"github.com/spf13/viper"
)

// ResolveModelParam receives already-resolved knobs (the handler prefills unset
// request fields from the server defaults), so these tests pass the final
// values directly.

// TestResolveModelParam_PassesLlamaCppValuesThrough verifies that nctx / ngl are
// forwarded verbatim for llama_cpp and the compute alias resolves to a device.
func TestResolveModelParam_PassesLlamaCppValuesThrough(t *testing.T) {
	got, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 2048, 10, "gpu", "", types.SpecParam{})
	if err != nil {
		t.Fatalf("ResolveModelParam: %v", err)
	}
	if got.NCtx != 2048 {
		t.Errorf("NCtx = %d, want 2048", got.NCtx)
	}
	if got.NGpuLayers != 10 {
		t.Errorf("NGpuLayers = %d, want 10", got.NGpuLayers)
	}
}

// TestResolveModelParam_NpuAliasResolvesDevice verifies the npu alias pins HTP0
// and passes ngl through (-1 = all layers).
func TestResolveModelParam_NpuAliasResolvesDevice(t *testing.T) {
	got, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 4096, -1, "npu", "", types.SpecParam{})
	if err != nil {
		t.Fatalf("ResolveModelParam: %v", err)
	}
	if got.DeviceID != "HTP0" {
		t.Errorf("DeviceID = %q, want HTP0", got.DeviceID)
	}
	if got.NGpuLayers != -1 {
		t.Errorf("NGpuLayers = %d, want -1 (all layers)", got.NGpuLayers)
	}
}

// TestResolveModelParam_CpuAliasZeroesGpuLayers verifies ngl 0 (pure CPU) is a
// valid value that survives resolution.
func TestResolveModelParam_CpuAliasZeroesGpuLayers(t *testing.T) {
	got, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 4096, 0, "cpu", "", types.SpecParam{})
	if err != nil {
		t.Fatalf("ResolveModelParam: %v", err)
	}
	if got.NGpuLayers != 0 {
		t.Errorf("NGpuLayers = %d, want 0 (pure CPU)", got.NGpuLayers)
	}
}

// TestResolveModelParam_NonLlamaCppZeroesNCtx verifies that for non-llama_cpp
// runtimes NCtx is zeroed so the plugin's param-guard is not tripped, even when
// the caller passes a non-zero value.
func TestResolveModelParam_NonLlamaCppZeroesNCtx(t *testing.T) {
	got, err := ResolveModelParam(geniex_sdk.RuntimeQairt, "some-model", 8192, 42, "", "", types.SpecParam{})
	if err != nil {
		t.Fatalf("ResolveModelParam: %v", err)
	}
	if got.NCtx != 0 {
		t.Errorf("NCtx = %d, want 0 for non-llama_cpp", got.NCtx)
	}
	if got.NGpuLayers != 0 {
		t.Errorf("NGpuLayers = %d, want 0 (SDK zeroes ngl for qairt)", got.NGpuLayers)
	}
}

// Regression test for #1322: model destruction shares the request GIL, so
// the cleanup goroutine can never destroy a model a handler is still using.

type fakeModel struct{ destroyed int }

func (f *fakeModel) Destroy() error { f.destroyed++; return nil }

func TestSweepNeverDestroysMidRequest(t *testing.T) {
	viper.Set("keepalive", -1) // any idle time counts as expired
	defer viper.Set("keepalive", nil)

	f := &fakeModel{}
	keepAlive.name = "m"
	keepAlive.model = f
	defer keepAlive.destroy()

	// A request in flight holds the GIL; the sweep must be a no-op.
	middleware.GILock.Lock()
	keepAlive.sweep()
	middleware.GILock.Unlock()
	if f.destroyed != 0 {
		t.Fatal("sweep destroyed the model while a request was in flight")
	}

	// Idle past the timeout with no request in flight: the model is freed.
	keepAlive.sweep()
	if f.destroyed != 1 {
		t.Fatal("sweep kept an idle model past the timeout")
	}
	if keepAlive.model != nil {
		t.Fatal("destroyed model was left in the cache")
	}
}
