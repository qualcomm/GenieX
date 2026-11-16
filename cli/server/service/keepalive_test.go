// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package service

import (
	"fmt"
	"os"
	"path/filepath"
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
	got, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 2048, 10, "gpu", "", "", "", types.SpecParam{})
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
	got, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 4096, -1, "npu", "", "", "", types.SpecParam{})
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
	got, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 4096, 0, "cpu", "", "", "", types.SpecParam{})
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
	got, err := ResolveModelParam(geniex_sdk.RuntimeQairt, "some-model", 8192, 42, "", "", "", "", types.SpecParam{})
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

// TestResolveModelParam_PowerModeResolves verifies a power_mode alias is
// resolved to the geniex_PowerMode the SDK expects.
func TestResolveModelParam_PowerModeResolves(t *testing.T) {
	got, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 4096, -1, "npu", "", "sustained_high_performance", "", types.SpecParam{})
	if err != nil {
		t.Fatalf("ResolveModelParam: %v", err)
	}
	if got.PowerMode != geniex_sdk.PowerModeSustainedHighPerformance {
		t.Errorf("PowerMode = %v, want PowerModeSustainedHighPerformance", got.PowerMode)
	}
}

// TestResolveModelParam_InvalidPowerMode verifies an unknown power_mode alias
// fails fast rather than reaching the plugin.
func TestResolveModelParam_InvalidPowerMode(t *testing.T) {
	_, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 4096, -1, "npu", "", "turbo", "", types.SpecParam{})
	if err == nil {
		t.Fatal("ResolveModelParam: want error for invalid power_mode, got nil")
	}
}

// Regression test for #1322: model destruction shares the request GIL, so
// the cleanup goroutine can never destroy a model a handler is still using.

type fakeModel struct{ destroyed, resets int }

func (f *fakeModel) Destroy() error { f.destroyed++; return nil }
func (f *fakeModel) Reset() error   { f.resets++; return nil }

func TestKeepAliveSessionReset(t *testing.T) {
	root := t.TempDir()
	if err := geniex_sdk.ModelInit(root); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { geniex_sdk.ModelDeinit() })

	for _, tc := range []struct {
		name, runtime, kind string
		reset               bool
	}{
		{"llama-llm", geniex_sdk.RuntimeLlamaCpp, "llm", false},
		{"qairt-llm", geniex_sdk.RuntimeQairt, "llm", true},
		{"llama-vlm", geniex_sdk.RuntimeLlamaCpp, "vlm", true},
		{"qairt-vlm", geniex_sdk.RuntimeQairt, "vlm", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			name := "tests/" + tc.name
			dir := filepath.Join(root, "models", "tests", tc.name)
			if err := os.MkdirAll(dir, 0755); err != nil {
				t.Fatal(err)
			}
			manifest := fmt.Sprintf(`{"Name":%q,"ModelName":"test","ModelType":%q,"PluginId":%q,"ModelFile":{"Q4_0":{"Name":"model.gguf","Downloaded":true,"Size":0}}}`, name, tc.kind, tc.runtime)
			if err := os.WriteFile(filepath.Join(dir, "geniex.json"), []byte(manifest), 0600); err != nil {
				t.Fatal(err)
			}

			f := &fakeModel{}
			keepAlive.name = name
			keepAlive.model = f
			keepAlive.param = types.ModelParam{}
			keepAlive.lastSession = []uint64{1}
			t.Cleanup(keepAlive.destroy)
			get := func(session []uint64) (any, bool, error) {
				if tc.kind == "vlm" {
					return keepAliveGet[geniex_sdk.VLM](name, types.ModelParam{}, session)
				}
				return keepAliveGet[geniex_sdk.LLM](name, types.ModelParam{}, session)
			}
			for i, session := range [][]uint64{{1, 2}, {3}} {
				model, fresh, err := get(session)
				wantReset := tc.reset && i == 1
				if err != nil || model != f || fresh != wantReset || (f.resets > 0) != wantReset {
					t.Fatalf("session %v: model=%v fresh=%v resets=%d err=%v; want reset=%v", session, model, fresh, f.resets, err, wantReset)
				}
			}
		})
	}
}

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

func TestStatusLifecycle(t *testing.T) {
	k := &keepAliveService{}
	k.setStatus(&ModelStatus{ID: "m", Type: "llm", State: StateLoading})
	if k.status.State != StateLoading || !k.status.LoadedAt.IsZero() {
		t.Fatalf("loading: %+v", k.status)
	}
	k.setState(StateActive)
	if k.status.State != StateActive || k.status.LoadedAt.IsZero() {
		t.Fatalf("active: %+v", k.status)
	}
	loadedAt := k.status.LoadedAt
	k.setState(StateIdle)
	k.setState(StateActive)
	if !k.status.LoadedAt.Equal(loadedAt) {
		t.Errorf("LoadedAt changed across activity")
	}
	k.setState(StateIdle)
	if k.status.State != StateIdle || k.status.LastActivity.IsZero() {
		t.Fatalf("idle: %+v", k.status)
	}
	k.setStatus(nil)
	k.setState(StateActive)
	if k.status != nil {
		t.Errorf("setState on empty status must be a no-op")
	}
}
