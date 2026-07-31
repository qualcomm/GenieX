// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package service

import (
	"bytes"
	"strings"
	"testing"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// ResolveModelParam receives already-resolved knobs (the handler prefills unset
// request fields from the server defaults), so these tests pass the final
// values directly.

// TestResolveModelParam_PassesLlamaCppValuesThrough verifies that nctx / ngl are
// forwarded verbatim for llama_cpp and the compute alias resolves to a device.
func TestResolveModelParam_PassesLlamaCppValuesThrough(t *testing.T) {
	got, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 2048, 10, "gpu", SpecParam{})
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
	got, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 4096, -1, "npu", SpecParam{})
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
	got, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 4096, 0, "cpu", SpecParam{})
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
	got, err := ResolveModelParam(geniex_sdk.RuntimeQairt, "some-model", 8192, 42, "", SpecParam{})
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

// TestResolveModelParam_QairtGpuEmitsCoercionWarning verifies that when the
// qairt plugin coerces a non-NPU compute unit to NPU, the SDK warning is
// printed to the configured sink so serve users can see the redirection.
func TestResolveModelParam_QairtGpuEmitsCoercionWarning(t *testing.T) {
	var buf bytes.Buffer
	restore := SetComputeWarningSinkForTest(&buf)
	defer restore()

	if _, err := ResolveModelParam(geniex_sdk.RuntimeQairt, "some-model", 0, 0, "gpu", SpecParam{}); err != nil {
		t.Fatalf("ResolveModelParam: %v", err)
	}
	got := buf.String()
	if !strings.Contains(got, "qairt plugin only supports NPU") {
		t.Errorf("sink missing coercion warning; got %q", got)
	}
}

// TestResolveModelParam_QairtWarningDedupPerRuntimeCompute verifies that repeat
// requests with the same (runtime, compute) print the warning only once so
// long-running serve processes are not spammed.
func TestResolveModelParam_QairtWarningDedupPerRuntimeCompute(t *testing.T) {
	var buf bytes.Buffer
	restore := SetComputeWarningSinkForTest(&buf)
	defer restore()

	for i := 0; i < 3; i++ {
		if _, err := ResolveModelParam(geniex_sdk.RuntimeQairt, "some-model", 0, 0, "gpu", SpecParam{}); err != nil {
			t.Fatalf("ResolveModelParam[%d]: %v", i, err)
		}
	}
	if n := strings.Count(buf.String(), "qairt plugin only supports NPU"); n != 1 {
		t.Errorf("warning printed %d times, want 1 (dedup); output=%q", n, buf.String())
	}
}

// TestResolveModelParam_LlamaCppNoCoercionWarning verifies llama_cpp with a
// resolvable compute unit does not print a coercion warning.
func TestResolveModelParam_LlamaCppNoCoercionWarning(t *testing.T) {
	var buf bytes.Buffer
	restore := SetComputeWarningSinkForTest(&buf)
	defer restore()

	if _, err := ResolveModelParam(geniex_sdk.RuntimeLlamaCpp, "some-model", 2048, 10, "cpu", SpecParam{}); err != nil {
		t.Fatalf("ResolveModelParam: %v", err)
	}
	if buf.Len() != 0 {
		t.Errorf("expected no output on non-coerced path; got %q", buf.String())
	}
}
