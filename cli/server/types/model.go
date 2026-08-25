// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package types

// ModelParam holds the model-load knobs the keep-alive cache keys instances on.
// NCtx / NGpuLayers are llama_cpp-only; DeviceID is the compute unit resolved by
// the SDK (empty = the SDK's own default).
type ModelParam struct {
	NCtx       int32
	NGpuLayers int32
	DeviceID   string
	Spec       SpecParam
}

// SpecParam bundles the speculative-decoding knobs sourced from a request; all
// zero-values mean "spec disabled". Only llama_cpp consumes these fields.
type SpecParam struct {
	Type       string
	DraftModel string
	NMax       int32
	NMin       int32
	PMin       float32
}
