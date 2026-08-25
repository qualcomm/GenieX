// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package geniex_sdk

/*
#include <stdlib.h>
#include "geniex.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// LCOV_EXCL_START

// Friendly compute-unit aliases that downstream callers pass on their
// `--compute` / `device_map` option. The SDK (`sdk/src/device.cpp`)
// owns the alias table; this file is just the Go-side thin wrapper.
const (
	ComputeUnitCPU    = "cpu"
	ComputeUnitGPU    = "gpu"
	ComputeUnitNPU    = "npu"
	ComputeUnitHybrid = "hybrid"
)

// Runtime IDs. Kept here so CLI / pybind / android agree on the strings
// the SDK runtime registry uses.
const (
	RuntimeLlamaCpp = "llama_cpp"
	RuntimeQairt    = "qairt"
)

type ResolveDeviceInput struct {
	RuntimeID   string
	ModelName   string
	ComputeUnit string
	NglDefault  int32
}

func (rdi ResolveDeviceInput) toCPtr() *C.geniex_ResolveDeviceInput {
	cPtr := (*C.geniex_ResolveDeviceInput)(cMalloc(C.sizeof_geniex_ResolveDeviceInput))
	*cPtr = C.geniex_ResolveDeviceInput{
		plugin_id:   cStringIfSet(rdi.RuntimeID),
		model_name:  cStringIfSet(rdi.ModelName),
		mode:        cStringIfSet(rdi.ComputeUnit),
		ngl_default: C.int32_t(rdi.NglDefault),
	}
	return cPtr
}

func freeResolveDeviceInput(cPtr *C.geniex_ResolveDeviceInput) {
	if cPtr == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(cPtr.plugin_id))
	cFreeIfSet(unsafe.Pointer(cPtr.model_name))
	cFreeIfSet(unsafe.Pointer(cPtr.mode))
	C.free(unsafe.Pointer(cPtr))
}

type ResolveDeviceOutput struct {
	DeviceID string
	Ngl      int32
	Warning  string
}

func newResolveDeviceOutputFromCPtr(c *C.geniex_ResolveDeviceOutput) ResolveDeviceOutput {
	if c == nil {
		return ResolveDeviceOutput{}
	}
	return ResolveDeviceOutput{
		DeviceID: C.GoString(c.device_id),
		Ngl:      int32(c.ngl),
		Warning:  C.GoString(c.warning),
	}
}

func freeResolveDeviceOutput(c *C.geniex_ResolveDeviceOutput) {
	if c == nil {
		return
	}
	if c.device_id != nil {
		free(unsafe.Pointer(c.device_id))
	}
	if c.warning != nil {
		free(unsafe.Pointer(c.warning))
	}
}

// ResolveDevice maps a (RuntimeID, ModelName, ComputeUnit) triple onto the
// concrete (DeviceID, Ngl) pair the runtimes expect. See `geniex_resolve_device`
// in the C API for alias semantics — the SDK is the single source of truth.
//
// ModelName may be empty if the caller doesn't know it; it's currently
// unused by the resolver and reserved for future model-specific defaults.
//
// A non-nil error means ComputeUnit was neither a known alias nor a valid
// device list; the SDK returned GENIEX_ERROR_COMMON_INVALID_DEVICE. Warning is
// non-empty when the value was coerced (e.g. qairt ↦ NPU regardless of user input).
func ResolveDevice(input ResolveDeviceInput) (*ResolveDeviceOutput, error) {
	cInput := input.toCPtr()
	defer freeResolveDeviceInput(cInput)

	var cOutput C.geniex_ResolveDeviceOutput
	res := C.geniex_resolve_device(cInput, &cOutput)
	if res != C.GENIEX_SUCCESS {
		if res == C.GENIEX_ERROR_COMMON_INVALID_DEVICE {
			return nil, fmt.Errorf("invalid compute unit %q, must be one of: cpu, gpu, npu, hybrid, or a device list like HTP0,HTP1", input.ComputeUnit)
		}
		return nil, SDKError(res)
	}
	defer freeResolveDeviceOutput(&cOutput)

	output := newResolveDeviceOutputFromCPtr(&cOutput)
	return &output, nil
}

// LCOV_EXCL_STOP
