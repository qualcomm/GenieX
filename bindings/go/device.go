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

// Friendly device aliases that downstream callers pass on their
// `--device` / `device_map` option. The SDK (`sdk/src/device.cpp`)
// owns the alias table; this file is just the Go-side thin wrapper.
const (
	DeviceCPU    = "cpu"
	DeviceGPU    = "gpu"
	DeviceNPU    = "npu"
	DeviceHybrid = "hybrid"
)

// Plugin IDs. Kept here so CLI / pybind / android agree on the strings
// the SDK plugin registry uses.
const (
	PluginLlamaCpp = "llama_cpp"
	PluginQairt    = "qairt"
)

type ResolveDeviceInput struct {
	PluginID   string
	ModelName  string
	Mode       string
	NglDefault int32
}

func (rdi ResolveDeviceInput) toCPtr() *C.geniex_ResolveDeviceInput {
	cPtr := (*C.geniex_ResolveDeviceInput)(cMalloc(C.sizeof_geniex_ResolveDeviceInput))
	*cPtr = C.geniex_ResolveDeviceInput{
		plugin_id:   cStringIfSet(rdi.PluginID),
		model_name:  cStringIfSet(rdi.ModelName),
		mode:        cStringIfSet(rdi.Mode),
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
		mlFree(unsafe.Pointer(c.device_id))
	}
	if c.warning != nil {
		mlFree(unsafe.Pointer(c.warning))
	}
}

// ResolveDevice maps a (PluginID, ModelName, Mode) triple onto the concrete
// (DeviceID, Ngl) pair the plugins expect. See `geniex_resolve_device` in
// the C API for alias semantics — the SDK is the single source of truth.
//
// ModelName may be empty if the caller doesn't know it; it's only consulted
// for model-specific default overrides (e.g. llama_cpp gpt-oss → npu).
//
// A non-nil error means Mode was a non-empty unknown alias; the SDK returned
// GENIEX_ERROR_COMMON_INVALID_DEVICE. Warning is non-empty when the alias
// was coerced (e.g. qairt ↦ NPU regardless of user input).
func ResolveDevice(input ResolveDeviceInput) (*ResolveDeviceOutput, error) {
	cInput := input.toCPtr()
	defer freeResolveDeviceInput(cInput)

	var cOutput C.geniex_ResolveDeviceOutput
	res := C.geniex_resolve_device(cInput, &cOutput)
	if res != C.GENIEX_SUCCESS {
		if res == C.GENIEX_ERROR_COMMON_INVALID_DEVICE {
			return nil, fmt.Errorf("invalid device %q, must be one of: cpu, gpu, npu, hybrid", input.Mode)
		}
		return nil, SDKError(res)
	}
	defer freeResolveDeviceOutput(&cOutput)

	output := newResolveDeviceOutputFromCPtr(&cOutput)
	return &output, nil
}
