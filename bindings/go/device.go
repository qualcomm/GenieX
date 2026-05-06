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

// ResolveDevice calls into the SDK's `geniex_resolve_device` to map a
// (pluginID, modelName, mode) triple onto the concrete (device_id,
// n_gpu_layers) pair the plugins expect. See the C API doc for alias
// semantics — the SDK is the single source of truth.
//
// `modelName` may be empty if the caller doesn't know it; it's only
// consulted for model-specific default overrides (e.g. llama_cpp
// gpt-oss models default to `npu` instead of `hybrid`).
//
// A non-nil `err` means the mode was a non-empty unknown alias; the
// SDK returned GENIEX_ERROR_COMMON_INVALID_DEVICE. `warning` is
// non-empty when the alias was coerced (e.g. qairt ↦ NPU regardless
// of user input).
func ResolveDevice(pluginID, modelName, mode string, nglDefault int32) (deviceID string, ngl int32, warning string, err error) {
	cPlugin := C.CString(pluginID)
	defer C.free(unsafe.Pointer(cPlugin))

	var cModel *C.char
	if modelName != "" {
		cModel = C.CString(modelName)
		defer C.free(unsafe.Pointer(cModel))
	}
	var cMode *C.char
	if mode != "" {
		cMode = C.CString(mode)
		defer C.free(unsafe.Pointer(cMode))
	}

	input := C.geniex_ResolveDeviceInput{
		plugin_id:   cPlugin,
		model_name:  cModel,
		mode:        cMode,
		ngl_default: C.int32_t(nglDefault),
	}
	var output C.geniex_ResolveDeviceOutput
	rc := C.geniex_resolve_device(&input, &output)
	if rc != C.GENIEX_SUCCESS {
		if rc == C.GENIEX_ERROR_COMMON_INVALID_DEVICE {
			return "", nglDefault, "", fmt.Errorf("invalid device %q, must be one of: cpu, gpu, npu, hybrid", mode)
		}
		return "", nglDefault, "", SDKError(rc)
	}

	if output.device_id != nil {
		deviceID = C.GoString(output.device_id)
		C.geniex_free(unsafe.Pointer(output.device_id))
	}
	if output.warning != nil {
		warning = C.GoString(output.warning)
		C.geniex_free(unsafe.Pointer(output.warning))
	}
	ngl = int32(output.ngl)
	return
}
