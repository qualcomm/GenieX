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

import (
	"fmt"
	"strings"
)

// Friendly device aliases that downstream callers (CLI, Python bindings,
// Android bindings) accept on their `--device` / `device_map` option. They
// translate to a plugin-specific `(device_id, n_gpu_layers)` pair via
// [ResolveDevice].
const (
	// DeviceCPU forces pure-CPU inference: no accelerator device is
	// selected and `n_gpu_layers` is clamped to 0.
	DeviceCPU = "cpu"
	// DeviceGPU pins inference to the first GPU backend (for llama_cpp
	// on Snapdragon this is Adreno via OpenCL).
	DeviceGPU = "gpu"
	// DeviceNPU pins inference to a single NPU session. For llama_cpp
	// this is `HTP0`, and the plugin falls into its single-device path
	// (see `sdk/plugins/llama_cpp/src/llm.cpp`).
	DeviceNPU = "npu"
	// DeviceHybrid leaves `device_id` empty with `n_gpu_layers=999` so
	// llama.cpp's per-tensor backend scheduler places each op on HTP or
	// CPU as appropriate. This is the fast path for LLMs on Snapdragon.
	DeviceHybrid = "hybrid"
)

// Plugin IDs. Kept here (rather than in every caller) so CLI / pybind /
// android agree on the strings the SDK plugin registry uses.
const (
	PluginLlamaCpp = "llama_cpp"
	PluginQairt    = "qairt"
)

// Concrete device_id strings the SDK plugins expect.
const (
	deviceIDHTP0      = "HTP0"
	deviceIDGPUOpenCL = "GPUOpenCL"
	deviceIDQairtNPU  = "NPU"
)

// ResolveDevice maps a user-facing (pluginID, mode) pair to the
// `(device_id, n_gpu_layers)` values that go into [LlmCreateInput] /
// [VlmCreateInput].
//
//   - `mode` is one of the Device* constants above, or an empty string
//     / "auto" for "pick the plugin's default" (llama_cpp → hybrid,
//     qairt → npu).
//   - `nglDefault` is the caller's default `n_gpu_layers` and is returned
//     unchanged unless the mode forces a value (cpu → 0, hybrid → 999).
//
// The returned `warning` is non-empty when the mode had to be coerced
// (e.g. qairt only runs on its NPU device, so cpu/gpu/hybrid fall back
// to NPU). Callers should surface it and continue — there is no early
// exit for coerced modes, by design.
//
// An `err` is returned only when `mode` is not one of the known
// aliases. CLI / binding callers may still choose to treat that as a
// warning and pass through, but the default expectation is a hard
// validation failure.
func ResolveDevice(pluginID, mode string, nglDefault int32) (deviceID string, ngl int32, warning string, err error) {
	mode = strings.ToLower(strings.TrimSpace(mode))
	ngl = nglDefault

	if mode != "" && mode != "auto" && !isKnownMode(mode) {
		return "", ngl, "", fmt.Errorf("invalid device %q, must be one of: cpu, gpu, npu, hybrid", mode)
	}

	// Empty / "auto" → plugin-specific default.
	if mode == "" || mode == "auto" {
		if pluginID == PluginQairt {
			mode = DeviceNPU
		} else {
			mode = DeviceHybrid
		}
	}

	// QAIRT only exposes a single NPU device. Coerce other modes with
	// a warning so the caller can surface it.
	if pluginID == PluginQairt {
		if mode != DeviceNPU {
			warning = fmt.Sprintf("qairt plugin only supports NPU inference; ignoring --device=%s and running on NPU", mode)
		}
		return deviceIDQairtNPU, ngl, warning, nil
	}

	switch mode {
	case DeviceCPU:
		return "", 0, "", nil
	case DeviceGPU:
		return deviceIDGPUOpenCL, ngl, "", nil
	case DeviceNPU:
		return deviceIDHTP0, ngl, "", nil
	case DeviceHybrid:
		return "", 999, "", nil
	}
	// Unreachable: isKnownMode above guards this switch.
	return "", ngl, "", nil
}

func isKnownMode(s string) bool {
	switch s {
	case DeviceCPU, DeviceGPU, DeviceNPU, DeviceHybrid:
		return true
	}
	return false
}
