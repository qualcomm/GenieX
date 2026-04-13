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

package geniex_bridge

/*
#cgo linux LDFLAGS: -Wl,-rpath,'$ORIGIN' -Wl,-rpath,${SRCDIR}/../build
#cgo darwin LDFLAGS: -Wl,-rpath,${SRCDIR}/../build

#cgo CFLAGS: -I../build
#cgo LDFLAGS: -L../build -lnexa_bridge

#include <stdlib.h>
#include "ml.h"

#if defined(_WIN32)
__declspec(dllexport) void go_log_wrap(ml_LogLevel level, char *msg);
#else
extern void go_log_wrap(ml_LogLevel level, char *msg);
#endif

static void set_token(const char* token) {
#if defined(_WIN32)
	_putenv_s("GENIEX_TOKEN", token);
#else
	setenv("GENIEX_TOKEN", token, 1);
#endif
}
*/
import "C"

import (
	"fmt"
	"log/slog"
	"os"
	"unsafe"
)

var bridgeLogEnabled = false

type SDKError int32

func (s SDKError) Error() string {
	return fmt.Sprintf("SDKError(%s)", C.GoString(C.ml_get_error_message(C.ml_ErrorCode(s))))
}

func SDKErrorCode(err error) int32 {
	if sdkErr, ok := err.(SDKError); ok {
		return int32(sdkErr)
	}
	return -1
}

var (
	ErrCommonNotSupport             = SDKError(C.ML_ERROR_COMMON_NOT_SUPPORTED)
	ErrCommonModelLoad              = SDKError(C.ML_ERROR_COMMON_MODEL_LOAD)
	ErrCommonPluginLoad             = SDKError(C.ML_ERROR_COMMON_PLUGIN_LOAD)
	ErrCommonPluginInvalid          = SDKError(C.ML_ERROR_COMMON_PLUGIN_INVALID)
	ErrLlmTokenizationContextLength = SDKError(C.ML_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH)
)

// Init initializes the GenieX-CLI by calling the underlying C library initialization
// This must be called before using any other SDK functions
func Init() {
	slog.Debug("[ML] Init", "bridgeLogEnabled", bridgeLogEnabled)
	if bridgeLogEnabled {
		C.ml_set_log((C.ml_log_callback)(C.go_log_wrap))
	}
	C.set_token(C.CString(os.Getenv("GENIEX_TOKEN"))) // sync token to C env
	C.ml_init()
}

// DeInit cleans up resources allocated by the GenieX-CLI
// This should be called when the SDK is no longer needed
func DeInit() {
	C.ml_deinit()
}

// Get SDK Version
func Version() string {
	return C.GoString(C.ml_version())
}

type PluginListOutput struct {
	PluginIDs []string
}

func newPluginListOutputFromCPtr(c *C.ml_GetPluginListOutput) PluginListOutput {
	return PluginListOutput{
		PluginIDs: cCharArrayToSlice((**C.char)(unsafe.Pointer(c.plugin_ids)), c.plugin_count),
	}
}

func GetPluginList() (*PluginListOutput, error) {
	var cOutput C.ml_GetPluginListOutput

	res := C.ml_get_plugin_list(&cOutput)
	if res < 0 {
		return nil, SDKError(res)
	}

	output := newPluginListOutputFromCPtr(&cOutput)

	if cOutput.plugin_ids != nil {
		mlFreeCCharArray((**C.char)(unsafe.Pointer(cOutput.plugin_ids)), cOutput.plugin_count)
	}

	return &output, nil
}

type DeviceListInput struct {
	PluginID string
}

func (di DeviceListInput) toCPtr() *C.ml_GetDeviceListInput {
	cPtr := (*C.ml_GetDeviceListInput)(C.malloc(C.sizeof_ml_GetDeviceListInput))
	cPtr.plugin_id = C.CString(di.PluginID)
	return cPtr
}

func freeDeviceListInput(cPtr *C.ml_GetDeviceListInput) {
	if cPtr == nil {
		return
	}
	if cPtr.plugin_id != nil {
		C.free(unsafe.Pointer(cPtr.plugin_id))
	}
	C.free(unsafe.Pointer(cPtr))
}

type Device struct {
	ID   string
	Name string
}

type DeviceListOutput struct {
	Devices []Device
}

func freeDeviceListOutput(c *C.ml_GetDeviceListOutput) {
	if c == nil {
		return
	}
	mlFree(unsafe.Pointer(c.device_ids))
	mlFree(unsafe.Pointer(c.device_names))
}

func newDeviceListOutputFromCPtr(c *C.ml_GetDeviceListOutput) DeviceListOutput {
	devices := make([]Device, c.device_count)

	deviceIDs := unsafe.Slice(c.device_ids, int(c.device_count))
	deviceNames := unsafe.Slice(c.device_names, int(c.device_count))
	for i := range devices {
		devices[i] = Device{
			ID:   C.GoString(deviceIDs[i]),
			Name: C.GoString(deviceNames[i]),
		}
	}

	return DeviceListOutput{
		Devices: devices,
	}
}

func GetDeviceList(input DeviceListInput) (*DeviceListOutput, error) {
	cInput := input.toCPtr()
	defer freeDeviceListInput(cInput)

	var cOutput C.ml_GetDeviceListOutput
	defer freeDeviceListOutput(&cOutput)

	res := C.ml_get_device_list(cInput, &cOutput)
	if res < 0 {
		return nil, SDKError(res)
	}

	output := newDeviceListOutputFromCPtr(&cOutput)

	return &output, nil
}

// go_log_wrap is exported to C and handles log messages from the C library
// It converts C strings to Go strings and prints them to stdout
//
//export go_log_wrap
func go_log_wrap(level C.ml_LogLevel, msg *C.char) {
	msgStr := C.GoString(msg)
	switch level {
	case C.ML_LOG_LEVEL_INFO:
		slog.Info("[ML] " + msgStr)
	case C.ML_LOG_LEVEL_WARN:
		slog.Warn("[ML] " + msgStr)
	case C.ML_LOG_LEVEL_ERROR:
		slog.Error("[ML] " + msgStr)
	default:
		slog.Debug("[ML] " + msgStr)
	}
}

// EnableBridgeLog enables or disables the bridge log
func EnableBridgeLog(enable bool) {
	bridgeLogEnabled = enable
}
