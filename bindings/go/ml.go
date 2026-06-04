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
#cgo CFLAGS: -I${SRCDIR}/../../sdk/pkg-geniex/include

#include <stdlib.h>
#include "geniex.h"

#if defined(_WIN32)
__declspec(dllexport) void go_log_wrap(geniex_LogLevel level, char *msg);
#else
extern void go_log_wrap(geniex_LogLevel level, char *msg);
#endif
*/
import "C"

import (
	"errors"
	"fmt"
	"log/slog"
	"unsafe"
)

// LCOV_EXCL_START

type SDKError int32

func (s SDKError) Error() string {
	return fmt.Sprintf("SDKError(%s)",
		C.GoString(C.geniex_get_error_message(C.geniex_ErrorCode(s))))
}

func SDKErrorCode(err error) int32 {
	var sdkErr SDKError
	if errors.As(err, &sdkErr) {
		return int32(sdkErr)
	}
	return -1
}

var (
	ErrCommonNotSupport             = SDKError(C.GENIEX_ERROR_COMMON_NOT_SUPPORTED)
	ErrCommonParamNotSupported      = SDKError(C.GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED)
	ErrCommonModelLoad              = SDKError(C.GENIEX_ERROR_COMMON_MODEL_LOAD)
	ErrCommonPluginLoad             = SDKError(C.GENIEX_ERROR_COMMON_PLUGIN_LOAD)
	ErrCommonPluginInvalid          = SDKError(C.GENIEX_ERROR_COMMON_PLUGIN_INVALID)
	ErrLlmTokenizationContextLength = SDKError(C.GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH)
)

// Init must be called before any other SDK function.
func Init() {
	C.geniex_init()
}

func DeInit() {
	C.geniex_deinit()
}

func Version() string {
	return C.GoString(C.geniex_version())
}

func SetLog(enable bool) {
	if enable {
		C.geniex_set_log((C.geniex_log_callback)(C.go_log_wrap))
	} else {
		C.geniex_set_log(nil)
	}
}

// GetPluginVersion returns the version the plugin reports for itself (QAIRT
// runtime version, llama.cpp build commit, …). Empty string if not registered.
func GetPluginVersion(pluginID string) string {
	cID := C.CString(pluginID)
	defer C.free(unsafe.Pointer(cID))
	return C.GoString(C.geniex_get_plugin_version(cID))
}

type GetPluginListOutput struct {
	PluginIDs []string
}

func newGetPluginListOutputFromCPtr(c *C.geniex_GetPluginListOutput) GetPluginListOutput {
	if c == nil {
		return GetPluginListOutput{}
	}
	return GetPluginListOutput{
		PluginIDs: cCharArrayToSlice((**C.char)(unsafe.Pointer(c.plugin_ids)), c.plugin_count),
	}
}

func freeGetPluginListOutput(c *C.geniex_GetPluginListOutput) {
	if c == nil {
		return
	}
	mlFreeCCharArray((**C.char)(unsafe.Pointer(c.plugin_ids)), c.plugin_count)
}

func GetPluginList() (*GetPluginListOutput, error) {
	var cOutput C.geniex_GetPluginListOutput
	res := C.geniex_get_plugin_list(&cOutput)
	if res < 0 {
		return nil, SDKError(res)
	}
	defer freeGetPluginListOutput(&cOutput)

	output := newGetPluginListOutputFromCPtr(&cOutput)
	return &output, nil
}

type GetDeviceListInput struct {
	PluginID string
}

func (gdli GetDeviceListInput) toCPtr() *C.geniex_GetDeviceListInput {
	cPtr := (*C.geniex_GetDeviceListInput)(cMalloc(C.sizeof_geniex_GetDeviceListInput))
	*cPtr = C.geniex_GetDeviceListInput{plugin_id: cStringIfSet(gdli.PluginID)}
	return cPtr
}

func freeGetDeviceListInput(cPtr *C.geniex_GetDeviceListInput) {
	if cPtr == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(cPtr.plugin_id))
	C.free(unsafe.Pointer(cPtr))
}

type Device struct {
	ID   string
	Name string
}

type GetDeviceListOutput struct {
	Devices []Device
}

func newGetDeviceListOutputFromCPtr(c *C.geniex_GetDeviceListOutput) GetDeviceListOutput {
	if c == nil {
		return GetDeviceListOutput{}
	}
	count := int(c.device_count)
	devices := make([]Device, count)
	if count > 0 {
		ids := unsafe.Slice(c.device_ids, count)
		names := unsafe.Slice(c.device_names, count)
		for i := range devices {
			devices[i] = Device{
				ID:   C.GoString(ids[i]),
				Name: C.GoString(names[i]),
			}
		}
	}
	return GetDeviceListOutput{Devices: devices}
}

func freeGetDeviceListOutput(c *C.geniex_GetDeviceListOutput) {
	if c == nil {
		return
	}
	if c.device_ids != nil {
		free(unsafe.Pointer(c.device_ids))
	}
	if c.device_names != nil {
		free(unsafe.Pointer(c.device_names))
	}
}

func GetDeviceList(input GetDeviceListInput) (*GetDeviceListOutput, error) {
	cInput := input.toCPtr()
	defer freeGetDeviceListInput(cInput)

	var cOutput C.geniex_GetDeviceListOutput
	res := C.geniex_get_device_list(cInput, &cOutput)
	if res < 0 {
		return nil, SDKError(res)
	}
	defer freeGetDeviceListOutput(&cOutput)

	output := newGetDeviceListOutputFromCPtr(&cOutput)
	return &output, nil
}

//export go_log_wrap
func go_log_wrap(level C.geniex_LogLevel, msg *C.char) {
	msgStr := "[ML] " + C.GoString(msg)
	switch level {
	case C.GENIEX_LOG_LEVEL_INFO:
		slog.Info(msgStr)
	case C.GENIEX_LOG_LEVEL_WARN:
		slog.Warn(msgStr)
	case C.GENIEX_LOG_LEVEL_ERROR:
		slog.Error(msgStr)
	default:
		slog.Debug(msgStr)
	}
}

// LCOV_EXCL_STOP
