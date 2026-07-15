// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

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
	ErrCommonNetwork                = SDKError(C.GENIEX_ERROR_COMMON_NETWORK)
	ErrCommonAuth                   = SDKError(C.GENIEX_ERROR_COMMON_AUTH)
	ErrCommonHubModelNotFound       = SDKError(C.GENIEX_ERROR_COMMON_HUB_MODEL_NOT_FOUND)
	ErrCommonRateLimited            = SDKError(C.GENIEX_ERROR_COMMON_RATE_LIMITED)
	ErrCommonHubServer              = SDKError(C.GENIEX_ERROR_COMMON_HUB_SERVER)
	ErrLlmTokenizationContextLength = SDKError(C.GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH)
)

// Init must be called before any other SDK function.
func Init() error {
	res := C.geniex_init()
	if res < 0 {
		return SDKError(res)
	}
	return nil
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

type GetRuntimeListOutput struct {
	RuntimeIDs []string
}

func newGetRuntimeListOutputFromCPtr(c *C.geniex_GetPluginListOutput) GetRuntimeListOutput {
	if c == nil {
		return GetRuntimeListOutput{}
	}
	return GetRuntimeListOutput{
		RuntimeIDs: cCharArrayToSlice((**C.char)(unsafe.Pointer(c.plugin_ids)), c.plugin_count),
	}
}

func freeGetRuntimeListOutput(c *C.geniex_GetPluginListOutput) {
	if c == nil {
		return
	}
	mlFreeCCharArray((**C.char)(unsafe.Pointer(c.plugin_ids)), c.plugin_count)
}

func GetRuntimeList() (*GetRuntimeListOutput, error) {
	var cOutput C.geniex_GetPluginListOutput
	res := C.geniex_get_plugin_list(&cOutput)
	if res < 0 {
		return nil, SDKError(res)
	}
	defer freeGetRuntimeListOutput(&cOutput)

	output := newGetRuntimeListOutputFromCPtr(&cOutput)
	return &output, nil
}

type GetComputeUnitListInput struct {
	RuntimeID string
}

func (gculi GetComputeUnitListInput) toCPtr() *C.geniex_GetDeviceListInput {
	cPtr := (*C.geniex_GetDeviceListInput)(cMalloc(C.sizeof_geniex_GetDeviceListInput))
	*cPtr = C.geniex_GetDeviceListInput{plugin_id: cStringIfSet(gculi.RuntimeID)}
	return cPtr
}

func freeGetComputeUnitListInput(cPtr *C.geniex_GetDeviceListInput) {
	if cPtr == nil {
		return
	}
	cFreeIfSet(unsafe.Pointer(cPtr.plugin_id))
	C.free(unsafe.Pointer(cPtr))
}

type ComputeUnit struct {
	ID   string
	Name string
}

type GetComputeUnitListOutput struct {
	ComputeUnits []ComputeUnit
}

func newGetComputeUnitListOutputFromCPtr(c *C.geniex_GetDeviceListOutput) GetComputeUnitListOutput {
	if c == nil {
		return GetComputeUnitListOutput{}
	}
	count := int(c.device_count)
	units := make([]ComputeUnit, count)
	if count > 0 {
		ids := unsafe.Slice(c.device_ids, count)
		names := unsafe.Slice(c.device_names, count)
		for i := range units {
			units[i] = ComputeUnit{
				ID:   C.GoString(ids[i]),
				Name: C.GoString(names[i]),
			}
		}
	}
	return GetComputeUnitListOutput{ComputeUnits: units}
}

func freeGetComputeUnitListOutput(c *C.geniex_GetDeviceListOutput) {
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

func GetComputeUnitList(input GetComputeUnitListInput) (*GetComputeUnitListOutput, error) {
	cInput := input.toCPtr()
	defer freeGetComputeUnitListInput(cInput)

	var cOutput C.geniex_GetDeviceListOutput
	res := C.geniex_get_device_list(cInput, &cOutput)
	if res < 0 {
		return nil, SDKError(res)
	}
	defer freeGetComputeUnitListOutput(&cOutput)

	output := newGetComputeUnitListOutputFromCPtr(&cOutput)
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
