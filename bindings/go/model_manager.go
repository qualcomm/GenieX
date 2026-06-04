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
#include "geniex_model.h"

extern bool go_model_progress(geniex_FileProgress*, int32_t, void*);
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime/cgo"
	"strings"
	"unsafe"
)

// LCOV_EXCL_START

// QuantNA is the precision placeholder the SDK records for non-quantized
// (e.g. QAIRT) models.
const QuantNA = "N/A"

// ModelType mirrors geniex_ModelType.
type ModelType int32

const (
	ModelTypeLLM ModelType = C.GENIEX_MODEL_TYPE_LLM
	ModelTypeVLM ModelType = C.GENIEX_MODEL_TYPE_VLM
)

func (t ModelType) String() string {
	switch t {
	case ModelTypeLLM:
		return "llm"
	case ModelTypeVLM:
		return "vlm"
	default:
		return "unknown"
	}
}

// ParseModelType maps "llm"/"vlm" (case-insensitive) to a ModelType. The
// second return is false for any other value.
func ParseModelType(s string) (ModelType, bool) {
	switch strings.ToLower(s) {
	case "llm":
		return ModelTypeLLM, true
	case "vlm":
		return ModelTypeVLM, true
	default:
		return 0, false
	}
}

// SplitNameQuant splits "name[:quant]" into (name, quant), upper-casing the
// quant. A pasted HuggingFace URL prefix is stripped first so its scheme colon
// isn't mistaken for the quant separator. Bare names are canonicalized by the
// SDK; this only handles the URL strip + ':' split so callers can pass name
// and quant to the FFI separately.
func SplitNameQuant(arg string) (string, string) {
	arg = strings.TrimPrefix(arg, "https://huggingface.co/")
	arg = strings.TrimPrefix(arg, "http://huggingface.co/")
	name, quant, found := strings.Cut(arg, ":")
	if !found || quant == "" {
		return name, ""
	}
	return name, strings.ToUpper(quant)
}

// HubSource mirrors geniex_HubSource.
type HubSource int32

const (
	HubAuto        HubSource = C.GENIEX_HUB_AUTO
	HubHuggingFace HubSource = C.GENIEX_HUB_HUGGINGFACE
	HubModelScope  HubSource = C.GENIEX_HUB_MODELSCOPE
	HubAIHub       HubSource = C.GENIEX_HUB_AIHUB
	HubVolces      HubSource = C.GENIEX_HUB_VOLCES
	HubLocalFS     HubSource = C.GENIEX_HUB_LOCALFS
)

// FileProgress mirrors geniex_FileProgress.
type FileProgress struct {
	FileName        string
	DownloadedBytes int64
	TotalBytes      int64 // -1 if unknown
}

// ModelProgressCallback is invoked periodically during a pull; returning
// false cancels the download.
type ModelProgressCallback func(files []FileProgress) bool

// ModelInit initializes the model manager. dataDir precedence:
// argument → GENIEX_DATADIR env → ~/.cache/geniex. Idempotent: a second
// call returns nil (the SDK reports ALREADY_INITIALIZED, treated as success).
func ModelInit(dataDir string) error {
	cDir := cStringIfSet(dataDir)
	defer cFreeIfSet(unsafe.Pointer(cDir))
	res := C.geniex_model_init(cDir)
	if res != C.GENIEX_SUCCESS && res != C.GENIEX_ERROR_COMMON_ALREADY_INITIALIZED {
		return modelError(res)
	}
	return nil
}

// ModelDeinit releases model-manager resources (a no-op in the SDK today).
func ModelDeinit() error {
	if res := C.geniex_model_deinit(); res != C.GENIEX_SUCCESS {
		return modelError(res)
	}
	return nil
}

// ModelPullInput mirrors geniex_ModelPullInput (minus the ABI gate, which
// ModelPull sets automatically).
type ModelPullInput struct {
	ModelName   string
	Quant       string
	Hub         HubSource
	LocalPath   string
	HfToken     string
	Chipset     string
	DisplayName string
	OnProgress  ModelProgressCallback
	// ModelType selects the model type to pull; nil means auto-detect.
	ModelType *ModelType
}

//export go_model_progress
func go_model_progress(files *C.geniex_FileProgress, count C.int32_t, userData unsafe.Pointer) C.bool {
	if userData == nil {
		return C.bool(true)
	}
	cb, ok := userDataToHandle(userData).Value().(ModelProgressCallback)
	if !ok || cb == nil {
		return C.bool(true)
	}
	entries := unsafe.Slice(files, int(count))
	items := make([]FileProgress, count)
	for i, e := range entries {
		items[i] = FileProgress{
			FileName:        C.GoString(e.file_name),
			DownloadedBytes: int64(e.downloaded_bytes),
			TotalBytes:      int64(e.total_bytes),
		}
	}
	return C.bool(cb(items))
}

// ModelPull downloads a model into the local cache (blocking, resumable).
func ModelPull(input ModelPullInput) error {
	cModelName := cStringIfSet(input.ModelName)
	cQuant := cStringIfSet(input.Quant)
	cLocalPath := cStringIfSet(input.LocalPath)
	cHfToken := cStringIfSet(input.HfToken)
	cChipset := cStringIfSet(input.Chipset)
	cDisplayName := cStringIfSet(input.DisplayName)
	defer func() {
		cFreeIfSet(unsafe.Pointer(cModelName))
		cFreeIfSet(unsafe.Pointer(cQuant))
		cFreeIfSet(unsafe.Pointer(cLocalPath))
		cFreeIfSet(unsafe.Pointer(cHfToken))
		cFreeIfSet(unsafe.Pointer(cChipset))
		cFreeIfSet(unsafe.Pointer(cDisplayName))
	}()

	in := C.geniex_ModelPullInput{
		struct_size:  C.uint32_t(C.sizeof_geniex_ModelPullInput),
		model_name:   cModelName,
		quant:        cQuant,
		hub:          C.geniex_HubSource(input.Hub),
		local_path:   cLocalPath,
		hf_token:     cHfToken,
		chipset:      cChipset,
		display_name: cDisplayName,
		model_type:   C.int32_t(C.GENIEX_MODEL_TYPE_AUTO),
	}
	if input.ModelType != nil {
		in.model_type = C.int32_t(*input.ModelType)
	}

	if input.OnProgress != nil {
		h := cgo.NewHandle(input.OnProgress)
		defer h.Delete()
		in.on_progress = C.geniex_download_progress_cb(C.go_model_progress)
		in.user_data = handleToUserData(h)
	}

	if res := C.geniex_model_pull(&in); res != C.GENIEX_SUCCESS {
		return modelError(res)
	}
	return nil
}

// ModelDetail mirrors geniex_ModelDetail.
type ModelDetail struct {
	Name       string
	ModelName  string
	PluginID   string
	ModelType  ModelType
	TotalSize  int64
	Precisions []string
}

// ModelListDetailed returns cached models with full metadata.
func ModelListDetailed() ([]ModelDetail, error) {
	var out C.geniex_ModelListDetailedOutput
	if res := C.geniex_model_list_detailed(&out); res != C.GENIEX_SUCCESS {
		return nil, modelError(res)
	}
	defer C.geniex_model_list_detailed_free(&out)

	if out.models == nil || out.count == 0 {
		return nil, nil
	}
	models := unsafe.Slice(out.models, int(out.count))
	result := make([]ModelDetail, out.count)
	for i, m := range models {
		result[i] = ModelDetail{
			Name:       C.GoString(m.name),
			ModelName:  C.GoString(m.model_name),
			PluginID:   C.GoString(m.plugin_id),
			ModelType:  ModelType(m.model_type),
			TotalSize:  int64(m.total_size),
			Precisions: cCharArrayToSlice(m.precisions, m.precision_count),
		}
	}
	return result, nil
}

// ModelRemove deletes a cached model ("org/repo" or "org/repo:quant").
func ModelRemove(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	if res := C.geniex_model_remove(cName); res != C.GENIEX_SUCCESS {
		return modelError(res)
	}
	return nil
}

// ModelClean removes all cached models, returning the number removed.
func ModelClean() (int, error) {
	var count C.int32_t
	if res := C.geniex_model_clean(&count); res != C.GENIEX_SUCCESS {
		return 0, modelError(res)
	}
	return int(count), nil
}

// ModelGetType returns the model type of a cached model.
func ModelGetType(name string) (ModelType, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var t C.geniex_ModelType
	if res := C.geniex_model_get_type(cName, &t); res != C.GENIEX_SUCCESS {
		return 0, modelError(res)
	}
	return ModelType(t), nil
}

// ModelSetType overrides the stored model type of a cached model.
func ModelSetType(name string, t ModelType) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	if res := C.geniex_model_set_type(cName, C.geniex_ModelType(t)); res != C.GENIEX_SUCCESS {
		return modelError(res)
	}
	return nil
}

// ModelPaths mirrors geniex_ModelPaths. Optional fields are empty when unset.
type ModelPaths struct {
	ModelPath     string
	MmprojPath    string
	TokenizerPath string
	ModelDir      string
	ModelName     string
	PluginID      string
	ModelType     ModelType
}

// ModelGetPaths resolves "org/repo[:quant]" (or alias) to absolute on-disk paths.
func ModelGetPaths(name string) (*ModelPaths, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var out C.geniex_ModelPaths
	if res := C.geniex_model_get_paths(cName, &out); res != C.GENIEX_SUCCESS {
		return nil, modelError(res)
	}
	defer C.geniex_model_paths_free(&out)
	return &ModelPaths{
		ModelPath:     C.GoString(out.model_path),
		MmprojPath:    C.GoString(out.mmproj_path),
		TokenizerPath: C.GoString(out.tokenizer_path),
		ModelDir:      C.GoString(out.model_dir),
		ModelName:     C.GoString(out.model_name),
		PluginID:      C.GoString(out.plugin_id),
		ModelType:     ModelType(out.model_type),
	}, nil
}

// QuantCandidate mirrors geniex_QuantCandidate.
type QuantCandidate struct {
	Quant     string
	Size      int64
	IsDefault bool
}

// ModelQueryResult mirrors geniex_ModelQueryOutput.
type ModelQueryResult struct {
	ModelName  string
	PluginID   string
	ModelType  ModelType
	Candidates []QuantCandidate
}

// ModelQuery resolves a model's remote candidate quantizations without
// downloading. Reuses ModelPullInput's fields; OnProgress and Quant are ignored.
func ModelQuery(input ModelPullInput) (*ModelQueryResult, error) {
	cModelName := cStringIfSet(input.ModelName)
	cLocalPath := cStringIfSet(input.LocalPath)
	cHfToken := cStringIfSet(input.HfToken)
	cChipset := cStringIfSet(input.Chipset)
	cDisplayName := cStringIfSet(input.DisplayName)
	defer func() {
		cFreeIfSet(unsafe.Pointer(cModelName))
		cFreeIfSet(unsafe.Pointer(cLocalPath))
		cFreeIfSet(unsafe.Pointer(cHfToken))
		cFreeIfSet(unsafe.Pointer(cChipset))
		cFreeIfSet(unsafe.Pointer(cDisplayName))
	}()

	// Query reuses the pull input struct; quant / progress / model_type are
	// ignored by the SDK for a plan-only call.
	in := C.geniex_ModelPullInput{
		struct_size:  C.uint32_t(C.sizeof_geniex_ModelPullInput),
		model_name:   cModelName,
		hub:          C.geniex_HubSource(input.Hub),
		local_path:   cLocalPath,
		hf_token:     cHfToken,
		chipset:      cChipset,
		display_name: cDisplayName,
		model_type:   C.int32_t(C.GENIEX_MODEL_TYPE_AUTO),
	}

	var out C.geniex_ModelQueryOutput
	if res := C.geniex_model_query(&in, &out); res != C.GENIEX_SUCCESS {
		return nil, modelError(res)
	}
	defer C.geniex_model_query_free(&out)

	result := &ModelQueryResult{
		ModelName: C.GoString(out.model_name),
		PluginID:  C.GoString(out.plugin_id),
		ModelType: ModelType(out.model_type),
	}
	if out.candidates != nil && out.candidate_count > 0 {
		cands := unsafe.Slice(out.candidates, int(out.candidate_count))
		result.Candidates = make([]QuantCandidate, out.candidate_count)
		for i, c := range cands {
			result.Candidates[i] = QuantCandidate{
				Quant:     C.GoString(c.quant),
				Size:      int64(c.size),
				IsDefault: bool(c.is_default),
			}
		}
	}
	return result, nil
}

// ModelLastErrorMessage returns the library-owned (do not free) message for the
// last failing geniex_model_* call on this thread, or "" if none.
func ModelLastErrorMessage() string {
	return C.GoString(C.geniex_model_last_error_message())
}

// modelError wraps an FFI error code with the SDK's detailed last-error message
// (e.g. "quantization 'Q2_K' not found for model 'org/repo'") when available,
// while keeping the underlying SDKError matchable via errors.Is / errors.As.
func modelError(res C.int32_t) error {
	err := SDKError(res)
	if msg := ModelLastErrorMessage(); msg != "" {
		return fmt.Errorf("%s: %w", msg, err)
	}
	return err
}

// IsModelNotFound reports whether err is the SDK's "model not cached" code.
func IsModelNotFound(err error) bool {
	var sdkErr SDKError
	return errors.As(err, &sdkErr) && int32(sdkErr) == int32(C.GENIEX_ERROR_COMMON_FILE_NOT_FOUND)
}

// LCOV_EXCL_STOP
